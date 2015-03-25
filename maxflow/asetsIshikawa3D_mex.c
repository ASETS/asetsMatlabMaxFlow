/*  Martin Rajchl, Imperial College London, 2015
 *
 * Re-implementation with of [1]
 *
 * [1] Rajchl M., J. Yuan, E. Ukwatta, and T. Peters (2012).
 *     Fast Interactive Multi-Region Cardiac Segmentation With
 *     Linearly Ordered Labels.
 *     ISBI, 2012. pp.1409–1412.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mex.h>
#include <math.h>
#include <time.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define ABS(x) ( (x) > 0.0 ? x : -(x) )

void runMaxFlow( float *alpha, float *Ct,
        int Nx, int Ny, int Nz, int nLab, int maxIt,
        float errbound, float cc, float steps,
        float *u, float *cvg, int *itNum);

void init(float *Ct, float *pt, float *u, int Nx, int Ny, int Nz, int nLab);

void updateP1(float *gk, float *dv, float *pt, float *u, int Nx, int Ny, int Nz, float cc, int lbl_id);
void updatePX(float *gk, float *bx, int Nx, int Ny, int Nz, float steps, int lbl_id);
void updatePY(float *gk, float *by, int Nx, int Ny, int Nz, float steps, int lbl_id);
void updatePZ(float *gk, float *bz, int Nx, int Ny, int Nz, float steps, int lbl_id);
void projStep(float *bx, float *by, float *bz, float *alpha, float *gk, int Nx, int Ny, int Nz, int lbl_id);
void updateBX(float *bx, float *gk, int Nx, int Ny, int Nz, int lbl_id);
void updateBY(float *by, float *gk, int Nx, int Ny, int Nz, int lbl_id);
void updateBZ(float *bz, float *gk, int Nx, int Ny, int Nz, int lbl_id);
void updateDIV(float *dv, float *bx, float *by, float *bz, int Nx, int Ny, int Nz, float cc, int lbl_id);
void updateTopLayerPT(float *gk, float *dv, float *pt, float *u, float *Ct, int Nx,  int Ny, int Nz, float cc, int lbl_id);
void updateMidLayerPT(float *gk, float *dv, float *pt, float *u, float *Ct, int Nx,  int Ny, int Nz, float cc, int lbl_id);
void updateBottomLayerPT(float *gk, float *dv, float *pt, float *u, float *Ct, int Nx,  int Ny, int Nz, float cc, int lbl_id);
float updateU(float *dv, float *pt, float *u, int Nx, int Ny, int Nz, int nLab, float cc);

extern void mexFunction(int iNbOut, mxArray *pmxOut[],
        int iNbIn, const mxArray *pmxIn[])
{
    /* vars in */
    float *alpha, *Ct, *pars;
    int Nx, Ny, Nz, nLab, maxIt;
    float errBound, cc, steps;
    
    /* vars out */
    float *u, *cvg;
    int *itNum;
    double *runTime;
    int nDim;
    int dim[4];
    
    /* others */
    time_t  start_time, end_time;
    
    /* compute Max-Flow */
    start_time = clock();
    
    /* Inputs */
    Ct = mxGetData(pmxIn[0]);             /* bound of sink flows */
    alpha = mxGetData(pmxIn[1]);        /* penalty parameters */
    pars = mxGetData(pmxIn[2]);  /* Vector of parameters */
    
    /*
     *pfVecParameters Setting
     * [0] : number of columns
     * [1] : number of rows
     * [2] : number of rows
     * [3] : number of labels
     * [4] : the maximum iteration number
     * [5] : error criterion
     * [6] : cc for the step-size of ALM
     * [7] : steps for the step-size of projected-gradient of p
     */
    
    /* pars */
    Ny = (int) pars[0];
    Nx = (int) pars[1];
    Nz = (int) pars[2];
    nLab = (int) pars[3];
    maxIt = (int) pars[4];
    errBound = (float) pars[5];
    cc = (float) pars[6];
    steps = (float) pars[7];
    
    /* Outputs */
    /* outputs the computed u(x)  */
    dim[0] = Ny;
    dim[1] = Nx;
    dim[2] = Nz;
    dim[3] = nLab-1;
    nDim = 4;
    
    pmxOut[0] = mxCreateNumericArray(nDim,(const int*)dim,mxSINGLE_CLASS,mxREAL);
    u = mxGetData(pmxOut[0]);
    
    /* outputs the convergence rate  */
    nDim = 2;
    dim[0] = maxIt;
    dim[1] = 1;
    pmxOut[1] = mxCreateNumericArray(nDim,(const int*)dim,mxSINGLE_CLASS,mxREAL);
    cvg = mxGetData(pmxOut[1]);
    
    /* outputs the iteration number  */
    nDim = 2;
    dim[0] = 1;
    dim[1] = 1;
    pmxOut[2] = mxCreateNumericArray(nDim,(const int*)dim,mxUINT16_CLASS,mxREAL);
    itNum = mxGetData(pmxOut[2]);
    
    /* outputs the computation time  */
    nDim = 2;
    dim[0] = 1;
    dim[1] = 1;
    pmxOut[3] = mxCreateNumericArray(nDim,(const int*)dim,mxSINGLE_CLASS,mxREAL);
    runTime = mxGetData(pmxOut[3]);
    
    
    /* compute Max-Flow */
    start_time = clock();
    
    
    runMaxFlow(alpha, Ct,
            Nx, Ny, Nz, nLab, maxIt, errBound, cc, steps,
            u, cvg, itNum);
    
    
    end_time = clock();
    
    runTime[0] = difftime(end_time, start_time)/1000000;
    
    mexPrintf("ishikawa model max flow 3D: number of iterations = %i; time = %.4f sec\n",itNum[0],runTime[0]);
    
}


void runMaxFlow( float *alpha, float *Ct,
        int Nx, int Ny, int Nz, int nLab, int maxIt,
        float errBound, float cc, float steps,
        float *u, float *cvg, int *itNum){
    
    
    float   *bx, *by, *bz, *dv, *gk, *pt;
    int i;
    float total_err;
    
    /* alloc buffers */
    bx = (float *) calloc( (unsigned)((Nx+1)*Ny*Nz*(nLab-1)), sizeof(float) );
    by = (float *) calloc( (unsigned)(Nx*(Ny+1)*Nz*(nLab-1)), sizeof(float) );
    bz = (float *) calloc( (unsigned)(Nx*Ny*(Nz+1)*(nLab-1)), sizeof(float) );
    dv = (float *) calloc( (unsigned)(Nx*Ny*Nz*(nLab-1)), sizeof(float) );
    gk = (float *) calloc( (unsigned)(Nx*Ny*Nz), sizeof(float) );
    pt = (float *) calloc( (unsigned)(Nx*Ny*Nz*nLab), sizeof(float) );
    if (!(bx || by || bz || dv || gk || pt))
        mexPrintf("malloc error.\n");
    
    init(Ct, pt, u, Nx, Ny, Nz, nLab);
    
    /* iterate */
    i = 0;
    for (i = 0; i < maxIt; i++){
        
        int k = 0;
        for (k = 0; k < (nLab-1); k++){
            
            /* update the spatial flow field p(x,lbl) = (bx(x,lbl),by(x,lbl)) */
            updateP1(gk, dv, pt, u, Nx, Ny, Nz, cc, k);
            updatePX(gk, bx, Nx, Ny, Nz, steps, k);
            updatePY(gk, by, Nx, Ny, Nz, steps, k);
            updatePZ(gk, bz, Nx, Ny, Nz, steps, k);
            
            /* projection step to make |p(x,i)| <= alpha(x,lbl)*/
            projStep(bx, by, bz, alpha, gk, Nx, Ny, Nz, k);
            
            
            /* update the component bx, by */
            updateBX(bx, gk, Nx, Ny, Nz, k);
            updateBY(by, gk, Nx, Ny, Nz, k);
            updateBZ(bz, gk, Nx, Ny, Nz, k);
            
            /* div(x,lbl)  */
            updateDIV(dv, bx, by, bz, Nx, Ny, Nz, cc, k);
        }
        
        for (k = 0; k < nLab; k++){
            
            if (k == 0){
                updateBottomLayerPT(gk, dv, pt, u, Ct, Nx, Ny, Nz, cc, k);
            }
            else if ((k >= 1) && ( k < (nLab-1) )){
                updateMidLayerPT(gk, dv, pt, u, Ct, Nx, Ny, Nz, cc, k);
            }
            else{
                updateTopLayerPT(gk, dv, pt, u, Ct, Nx, Ny, Nz, cc, k);
            }
            
        }
        
        
        /* multiplier/labeling functions u(x,lbl) */
        total_err = updateU(dv, pt, u, Nx, Ny, Nz, nLab, cc);
        
        
        /* evaluate the convergence error */
        cvg[i] = total_err / (float)(Nx*Ny*Nz*(nLab-1));
        /* mexPrintf("it= %d, cvg = %f\n", i,cvg[i] ); */
        
        /* check if converged */
        if (cvg[i] <= errBound)
            break;
        
    }
    /* update iteration number */
    itNum[0] = i;
    
    /* free mem */
    free( (float *) bx );
    free( (float *) by );
    free( (float *) bz );
    free( (float *) dv );
    free( (float *) gk );
    free( (float *) pt );
    
}

void init(float *Ct, float *pt, float *u, int Nx, int Ny, int Nz, int nLab){
    /* init */
    int x, y, z;
    
    for (z=0; z < Nz; z++){
        for (x=0; x < Nx; x++){
            for (y=0; y < Ny; y++){
                
                int g_idx = z*Nx*Ny + x*Ny + y;
                
                float minVal = 1e30;
                int minId = 1e9;
                
                /* find the minimum Ct(x,l) */
                int l;
                for (l = 0; l < nLab; l++){
                    int l_idx = g_idx + l*Nx*Ny*Nz;
                    
                    if ( minVal > Ct[l_idx] ){
                        minVal = Ct[l_idx];
                        minId = l;
                    }
                }
                
                /* init pt, u */
                for (l = 0; l < nLab-1; l++){
                    int l_idx = g_idx + l*Nx*Ny*Nz;
                    
                    pt[l_idx] = minVal;
                    
                    if (l >= minId)
                        u[l_idx] = 0.0f;
                    else
                        u[l_idx] = 1.0f;
                }
                
            }
        }
        
    }
}

void updateP1(float *gk, float *dv, float *pt, float *u, int Nx, int Ny, int Nz, float cc, int lbl_id){
    
    int x = 0;
    int y = 0;
    int z = 0;
    
    int graphSize = Nx*Ny*Nz;
    
    for (z=0; z < Nz; z++){
        for (x=0; x < Nx; x++){
            for (y=0; y < Ny; y++){
                
                int g_idx = z*Nx*Ny + x*Ny + y;
                int l_idx = g_idx + lbl_id*graphSize;
                
                gk[g_idx] = dv[l_idx] - (pt[l_idx]
                        - pt[l_idx+graphSize] + u[l_idx]/cc);
                
            }
        }
    }
}


void updatePX(float *gk, float *bx, int Nx, int Ny, int Nz, float steps, int lbl_id){
    
    int x = 0;
    int y = 0;
    int z = 0;
    int graphSize = Nx*Ny*Nz;
    
    for (z=0; z < Nz; z++){
        for (x=1; x < Nx; x++){
            for (y=0; y < Ny; y++){
                
                
                int g_idx = z*Nx*Ny + x*Ny + y;
                int l_idx = g_idx + lbl_id*graphSize;
                
                bx[l_idx] = steps*(gk[g_idx] - gk[g_idx-Ny]) + bx[l_idx];
                
            }
        }
    }
    
}

void updatePY(float *gk, float *by, int Nx, int Ny, int Nz, float steps, int lbl_id){
    
    int x = 0;
    int y = 0;
    int z = 0;
    int graphSize = Nx*Ny*Nz;
    
    for (z=0; z < Nz; z++){
        for(x = 0; x < Nx; x ++){
            for(y = 1; y < Ny; y++){
                
                
                int g_idx = z*Nx*Ny + x*Ny + y;
                int l_idx = g_idx + lbl_id*graphSize;
                
                by[l_idx] = steps*(gk[g_idx] - gk[g_idx-1]) + by[l_idx];
                
            }
        }
    }
}

void updatePZ(float *gk, float *bz, int Nx, int Ny, int Nz, float steps, int lbl_id){
    
    int x = 0;
    int y = 0;
    int z = 0;
    int graphSize = Nx*Ny*Nz;
    
    for (z=1; z < Nz; z++){
        for(x = 0; x < Nx; x ++){
            for(y = 0; y < Ny; y++){
                
                
                int g_idx = z*Nx*Ny + x*Ny + y;
                int l_idx = g_idx + lbl_id*graphSize;
                
                bz[l_idx] = steps*(gk[g_idx] - gk[g_idx-(Nx*Ny)]) + bz[l_idx];
            }
        }
    }
}
void projStep(float *bx, float *by, float *bz, float *alpha, float *gk, int Nx, int Ny, int Nz, int lbl_id){
    
    float fpt;
    int x = 0;
    int y = 0;
    int z = 0;
    int graphSize = Nx*Ny*Nz;
    
    for (z=0; z < Nz; z++){
        for (x=0; x< Nx; x++){
            for (y=0; y< Ny; y++){
                
                
                int g_idx = z*Nx*Ny + x*Ny + y;
                int l_idx = g_idx + lbl_id*graphSize;
                
                if( alpha[l_idx] <= 0 ){
                    mexErrMsgTxt("alpha(x,l) must be positive. Exiting...");
                }
                
                fpt = sqrt((pow(bx[l_idx+Ny],2) + pow(bx[l_idx],2) +
                        pow(by[l_idx+1],2) + pow(by[l_idx],2) +
                        pow(bz[l_idx+(Nx*Ny)],2) + pow(bz[l_idx],2)
                        )*0.5);
                
                if (fpt > alpha[l_idx])
                    fpt = fpt / alpha[l_idx];
                else
                    fpt = 1;
                
                gk[g_idx] = 1/fpt;
                
            }
        }
    }
}


void updateBX(float *bx, float *gk, int Nx, int Ny, int Nz, int lbl_id){
    
    int g_idx;
    int l_idx;
    int x = 0;
    int y = 0;
    int z = 0;
    int graphSize = Nx*Ny*Nz;
    
    for (z=0; z < Nz; z++){
        for (x=1; x< Nx; x++){
            for (y=0; y< Ny; y++){
                
                
                g_idx = z*Nx*Ny + x*Ny + y;
                l_idx = g_idx + lbl_id*graphSize;
                
                bx[l_idx] = (gk[g_idx] + gk[g_idx-Ny])
                *0.5*bx[l_idx];
            }
        }
    }
}

void updateBY(float *by, float *gk, int Nx, int Ny, int Nz, int lbl_id){
    
    int g_idx;
    int l_idx;
    int x = 0;
    int y = 0;
    int z = 0;
    int graphSize = Nx*Ny*Nz;
    
    for (z=0; z < Nz; z++){
        for (x=0; x<Nx; x++){
            for (y=1; y< Ny; y++){
                
                
                g_idx = z*Nx*Ny + x*Ny + y;
                l_idx = g_idx + lbl_id*graphSize;
                
                by[l_idx] = 0.5*(gk[g_idx-1] + gk[g_idx])
                *by[l_idx];
            }
        }
    }
}
void updateBZ(float *bz, float *gk, int Nx, int Ny, int Nz, int lbl_id){
    
    int g_idx;
    int l_idx;
    int x = 0;
    int y = 0;
    int z = 0;
    int graphSize = Nx*Ny*Nz;
    
    for (z=1; z < Nz; z++){
        for (x=0; x<Nx; x++){
            for (y=0; y< Ny; y++){
                
                
                g_idx = z*Nx*Ny + x*Ny + y;
                l_idx = g_idx + lbl_id*graphSize;
                
                bz[l_idx] = 0.5*(gk[g_idx-(Nx*Ny)] + gk[g_idx])
                *bz[l_idx];
            }
        }
    }
}

void updateDIV(float *dv, float *bx, float *by, float *bz, int Nx, int Ny, int Nz, float cc, int lbl_id){
    
    float fpt;
    int g_idx;
    int l_idx;
    int x = 0;
    int y = 0;
    int z = 0;
    int graphSize = Nx*Ny*Nz;
    
    for (z=0; z < Nz; z++){
        for (x=0; x< Nx; x++){
            for (y=0; y< Ny; y++){
                
                g_idx = z*Nx*Ny + x*Ny + y;
                l_idx = g_idx + lbl_id*graphSize;
                
                /* update the divergence field dv(x,l)  */
                dv[l_idx] = by[l_idx+1] - by[l_idx]
                        + bx[l_idx+Ny] - bx[l_idx]
                        + bz[l_idx+(Nx*Ny)] - bz[l_idx];
                
            }
        }
    }
    
}

void updateBottomLayerPT(float *gk, float *dv, float *pt, float *u, float *Ct, int Nx,  int Ny, int Nz, float cc, int lbl_id){
    
    float fpt = 0;
    int g_idx;
    int l_idx;
    int x = 0;
    int y = 0;
    int z = 0;
    int graphSize = Nx*Ny*Nz;
    
    for (z=0; z < Nz; z++){
        for (x=0; x< Nx; x++){
            for (y=0; y< Ny; y++){
                
                g_idx = z*Nx*Ny + x*Ny + y;
                l_idx = g_idx + lbl_id*Nx*Ny*Nz;
                
                /* update pt(x,l)  */
                fpt = dv[l_idx] + pt[l_idx+graphSize] - u[l_idx]/cc + 1/cc;
                pt[l_idx] = MIN(fpt, Ct[l_idx]);
                
            }
        }
    }
}

void updateMidLayerPT(float *gk, float *dv, float *pt, float *u, float *Ct, int Nx,  int Ny, int Nz, float cc, int lbl_id){
    
    float fpt = 0;
    int g_idx;
    int l_idx;
    int x = 0;
    int y = 0;
    int z = 0;
    int graphSize = Nx*Ny*Nz;
    
    for (z=0; z < Nz; z++){
        for (x=0; x< Nx; x++){
            for (y=0; y< Ny; y++){
                
                g_idx = z*Nx*Ny + x*Ny + y;
                l_idx = g_idx + lbl_id*Nx*Ny*Nz;
                
                /* update pt(x,l)  */
                fpt =  - dv[l_idx-graphSize] + pt[l_idx-graphSize] + u[l_idx-graphSize]/cc;
                fpt +=   dv[l_idx] + pt[l_idx+graphSize] - u[l_idx]/cc;
                fpt /= 2.0f;
                
                pt[l_idx] = MIN(fpt, Ct[l_idx]);
            }
        }
    }
}

void updateTopLayerPT(float *gk, float *dv, float *pt, float *u, float *Ct, int Nx,  int Ny, int Nz, float cc, int lbl_id){
    
    float fpt = 0;
    int g_idx;
    int l_idx;
    int x = 0;
    int y = 0;
    int z = 0;
    int graphSize = Nx*Ny*Nz;
    
    for (z=0; z < Nz; z++){
        for (x=0; x< Nx; x++){
            for (y=0; y< Ny; y++){
                
                g_idx = z*Nx*Ny + x*Ny + y;
                l_idx = g_idx + lbl_id*Nx*Ny*Nz;
                
                /* update pt(x,l)  */
                fpt = - dv[l_idx-graphSize] + pt[l_idx-graphSize] + u[l_idx-graphSize]/cc;
                pt[l_idx] = MIN(fpt, Ct[l_idx]);
                
            }
        }
    }
}

float updateU(float *dv, float *pt, float *u, int Nx, int Ny, int Nz, int nLab, float cc){
    
    float fpt;
    int l;
    int g_idx;
    int x = 0;
    int y = 0;
    int z = 0;
    int graphSize = Nx*Ny*Nz;
    
    float erru = 0;
    
    for (z=0; z < Nz; z++){
        for (x=0; x< Nx; x++){
            for (y=0; y< Ny; y++){
                
                g_idx = z*Nx*Ny + x*Ny + y;
                
                fpt = 0;
                
                /* update the multipliers u(x,l) */
                for (l = 0; l < (nLab-1); l++){
                    fpt = cc*(dv[g_idx+l*graphSize] + pt[g_idx+((l+1)*graphSize)] - pt[g_idx+l*graphSize]);
                    
                    u[g_idx+l*graphSize] -= fpt;
                    erru += ABS(fpt);
                    
                }
            }
        }
    }
    return erru;
}


