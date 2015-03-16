/*  Martin Rajchl, Imperial College London, 2015
 *
 * Re-implementation with of [1] with a horizontal model
 * according to [2]
 *
 * [1] Yuan, J.; Bae, E.; Tai, X.-C.; Boykov, Y.
 * A Continuous Max-Flow Approach to Potts Model
 * ECCV, 2010
 *
 * [2] Baxter, JSH.; Rajchl, M.; Yuan, J.; Peters, TM.
 * A Continuous Max-Flow Approach to General
 * Hierarchical Multi-Labelling Problems
 * arXiv preprint arXiv:1404.0336
 */

#include <stdio.h>
#include <stdlib.h>
#include <mex.h>
#include <math.h>
#include <time.h>

void runMaxFlow( float *alpha, float *Ct,
        int Nx, int Ny, int Nz, int nLab, int maxIt,
        float errbound, float cc, float steps,
        float *u, float *cvg, int *itNum);

void init();

void updateP1(float *gk, float *dv, float *ps, float *pt, float *u, int Nx, int Ny, int Nz, float cc, int lbl_id);
void updatePX(float *gk, float *bx, int Nx, int Ny, int Nz, float steps, int lbl_id);
void updatePY(float *gk, float *by, int Nx, int Ny, int Nz, float steps, int lbl_id);
void updatePZ(float *gk, float *bz, int Nx, int Ny, int Nz, float steps, int lbl_id);
void projStep(float *bx, float *by, float *bz, float *alpha, float *gk, int Nx, int Ny, int Nz, int lbl_id);
void updateBX(float *bx, float *gk, int Nx, int Ny, int Nz, int lbl_id);
void updateBY(float *by, float *gk, int Nx, int Ny, int Nz, int lbl_id);
void updateBZ(float *bz, float *gk, int Nx, int Ny, int Nz, int lbl_id);
void updatePTDIV(float *dv, float *bx, float *by, float *bz, float *ps, float *u, float *pt, float *Ct, int Nx, int Ny, int Nz, float cc, int lbl_id);
float updatePSU(float *dv, float *pt, float *u, float *ps, int Nx, int Ny, int Nz, int nLab, float cc);

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
    
    int dim[4];
    int nDim;
    
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
     * [2] : number of slices
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
    dim[3] = nLab;
    nDim = 4;
        
    pmxOut[0] = mxCreateNumericArray(nDim,(const int*)dim,mxSINGLE_CLASS,mxREAL);
    u = mxGetData(pmxOut[0]);
    
    /* outputs the convergence rate  */
    nDim = 2;
    dim[0] = 1;
    dim[1] = maxIt;
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
    mexPrintf("\n nIt = %i\n",itNum[0]);
    
    mexPrintf("runtime = %.4f sec\n", runTime[0]);
    
}


void runMaxFlow( float *alpha, float *Ct,
        int Nx, int Ny, int Nz, int nLab, int maxIt,
        float errBound, float cc, float steps,
        float *u, float *cvg, int *itNum){
    
    
    float   *bx, *by, *bz, *dv, *gk, *ps, *pt;
    int i;
    float total_err;
    
    /* alloc buffers */
    bx = (float *) calloc( (unsigned)((Nx+1)*Ny*Nz*nLab), sizeof(float) );
    by = (float *) calloc( (unsigned)(Nx*(Ny+1)*Nz*nLab), sizeof(float) );
    bz = (float *) calloc( (unsigned)(Nx*Ny*(Nz+1)*nLab), sizeof(float) );
    dv = (float *) calloc( (unsigned)(Nx*Ny*Nz*nLab), sizeof(float) );
    gk = (float *) calloc( (unsigned)(Nx*Ny*Nz), sizeof(float) );
    ps = (float *) calloc( (unsigned)(Nx*Ny*Nz), sizeof(float) );
    pt = (float *) calloc( (unsigned)(Nx*Ny*Nz*nLab), sizeof(float) );
    if (!(bx || by || bz || dv || gk || ps || pt))
        mexPrintf("malloc error.\n");
    
    init();
    
    
    /* iterate */
    i = 0;
    for (i = 0; i < maxIt; i++){
        
        int k = 0;
        for (k = 0; k < nLab; k++){
            
            /* update the spatial flow field p(x,lbl) = (bx(x,lbl),by(x,lbl)) */
            updateP1(gk, dv, ps, pt, u, Nx, Ny, Nz, cc, k);
            updatePX(gk, bx, Nx, Ny, Nz, steps, k);
            updatePY(gk, by, Nx, Ny, Nz, steps, k);
            updatePZ(gk, bz, Nx, Ny, Nz, steps, k);
            
            /* projection step to make |p(x,i)| <= alpha(x,lbl)*/
            projStep(bx, by, bz, alpha, gk, Nx, Ny, Nz, k);
                        
            /* update the component bx, by */
            updateBX(bx, gk, Nx, Ny, Nz, k);
            updateBY(by, gk, Nx, Ny, Nz, k);
            updateBZ(bz, gk, Nx, Ny, Nz, k);
            
            /* update the sink flow field pt(x,lbl) pd and div(x,lbl)  */
            updatePTDIV(dv, bx, by, bz, ps, u, pt, Ct, Nx, Ny, Nz, cc, k);
        }
        
        /* update ps(x) and the multiplier/labeling functions u(x,lbl) */
        total_err = updatePSU(dv, pt, u, ps, Nx, Ny, Nz, nLab, cc);
        
        /* to be implemented */
        /* evaluate the convergence error */
        /*cvg[i] = total_err / (float)(Nx*Ny*Nz*nLab); */
        /*mexPrintf("it= %d, cvg = %f\n", i,cvg[i] ); */
        
        /*
         * if (cvg[i] <= errBound)
         * break;
         */
        
    }
    /* update iteration number */
    itNum[0] = i;
    
    /* free mem */
    free( (float *) bx );
    free( (float *) by );
    free( (float *) bz );
    free( (float *) dv );
    free( (float *) gk );
    free( (float *) ps );
    free( (float *) pt );
    
}

/* to be implemented */
void init(){
    /* init */
    /*
     * for (x=0; x < Nx; x++){
     * for (y=0; y < Ny; y++){
     *
     * idx = x + (y*Nx);
     * graphSz = Nx*Ny;
     *
     * for (k = 0; k < nLab; k++){
     * float tmp = 10-7;
     * tmp = min(Ct[idx+k*graphSz], tmp);
     * ps[idx] = tmp;
     *
     * pt[idx+k*graphSz] = ;
     * }
     *
     * }
     * }
     */
}

void updateP1(float *gk, float *dv, float *ps, float *pt, float *u, int Nx, int Ny, int Nz, float cc, int lbl_id){
    
    int x = 0;
    int y = 0;
    int z = 0;
    
    for (z=0; z < Nz; z++){
        for (x=0; x < Nx; x++){
            for (y=0; y < Ny; y++){
                    
                int g_idx = z*Nx*Ny + x*Ny + y;
                int l_idx = g_idx + lbl_id*Nx*Ny*Nz;
                
                gk[g_idx] = dv[l_idx] - (ps[g_idx]
                        - pt[l_idx] + u[l_idx]/cc);
               
            }
        }
    }
}

void updatePX(float *gk, float *bx, int Nx, int Ny, int Nz, float steps, int lbl_id){
    
    int x = 0;
    int y = 0;
    int z = 0;
    for (z=0; z < Nz; z++){
        for (x=1; x < Nx; x++){
            for (y=0; y < Ny; y++){
                
                
                int g_idx = z*Nx*Ny + x*Ny + y;
                int l_idx = g_idx + lbl_id*Nx*Ny*Nz;
                
                bx[l_idx] = steps*(gk[g_idx] - gk[g_idx-Ny]) + bx[l_idx];
                
            }
        }
    }
    
}

void updatePY(float *gk, float *by, int Nx, int Ny, int Nz, float steps, int lbl_id){
    
    int x = 0;
    int y = 0;
    int z = 0;
    for (z=0; z < Nz; z++){
        for(x = 0; x < Nx; x ++){
            for(y = 1; y < Ny; y++){
                
                
                int g_idx = z*Nx*Ny + x*Ny + y;
                int l_idx = g_idx + lbl_id*Nx*Ny*Nz;
                
                by[l_idx] = steps*(gk[g_idx] - gk[g_idx-1]) + by[l_idx];
                
            }
        }
    }
     }

void updatePZ(float *gk, float *bz, int Nx, int Ny, int Nz, float steps, int lbl_id){
    
    int x = 0;
    int y = 0;
    int z = 0;
    for (z=1; z < Nz; z++){
        for(x = 0; x < Nx; x ++){
            for(y = 0; y < Ny; y++){
                
                
                int g_idx = z*Nx*Ny + x*Ny + y;
                int l_idx = g_idx + lbl_id*Nx*Ny*Nz;
                
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
    for (z=0; z < Nz; z++){
        for (x=0; x< Nx; x++){
            for (y=0; y< Ny; y++){
                
                
                int g_idx = z*Nx*Ny + x*Ny + y;
                int l_idx = g_idx + lbl_id*Nx*Ny*Nz;
                
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
    for (z=0; z < Nz; z++){
        for (x=1; x< Nx; x++){
            for (y=0; y< Ny; y++){
                
                
                g_idx = z*Nx*Ny + x*Ny + y;
                l_idx = g_idx + lbl_id*Nx*Ny*Nz;
                
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
    for (z=0; z < Nz; z++){
        for (x=0; x<Nx; x++){
            for (y=1; y< Ny; y++){
                
                
                g_idx = z*Nx*Ny + x*Ny + y;
                l_idx = g_idx + lbl_id*Nx*Ny*Nz;
                
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
    for (z=1; z < Nz; z++){
        for (x=0; x<Nx; x++){
            for (y=0; y< Ny; y++){
                
                
                g_idx = z*Nx*Ny + x*Ny + y;
                l_idx = g_idx + lbl_id*Nx*Ny*Nz;
                
                bz[l_idx] = 0.5*(gk[g_idx-(Nx*Ny)] + gk[g_idx])
                *bz[l_idx];
            }
        }
    }
}
void updatePTDIV(float *dv, float *bx, float *by, float *bz, float *ps, float *u, float *pt, float *Ct, int Nx, int Ny, int Nz, float cc, int lbl_id){
    
    float fpt;
    int g_idx;
    int l_idx;
    int x = 0;
    int y = 0;
    int z = 0;
    for (z=0; z < Nz; z++){
        for (x=0; x< Nx; x++){
            for (y=0; y< Ny; y++){
                
                
                g_idx = z*Nx*Ny + x*Ny + y;
                l_idx = g_idx + lbl_id*Nx*Ny*Nz;
                
                /* update the divergence field dv(x,l)  */
                dv[l_idx] = by[l_idx+1] - by[l_idx] +
                        bx[l_idx+Ny] - bx[l_idx] +
                        bz[l_idx+(Nx*Ny)] - bz[l_idx];
                
                fpt = ps[g_idx] + u[l_idx]/cc - dv[l_idx];
                
                if (fpt < Ct[l_idx])
                    pt[l_idx] = fpt;
                else
                    pt[l_idx] = Ct[l_idx];
                
            }
        }
    }
}


float updatePSU(float *dv, float *pt, float *u, float *ps, int Nx, int Ny, int Nz, int nLab, float cc){
    
    int g_idx;
    int l_idx;
    int l;
    float fpt;
    float* ft = malloc(sizeof(float)*nLab);
    int x = 0;
    int y = 0;
    int z = 0;
    
    float erru = 0;
    
    for (z=0; z < Nz; z++){
        for (x=0; x< Nx; x++){
            for (y=0; y< Ny; y++){
                
                g_idx = z*Nx*Ny + x*Ny + y;
                
                fpt = 0;
                
                for (l = 0; l < nLab; l++){
                    l_idx = g_idx + l*(Nx*Ny*Nz);
                    
                    ft[l] = dv[l_idx] + pt[l_idx];
                    
                    fpt += (ft[l] - u[l_idx]/cc);
                    
                }
                
                ps[g_idx] = fpt/nLab + 1/(cc*nLab);
                
                /* update the multipliers u(x,l) */
                for (l = 0; l < nLab; l++){
                    l_idx = g_idx + l*(Nx*Ny*Nz);
                    
                    fpt = cc*(ft[l] - ps[g_idx]);
                    u[l_idx] -= fpt;
                    erru += abs(fpt);
                    
                }
                
            }
        }
    }
    free(ft);
    return erru;
}


