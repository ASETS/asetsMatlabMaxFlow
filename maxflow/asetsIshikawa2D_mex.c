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

void runMaxFlow( float *alpha, float *Ct,
        int Nx, int Ny, int nLab, int maxIt,
        float errbound, float cc, float steps,
        float *u, float *cvg, int *itNum);

void init();

void updateP1(float *gk, float *dv, float *pt, float *u, int Nx, int Ny, float cc, int lbl_id);
void updatePX(float *gk, float *bx, int Nx, int Ny, float steps, int lbl_id);
void updatePY(float *gk, float *by, int Nx, int Ny, float steps, int lbl_id);
void projStep(float *bx, float *by, float *alpha, float *gk, int Nx, int Ny, int lbl_id);
void updateBX(float *bx, float *gk, int Nx, int Ny, int lbl_id);
void updateBY(float *by, float *gk, int Nx, int Ny, int lbl_id);
void updateDIV(float *dv, float *bx, float *by, int Nx, int Ny, float cc, int lbl_id);
void updateTopLayerPT(float *gk, float *dv, float *pt, float *u, float *Ct, int Nx,  int Ny, float cc, int lbl_id);
void updateMidLayerPT(float *gk, float *dv, float *pt, float *u, float *Ct, int Nx,  int Ny, float cc, int lbl_id);
void updateBottomLayerPT(float *gk, float *dv, float *pt, float *u, float *Ct, int Nx,  int Ny, float cc, int lbl_id);
float updateU(float *dv, float *pt, float *u, int Nx, int Ny, int nLab, float cc);

extern void mexFunction(int iNbOut, mxArray *pmxOut[],
        int iNbIn, const mxArray *pmxIn[])
{
    /* vars in */
    float *alpha, *Ct, *pars;
    int Nx, Ny, nLab, maxIt;
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
     * [2] : number of labels
     * [3] : the maximum iteration number
     * [4] : error criterion
     * [5] : cc for the step-size of ALM
     * [6] : steps for the step-size of projected-gradient of p
     */
    
    /* pars */
    Ny = (int) pars[0];
    Nx = (int) pars[1];
    nLab = (int) pars[2];
    maxIt = (int) pars[3];
    errBound = (float) pars[4];
    cc = (float) pars[5];
    steps = (float) pars[6];
    
    /* Outputs */
    /* outputs the computed u(x)  */
    dim[0] = Ny;
    dim[1] = Nx;
    dim[2] = nLab-1;
    nDim = 3;
    
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
            Nx, Ny, nLab, maxIt, errBound, cc, steps,
            u, cvg, itNum);
    
    
    end_time = clock();
    
    runTime[0] = difftime(end_time, start_time)/1000000;

    mexPrintf("ishikawa model max flow 2D: number of iterations = %i; time = %.4f sec\n",itNum[0],runTime[0]);
    
}


void runMaxFlow( float *alpha, float *Ct,
        int Nx, int Ny, int nLab, int maxIt,
        float errBound, float cc, float steps,
        float *u, float *cvg, int *itNum){
    
    
    float   *bx, *by, *dv, *gk, *pt;
    int i;
    float total_err;
    
    /* alloc buffers */
    bx = (float *) calloc( (unsigned)((Nx+1)*Ny*(nLab-1)), sizeof(float) );
    by = (float *) calloc( (unsigned)(Nx*(Ny+1)*(nLab-1)), sizeof(float) );
    dv = (float *) calloc( (unsigned)(Nx*Ny*(nLab-1)), sizeof(float) );
    gk = (float *) calloc( (unsigned)(Nx*Ny), sizeof(float) );
    pt = (float *) calloc( (unsigned)(Nx*Ny*nLab), sizeof(float) );
    if (!(bx || by || dv || gk || pt))
        mexPrintf("malloc error.\n");
    
    init();
    
    /* iterate */
    i = 0;
    for (i = 0; i < maxIt; i++){
                
        int k = 0;
        for (k = 0; k < (nLab-1); k++){
            
            /* update the spatial flow field p(x,lbl) = (bx(x,lbl),by(x,lbl)) */
            updateP1(gk, dv, pt, u, Nx, Ny, cc, k);
            updatePX(gk, bx, Nx, Ny, steps, k);
            updatePY(gk, by, Nx, Ny, steps, k);
        
            /* projection step to make |p(x,i)| <= alpha(x,lbl)*/
            projStep(bx, by, alpha, gk, Nx, Ny, k);
        
            
            /* update the component bx, by */
            updateBX(bx, gk, Nx, Ny, k);
            updateBY(by, gk, Nx, Ny, k);
        
            
            /* div(x,lbl)  */
            updateDIV(dv, bx, by, Nx, Ny, cc, k);
        }
        
        for (k = 0; k < nLab; k++){
            
            if (k == 0){
                updateBottomLayerPT(gk, dv, pt, u, Ct, Nx, Ny, cc, k);    
            }
            else if ((k >= 1) && ( k < (nLab-1) )){
                updateMidLayerPT(gk, dv, pt, u, Ct, Nx, Ny, cc, k);    
            }
            else{
                updateTopLayerPT(gk, dv, pt, u, Ct, Nx, Ny, cc, k);    
            }
            
        }
        

        /* multiplier/labeling functions u(x,lbl) */
        total_err = updateU(dv, pt, u, Nx, Ny, nLab, cc);
        
        /* evaluate the convergence error */
        cvg[i] = total_err / (float)(Nx*Ny*(nLab-1)); 
        /*mexPrintf("it= %d, cvg = %f\n", i,cvg[i] ); */
        
        /* check if converged */
        if (cvg[i] < errBound)
            break;
        
    }
    /* update iteration number */
    itNum[0] = i;
    
    /* free mem */
    free( (float *) bx );
    free( (float *) by );
    free( (float *) dv );
    free( (float *) gk );
    free( (float *) pt );
    
}

/* to be implemented */
void init(){
    /* init */
    /*
    for (x=0; x < Nx; x++){
        for (y=0; y < Ny; y++){
            
            idx = x + (y*Nx);
            graphSz = Nx*Ny;
            
            for (k = 0; k < nLab; k++){
                float tmp = 10-7;
                tmp = min(Ct[idx+k*graphSz], tmp);
                ps[idx] = tmp;
                
                pt[idx+k*graphSz] = ;
            }
            
        }
    }
     */
}

void updateP1(float *gk, float *dv, float *pt, float *u, int Nx, int Ny, float cc, int lbl_id){
    
    int x = 0;
    int y = 0;
    for (x=0; x < Nx; x++){
        for (y=0; y < Ny; y++){
            
            int g_idx = x*Ny + y;
            int l_idx = g_idx + lbl_id*Nx*Ny;
            
            gk[g_idx] = dv[l_idx] - (pt[l_idx]
                    - pt[l_idx+(Nx*Ny)] + u[l_idx]/cc);
            
             if(!finite(gk[g_idx]))
                mexErrMsgTxt("Caught gk !finite. Exiting...");
            
        }
    }
}

void updatePX(float *gk, float *bx, int Nx, int Ny, float steps, int lbl_id){
    
    int x = 0;
    int y = 0;
    for (x=1; x < Nx; x++){
        for (y=0; y < Ny; y++){
            
            int g_idx = x*Ny + y;
            int l_idx = g_idx + lbl_id*Nx*Ny;
            
            bx[l_idx] = steps*(gk[g_idx] - gk[g_idx-Ny]) + bx[l_idx];
        }
    }
}

void updatePY(float *gk, float *by, int Nx, int Ny, float steps, int lbl_id){
    
    int x = 0;
    int y = 0;
    
    for(x = 0; x < Nx; x ++){
        for(y = 1; y < Ny; y++){

            int g_idx = x*Ny + y;
            int l_idx = g_idx + lbl_id*Nx*Ny;
            
            by[l_idx] = steps*(gk[g_idx] - gk[g_idx-1]) + by[l_idx];
        }
    }
}



void projStep(float *bx, float *by, float *alpha, float *gk, int Nx, int Ny, int lbl_id){
    
    float fpt;
    int x = 0;
    int y = 0;
    for (x=0; x< Nx; x++){
        for (y=0; y< Ny; y++){
            
            int g_idx = x*Ny + y;
            int l_idx = g_idx + lbl_id*Nx*Ny;
            
            if( alpha[l_idx] <= 0 ){
                mexErrMsgTxt("alpha(x,l) must be positive. Exiting...");
            }                        
            
            fpt = sqrt((pow(bx[l_idx+Ny],2) + pow(bx[l_idx],2) +
                    pow(by[l_idx+1],2) + pow(by[l_idx],2))*0.5);
            
            if (fpt > alpha[l_idx])
                fpt = fpt / alpha[l_idx];
            else
                fpt = 1;
            
            gk[g_idx] = 1/fpt;
            
            if(!finite(gk[g_idx]))
                mexErrMsgTxt("Caught gk !finite. Exiting...");
        }
    }
}

void updateBX(float *bx, float *gk, int Nx, int Ny, int lbl_id){
    
    int x = 0;
    int y = 0;
    
    for (x=1; x< Nx; x++){
        for (y=0; y< Ny; y++){
            
            int g_idx = x*Ny + y;
            int l_idx = g_idx + lbl_id*Nx*Ny;
            
            bx[l_idx] = (gk[g_idx] + gk[g_idx-Ny])
            *0.5*bx[l_idx];
        }
    }
}

void updateBY(float *by, float *gk, int Nx, int Ny, int lbl_id){
    
    int x = 0;
    int y = 0;
    
    for (x=0; x<Nx; x++){
        for (y=1; y< Ny; y++){

            int g_idx = x*Ny + y;
            int l_idx = g_idx + lbl_id*Nx*Ny;
            
            by[l_idx] = 0.5*(gk[g_idx-1] + gk[g_idx])
            *by[l_idx];
        }
    }
}

void updateDIV(float *dv, float *bx, float *by, int Nx, int Ny, float cc, int lbl_id){
    
    float fpt;
    int x = 0;
    int y = 0;
    for (x=0; x< Nx; x++){
        for (y=0; y< Ny; y++){

            int g_idx = x*Ny + y;
            int l_idx = g_idx + lbl_id*Nx*Ny;
            
            /* update the divergence field dv(x,l)  */  
            dv[l_idx] = by[l_idx+1] - by[l_idx]
                    + bx[l_idx+Ny] - bx[l_idx];
            
            if(!finite(dv[l_idx])){
                mexErrMsgTxt("Caught div !finite. Exiting...");
        }
        }
    }

}

void updateBottomLayerPT(float *gk, float *dv, float *pt, float *u, float *Ct, int Nx,  int Ny, float cc, int lbl_id){
    
    float fpt = 0;
    int x = 0;
    int y = 0;
    
    for (x=0; x< Nx; x++){
        for (y=0; y< Ny; y++){

            int g_idx = x*Ny + y;
            int l_idx = g_idx + lbl_id*Nx*Ny;
            
            /* update pt(x,l)  */
            fpt = dv[l_idx] + pt[l_idx+(Nx*Ny)] - u[l_idx]/cc + 1/cc;
            pt[l_idx] = MIN(fpt, Ct[l_idx]);
            
            
            if(!finite(fpt))
                mexErrMsgTxt("Caught pt bl !finite. Exiting...");
            
        }
    }
}

void updateMidLayerPT(float *gk, float *dv, float *pt, float *u, float *Ct, int Nx,  int Ny, float cc, int lbl_id){
    
    float fpt = 0;
    int x = 0;
    int y = 0;
    
    for (x=0; x< Nx; x++){
        for (y=0; y< Ny; y++){

            int g_idx = x*Ny + y;
            int l_idx = g_idx + lbl_id*Nx*Ny;
            
            /* update pt(x,l)  */
            fpt =  - dv[l_idx-(Nx*Ny)] + pt[l_idx-(Nx*Ny)] + u[l_idx-(Nx*Ny)]/cc;
            fpt +=   dv[l_idx] + pt[l_idx+(Nx*Ny)] - u[l_idx]/cc;
            fpt /= 2.0f;
            
            pt[l_idx] = MIN(fpt, Ct[l_idx]);
            
             if(!finite(fpt))
                mexErrMsgTxt("Caught pt ml !finite. Exiting...");
            
        }
    }
}

void updateTopLayerPT(float *gk, float *dv, float *pt, float *u, float *Ct, int Nx,  int Ny, float cc, int lbl_id){
    
    float fpt = 0;
    int x = 0;
    int y = 0;
    
    for (x=0; x< Nx; x++){
        for (y=0; y< Ny; y++){

            int g_idx = x*Ny + y;
            int l_idx = g_idx + lbl_id*Nx*Ny;
            
            /* update pt(x,l)  */
            fpt = - dv[l_idx-(Nx*Ny)] + pt[l_idx-(Nx*Ny)] + u[l_idx-(Nx*Ny)]/cc;
            pt[l_idx] = MIN(fpt, Ct[l_idx]);
            
             if(!finite(fpt))
                mexErrMsgTxt("Caught pt tl !finite. Exiting...");
        }
    }
}


float updateU(float *dv, float *pt, float *u, int Nx, int Ny, int nLab, float cc){
    
    float fpt;
    int l;
    int x = 0;
    int y = 0;
    
    float erru = 0;
    
    for (x=0; x< Nx; x++){
        for (y=0; y< Ny; y++){
            
            int g_idx = x*Ny + y;
            
            fpt = 0;
                       
            /* update the multipliers u(x,l) */
            for (l = 0; l < (nLab-1); l++){
                fpt = cc*(dv[g_idx+l*Nx*Ny] + pt[g_idx+((l+1)*Nx*Ny)] - pt[g_idx+l*Nx*Ny]);
                
                u[g_idx+l*Nx*Ny] -= fpt;
                erru += fabsf(fpt);
                
                if(!finite(pt[g_idx+((l+1)*Nx*Ny)]))
                    mexErrMsgTxt("Caught pt+1 !finite. Exiting...");
                
                if(!finite(pt[g_idx+(l*Nx*Ny)]))
                    mexErrMsgTxt("Caught pt !finite. Exiting...");
                
                if(!finite(dv[g_idx+l*Nx*Ny]))
                    mexErrMsgTxt("Caught dv !finite. Exiting...");
                
                if(!finite(cc))
                    mexErrMsgTxt("Caught cc !finite. Exiting...");
                
                if(!finite(fpt)){
                    mexPrintf("%.2f, %.2f, %.2f, %.2f, %.2f", pt[g_idx+((l+1)*Nx*Ny)], pt[g_idx+(l*Nx*Ny)], dv[g_idx+l*Nx*Ny], cc, fpt);
                    mexErrMsgTxt("Caught fpt !finite. Exiting...");
                }
                
            }
            
        }
    }
    return erru;
}


