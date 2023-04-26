/* Weak order stochastic exponential RK schemes for
   Ito SDEs with diagonal noise */
/* This file was made to put on GitHub (26-Apr-2023). */
/*************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include "For_Kry_sub_tech.h"

#define SQ2 1.4142135623730950 /* sqrt(2) */
#define SQ3 1.7320508075688773 /* sqrt(3) */
#define SQ6 2.4494897427831781 /* sqrt(6) */

#define MaxCoreNumForSERK_Kry 8 /* 8, 1, 4 *//* Maximum number of multi-core. */

extern struct commonFort {
  double a[NZMAX];
  int ia[NZMAX], ja[NZMAX];
  int nz, n;
 } RMAT;

extern void DGCOOV(double x[], double y[]);
extern void DSEXTPHIV(int *n, int *m, double *t, double u[], double v[],
		      double w[], double wht[], double *tol, double *anorm,
		      double wsp[], int *lwsp, int *iwsp, int *liwsp,
		      void (*matvec)(double *,double *),
		      int *itrace, int *iflag );
extern void DSEXTPHI2VBIG(int *n, int *m, double *t, double u[], double w[],
			  double wht[], double *tol, double *anorm,
			  double wsp[], int *lwsp, double wspA[], int *lwspA,
			  double w_aux[],
			  int *iwsp, int *liwsp,
			  void (*matvec)(double *,double *),
			  int *itrace, int *iflag, int *errChkFlag );
extern void DSEXPV(int *n, int *m, double *t, double u[], double w[],
		   double *tol, double *anorm, double wsp[],
		   int *lwsp, int *iwsp, int *liwsp,
		   void (*matvec)(double *,double *),
		   int *itrace, int *iflag );
extern void DSPHI2VBIG(int *n, int *m, double *t, double u[], double w[],
		       double *tol, double *anorm, double wsp[],
		       int *lwsp, double w_cp[], double w_aux[],
		       int *iwsp, int *liwsp,
		       void (*matvec)(double *,double *),
		       int *itrace, int *iflag );
extern void DSEXPMVTRAY2(int *n, int *mMin, double *t, double v[], double w[],
			 double wsp[], int *lwsp, double w_aux[], 
			 int *m, int *npmv, void (*matvec)(double *,double *),
			 int *iflag);

extern int OMP_wo2_Ef_AstERK3_Kry_sym_for_DNoiseSDEs_WinMulti(int ydim,
							   unsigned long traj,
							   double *yvec,
							   double step,
							   char *ran2pFull,
							   char *ran3p,
							   void (*ffunc)(),
							   void (*gfunc_diag)(),
							   int mMax,
							   double anorm,
							   int lwsp,
							   int lwspA,
							   int liwsp,
							   int itrace,
							   int errChkFlag,
							   double tol,
							   double work[],
							   double work_K[],
							   int iWork_K[],
							   double workA_K[],
							   double *ynew)
/* wo2_AstERK3_Kry_sym_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs an Efficient, A-stable and weak second order
   exponential RK scheme.
   In the function, Krylov subspace techniques for a symmetric matrix are used.
   It gives all trajectries for one step concerning SDEs with
   diagonal noise. If an error occurs, it will return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   ran3p: pointer of the head of wdim*traj three-point distributed RVs
          with P(-1)=P(1)=1/6 and P(0)=2/3,
   ffunc: nonlinear part in the drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,
   mMax: maximum size for the Krylov basis.
   anorm: an approximation of some norm of A.
   lwsp: length of workspace "wsp" (see below).
   lwspA: length of workspace "wspA" (see below).
   liwsp: length of workspace "iwsp" (see below).
   itrace: running mode. 0=silent, 1=print step-by-step info.
   tol: requested accuracy tolerance for Krylov subspace projection techniques.
   errChkFlag: err estimatation flag for Phi1.
	      
   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Kry*16*ydim.
   work_K: workspace of length MaxCoreNumForSERK_Kry*LWSP for Krylov subspace techniques.
   iWork_K: workspace of length MaxCoreNumForSERK_Kry*NMAX for Krylov subspace techniques.
   workA_K: workspace of length MaxCoreNumForSERK_Kry*LWSP for Krylov subspace techniques.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0,
    b1=(6+SQ6)/10.0,
    b2=(3-2*SQ6)/5.0;

  int ii, jj, kk;
  double sqstep, glTmp1, glTmp2, step_tmp;

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Kry) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Kry);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Kry;
    static_flag=1;
  }

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    int *iwsp, iflag=0;
    double *yn, *yn1, *fn, *Y1, *Y2, *fY2add_fY4add_Minus_fn,
      *sum_sqhgjY2_gzai, *sumH, *wsp, *wspA, *w_aux, *w_cp, tmp1, tmp2, tmpTol;
    double *gn_diag, *sqhg_diag_Y2, *hg_diag_Y2, *g_diag_Y2_Plus,
      *g_diag_Y2_Minus, *sqhg_diag_Y2_Plus, *sqhg_diag_Y2_Minus;
    char *wj, *wtj;

    wsp=&work_K[LWSP*ii_par];
    iwsp=&iWork_K[NMAX*ii_par];
    wspA=&workA_K[LWSP*ii_par];

    ibase_work_step=16;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fn=&work[ibase_work+ii];
    ii+=ydim;
    Y1=&work[ibase_work+ii];
    ii+=ydim;
    Y2=&work[ibase_work+ii];
    ii+=ydim;
    fY2add_fY4add_Minus_fn=&work[ibase_work+ii];
    ii+=ydim;
    sum_sqhgjY2_gzai=&work[ibase_work+ii];
    ii+=ydim;
    sumH=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_Y2=&work[ibase_work+ii];
    ii+=ydim;
    hg_diag_Y2=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_Y2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_Y2_Minus=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_Y2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_Y2_Minus=&work[ibase_work+ii];
    ii+=ydim;
    w_aux=&work[ibase_work+ii];
    ii+=ydim;
    w_cp=&work[ibase_work+ii];
    ii+=ydim;

    ibase=(ydim*traj_mini)*ii_par;
    ibase3p=(wdim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wj=&ran3p[ibase3p];
      ibase3p+=wdim;
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;
      
      /* See No. 69 in Note '12/'13. */
      ffunc(yn,fn);
      tmpTol=tol;
      DSEXTPHIV(&RMAT.n,&mMax,&step,fn,yn,Y1,Y2,&tmpTol,&anorm,wsp,&lwsp,
		iwsp,&liwsp,DGCOOV,&itrace,&iflag); /* completed */
      if (0 != iflag) {
	printf("There was a problem in DSEXTPHIV\n");
	exit(0);
      }

      gfunc_diag(Y2,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_Y2[ii]=sqstep*gn_diag[ii]; /* completed */
	hg_diag_Y2[ii]=step*gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1:
	  tmp1=sqhg_diag_Y2[ii];
	    break;
	case -1:
	  tmp1=-sqhg_diag_Y2[ii];
	  break;
	default:
	  tmp1=0;
	}
	sum_sqhgjY2_gzai[ii]=SQ3*tmp1;
      }
      
      for (ii=0; ii<ydim; ii++) {
	sumH[ii]=fn[ii]; /* a temporary use of sumH */
      }
      ffunc(Y2,fn);
      for(ii=0; ii<ydim; ii++) {
	fn[ii]-=sumH[ii];
      }
      tmpTol=tol;
      /* a temporary use of yn1 */
      DSEXTPHI2VBIG(&RMAT.n,&mMax,&step,fn,fY2add_fY4add_Minus_fn,yn1,&tmpTol,
		    &anorm,wsp,&lwsp,wspA,&lwspA,w_aux,
		    iwsp,&liwsp,DGCOOV,&itrace,&iflag,&errChkFlag);
      if (0 != iflag) {
	printf("There was a problem in DSEXTPHI2VBIG\n");
	exit(0);
      }
      step_tmp=step/2.0;
      for(ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]=2.0*fY2add_fY4add_Minus_fn[ii]/step;
	yn1[ii]=4.0*yn1[ii]/step_tmp;
	/* The next is for (Y4 + an addi. term). */
	fY2add_fY4add_Minus_fn[ii]=Y1[ii]+fY2add_fY4add_Minus_fn[ii]
	  +yn1[ii]+b2*sum_sqhgjY2_gzai[ii];
      }
      ffunc(fY2add_fY4add_Minus_fn,fn);
      for (ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]=fn[ii]-5*sumH[ii];
      }
      for (ii=0; ii<ydim; ii++) { /* a temporary use of sumH */
	sumH[ii]=Y2[ii]+b1*sum_sqhgjY2_gzai[ii];
      }
      ffunc(sumH,fn);
      for (ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]+=4*fn[ii]; /* completed */
      }

      /* For details, see (337) and No. 69 in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[jj] */
	  tmp1=2.0*hg_diag_Y2[ii];
	  break;
	case -1: /* -1==wj[jj] */
	  tmp1=2.0*hg_diag_Y2[ii];
	  break;
	default: /* 0==wj[jj] */
	  tmp1=-hg_diag_Y2[ii];
	}
	g_diag_Y2_Plus[ii]=Y2[ii]+tmp1/2.0;
	g_diag_Y2_Minus[ii]=Y2[ii]-tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_Y2_Plus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_Y2_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_Y2_Minus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_Y2_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=sqhg_diag_Y2[ii];
	} else {
	  tmp1=-sqhg_diag_Y2[ii];
	}
	sqhg_diag_Y2_Plus[ii]=Y2[ii]+tmp1/SQ2;
	sqhg_diag_Y2_Minus[ii]=Y2[ii]-tmp1/SQ2;
      }
      gfunc_diag(sqhg_diag_Y2_Plus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_Y2_Plus[ii]=sqstep*gn_diag[ii]; /* completed */
      }
      gfunc_diag(sqhg_diag_Y2_Minus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_Y2_Minus[ii]=sqstep*gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=g_diag_Y2_Plus[ii]-g_diag_Y2_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp2=(sqhg_diag_Y2_Plus[ii]+sqhg_diag_Y2_Minus[ii]);
	  break;
	case -1:
	  tmp2=-(sqhg_diag_Y2_Plus[ii]+sqhg_diag_Y2_Minus[ii]);
	  break;
	default:
	  tmp2=0;
	}
	sumH[ii]=tmp1/2.0+tmp2/2.0*SQ3; /* completed */
      }

      for(ii=0; ii<ydim; ii++) {
	yn1[ii]=Y1[ii];
      }
      tmpTol=tol;
      DSPHI2VBIG(&RMAT.n,&mMax,&step,fY2add_fY4add_Minus_fn,Y1,&tmpTol,&anorm,
		 wsp,&lwsp,w_cp,w_aux,iwsp,&liwsp,DGCOOV,&itrace,&iflag);
      if (0 != iflag) {
	printf("There was a problem in DSPHI2VBIG\n");
	exit(0);
      }
      for(ii=0; ii<ydim; ii++) {
	yn1[ii]+=Y1[ii]/step/3.0;
      }
      step_tmp=step/2.0;
      tmpTol=tol;
      DSEXPV(&RMAT.n,&mMax,&step_tmp,sumH,Y1,&tmpTol,&anorm,
	     wsp,&lwsp,iwsp,&liwsp,DGCOOV,&itrace,&iflag);
      if (0 != iflag) {
	printf("There was a problem in DSEXPV\n");
	exit(0);
      }
      for(ii=0; ii<ydim; ii++) {
	yn1[ii]+=Y1[ii];
      }
    } /* End of loop for itr */
  }
  return 0;
}

extern int
OMP_wo2_Ef_AstERK3_Kry_sym_for_DNoiseSDEs_WinMulti_withCnt(int ydim,
							   unsigned long traj,
							   double *yvec,
							   double step,
							   char *ran2pFull,
							   char *ran3p,
							   void (*ffunc)(),
							   void (*gfunc_diag)(),
							   int mMax,
							   double anorm,
							   int lwsp,
							   int lwspA,
							   int liwsp,
							   int itrace,
							   int errChkFlag,
							   double tol,
							   double work[],
							   double work_K[],
							   int iWork_K[],
							   double workA_K[],
							   double *ynew,
							   unsigned long long *ev_cnt)
/* wo2_AstERK3_Kry_sym_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs an Efficient, A-stable and weak second order
   exponential RK scheme.
   In the function, Krylov subspace techniques for a symmetric matrix are used.
   It gives all trajectries for one step concerning SDEs with
   diagonal noise. If an error occurs, it will return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   ran3p: pointer of the head of wdim*traj three-point distributed RVs
          with P(-1)=P(1)=1/6 and P(0)=2/3,
   ffunc: nonlinear part in the drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,
   mMax: maximum size for the Krylov basis.
   anorm: an approximation of some norm of A.
   lwsp: length of workspace "wsp" (see below).
   lwspA: length of workspace "wspA" (see below).
   liwsp: length of workspace "iwsp" (see below).
   itrace: running mode. 0=silent, 1=print step-by-step info.
   tol: requested accuracy tolerance for Krylov subspace projection techniques.
   errChkFlag: err estimatation flag for Phi1.
	      
   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Kry*16*ydim.
   work_K: workspace of length MaxCoreNumForSERK_Kry*LWSP for Krylov subspace techniques.
   iWork_K: workspace of length MaxCoreNumForSERK_Kry*NMAX for Krylov subspace techniques.
   workA_K: workspace of length MaxCoreNumForSERK_Kry*LWSP for Krylov subspace techniques.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function evaluations.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0,
    b1=(6+SQ6)/10.0,
    b2=(3-2*SQ6)/5.0;

  int ii, jj, kk;
  double sqstep, glTmp1, glTmp2, step_tmp;
  unsigned long long func_ev_num[MaxCoreNumForSERK_Kry];

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Kry) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Kry);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Kry;
    static_flag=1;
  }

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    int *iwsp, iflag=0;
    double *yn, *yn1, *fn, *Y1, *Y2, *fY2add_fY4add_Minus_fn,
      *sum_sqhgjY2_gzai, *sumH, *wsp, *wspA, *w_aux, *w_cp, tmp1, tmp2, tmpTol;
    double *gn_diag, *sqhg_diag_Y2, *hg_diag_Y2, *g_diag_Y2_Plus,
      *g_diag_Y2_Minus, *sqhg_diag_Y2_Plus, *sqhg_diag_Y2_Minus;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;

    wsp=&work_K[LWSP*ii_par];
    iwsp=&iWork_K[NMAX*ii_par];
    wspA=&workA_K[LWSP*ii_par];

    ibase_work_step=16;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fn=&work[ibase_work+ii];
    ii+=ydim;
    Y1=&work[ibase_work+ii];
    ii+=ydim;
    Y2=&work[ibase_work+ii];
    ii+=ydim;
    fY2add_fY4add_Minus_fn=&work[ibase_work+ii];
    ii+=ydim;
    sum_sqhgjY2_gzai=&work[ibase_work+ii];
    ii+=ydim;
    sumH=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_Y2=&work[ibase_work+ii];
    ii+=ydim;
    hg_diag_Y2=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_Y2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_Y2_Minus=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_Y2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_Y2_Minus=&work[ibase_work+ii];
    ii+=ydim;
    w_aux=&work[ibase_work+ii];
    ii+=ydim;
    w_cp=&work[ibase_work+ii];
    ii+=ydim;

    ibase=(ydim*traj_mini)*ii_par;
    ibase3p=(wdim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wj=&ran3p[ibase3p];
      ibase3p+=wdim;
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;
      
      /* See No. 69 in Note '12/'13. */
      ffunc(yn,fn); func_ev_num[ii_par]++;
      tmpTol=tol;
      DSEXTPHIV(&RMAT.n,&mMax,&step,fn,yn,Y1,Y2,&tmpTol,&anorm,wsp,&lwsp,
		iwsp,&liwsp,DGCOOV,&itrace,&iflag); /* completed */
      if (0 != iflag) {
	printf("There was a problem in DSEXTPHIV\n");
	exit(0);
      }

      gfunc_diag(Y2,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_Y2[ii]=sqstep*gn_diag[ii]; /* completed */
	hg_diag_Y2[ii]=step*gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1:
	  tmp1=sqhg_diag_Y2[ii];
	    break;
	case -1:
	  tmp1=-sqhg_diag_Y2[ii];
	  break;
	default:
	  tmp1=0;
	}
	sum_sqhgjY2_gzai[ii]=SQ3*tmp1;
      }
      
      for (ii=0; ii<ydim; ii++) {
	sumH[ii]=fn[ii]; /* a temporary use of sumH */
      }
      ffunc(Y2,fn); func_ev_num[ii_par]++;
      for(ii=0; ii<ydim; ii++) {
	fn[ii]-=sumH[ii];
      }
      tmpTol=tol;
      /* a temporary use of yn1 */
      DSEXTPHI2VBIG(&RMAT.n,&mMax,&step,fn,fY2add_fY4add_Minus_fn,yn1,&tmpTol,
		    &anorm,wsp,&lwsp,wspA,&lwspA,w_aux,
		    iwsp,&liwsp,DGCOOV,&itrace,&iflag,&errChkFlag);
      if (0 != iflag) {
	printf("There was a problem in DSEXTPHI2VBIG\n");
	exit(0);
      }
      step_tmp=step/2.0;
      for(ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]=2.0*fY2add_fY4add_Minus_fn[ii]/step;
	yn1[ii]=4.0*yn1[ii]/step_tmp;
	/* The next is for (Y4 + an addi. term). */
	fY2add_fY4add_Minus_fn[ii]=Y1[ii]+fY2add_fY4add_Minus_fn[ii]
	  +yn1[ii]+b2*sum_sqhgjY2_gzai[ii];
      }
      ffunc(fY2add_fY4add_Minus_fn,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]=fn[ii]-5*sumH[ii];
      }
      for (ii=0; ii<ydim; ii++) { /* a temporary use of sumH */
	sumH[ii]=Y2[ii]+b1*sum_sqhgjY2_gzai[ii];
      }
      ffunc(sumH,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]+=4*fn[ii]; /* completed */
      }

      /* For details, see (337) and No. 69 in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[jj] */
	  tmp1=2.0*hg_diag_Y2[ii];
	  break;
	case -1: /* -1==wj[jj] */
	  tmp1=2.0*hg_diag_Y2[ii];
	  break;
	default: /* 0==wj[jj] */
	  tmp1=-hg_diag_Y2[ii];
	}
	g_diag_Y2_Plus[ii]=Y2[ii]+tmp1/2.0;
	g_diag_Y2_Minus[ii]=Y2[ii]-tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_Y2_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_Y2_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_Y2_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_Y2_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=sqhg_diag_Y2[ii];
	} else {
	  tmp1=-sqhg_diag_Y2[ii];
	}
	sqhg_diag_Y2_Plus[ii]=Y2[ii]+tmp1/SQ2;
	sqhg_diag_Y2_Minus[ii]=Y2[ii]-tmp1/SQ2;
      }
      gfunc_diag(sqhg_diag_Y2_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_Y2_Plus[ii]=sqstep*gn_diag[ii]; /* completed */
      }
      gfunc_diag(sqhg_diag_Y2_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_Y2_Minus[ii]=sqstep*gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=g_diag_Y2_Plus[ii]-g_diag_Y2_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp2=(sqhg_diag_Y2_Plus[ii]+sqhg_diag_Y2_Minus[ii]);
	  break;
	case -1:
	  tmp2=-(sqhg_diag_Y2_Plus[ii]+sqhg_diag_Y2_Minus[ii]);
	  break;
	default:
	  tmp2=0;
	}
	sumH[ii]=tmp1/2.0+tmp2/2.0*SQ3; /* completed */
      }

      for(ii=0; ii<ydim; ii++) {
	yn1[ii]=Y1[ii];
      }
      tmpTol=tol;
      DSPHI2VBIG(&RMAT.n,&mMax,&step,fY2add_fY4add_Minus_fn,Y1,&tmpTol,&anorm,
		 wsp,&lwsp,w_cp,w_aux,iwsp,&liwsp,DGCOOV,&itrace,&iflag);
      if (0 != iflag) {
	printf("There was a problem in DSPHI2VBIG\n");
	exit(0);
      }
      for(ii=0; ii<ydim; ii++) {
	yn1[ii]+=Y1[ii]/step/3.0;
      }
      step_tmp=step/2.0;
      tmpTol=tol;
      DSEXPV(&RMAT.n,&mMax,&step_tmp,sumH,Y1,&tmpTol,&anorm,
	     wsp,&lwsp,iwsp,&liwsp,DGCOOV,&itrace,&iflag);
      if (0 != iflag) {
	printf("There was a problem in DSEXPV\n");
	exit(0);
      }
      for(ii=0; ii<ydim; ii++) {
	yn1[ii]+=Y1[ii];
      }
    } /* End of loop for itr */
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
  }
  return 0;
}

extern int
OMP_wo2_Ef_AstERK3_Kry_sym_for_DNoiseSDEs_WinMulti_withCntMatProd(int ydim,
							   unsigned long traj,
							   double *yvec,
							   double step,
							   char *ran2pFull,
							   char *ran3p,
							   void (*ffunc)(),
							   void (*gfunc_diag)(),
							   int mMax,
							   double anorm,
							   int lwsp,
							   int lwspA,
							   int liwsp,
							   int itrace,
							   int errChkFlag,
							   double tol,
							   double work[],
							   double work_K[],
							   int iWork_K[],
							   double workA_K[],
							   double *ynew,
							   unsigned long long *ev_cnt,
							   unsigned long long *Kry_mat_prod_cnt)
/* wo2_AstERK3_Kry_sym_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs an Efficient, A-stable and weak second order
   exponential RK scheme.
   In the function, Krylov subspace techniques for a symmetric matrix are used.
   It gives all trajectries for one step concerning SDEs with
   diagonal noise. If an error occurs, it will return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   ran3p: pointer of the head of wdim*traj three-point distributed RVs
          with P(-1)=P(1)=1/6 and P(0)=2/3,
   ffunc: nonlinear part in the drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,
   mMax: maximum size for the Krylov basis.
   anorm: an approximation of some norm of A.
   lwsp: length of workspace "wsp" (see below).
   lwspA: length of workspace "wspA" (see below).
   liwsp: length of workspace "iwsp" (see below).
   itrace: running mode. 0=silent, 1=print step-by-step info.
   tol: requested accuracy tolerance for Krylov subspace projection techniques.
   errChkFlag: err estimatation flag for Phi1.
	      
   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Kry*16*ydim.
   work_K: workspace of length MaxCoreNumForSERK_Kry*LWSP for Krylov subspace techniques.
   iWork_K: workspace of length MaxCoreNumForSERK_Kry*NMAX for Krylov subspace techniques.
   workA_K: workspace of length MaxCoreNumForSERK_Kry*LWSP for Krylov subspace techniques.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function evaluations.
   Kry_mat_prod_cnt: the number of matrix products in Krylov subspace techniques.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0,
    b1=(6+SQ6)/10.0,
    b2=(3-2*SQ6)/5.0;

  int ii, jj, kk;
  double sqstep, glTmp1, glTmp2, step_tmp;
  unsigned long long func_ev_num[MaxCoreNumForSERK_Kry], mat_proc_num[MaxCoreNumForSERK_Kry];

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Kry) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Kry);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Kry;
    static_flag=1;
  }

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    int *iwsp, iflag=0;
    double *yn, *yn1, *fn, *Y1, *Y2, *fY2add_fY4add_Minus_fn,
      *sum_sqhgjY2_gzai, *sumH, *wsp, *wspA, *w_aux, *w_cp, tmp1, tmp2, tmpTol;
    double *gn_diag, *sqhg_diag_Y2, *hg_diag_Y2, *g_diag_Y2_Plus,
      *g_diag_Y2_Minus, *sqhg_diag_Y2_Plus, *sqhg_diag_Y2_Minus;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;
    mat_proc_num[ii_par]=0;

    wsp=&work_K[LWSP*ii_par];
    iwsp=&iWork_K[NMAX*ii_par];
    wspA=&workA_K[LWSP*ii_par];

    ibase_work_step=16;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fn=&work[ibase_work+ii];
    ii+=ydim;
    Y1=&work[ibase_work+ii];
    ii+=ydim;
    Y2=&work[ibase_work+ii];
    ii+=ydim;
    fY2add_fY4add_Minus_fn=&work[ibase_work+ii];
    ii+=ydim;
    sum_sqhgjY2_gzai=&work[ibase_work+ii];
    ii+=ydim;
    sumH=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_Y2=&work[ibase_work+ii];
    ii+=ydim;
    hg_diag_Y2=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_Y2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_Y2_Minus=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_Y2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_Y2_Minus=&work[ibase_work+ii];
    ii+=ydim;
    w_aux=&work[ibase_work+ii];
    ii+=ydim;
    w_cp=&work[ibase_work+ii];
    ii+=ydim;

    ibase=(ydim*traj_mini)*ii_par;
    ibase3p=(wdim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wj=&ran3p[ibase3p];
      ibase3p+=wdim;
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;
      
      /* See No. 69 in Note '12/'13. */
      ffunc(yn,fn); func_ev_num[ii_par]++;
      tmpTol=tol;
      DSEXTPHIV(&RMAT.n,&mMax,&step,fn,yn,Y1,Y2,&tmpTol,&anorm,wsp,&lwsp,
		iwsp,&liwsp,DGCOOV,&itrace,&iflag); /* completed */
      mat_proc_num[ii_par]+=iwsp[0];
      if (0 != iflag) {
	printf("There was a problem in DSEXTPHIV\n");
	exit(0);
      }

      gfunc_diag(Y2,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_Y2[ii]=sqstep*gn_diag[ii]; /* completed */
	hg_diag_Y2[ii]=step*gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1:
	  tmp1=sqhg_diag_Y2[ii];
	    break;
	case -1:
	  tmp1=-sqhg_diag_Y2[ii];
	  break;
	default:
	  tmp1=0;
	}
	sum_sqhgjY2_gzai[ii]=SQ3*tmp1;
      }
      
      for (ii=0; ii<ydim; ii++) {
	sumH[ii]=fn[ii]; /* a temporary use of sumH */
      }
      ffunc(Y2,fn); func_ev_num[ii_par]++;
      for(ii=0; ii<ydim; ii++) {
	fn[ii]-=sumH[ii];
      }
      tmpTol=tol;
      /* a temporary use of yn1 */
      DSEXTPHI2VBIG(&RMAT.n,&mMax,&step,fn,fY2add_fY4add_Minus_fn,yn1,&tmpTol,
		    &anorm,wsp,&lwsp,wspA,&lwspA,w_aux,
		    iwsp,&liwsp,DGCOOV,&itrace,&iflag,&errChkFlag);
      mat_proc_num[ii_par]+=iwsp[0];
      if (0 != iflag) {
	printf("There was a problem in DSEXTPHI2VBIG\n");
	exit(0);
      }
      step_tmp=step/2.0;
      for(ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]=2.0*fY2add_fY4add_Minus_fn[ii]/step;
	yn1[ii]=4.0*yn1[ii]/step_tmp;
	/* The next is for (Y4 + an addi. term). */
	fY2add_fY4add_Minus_fn[ii]=Y1[ii]+fY2add_fY4add_Minus_fn[ii]
	  +yn1[ii]+b2*sum_sqhgjY2_gzai[ii];
      }
      ffunc(fY2add_fY4add_Minus_fn,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]=fn[ii]-5*sumH[ii];
      }
      for (ii=0; ii<ydim; ii++) { /* a temporary use of sumH */
	sumH[ii]=Y2[ii]+b1*sum_sqhgjY2_gzai[ii];
      }
      ffunc(sumH,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]+=4*fn[ii]; /* completed */
      }

      /* For details, see (337) and No. 69 in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[jj] */
	  tmp1=2.0*hg_diag_Y2[ii];
	  break;
	case -1: /* -1==wj[jj] */
	  tmp1=2.0*hg_diag_Y2[ii];
	  break;
	default: /* 0==wj[jj] */
	  tmp1=-hg_diag_Y2[ii];
	}
	g_diag_Y2_Plus[ii]=Y2[ii]+tmp1/2.0;
	g_diag_Y2_Minus[ii]=Y2[ii]-tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_Y2_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_Y2_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_Y2_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_Y2_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=sqhg_diag_Y2[ii];
	} else {
	  tmp1=-sqhg_diag_Y2[ii];
	}
	sqhg_diag_Y2_Plus[ii]=Y2[ii]+tmp1/SQ2;
	sqhg_diag_Y2_Minus[ii]=Y2[ii]-tmp1/SQ2;
      }
      gfunc_diag(sqhg_diag_Y2_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_Y2_Plus[ii]=sqstep*gn_diag[ii]; /* completed */
      }
      gfunc_diag(sqhg_diag_Y2_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_Y2_Minus[ii]=sqstep*gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=g_diag_Y2_Plus[ii]-g_diag_Y2_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp2=(sqhg_diag_Y2_Plus[ii]+sqhg_diag_Y2_Minus[ii]);
	  break;
	case -1:
	  tmp2=-(sqhg_diag_Y2_Plus[ii]+sqhg_diag_Y2_Minus[ii]);
	  break;
	default:
	  tmp2=0;
	}
	sumH[ii]=tmp1/2.0+tmp2/2.0*SQ3; /* completed */
      }

      for(ii=0; ii<ydim; ii++) {
	yn1[ii]=Y1[ii];
      }
      tmpTol=tol;
      DSPHI2VBIG(&RMAT.n,&mMax,&step,fY2add_fY4add_Minus_fn,Y1,&tmpTol,&anorm,
		 wsp,&lwsp,w_cp,w_aux,iwsp,&liwsp,DGCOOV,&itrace,&iflag);
      mat_proc_num[ii_par]+=iwsp[0];
      if (0 != iflag) {
	printf("There was a problem in DSPHI2VBIG\n");
	exit(0);
      }
      for(ii=0; ii<ydim; ii++) {
	yn1[ii]+=Y1[ii]/step/3.0;
      }
      step_tmp=step/2.0;
      tmpTol=tol;
      DSEXPV(&RMAT.n,&mMax,&step_tmp,sumH,Y1,&tmpTol,&anorm,
	     wsp,&lwsp,iwsp,&liwsp,DGCOOV,&itrace,&iflag);
      mat_proc_num[ii_par]+=iwsp[0];
      if (0 != iflag) {
	printf("There was a problem in DSEXPV\n");
	exit(0);
      }
      for(ii=0; ii<ydim; ii++) {
	yn1[ii]+=Y1[ii];
      }
    } /* End of loop for itr */
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
    *Kry_mat_prod_cnt+=mat_proc_num[ii_par];
  }
  return 0;
}

extern int OMP_wo2_SSDFMT_Kry_sym_for_DNoiseSDEs_WinMulti(int ydim,
						   unsigned long traj,
						   double *yvec,
						   double step,
						   char *ran2pFull,
						   char *ran3p,
						   void (*ffunc)(),
						   void (*gfunc_diag)(),
					           int mMax,
					           double anorm,
						   int lwsp,
						   int lwspA,
						   int liwsp,
						   int itrace,
						   int errChkFlag,
						   double tol,
						   double work[],
						   double work_K[],
						   int iWork_K[],
						   double workA_K[],
						   double *ynew)
/* wo2_SSDFMT_Kry_sym_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs the Strang splitting DFMT method.
   It gives all trajectries for one step concerning SDEs with
   a multi-dimentional Winer process. If an error occurs, it will
   return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   ran3p: pointer of the head of wdim*traj three-point distributed RVs
          with P(-1)=P(1)=1/6 and P(0)=2/3,
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,
   mMax: maximum size for the Krylov basis.
   anorm: an approximation of some norm of A.
   lwsp: length of workspace "wsp" (see below).
   lwspA: length of workspace "wspA" (see below).
   liwsp: length of workspace "iwsp" (see below).
   itrace: running mode. 0=silent, 1=print step-by-step info.
   tol: requested accuracy tolerance for Krylov subspace projection techniques.
   errChkFlag: err estimatation flag for Phi1.
   
   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Kry*11*ydim.
   work_K: workspace of length MaxCoreNumForSERK_Kry*LWSP for Krylov subspace techniques.
   iWork_K: workspace of length MaxCoreNumForSERK_Kry*NMAX for Krylov subspace techniques.
   workA_K: workspace of length MaxCoreNumForSERK_Kry*LWSP for Krylov subspace techniques.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0;

  int ii, jj, kk;
  double sqstep, step_tmp;

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Kry) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Kry);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Kry;
    static_flag=1;
  }

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    int *iwsp, iflag=0;
    double *yn, *yn1, *fyn, *K1, *fK2, *exYn, tmp1, tmp2, *wsp, *wspA, tmpTol;
    double *gn_diag, *g_diag_yn, *g_diag_yn_Plus, *g_diag_yn_Minus,
      *g_diag_yn_K1_dev2_Plus, *g_diag_yn_K1_dev2_Minus;
    char *wj, *wtj;

    wsp=&work_K[LWSP*ii_par];
    iwsp=&iWork_K[NMAX*ii_par];
    wspA=&workA_K[LWSP*ii_par];

    ibase_work_step=11;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fyn=&work[ibase_work+ii];
    ii+=ydim;
    K1=&work[ibase_work+ii];
    ii+=ydim;
    fK2=&work[ibase_work+ii];
    ii+=ydim;
    exYn=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Minus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Minus=&work[ibase_work+ii];
    ii+=ydim;

    ibase=(ydim*traj_mini)*ii_par;
    ibase3p=(wdim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wj=&ran3p[ibase3p];
      ibase3p+=wdim;
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;
      
      /* See No. 42 and No. 56 in Note '12/'13. */
      step_tmp=step/2.0;
      tmpTol=tol;
      DSEXPV(&RMAT.n,&mMax,&step_tmp,yn,exYn,&tmpTol,&anorm,
	     wsp,&lwsp,iwsp,&liwsp,DGCOOV,&itrace,&iflag);
      if (0 != iflag) {
	printf("There was a problem in DGEXPV\n");
	exit(0);
      }

      ffunc(exYn,fyn); /* completed */
      for(ii=0; ii<ydim; ii++) {
	K1[ii]=exYn[ii]+step*fyn[ii]; /* completed */
      }

      for (jj=0; jj<wdim; jj++) {
	gfunc_diag(exYn,g_diag_yn); /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1:
	  tmp1=g_diag_yn[ii];
	  break;
	case -1:
	  tmp1=-g_diag_yn[ii];
	  break;
	default:
	  tmp1=0;
	}
	yn1[ii]=K1[ii]+sqstep*SQ3*tmp1;
      }
      ffunc(yn1,fK2); /* completed */

      /* For details, see (337) and (338) in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	case -1: /* -1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	default: /* 0==wj[jj] */
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_Plus[ii]=exYn[ii]+step*tmp1/2.0;
	g_diag_yn_Minus[ii]=exYn[ii]-step*tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_yn_Plus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_Minus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=g_diag_yn[ii];
	} else {
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_K1_dev2_Plus[ii]=(exYn[ii]+K1[ii])/2.0+sqstep*tmp1/SQ2;
	g_diag_yn_K1_dev2_Minus[ii]=(exYn[ii]+K1[ii])/2.0-sqstep*tmp1/SQ2;
      }
      gfunc_diag(g_diag_yn_K1_dev2_Plus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_K1_dev2_Minus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=g_diag_yn_Plus[ii]-g_diag_yn_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp2=(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	case -1:
	  tmp2=-(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	default:
	  tmp2=0;
	}
	exYn[ii]+=step*(fyn[ii]+fK2[ii])/2.0+tmp1/2.0+sqstep*tmp2/2.0*SQ3;
      }

      step_tmp=step/2.0;
      tmpTol=tol;
      DSEXPV(&RMAT.n,&mMax,&step_tmp,exYn,yn1,&tmpTol,&anorm,
	     wsp,&lwsp,iwsp,&liwsp,DGCOOV,&itrace,&iflag);
      if (0 != iflag) {
	printf("There was a problem in DGEXPV\n");
	exit(0);
      }
      
    } /* End of loop for itr */
  }
  return 0;
}

extern
int OMP_wo2_SSDFMT_Kry_sym_for_DNoiseSDEs_WinMulti_withCnt(int ydim,
							   unsigned long traj,
							   double *yvec,
							   double step,
							   char *ran2pFull,
							   char *ran3p,
							   void (*ffunc)(),
							   void (*gfunc_diag)(),
							   int mMax,
							   double anorm,
							   int lwsp,
							   int lwspA,
							   int liwsp,
							   int itrace,
							   int errChkFlag,
							   double tol,
							   double work[],
							   double work_K[],
							   int iWork_K[],
							   double workA_K[],
							   double *ynew,
							   unsigned long long *ev_cnt)
/* wo2_SSDFMT_Kry_sym_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs the Strang splitting DFMT method.
   It gives all trajectries for one step concerning SDEs with
   a multi-dimentional Winer process. If an error occurs, it will
   return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   ran3p: pointer of the head of wdim*traj three-point distributed RVs
          with P(-1)=P(1)=1/6 and P(0)=2/3,
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,
   mMax: maximum size for the Krylov basis.
   anorm: an approximation of some norm of A.
   lwsp: length of workspace "wsp" (see below).
   lwspA: length of workspace "wspA" (see below).
   liwsp: length of workspace "iwsp" (see below).
   itrace: running mode. 0=silent, 1=print step-by-step info.
   tol: requested accuracy tolerance for Krylov subspace projection techniques.
   errChkFlag: err estimatation flag for Phi1.
   
   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Kry*11*ydim.
   work_K: workspace of length MaxCoreNumForSERK_Kry*LWSP for Krylov subspace techniques.
   iWork_K: workspace of length MaxCoreNumForSERK_Kry*NMAX for Krylov subspace techniques.
   workA_K: workspace of length MaxCoreNumForSERK_Kry*LWSP for Krylov subspace techniques.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function evaluations.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0;

  int ii, jj, kk;
  double sqstep, step_tmp;
  unsigned long long func_ev_num[MaxCoreNumForSERK_Kry];

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Kry) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Kry);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Kry;
    static_flag=1;
  }

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    int *iwsp, iflag=0;
    double *yn, *yn1, *fyn, *K1, *fK2, *exYn, tmp1, tmp2, *wsp, *wspA, tmpTol;
    double *gn_diag, *g_diag_yn, *g_diag_yn_Plus, *g_diag_yn_Minus,
      *g_diag_yn_K1_dev2_Plus, *g_diag_yn_K1_dev2_Minus;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;

    wsp=&work_K[LWSP*ii_par];
    iwsp=&iWork_K[NMAX*ii_par];
    wspA=&workA_K[LWSP*ii_par];

    ibase_work_step=11;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fyn=&work[ibase_work+ii];
    ii+=ydim;
    K1=&work[ibase_work+ii];
    ii+=ydim;
    fK2=&work[ibase_work+ii];
    ii+=ydim;
    exYn=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Minus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Minus=&work[ibase_work+ii];
    ii+=ydim;

    ibase=(ydim*traj_mini)*ii_par;
    ibase3p=(wdim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wj=&ran3p[ibase3p];
      ibase3p+=wdim;
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;
      
      /* See No. 42 and No. 56 in Note '12/'13. */
      step_tmp=step/2.0;
      tmpTol=tol;
      DSEXPV(&RMAT.n,&mMax,&step_tmp,yn,exYn,&tmpTol,&anorm,
	     wsp,&lwsp,iwsp,&liwsp,DGCOOV,&itrace,&iflag);
      if (0 != iflag) {
	printf("There was a problem in DGEXPV\n");
	exit(0);
      }

      ffunc(exYn,fyn); func_ev_num[ii_par]++; /* completed */
      for(ii=0; ii<ydim; ii++) {
	K1[ii]=exYn[ii]+step*fyn[ii]; /* completed */
      }

      for (jj=0; jj<wdim; jj++) {
	gfunc_diag(exYn,g_diag_yn); func_ev_num[ii_par]++; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1:
	  tmp1=g_diag_yn[ii];
	  break;
	case -1:
	  tmp1=-g_diag_yn[ii];
	  break;
	default:
	  tmp1=0;
	}
	yn1[ii]=K1[ii]+sqstep*SQ3*tmp1;
      }
      ffunc(yn1,fK2); func_ev_num[ii_par]++; /* completed */

      /* For details, see (337) and (338) in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	case -1: /* -1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	default: /* 0==wj[jj] */
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_Plus[ii]=exYn[ii]+step*tmp1/2.0;
	g_diag_yn_Minus[ii]=exYn[ii]-step*tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_yn_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=g_diag_yn[ii];
	} else {
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_K1_dev2_Plus[ii]=(exYn[ii]+K1[ii])/2.0+sqstep*tmp1/SQ2;
	g_diag_yn_K1_dev2_Minus[ii]=(exYn[ii]+K1[ii])/2.0-sqstep*tmp1/SQ2;
      }
      gfunc_diag(g_diag_yn_K1_dev2_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_K1_dev2_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=g_diag_yn_Plus[ii]-g_diag_yn_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp2=(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	case -1:
	  tmp2=-(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	default:
	  tmp2=0;
	}
	exYn[ii]+=step*(fyn[ii]+fK2[ii])/2.0+tmp1/2.0+sqstep*tmp2/2.0*SQ3;
      }

      step_tmp=step/2.0;
      tmpTol=tol;
      DSEXPV(&RMAT.n,&mMax,&step_tmp,exYn,yn1,&tmpTol,&anorm,
	     wsp,&lwsp,iwsp,&liwsp,DGCOOV,&itrace,&iflag);
      if (0 != iflag) {
	printf("There was a problem in DGEXPV\n");
	exit(0);
      }
      
    } /* End of loop for itr */
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
  }
  return 0;
}

extern
int OMP_wo2_SSDFMT_Kry_sym_for_DNoiseSDEs_WinMulti_withCntMatProd(int ydim,
							   unsigned long traj,
							   double *yvec,
							   double step,
							   char *ran2pFull,
							   char *ran3p,
							   void (*ffunc)(),
							   void (*gfunc_diag)(),
							   int mMax,
							   double anorm,
							   int lwsp,
							   int lwspA,
							   int liwsp,
							   int itrace,
							   int errChkFlag,
							   double tol,
							   double work[],
							   double work_K[],
							   int iWork_K[],
							   double workA_K[],
							   double *ynew,
							   unsigned long long *ev_cnt,
							   unsigned long long *Kry_mat_prod_cnt)
/* wo2_SSDFMT_Kry_sym_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs the Strang splitting DFMT method.
   It gives all trajectries for one step concerning SDEs with
   a multi-dimentional Winer process. If an error occurs, it will
   return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   ran3p: pointer of the head of wdim*traj three-point distributed RVs
          with P(-1)=P(1)=1/6 and P(0)=2/3,
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,
   mMax: maximum size for the Krylov basis.
   anorm: an approximation of some norm of A.
   lwsp: length of workspace "wsp" (see below).
   lwspA: length of workspace "wspA" (see below).
   liwsp: length of workspace "iwsp" (see below).
   itrace: running mode. 0=silent, 1=print step-by-step info.
   tol: requested accuracy tolerance for Krylov subspace projection techniques.
   errChkFlag: err estimatation flag for Phi1.
   
   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Kry*11*ydim.
   work_K: workspace of length MaxCoreNumForSERK_Kry*LWSP for Krylov subspace techniques.
   iWork_K: workspace of length MaxCoreNumForSERK_Kry*NMAX for Krylov subspace techniques.
   workA_K: workspace of length MaxCoreNumForSERK_Kry*LWSP for Krylov subspace techniques.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function evaluations.
   Kry_mat_prod_cnt: the number of matrix products in Krylov subspace techniques.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0;

  int ii, jj, kk;
  double sqstep, step_tmp;
  unsigned long long func_ev_num[MaxCoreNumForSERK_Kry], mat_prod_num[MaxCoreNumForSERK_Kry];

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Kry) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Kry);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Kry;
    static_flag=1;
  }

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    int *iwsp, iflag=0;
    double *yn, *yn1, *fyn, *K1, *fK2, *exYn, tmp1, tmp2, *wsp, *wspA, tmpTol;
    double *gn_diag, *g_diag_yn, *g_diag_yn_Plus, *g_diag_yn_Minus,
      *g_diag_yn_K1_dev2_Plus, *g_diag_yn_K1_dev2_Minus;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;
    mat_prod_num[ii_par]=0;

    wsp=&work_K[LWSP*ii_par];
    iwsp=&iWork_K[NMAX*ii_par];
    wspA=&workA_K[LWSP*ii_par];

    ibase_work_step=11;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fyn=&work[ibase_work+ii];
    ii+=ydim;
    K1=&work[ibase_work+ii];
    ii+=ydim;
    fK2=&work[ibase_work+ii];
    ii+=ydim;
    exYn=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Minus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Minus=&work[ibase_work+ii];
    ii+=ydim;

    ibase=(ydim*traj_mini)*ii_par;
    ibase3p=(wdim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wj=&ran3p[ibase3p];
      ibase3p+=wdim;
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;
      
      /* See No. 42 and No. 56 in Note '12/'13. */
      step_tmp=step/2.0;
      tmpTol=tol;
      DSEXPV(&RMAT.n,&mMax,&step_tmp,yn,exYn,&tmpTol,&anorm,
	     wsp,&lwsp,iwsp,&liwsp,DGCOOV,&itrace,&iflag);
      mat_prod_num[ii_par]+=iwsp[0];
      if (0 != iflag) {
	printf("There was a problem in DGEXPV\n");
	exit(0);
      }

      ffunc(exYn,fyn); func_ev_num[ii_par]++; /* completed */
      for(ii=0; ii<ydim; ii++) {
	K1[ii]=exYn[ii]+step*fyn[ii]; /* completed */
      }

      for (jj=0; jj<wdim; jj++) {
	gfunc_diag(exYn,g_diag_yn); func_ev_num[ii_par]++; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1:
	  tmp1=g_diag_yn[ii];
	  break;
	case -1:
	  tmp1=-g_diag_yn[ii];
	  break;
	default:
	  tmp1=0;
	}
	yn1[ii]=K1[ii]+sqstep*SQ3*tmp1;
      }
      ffunc(yn1,fK2); func_ev_num[ii_par]++; /* completed */

      /* For details, see (337) and (338) in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	case -1: /* -1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	default: /* 0==wj[jj] */
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_Plus[ii]=exYn[ii]+step*tmp1/2.0;
	g_diag_yn_Minus[ii]=exYn[ii]-step*tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_yn_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=g_diag_yn[ii];
	} else {
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_K1_dev2_Plus[ii]=(exYn[ii]+K1[ii])/2.0+sqstep*tmp1/SQ2;
	g_diag_yn_K1_dev2_Minus[ii]=(exYn[ii]+K1[ii])/2.0-sqstep*tmp1/SQ2;
      }
      gfunc_diag(g_diag_yn_K1_dev2_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_K1_dev2_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=g_diag_yn_Plus[ii]-g_diag_yn_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp2=(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	case -1:
	  tmp2=-(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	default:
	  tmp2=0;
	}
	exYn[ii]+=step*(fyn[ii]+fK2[ii])/2.0+tmp1/2.0+sqstep*tmp2/2.0*SQ3;
      }

      step_tmp=step/2.0;
      tmpTol=tol;
      DSEXPV(&RMAT.n,&mMax,&step_tmp,exYn,yn1,&tmpTol,&anorm,
	     wsp,&lwsp,iwsp,&liwsp,DGCOOV,&itrace,&iflag);
      mat_prod_num[ii_par]+=iwsp[0];
      if (0 != iflag) {
	printf("There was a problem in DGEXPV\n");
	exit(0);
      }
      
    } /* End of loop for itr */
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
    *Kry_mat_prod_cnt+=mat_prod_num[ii_par];
  }
  return 0;
}

extern int OMP_wo2_SSDFMT_Taylor_for_DNoiseSDEs_WinMulti(int ydim,
						   unsigned long traj,
						   double *yvec,
						   double step,
						   char *ran2pFull,
						   char *ran3p,
						   void (*ffunc)(),
						   void (*gfunc_diag)(),
					           int mMin,
						   int lwsp,
						   double work[],
						   double work_T[],
						   double workAux[],
						   double *ynew)
/* wo2_SSDFMT_Taylor_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs the Strang splitting DFMT method.
   It gives all trajectries for one step concerning SDEs with
   a multi-dimentional Winer process. If an error occurs, it will
   return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   ran3p: pointer of the head of wdim*traj three-point distributed RVs
          with P(-1)=P(1)=1/6 and P(0)=2/3,
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,
   mMin: minimum order of polynomial.
   lwsp: length of workspace "wsp" (see below).
   
   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Kry*11*ydim.
   work_T: workspace of length MaxCoreNumForSERK_Kry*LWSP for Taylor expansion.
   workAux: workspace of length MaxCoreNumForSERK_Kry*2*ydim for Taylor techniques.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0;

  int ii, jj, kk;
  double sqstep, step_tmp;

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Kry) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Kry);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Kry;
    static_flag=1;
  }

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step,
      mm, npmv, iflag;
    double *yn, *yn1, *fyn, *K1, *fK2, *exYn, tmp1, tmp2, *wsp, *wspAux;
    double *gn_diag, *g_diag_yn, *g_diag_yn_Plus, *g_diag_yn_Minus,
      *g_diag_yn_K1_dev2_Plus, *g_diag_yn_K1_dev2_Minus;
    char *wj, *wtj;

    wsp=&work_T[LWSP*ii_par];
    wspAux=&workAux[2*ydim*ii_par];

    ibase_work_step=11;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fyn=&work[ibase_work+ii];
    ii+=ydim;
    K1=&work[ibase_work+ii];
    ii+=ydim;
    fK2=&work[ibase_work+ii];
    ii+=ydim;
    exYn=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Minus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Minus=&work[ibase_work+ii];
    ii+=ydim;

    ibase=(ydim*traj_mini)*ii_par;
    ibase3p=(wdim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wj=&ran3p[ibase3p];
      ibase3p+=wdim;
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;
      
      /* See No. 42 and No. 56 in Note '12/'13. */
      step_tmp=step/2.0;
      DSEXPMVTRAY2(&RMAT.n,&mMin,&step_tmp,yn,exYn,wsp,&lwsp,wspAux,
		   &mm,&npmv,DGCOOV,&iflag);
      if (0 != iflag) {
	printf("There was a problem in DSEXPMVTRAY2\n");
	exit(0);
      }

      ffunc(exYn,fyn); /* completed */
      for(ii=0; ii<ydim; ii++) {
	K1[ii]=exYn[ii]+step*fyn[ii]; /* completed */
      }

      for (jj=0; jj<wdim; jj++) {
	gfunc_diag(exYn,g_diag_yn); /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1:
	  tmp1=g_diag_yn[ii];
	  break;
	case -1:
	  tmp1=-g_diag_yn[ii];
	  break;
	default:
	  tmp1=0;
	}
	yn1[ii]=K1[ii]+sqstep*SQ3*tmp1;
      }
      ffunc(yn1,fK2); /* completed */

      /* For details, see (337) and (338) in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	case -1: /* -1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	default: /* 0==wj[jj] */
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_Plus[ii]=exYn[ii]+step*tmp1/2.0;
	g_diag_yn_Minus[ii]=exYn[ii]-step*tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_yn_Plus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_Minus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=g_diag_yn[ii];
	} else {
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_K1_dev2_Plus[ii]=(exYn[ii]+K1[ii])/2.0+sqstep*tmp1/SQ2;
	g_diag_yn_K1_dev2_Minus[ii]=(exYn[ii]+K1[ii])/2.0-sqstep*tmp1/SQ2;
      }
      gfunc_diag(g_diag_yn_K1_dev2_Plus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_K1_dev2_Minus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=g_diag_yn_Plus[ii]-g_diag_yn_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp2=(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	case -1:
	  tmp2=-(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	default:
	  tmp2=0;
	}
	exYn[ii]+=step*(fyn[ii]+fK2[ii])/2.0+tmp1/2.0+sqstep*tmp2/2.0*SQ3;
      }

      step_tmp=step/2.0;
      DSEXPMVTRAY2(&RMAT.n,&mMin,&step_tmp,exYn,yn1,wsp,&lwsp,wspAux,
		   &mm,&npmv,DGCOOV,&iflag);
      if (0 != iflag) {
	printf("There was a problem in DSEXPMVTRAY2\n");
	exit(0);
      }
      
    } /* End of loop for itr */
  }
  return 0;
}

extern
int OMP_wo2_SSDFMT_Taylor_for_DNoiseSDEs_WinMulti_withCnt(int ydim,
							  unsigned long traj,
							  double *yvec,
							  double step,
							  char *ran2pFull,
							  char *ran3p,
							  void (*ffunc)(),
							  void (*gfunc_diag)(),
							  int mMin,
							  int lwsp,
							  double work[],
							  double work_T[],
							  double workAux[],
							  double *ynew,
							  unsigned long long *ev_cnt)
/* wo2_SSDFMT_Taylor_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs the Strang splitting DFMT method.
   It gives all trajectries for one step concerning SDEs with
   a multi-dimentional Winer process. If an error occurs, it will
   return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   ran3p: pointer of the head of wdim*traj three-point distributed RVs
          with P(-1)=P(1)=1/6 and P(0)=2/3,
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,
   mMin: minimum order of polynomial.
   lwsp: length of workspace "wsp" (see below).
   
   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Kry*11*ydim.
   work_T: workspace of length MaxCoreNumForSERK_Kry*LWSP for Taylor expansion.
   workAux: workspace of length MaxCoreNumForSERK_Kry*2*ydim for Taylor techniques.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function evaluations.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0;

  int ii, jj, kk;
  double sqstep, step_tmp;
  unsigned long long func_ev_num[MaxCoreNumForSERK_Kry];

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Kry) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Kry);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Kry;
    static_flag=1;
  }

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step,
      mm, npmv, iflag;
    double *yn, *yn1, *fyn, *K1, *fK2, *exYn, tmp1, tmp2, *wsp, *wspAux;
    double *gn_diag, *g_diag_yn, *g_diag_yn_Plus, *g_diag_yn_Minus,
      *g_diag_yn_K1_dev2_Plus, *g_diag_yn_K1_dev2_Minus;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;

    wsp=&work_T[LWSP*ii_par];
    wspAux=&workAux[2*ydim*ii_par];

    ibase_work_step=11;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fyn=&work[ibase_work+ii];
    ii+=ydim;
    K1=&work[ibase_work+ii];
    ii+=ydim;
    fK2=&work[ibase_work+ii];
    ii+=ydim;
    exYn=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Minus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Minus=&work[ibase_work+ii];
    ii+=ydim;

    ibase=(ydim*traj_mini)*ii_par;
    ibase3p=(wdim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wj=&ran3p[ibase3p];
      ibase3p+=wdim;
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;
      
      /* See No. 42 and No. 56 in Note '12/'13. */
      step_tmp=step/2.0;
      DSEXPMVTRAY2(&RMAT.n,&mMin,&step_tmp,yn,exYn,wsp,&lwsp,wspAux,
		   &mm,&npmv,DGCOOV,&iflag);
      if (0 != iflag) {
	printf("There was a problem in DSEXPMVTRAY2\n");
	exit(0);
      }

      ffunc(exYn,fyn); func_ev_num[ii_par]++; /* completed */
      for(ii=0; ii<ydim; ii++) {
	K1[ii]=exYn[ii]+step*fyn[ii]; /* completed */
      }

      for (jj=0; jj<wdim; jj++) {
	gfunc_diag(exYn,g_diag_yn); func_ev_num[ii_par]++; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1:
	  tmp1=g_diag_yn[ii];
	  break;
	case -1:
	  tmp1=-g_diag_yn[ii];
	  break;
	default:
	  tmp1=0;
	}
	yn1[ii]=K1[ii]+sqstep*SQ3*tmp1;
      }
      ffunc(yn1,fK2); func_ev_num[ii_par]++; /* completed */

      /* For details, see (337) and (338) in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	case -1: /* -1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	default: /* 0==wj[jj] */
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_Plus[ii]=exYn[ii]+step*tmp1/2.0;
	g_diag_yn_Minus[ii]=exYn[ii]-step*tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_yn_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=g_diag_yn[ii];
	} else {
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_K1_dev2_Plus[ii]=(exYn[ii]+K1[ii])/2.0+sqstep*tmp1/SQ2;
	g_diag_yn_K1_dev2_Minus[ii]=(exYn[ii]+K1[ii])/2.0-sqstep*tmp1/SQ2;
      }
      gfunc_diag(g_diag_yn_K1_dev2_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_K1_dev2_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=g_diag_yn_Plus[ii]-g_diag_yn_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp2=(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	case -1:
	  tmp2=-(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	default:
	  tmp2=0;
	}
	exYn[ii]+=step*(fyn[ii]+fK2[ii])/2.0+tmp1/2.0+sqstep*tmp2/2.0*SQ3;
      }

      step_tmp=step/2.0;
      DSEXPMVTRAY2(&RMAT.n,&mMin,&step_tmp,exYn,yn1,wsp,&lwsp,wspAux,
		   &mm,&npmv,DGCOOV,&iflag);
      if (0 != iflag) {
	printf("There was a problem in DSEXPMVTRAY2\n");
	exit(0);
      }
      
    } /* End of loop for itr */
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
  }
  return 0;
}

extern
int OMP_wo2_SSDFMT_Taylor_for_DNoiseSDEs_WinMulti_withCntMatProd(int ydim,
							  unsigned long traj,
							  double *yvec,
							  double step,
							  char *ran2pFull,
							  char *ran3p,
							  void (*ffunc)(),
							  void (*gfunc_diag)(),
							  int mMin,
							  int lwsp,
							  double work[],
							  double work_T[],
							  double workAux[],
							  double *ynew,
							  unsigned long long *ev_cnt,
							  unsigned long long *Taylor_mat_prod_cnt)
/* wo2_SSDFMT_Taylor_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs the Strang splitting DFMT method.
   It gives all trajectries for one step concerning SDEs with
   a multi-dimentional Winer process. If an error occurs, it will
   return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   ran3p: pointer of the head of wdim*traj three-point distributed RVs
          with P(-1)=P(1)=1/6 and P(0)=2/3,
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,
   mMin: minimum order of polynomial.
   lwsp: length of workspace "wsp" (see below).
   
   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Kry*11*ydim.
   work_T: workspace of length MaxCoreNumForSERK_Kry*LWSP for Taylor expansion.
   workAux: workspace of length MaxCoreNumForSERK_Kry*2*ydim for Taylor techniques.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function evaluations.
   Taylor_mat_prod_cnt: the number of matrix products in Taylor expansion.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0;

  int ii, jj, kk;
  double sqstep, step_tmp;
  unsigned long long func_ev_num[MaxCoreNumForSERK_Kry], mat_prod_num[MaxCoreNumForSERK_Kry];

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Kry) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Kry);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Kry;
    static_flag=1;
  }

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step,
      mm, npmv, iflag;
    double *yn, *yn1, *fyn, *K1, *fK2, *exYn, tmp1, tmp2, *wsp, *wspAux;
    double *gn_diag, *g_diag_yn, *g_diag_yn_Plus, *g_diag_yn_Minus,
      *g_diag_yn_K1_dev2_Plus, *g_diag_yn_K1_dev2_Minus;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;
    mat_prod_num[ii_par]=0;

    wsp=&work_T[LWSP*ii_par];
    wspAux=&workAux[2*ydim*ii_par];

    ibase_work_step=11;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fyn=&work[ibase_work+ii];
    ii+=ydim;
    K1=&work[ibase_work+ii];
    ii+=ydim;
    fK2=&work[ibase_work+ii];
    ii+=ydim;
    exYn=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Minus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Minus=&work[ibase_work+ii];
    ii+=ydim;

    ibase=(ydim*traj_mini)*ii_par;
    ibase3p=(wdim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wj=&ran3p[ibase3p];
      ibase3p+=wdim;
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;
      
      /* See No. 42 and No. 56 in Note '12/'13. */
      step_tmp=step/2.0;
      DSEXPMVTRAY2(&RMAT.n,&mMin,&step_tmp,yn,exYn,wsp,&lwsp,wspAux,
		   &mm,&npmv,DGCOOV,&iflag);
      mat_prod_num[ii_par]+=npmv;
      if (0 != iflag) {
	printf("There was a problem in DSEXPMVTRAY2\n");
	exit(0);
      }

      ffunc(exYn,fyn); func_ev_num[ii_par]++; /* completed */
      for(ii=0; ii<ydim; ii++) {
	K1[ii]=exYn[ii]+step*fyn[ii]; /* completed */
      }

      for (jj=0; jj<wdim; jj++) {
	gfunc_diag(exYn,g_diag_yn); func_ev_num[ii_par]++; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1:
	  tmp1=g_diag_yn[ii];
	  break;
	case -1:
	  tmp1=-g_diag_yn[ii];
	  break;
	default:
	  tmp1=0;
	}
	yn1[ii]=K1[ii]+sqstep*SQ3*tmp1;
      }
      ffunc(yn1,fK2); func_ev_num[ii_par]++; /* completed */

      /* For details, see (337) and (338) in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	case -1: /* -1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	default: /* 0==wj[jj] */
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_Plus[ii]=exYn[ii]+step*tmp1/2.0;
	g_diag_yn_Minus[ii]=exYn[ii]-step*tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_yn_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=g_diag_yn[ii];
	} else {
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_K1_dev2_Plus[ii]=(exYn[ii]+K1[ii])/2.0+sqstep*tmp1/SQ2;
	g_diag_yn_K1_dev2_Minus[ii]=(exYn[ii]+K1[ii])/2.0-sqstep*tmp1/SQ2;
      }
      gfunc_diag(g_diag_yn_K1_dev2_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_K1_dev2_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=g_diag_yn_Plus[ii]-g_diag_yn_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp2=(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	case -1:
	  tmp2=-(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	default:
	  tmp2=0;
	}
	exYn[ii]+=step*(fyn[ii]+fK2[ii])/2.0+tmp1/2.0+sqstep*tmp2/2.0*SQ3;
      }

      step_tmp=step/2.0;
      DSEXPMVTRAY2(&RMAT.n,&mMin,&step_tmp,exYn,yn1,wsp,&lwsp,wspAux,
		   &mm,&npmv,DGCOOV,&iflag);
      mat_prod_num[ii_par]+=npmv;
      if (0 != iflag) {
	printf("There was a problem in DSEXPMVTRAY2\n");
	exit(0);
      }
      
    } /* End of loop for itr */
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Kry; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
    *Taylor_mat_prod_cnt+=mat_prod_num[ii_par];
  }
  return 0;
}
