/* File name: srock1_skrock.c */
/* Weak first order SKROCK2method for
   Ito SDEs with diagonal noise */
/* This file was made to put on GitHub (1-Sep-2023). */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include "mkl_lapacke.h"

#define MaxCoreNumForSROCK1 8 /* 8, 1, 4 *//* Maximum number of multi-core. */

#define MaxStageNumForSROCK1 104 /* Maximum stage number for Abdulle's srock scheme */

int GetSKROCKVal_eta2(int ss, double coe[]) ;

extern int OMP_wo1_skrock_eta2_for_DNoiseSDEs_WinMulti(int ydim,
						  unsigned long traj,
						  double *yvec,
						  double step,
						  char *ran2p,
						  void (*ffunc)(),
						  void (*gfunc_diag)(),
						  int ss,
						  double work[],
						  double *ynew)
/* wo1_skrock_for_SDEs_WinMulti for Open MP */
/* This function performs SKROCK scheme.
   It gives all trajectries for one step concerning SDEs with
   a multi-dimensional Winer process. If an error occurs, it will
   return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2p: pointer of the head of wdim*traj two-point distributed RVs
          with P(-1)=P(1)=1/2,
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,
   ss: stage numer of SROCK scheme.

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSROCK1*6*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, sM1, sM2, ii_par, wdim;
  static double coe[3*(MaxStageNumForSROCK1-1)+3];
  static int static_ss=0;
  double sqstep;
  
  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSROCK1) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSROCK1);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSROCK1;
    static_flag=1;
  }

  if(static_ss!=ss) {
    errflag=GetSKROCKVal_eta2(ss, coe);
    if (1==errflag) {
      printf("Error: ss is not among our selection numbers!\n");
      printf("       it must satisfy 3, 25, 50 or 100.\n");
      exit(1);
    }
    static_ss=ss;
  }
  sM1=ss-1;
  sM2=ss-2;

#pragma omp parallel for
  for (ii_par=0; ii_par<MaxCoreNumForSROCK1; ii_par++) {
    unsigned long itr;
    int ii, jj, ibase2p, ibase, ibaseCoe, ibase_work, ibase_work_step;
    double *kjM2, *kjM1, *kj, *yn, *yn1, *fn,
      tmp1, tmp2, tmp3, tmp4,
      *sqhg_diag_KjM2;
    char *wj;

    ibase_work_step=6;
    ibase_work=(ibase_work_step*ydim)*ii_par;
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fn=&work[ibase_work+ii];
    ii+=ydim;
    kjM2=&work[ibase_work+ii];
    ii+=ydim;
    kjM1=&work[ibase_work+ii];
    ii+=ydim;
    kj=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_KjM2=&work[ibase_work+ii];
    ii+=ydim;

    ibase=(ydim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wj=&ran2p[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;

      /* See (21) on Page 7 in [Abdulle:2018_preprint]. */
      /* Calculations for K_{s-1} */
      for(ii=0; ii<ydim; ii++) {
	kjM2[ii]=yn[ii];
      }	

      gfunc_diag(kjM2,sqhg_diag_KjM2);
      for (ii=0; ii<ydim; ii++) {
	  sqhg_diag_KjM2[ii]=sqstep*sqhg_diag_KjM2[ii]; /* completed */
      }

      for(ii=0; ii<ydim; ii++) {
	if (0<wj[ii]) {
	  tmp4=sqhg_diag_KjM2[ii];
	} else {
	  tmp4=-sqhg_diag_KjM2[ii];
	}
	kjM2[ii]=yn[ii]+coe[1]*tmp4;
      }
      ffunc(kjM2,fn);
      tmp1=step*coe[0];
      for(ii=0; ii<ydim; ii++) {
	if (0<wj[ii]) {
	  tmp4=sqhg_diag_KjM2[ii];
	} else {
	  tmp4=-sqhg_diag_KjM2[ii];
	}
	kjM2[ii]=yn[ii];
	kjM1[ii]=yn[ii]+tmp1*fn[ii]+coe[2]*tmp4;;
      }
      ibaseCoe=2;
      for(jj=1; jj<sM1; jj++) {
	ffunc(kjM1,fn);
	tmp1=step*coe[ibaseCoe+1];
	tmp3=coe[ibaseCoe+2];
	tmp2=1-tmp3;
	ibaseCoe+=2;
	for (ii=0; ii<ydim; ii++) {
	  kj[ii]=tmp1*fn[ii]+tmp2*kjM1[ii]+tmp3*kjM2[ii];
	  if (jj<sM2) {
	    kjM2[ii]=kjM1[ii];
	    kjM1[ii]=kj[ii];
	  }
	}
      }
      if (2==ss) {
	for (ii=0; ii<ydim; ii++) {
	  kj[ii]=kjM1[ii];
	  kjM1[ii]=kjM2[ii];
	}
      }
      
      ffunc(kj,fn);
      
      tmp1=step*coe[ibaseCoe+1];
      tmp3=coe[ibaseCoe+2];
      tmp2=1-tmp3;
      ibaseCoe+=2;
      
      for (ii=0; ii<ydim; ii++) {
	yn1[ii]=tmp1*fn[ii]+tmp2*kj[ii]+tmp3*kjM1[ii];
      }
    } /* End of loop for itr */
  }
  return 0;
}

extern int OMP_wo1_skrock_eta2_for_DNoiseSDEs_WinMulti_withCnt(int ydim,
						  unsigned long traj,
						  double *yvec,
						  double step,
						  char *ran2p,
						  void (*ffunc)(),
						  void (*gfunc_diag)(),
						  int ss,
						  double work[],
						  double *ynew,
						  unsigned long long *ev_cnt)
/* wo1_skrock_for_SDEs_WinMulti for Open MP */
/* This function performs SKROCK scheme.
   It gives all trajectries for one step concerning SDEs with
   a multi-dimensional Winer process. If an error occurs, it will
   return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2p: pointer of the head of wdim*traj two-point distributed RVs
          with P(-1)=P(1)=1/2,
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,
   ss: stage numer of SROCK scheme.

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSROCK1*6*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function and random number evaluations.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, sM1, sM2, ii_par, wdim;
  static double coe[3*(MaxStageNumForSROCK1-1)+3];
  static int static_ss=0;
  double sqstep;
  unsigned long long func_ev_num[MaxCoreNumForSROCK1];
  
  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSROCK1) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSROCK1);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSROCK1;
    static_flag=1;
  }

  if(static_ss!=ss) {
    errflag=GetSKROCKVal_eta2(ss, coe);
    if (1==errflag) {
      printf("Error: ss is not among our selection numbers!\n");
      printf("       it must satisfy 3, 25, 50 or 100.\n");
      exit(1);
    }
    static_ss=ss;
  }
  sM1=ss-1;
  sM2=ss-2;

#pragma omp parallel for
  for (ii_par=0; ii_par<MaxCoreNumForSROCK1; ii_par++) {
    unsigned long itr;
    int ii, jj, ibase2p, ibase, ibaseCoe, ibase_work, ibase_work_step;
    double *kjM2, *kjM1, *kj, *yn, *yn1, *fn,
      tmp1, tmp2, tmp3, tmp4,
      *sqhg_diag_KjM2;
    char *wj;

    func_ev_num[ii_par]=0;

    ibase_work_step=6;
    ibase_work=(ibase_work_step*ydim)*ii_par;
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fn=&work[ibase_work+ii];
    ii+=ydim;
    kjM2=&work[ibase_work+ii];
    ii+=ydim;
    kjM1=&work[ibase_work+ii];
    ii+=ydim;
    kj=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_KjM2=&work[ibase_work+ii];
    ii+=ydim;

    ibase=(ydim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wj=&ran2p[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;

      /* See (21) on Page 7 in [Abdulle:2018_preprint]. */
      /* Calculations for K_{s-1} */
      for(ii=0; ii<ydim; ii++) {
	kjM2[ii]=yn[ii];
      }	

      gfunc_diag(kjM2,sqhg_diag_KjM2); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	  sqhg_diag_KjM2[ii]=sqstep*sqhg_diag_KjM2[ii]; /* completed */
      }

      for(ii=0; ii<ydim; ii++) {
	if (0<wj[ii]) {
	  tmp4=sqhg_diag_KjM2[ii];
	} else {
	  tmp4=-sqhg_diag_KjM2[ii];
	}
	kjM2[ii]=yn[ii]+coe[1]*tmp4;
      }
      ffunc(kjM2,fn); func_ev_num[ii_par]++;
      tmp1=step*coe[0];
      for(ii=0; ii<ydim; ii++) {
	if (0<wj[ii]) {
	  tmp4=sqhg_diag_KjM2[ii];
	} else {
	  tmp4=-sqhg_diag_KjM2[ii];
	}
	kjM2[ii]=yn[ii];
	kjM1[ii]=yn[ii]+tmp1*fn[ii]+coe[2]*tmp4;;
      }
      ibaseCoe=2;
      for(jj=1; jj<sM1; jj++) {
	ffunc(kjM1,fn); func_ev_num[ii_par]++;
	tmp1=step*coe[ibaseCoe+1];
	tmp3=coe[ibaseCoe+2];
	tmp2=1-tmp3;
	ibaseCoe+=2;
	for (ii=0; ii<ydim; ii++) {
	  kj[ii]=tmp1*fn[ii]+tmp2*kjM1[ii]+tmp3*kjM2[ii];
	  if (jj<sM2) {
	    kjM2[ii]=kjM1[ii];
	    kjM1[ii]=kj[ii];
	  }
	}
      }
      if (2==ss) {
	for (ii=0; ii<ydim; ii++) {
	  kj[ii]=kjM1[ii];
	  kjM1[ii]=kjM2[ii];
	}
      }
      
      ffunc(kj,fn); func_ev_num[ii_par]++;
      
      tmp1=step*coe[ibaseCoe+1];
      tmp3=coe[ibaseCoe+2];
      tmp2=1-tmp3;
      ibaseCoe+=2;
      
      for (ii=0; ii<ydim; ii++) {
	yn1[ii]=tmp1*fn[ii]+tmp2*kj[ii]+tmp3*kjM1[ii];
      }
    } /* End of loop for itr */
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSROCK1; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
  }
  return 0;
}
