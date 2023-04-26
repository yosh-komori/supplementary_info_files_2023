/* Filename: srock2_using_Ab_values.c */
/* Weak second order SROCK2 method for
   Ito SDEs with noncommutative noise or diagonal noise */
/* This file was made to put on GitHub (26-Apr-2023). */
     
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include "mkl_lapacke.h"

#define SQ2 1.4142135623730950 /* sqrt(2) */
#define SQ3 1.7320508075688773 /* sqrt(3) */
#define SQ6 2.4494897427831781 /* sqrt(6) */

#define MaxCoreNumForAb 8 /* 8, 1, 4 *//* Maximum number of multi-core. */

#define MaxStageNum 200 /* Maximum stage number for Abdulle's srock2 */

extern int GetSROCK2Val_from_recp(int ss, double Mu[], double Ka[],
				  double *sig, double *tau, double *alpha);

extern int OMP_wo2_srock2_Ab_values_for_NonCommSDEs_WinMulti (int ydim,
							      unsigned long traj,
							      double *yvec,
							      double step,
							      char *ran2pFull,
							      char *ran3p,
							      void (*ffunc)(),
							      void (*gfunc_gene)(),
							      int wdim, int ss,
							      double work[],
							      double *ynew)
/* wo2_srock2_Ab_values_for_NonCommSDEs_WinMulti for Open MP */
/* This function performs SROCK2 scheme using the parameter values
   of Abdulle's code.
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
   ran3p: pointer of the head of wdim*traj three-point distributed RVs
          with P(-1)=P(1)=1/6 and P(0)=2/3,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   ffunc: drift coefficient,
   gfunc_gene: diffusion coefficients,
   wdim: dimension of Wiener process.
   ss: stage numer of SRK scheme.

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForAb*(7+6*wdim)*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, sM1, ii_par, sM2;
  double Mu[MaxStageNum], Ka[MaxStageNum-1], sigma, tau, dP0sM1, alpha,
    sigma_alpha, tau_alpha, sqstep, tmp;
  
  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForAb) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForAb);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForAb;
    static_flag=1;
  }

  errflag=GetSROCK2Val_from_recp(ss, Mu, Ka, &sigma, &tau, &alpha);
  if (1==errflag) {
    printf("Error: ss is not among our selection numbers!\n");
    printf("       it must satisfy 3<=ss<=22, or it must be\n");
    printf("       24, 26, 28, 30, 32, 35, 38, 41, 45, 49,\n");
    printf("       53, 58, 63, 68, 74, 80, 87, 95, 104, 114,\n");
    printf("       125, 137, 150, 165, 182 or 200.\n");
    exit(1);
  }

  sM1=ss-1;
  sM2=ss-2;
  tmp=1-alpha;
  sigma_alpha=tmp/2.0+alpha*sigma;
  tau_alpha=tmp*tmp/2.0+2*alpha*tmp*sigma+alpha*alpha*tau;

#pragma omp parallel for
  for (ii_par=0; ii_par<MaxCoreNumForAb; ii_par++) {
    unsigned long itr;
    int ii, jj, rr, qq, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    double *yn, *yn1, *fn, *ksM2, *ksM1, *ks, *hfKsM2, *hfKsM1_ast,
      tmp1, tmp2, tmp3;
    double **gn, *dummy, **sqhgrKs, **grKs_Plus, **grKs_Minus,
      **sqhgrKsM1_Plus, **sqhgrKsM1_Minus;
    char *wj, *wtj;

    if (NULL == (gn = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_srock2_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgrKs = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_srock2_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (grKs_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_srock2_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (grKs_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_srock2_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgrKsM1_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_srock2_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgrKsM1_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_srock2_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }

    ibase_work_step=7+6*wdim;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fn=&work[ibase_work+ii];
    ii+=ydim;
    ksM2=&work[ibase_work+ii];
    ii+=ydim;
    ksM1=&work[ibase_work+ii];
    ii+=ydim;
    ks=&work[ibase_work+ii];
    ii+=ydim;
    hfKsM2=&work[ibase_work+ii];
    ii+=ydim;
    hfKsM1_ast=&work[ibase_work+ii];
    ii+=ydim;
    for (jj=0; jj<wdim; jj++) {
      gn[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgrKs[jj]=&work[ibase_work+ii];
      ii+=ydim;
      grKs_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      grKs_Minus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgrKsM1_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgrKsM1_Minus[jj]=&work[ibase_work+ii];
      ii+=ydim;
    }

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
      
      /* See (3.29) in [Abdulle:2013]. */
      /* Calculations for K_{s-2} and K_{s}  */
      ffunc(yn,fn);
      tmp1=alpha*step*Mu[0];
      for(ii=0; ii<ydim; ii++) {
	ksM2[ii]=yn[ii];
	ksM1[ii]=yn[ii]+tmp1*fn[ii];
	hfKsM2[ii]=step*fn[ii]; /* completed if ss=2 */
      }
      for(jj=1; jj<ss; jj++) {
	ffunc(ksM1,fn);
	if (jj==sM2) {
	  for(ii=0; ii<ydim; ii++) {
	    hfKsM2[ii]=step*fn[ii]; /* completed if ss>2 */
	  }
	}
	tmp1=alpha*step*Mu[jj];
	tmp2=1+Ka[jj-1];
	tmp3=-Ka[jj-1];
	for (ii=0; ii<ydim; ii++) {
	  ks[ii]=tmp1*fn[ii]+tmp2*ksM1[ii]+tmp3*ksM2[ii];
	  if (jj<sM1) {
	    ksM2[ii]=ksM1[ii];
	    ksM1[ii]=ks[ii];
	  }
	}
      }

      for (rr=0; rr<wdim; rr++) {
	gfunc_gene(ks,rr,gn[rr]);
	for (ii=0; ii<ydim; ii++) {
	  sqhgrKs[rr][ii]=sqstep*gn[rr][ii]; /* completed */
	}
      }

      tmp1=2*tau_alpha;
      for (ii=0; ii<ydim; ii++) {
	tmp2=0;
	for (rr=0; rr<wdim; rr++) {
	  switch (wj[rr]) {
	  case 1:
	    tmp2+=sqhgrKs[rr][ii];
	    break;
	  case -1:
	    tmp2-=sqhgrKs[rr][ii];
	    break;
	    /* default: */
	    /* Nothing to do */
	  }
	}
	hfKsM1_ast[ii]=ksM2[ii]+tmp1*hfKsM2[ii]+SQ3*tmp2;
      }
      ffunc(hfKsM1_ast,fn);
      for (ii=0; ii<ydim; ii++) {
	hfKsM1_ast[ii]=step*fn[ii]; /* completed */
      }

      /* For details, see (3.2) and (3.29) in [Abdulle:2013]. */
      for (ii=0; ii<ydim; ii++) {
	for (rr=0; rr<wdim; rr++) {
	  tmp1=0;
	  switch (wj[rr]) {
	  case 1: /* 1==wj[rr] */
	    for (qq=0; qq<rr; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(3.0+wtj[rr])*sqhgrKs[qq][ii];
		break;
	      case -1:
		tmp1+=(-3.0+wtj[rr])*sqhgrKs[qq][ii];
		break;
	      default:
		tmp1+=(wtj[rr])*sqhgrKs[qq][ii];
	      }
	    }
	    tmp1+=2.0*sqhgrKs[rr][ii];
	    for (qq=rr+1; qq<wdim; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(3.0-wtj[qq])*sqhgrKs[qq][ii];
		break;
	      case -1:
		tmp1+=(-3.0-wtj[qq])*sqhgrKs[qq][ii];
		break;
	      default:
		tmp1+=(-wtj[qq])*sqhgrKs[qq][ii];
	      }
	    }
	    break;
	  case -1: /* -1==wj[rr] */
	    for (qq=0; qq<rr; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(-3.0+wtj[rr])*sqhgrKs[qq][ii];
		break;
	      case -1:
		tmp1+=(3.0+wtj[rr])*sqhgrKs[qq][ii];
		break;
	      default:
		tmp1+=(wtj[rr])*sqhgrKs[qq][ii];
	      }
	    }
	    tmp1+=2.0*sqhgrKs[rr][ii];
	    for (qq=rr+1; qq<wdim; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(-3.0-wtj[qq])*sqhgrKs[qq][ii];
		break;
	      case -1:
		tmp1+=(3.0-wtj[qq])*sqhgrKs[qq][ii];
		break;
	      default:
		tmp1+=(-wtj[qq])*sqhgrKs[qq][ii];
	      }
	    }
	    break;
	  default: /* 0==wj[rr] */
	    /* When rr=0, the following loop will not be performed. */
	    for (qq=0; qq<rr; qq++) {
	      tmp1+=(wtj[rr])*sqhgrKs[qq][ii];
	    }
	    tmp1+=-sqhgrKs[rr][ii];
	    for (qq=rr+1; qq<wdim; qq++) {
	      tmp1+=(-wtj[qq])*sqhgrKs[qq][ii];
	    }
	  } /* End of the switch for rr */
	  grKs_Plus[rr][ii]=ks[ii]+sqstep*tmp1/2.0;
	  grKs_Minus[rr][ii]=ks[ii]-sqstep*tmp1/2.0;
	} /* End of the loop for rr */
      } /* End of the loop for ii */
      for (rr=0; rr<wdim; rr++) {
	gfunc_gene(grKs_Plus[rr],rr,gn[rr]);
	for (ii=0; ii<ydim; ii++) {
	  grKs_Plus[rr][ii]=gn[rr][ii]; /* completed */
	}
	gfunc_gene(grKs_Minus[rr],rr,gn[rr]);
	for (ii=0; ii<ydim; ii++) {
	  grKs_Minus[rr][ii]=gn[rr][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	for (rr=0; rr<wdim; rr++) {
	  tmp1=0;
	  for (qq=0; qq<wdim; qq++) {
	    if (0<wtj[qq]) {
	      tmp1+=sqhgrKs[qq][ii];
	    } else {
	      tmp1-=sqhgrKs[qq][ii];
	    }
	  }
	  sqhgrKsM1_Plus[rr][ii]=ksM1[ii]+tmp1/SQ2;
	  sqhgrKsM1_Minus[rr][ii]=ksM1[ii]-tmp1/SQ2;
	}
      }
      for (rr=0; rr<wdim; rr++) {
	gfunc_gene(sqhgrKsM1_Plus[rr],rr,gn[rr]);
	for (ii=0; ii<ydim; ii++) {
	  sqhgrKsM1_Plus[rr][ii]=sqstep*gn[rr][ii]; /* completed */
	}
	gfunc_gene(sqhgrKsM1_Minus[rr],rr,gn[rr]);
	for (ii=0; ii<ydim; ii++) {
	  sqhgrKsM1_Minus[rr][ii]=sqstep*gn[rr][ii]; /* completed */
	}
      }

      tmp1=2*sigma_alpha-1.0/2.0;
      for (ii=0; ii<ydim; ii++) {
	tmp2=tmp3=0;
	for (rr=0; rr<wdim; rr++) {
	  tmp2+=grKs_Plus[rr][ii]-grKs_Minus[rr][ii];
	  switch (wj[rr]) {
	  case 1:
	    tmp3+=(sqhgrKsM1_Plus[rr][ii]+sqhgrKsM1_Minus[rr][ii]);
	    break;
	  case -1:
	    tmp3-=(sqhgrKsM1_Plus[rr][ii]+sqhgrKsM1_Minus[rr][ii]);
	    break;
	    /* default: */
	    /* Nothing to do. */
	  }
	}
	yn1[ii]=ksM2[ii]+tmp1*hfKsM2[ii]+hfKsM1_ast[ii]/2.0
	  +tmp2/2.0+tmp3/2.0*SQ3;
      }
    } /* End of loop for itr */
    free(gn);
    free(sqhgrKs);
    free(grKs_Plus);
    free(grKs_Minus);
    free(sqhgrKsM1_Plus);
    free(sqhgrKsM1_Minus);
  }
  return 0;
}

extern int
OMP_wo2_srock2_Ab_values_for_NonCommSDEs_WinMulti_withCnt(int ydim,
							  unsigned long traj,
							  double *yvec,
							  double step,
							  char *ran2pFull,
							  char *ran3p,
							  void (*ffunc)(),
							  void (*gfunc_gene)(),
							  int wdim, int ss,
							  double work[],
							  double *ynew,
							  unsigned long long *ev_cnt)
/* wo2_srock2_Ab_values_for_NonCommSDEs_WinMulti for Open MP */
/* This function performs SROCK2 scheme using the parameter values
   of Abdulle's code.
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
   ran3p: pointer of the head of wdim*traj three-point distributed RVs
          with P(-1)=P(1)=1/6 and P(0)=2/3,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   ffunc: drift coefficient,
   gfunc_gene: diffusion coefficients,
   wdim: dimension of Wiener process.
   ss: stage numer of SRK scheme.

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForAb*(7+6*wdim)*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function and random number evaluations.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, sM1, ii_par, sM2;
  double Mu[MaxStageNum], Ka[MaxStageNum-1], sigma, tau, dP0sM1, alpha,
    sigma_alpha, tau_alpha, sqstep, tmp;
  unsigned long long func_ev_num[MaxCoreNumForAb];
  
  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForAb) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForAb);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForAb;
    static_flag=1;
  }

  errflag=GetSROCK2Val_from_recp(ss, Mu, Ka, &sigma, &tau, &alpha);
  if (1==errflag) {
    printf("Error: ss is not among our selection numbers!\n");
    printf("       it must satisfy 3<=ss<=22, or it must be\n");
    printf("       24, 26, 28, 30, 32, 35, 38, 41, 45, 49,\n");
    printf("       53, 58, 63, 68, 74, 80, 87, 95, 104, 114,\n");
    printf("       125, 137, 150, 165, 182 or 200.\n");
    exit(1);
  }

  sM1=ss-1;
  sM2=ss-2;
  tmp=1-alpha;
  sigma_alpha=tmp/2.0+alpha*sigma;
  tau_alpha=tmp*tmp/2.0+2*alpha*tmp*sigma+alpha*alpha*tau;

#pragma omp parallel for
  for (ii_par=0; ii_par<MaxCoreNumForAb; ii_par++) {
    unsigned long itr;
    int ii, jj, rr, qq, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    double *yn, *yn1, *fn, *ksM2, *ksM1, *ks, *hfKsM2, *hfKsM1_ast,
      tmp1, tmp2, tmp3;
    double **gn, *dummy, **sqhgrKs, **grKs_Plus, **grKs_Minus,
      **sqhgrKsM1_Plus, **sqhgrKsM1_Minus;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;

    if (NULL == (gn = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_srock2_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgrKs = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_srock2_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (grKs_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_srock2_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (grKs_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_srock2_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgrKsM1_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_srock2_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgrKsM1_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_srock2_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }

    ibase_work_step=7+6*wdim;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fn=&work[ibase_work+ii];
    ii+=ydim;
    ksM2=&work[ibase_work+ii];
    ii+=ydim;
    ksM1=&work[ibase_work+ii];
    ii+=ydim;
    ks=&work[ibase_work+ii];
    ii+=ydim;
    hfKsM2=&work[ibase_work+ii];
    ii+=ydim;
    hfKsM1_ast=&work[ibase_work+ii];
    ii+=ydim;
    for (jj=0; jj<wdim; jj++) {
      gn[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgrKs[jj]=&work[ibase_work+ii];
      ii+=ydim;
      grKs_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      grKs_Minus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgrKsM1_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgrKsM1_Minus[jj]=&work[ibase_work+ii];
      ii+=ydim;
    }

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
      
      /* See (3.29) in [Abdulle:2013]. */
      /* Calculations for K_{s-2} and K_{s}  */
      ffunc(yn,fn); func_ev_num[ii_par]++;
      tmp1=alpha*step*Mu[0];
      for(ii=0; ii<ydim; ii++) {
	ksM2[ii]=yn[ii];
	ksM1[ii]=yn[ii]+tmp1*fn[ii];
	hfKsM2[ii]=step*fn[ii]; /* completed if ss=2 */
      }
      for(jj=1; jj<ss; jj++) {
	ffunc(ksM1,fn); func_ev_num[ii_par]++;
	if (jj==sM2) {
	  for(ii=0; ii<ydim; ii++) {
	    hfKsM2[ii]=step*fn[ii]; /* completed if ss>2 */
	  }
	}
	tmp1=alpha*step*Mu[jj];
	tmp2=1+Ka[jj-1];
	tmp3=-Ka[jj-1];
	for (ii=0; ii<ydim; ii++) {
	  ks[ii]=tmp1*fn[ii]+tmp2*ksM1[ii]+tmp3*ksM2[ii];
	  if (jj<sM1) {
	    ksM2[ii]=ksM1[ii];
	    ksM1[ii]=ks[ii];
	  }
	}
      }

      for (rr=0; rr<wdim; rr++) {
	gfunc_gene(ks,rr,gn[rr]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  sqhgrKs[rr][ii]=sqstep*gn[rr][ii]; /* completed */
	}
      }

      tmp1=2*tau_alpha;
      for (ii=0; ii<ydim; ii++) {
	tmp2=0;
	for (rr=0; rr<wdim; rr++) {
	  switch (wj[rr]) {
	  case 1:
	    tmp2+=sqhgrKs[rr][ii];
	    break;
	  case -1:
	    tmp2-=sqhgrKs[rr][ii];
	    break;
	    /* default: */
	    /* Nothing to do */
	  }
	}
	hfKsM1_ast[ii]=ksM2[ii]+tmp1*hfKsM2[ii]+SQ3*tmp2;
      }
      ffunc(hfKsM1_ast,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	hfKsM1_ast[ii]=step*fn[ii]; /* completed */
      }

      /* For details, see (3.2) and (3.29) in [Abdulle:2013]. */
      for (ii=0; ii<ydim; ii++) {
	for (rr=0; rr<wdim; rr++) {
	  tmp1=0;
	  switch (wj[rr]) {
	  case 1: /* 1==wj[rr] */
	    for (qq=0; qq<rr; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(3.0+wtj[rr])*sqhgrKs[qq][ii];
		break;
	      case -1:
		tmp1+=(-3.0+wtj[rr])*sqhgrKs[qq][ii];
		break;
	      default:
		tmp1+=(wtj[rr])*sqhgrKs[qq][ii];
	      }
	    }
	    tmp1+=2.0*sqhgrKs[rr][ii];
	    for (qq=rr+1; qq<wdim; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(3.0-wtj[qq])*sqhgrKs[qq][ii];
		break;
	      case -1:
		tmp1+=(-3.0-wtj[qq])*sqhgrKs[qq][ii];
		break;
	      default:
		tmp1+=(-wtj[qq])*sqhgrKs[qq][ii];
	      }
	    }
	    break;
	  case -1: /* -1==wj[rr] */
	    for (qq=0; qq<rr; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(-3.0+wtj[rr])*sqhgrKs[qq][ii];
		break;
	      case -1:
		tmp1+=(3.0+wtj[rr])*sqhgrKs[qq][ii];
		break;
	      default:
		tmp1+=(wtj[rr])*sqhgrKs[qq][ii];
	      }
	    }
	    tmp1+=2.0*sqhgrKs[rr][ii];
	    for (qq=rr+1; qq<wdim; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(-3.0-wtj[qq])*sqhgrKs[qq][ii];
		break;
	      case -1:
		tmp1+=(3.0-wtj[qq])*sqhgrKs[qq][ii];
		break;
	      default:
		tmp1+=(-wtj[qq])*sqhgrKs[qq][ii];
	      }
	    }
	    break;
	  default: /* 0==wj[rr] */
	    /* When rr=0, the following loop will not be performed. */
	    for (qq=0; qq<rr; qq++) {
	      tmp1+=(wtj[rr])*sqhgrKs[qq][ii];
	    }
	    tmp1+=-sqhgrKs[rr][ii];
	    for (qq=rr+1; qq<wdim; qq++) {
	      tmp1+=(-wtj[qq])*sqhgrKs[qq][ii];
	    }
	  } /* End of the switch for rr */
	  grKs_Plus[rr][ii]=ks[ii]+sqstep*tmp1/2.0;
	  grKs_Minus[rr][ii]=ks[ii]-sqstep*tmp1/2.0;
	} /* End of the loop for rr */
      } /* End of the loop for ii */
      for (rr=0; rr<wdim; rr++) {
	gfunc_gene(grKs_Plus[rr],rr,gn[rr]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  grKs_Plus[rr][ii]=gn[rr][ii]; /* completed */
	}
	gfunc_gene(grKs_Minus[rr],rr,gn[rr]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  grKs_Minus[rr][ii]=gn[rr][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	for (rr=0; rr<wdim; rr++) {
	  tmp1=0;
	  for (qq=0; qq<wdim; qq++) {
	    if (0<wtj[qq]) {
	      tmp1+=sqhgrKs[qq][ii];
	    } else {
	      tmp1-=sqhgrKs[qq][ii];
	    }
	  }
	  sqhgrKsM1_Plus[rr][ii]=ksM1[ii]+tmp1/SQ2;
	  sqhgrKsM1_Minus[rr][ii]=ksM1[ii]-tmp1/SQ2;
	}
      }
      for (rr=0; rr<wdim; rr++) {
	gfunc_gene(sqhgrKsM1_Plus[rr],rr,gn[rr]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  sqhgrKsM1_Plus[rr][ii]=sqstep*gn[rr][ii]; /* completed */
	}
	gfunc_gene(sqhgrKsM1_Minus[rr],rr,gn[rr]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  sqhgrKsM1_Minus[rr][ii]=sqstep*gn[rr][ii]; /* completed */
	}
      }

      tmp1=2*sigma_alpha-1.0/2.0;
      for (ii=0; ii<ydim; ii++) {
	tmp2=tmp3=0;
	for (rr=0; rr<wdim; rr++) {
	  tmp2+=grKs_Plus[rr][ii]-grKs_Minus[rr][ii];
	  switch (wj[rr]) {
	  case 1:
	    tmp3+=(sqhgrKsM1_Plus[rr][ii]+sqhgrKsM1_Minus[rr][ii]);
	    break;
	  case -1:
	    tmp3-=(sqhgrKsM1_Plus[rr][ii]+sqhgrKsM1_Minus[rr][ii]);
	    break;
	    /* default: */
	    /* Nothing to do. */
	  }
	}
	yn1[ii]=ksM2[ii]+tmp1*hfKsM2[ii]+hfKsM1_ast[ii]/2.0
	  +tmp2/2.0+tmp3/2.0*SQ3;
      }
    } /* End of loop for itr */
    free(gn);
    free(sqhgrKs);
    free(grKs_Plus);
    free(grKs_Minus);
    free(sqhgrKsM1_Plus);
    free(sqhgrKsM1_Minus);
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForAb; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
  }
  return 0;
}

extern int OMP_wo2_srock2_Ab_values_for_DNoiseSDEs_WinMulti (int ydim,
							     unsigned long traj,
							     double *yvec,
							     double step,
							     char *ran2pFull,
							     char *ran3p,
							     void (*ffunc)(),
							     void (*gfunc_diag)(),
							     int ss,
							     double work[],
							     double *ynew)
/* wo2_srock2_Ab_values_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs SROCK2 scheme using the parameter values
   of Abdulle's code.
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
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,
   ss: stage numer of SRK scheme.

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForAb*13*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, sM1, ii_par, sM2, wdim;
  double Mu[MaxStageNum], Ka[MaxStageNum-1], sigma, tau, dP0sM1, alpha,
    sigma_alpha, tau_alpha, sqstep, tmp;
  
  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForAb) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForAb);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForAb;
    static_flag=1;
  }

  errflag=GetSROCK2Val_from_recp(ss, Mu, Ka, &sigma, &tau, &alpha);
  if (1==errflag) {
    printf("Error: ss is not among our selection numbers!\n");
    printf("       it must satisfy 3<=ss<=22, or it must be\n");
    printf("       24, 26, 28, 30, 32, 35, 38, 41, 45, 49,\n");
    printf("       53, 58, 63, 68, 74, 80, 87, 95, 104, 114,\n");
    printf("       125, 137, 150, 165, 182 or 200.\n");
    exit(1);
  }

  sM1=ss-1;
  sM2=ss-2;
  tmp=1-alpha;
  sigma_alpha=tmp/2.0+alpha*sigma;
  tau_alpha=tmp*tmp/2.0+2*alpha*tmp*sigma+alpha*alpha*tau;

#pragma omp parallel for
  for (ii_par=0; ii_par<MaxCoreNumForAb; ii_par++) {
    unsigned long itr;
    int ii, jj, rr, qq, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    double *yn, *yn1, *fn, *ksM2, *ksM1, *ks, *hfKsM2, *hfKsM1_ast,
      tmp1, tmp2, tmp3,
      *gn_diag, *sqhg_diag_Ks, *g_diag_Ks_Plus, *g_diag_Ks_Minus,
      *sqhg_diag_KsM1_Plus, *sqhg_diag_KsM1_Minus;
    char *wj, *wtj;

    ibase_work_step=13;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fn=&work[ibase_work+ii];
    ii+=ydim;
    ksM2=&work[ibase_work+ii];
    ii+=ydim;
    ksM1=&work[ibase_work+ii];
    ii+=ydim;
    ks=&work[ibase_work+ii];
    ii+=ydim;
    hfKsM2=&work[ibase_work+ii];
    ii+=ydim;
    hfKsM1_ast=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_Ks=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_Ks_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_Ks_Minus=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_KsM1_Plus=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_KsM1_Minus=&work[ibase_work+ii];
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
      
      /* See (3.29) and a memo on p. A1804 in [Abdulle:2013]. */
      /* Calculations for K_{s-2} and K_{s}  */
      ffunc(yn,fn);
      tmp1=alpha*step*Mu[0];
      for(ii=0; ii<ydim; ii++) {
	ksM2[ii]=yn[ii];
	ksM1[ii]=yn[ii]+tmp1*fn[ii];
	hfKsM2[ii]=step*fn[ii]; /* completed if ss=2 */
      }
      for(jj=1; jj<ss; jj++) {
	ffunc(ksM1,fn);
	if (jj==sM2) {
	  for(ii=0; ii<ydim; ii++) {
	    hfKsM2[ii]=step*fn[ii]; /* completed if ss>2 */
	  }
	}
	tmp1=alpha*step*Mu[jj];
	tmp2=1+Ka[jj-1];
	tmp3=-Ka[jj-1];
	for (ii=0; ii<ydim; ii++) {
	  ks[ii]=tmp1*fn[ii]+tmp2*ksM1[ii]+tmp3*ksM2[ii];
	  if (jj<sM1) {
	    ksM2[ii]=ksM1[ii];
	    ksM1[ii]=ks[ii];
	  }
	}
      }

      gfunc_diag(ks,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_Ks[ii]=sqstep*gn_diag[ii]; /* completed */
      }

      tmp1=2*tau_alpha;
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1:
	  tmp2=sqhg_diag_Ks[ii];
	  break;
	case -1:
	  tmp2=-sqhg_diag_Ks[ii];
	  break;
	default:
	  tmp2=0;
	}
	hfKsM1_ast[ii]=ksM2[ii]+tmp1*hfKsM2[ii]+SQ3*tmp2;
      }
      ffunc(hfKsM1_ast,fn);
      for (ii=0; ii<ydim; ii++) {
	hfKsM1_ast[ii]=step*fn[ii]; /* completed */
      }

      /* For details, see (3.2) and (3.29) in [Abdulle:2013]. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[ii] */
	  tmp1=2.0*sqhg_diag_Ks[ii];
	  break;
	case -1: /* -1==wj[ii] */
	  tmp1=2.0*sqhg_diag_Ks[ii];
	  break;
	default: /* 0==wj[ii] */
	  tmp1=-sqhg_diag_Ks[ii];
	}
	g_diag_Ks_Plus[ii]=ks[ii]+sqstep*tmp1/2.0;
	g_diag_Ks_Minus[ii]=ks[ii]-sqstep*tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_Ks_Plus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_Ks_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_Ks_Minus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_Ks_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=sqhg_diag_Ks[ii];
	} else {
	  tmp1=-sqhg_diag_Ks[ii];
	}
	sqhg_diag_KsM1_Plus[ii]=ksM1[ii]+tmp1/SQ2;
	sqhg_diag_KsM1_Minus[ii]=ksM1[ii]-tmp1/SQ2;
      }
      gfunc_diag(sqhg_diag_KsM1_Plus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_KsM1_Plus[ii]=sqstep*gn_diag[ii]; /* completed */
      }
      gfunc_diag(sqhg_diag_KsM1_Minus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_KsM1_Minus[ii]=sqstep*gn_diag[ii]; /* completed */
      }

      tmp1=2*sigma_alpha-1.0/2.0;
      for (ii=0; ii<ydim; ii++) {
	tmp2=g_diag_Ks_Plus[ii]-g_diag_Ks_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp3=(sqhg_diag_KsM1_Plus[ii]+sqhg_diag_KsM1_Minus[ii]);
	  break;
	case -1:
	  tmp3=-(sqhg_diag_KsM1_Plus[ii]+sqhg_diag_KsM1_Minus[ii]);
	  break;
	default:
	  tmp3=0;
	}
	yn1[ii]=ksM2[ii]+tmp1*hfKsM2[ii]+hfKsM1_ast[ii]/2.0
	  +tmp2/2.0+tmp3/2.0*SQ3;
      }
    } /* End of loop for itr */
  }
  return 0;
}

extern int
OMP_wo2_srock2_Ab_values_for_DNoiseSDEs_WinMulti_withCnt (int ydim,
							  unsigned long traj,
							  double *yvec,
							  double step,
							  char *ran2pFull,
							  char *ran3p,
							  void (*ffunc)(),
							  void (*gfunc_diag)(),
							  int ss,
							  double work[],
							  double *ynew,
							  unsigned long long *ev_cnt)
/* wo2_srock2_Ab_values_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs SROCK2 scheme using the parameter values
   of Abdulle's code.
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
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,
   ss: stage numer of SRK scheme.

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForAb*13*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function and random number evaluations.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, sM1, ii_par, sM2, wdim;
  double Mu[MaxStageNum], Ka[MaxStageNum-1], sigma, tau, dP0sM1, alpha,
    sigma_alpha, tau_alpha, sqstep, tmp;
  unsigned long long func_ev_num[MaxCoreNumForAb];
  
  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForAb) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForAb);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForAb;
    static_flag=1;
  }

  errflag=GetSROCK2Val_from_recp(ss, Mu, Ka, &sigma, &tau, &alpha);
  if (1==errflag) {
    printf("Error: ss is not among our selection numbers!\n");
    printf("       it must satisfy 3<=ss<=22, or it must be\n");
    printf("       24, 26, 28, 30, 32, 35, 38, 41, 45, 49,\n");
    printf("       53, 58, 63, 68, 74, 80, 87, 95, 104, 114,\n");
    printf("       125, 137, 150, 165, 182 or 200.\n");
    exit(1);
  }

  sM1=ss-1;
  sM2=ss-2;
  tmp=1-alpha;
  sigma_alpha=tmp/2.0+alpha*sigma;
  tau_alpha=tmp*tmp/2.0+2*alpha*tmp*sigma+alpha*alpha*tau;

#pragma omp parallel for
  for (ii_par=0; ii_par<MaxCoreNumForAb; ii_par++) {
    unsigned long itr;
    int ii, jj, rr, qq, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    double *yn, *yn1, *fn, *ksM2, *ksM1, *ks, *hfKsM2, *hfKsM1_ast,
      tmp1, tmp2, tmp3,
      *gn_diag, *sqhg_diag_Ks, *g_diag_Ks_Plus, *g_diag_Ks_Minus,
      *sqhg_diag_KsM1_Plus, *sqhg_diag_KsM1_Minus;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;

    ibase_work_step=13;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fn=&work[ibase_work+ii];
    ii+=ydim;
    ksM2=&work[ibase_work+ii];
    ii+=ydim;
    ksM1=&work[ibase_work+ii];
    ii+=ydim;
    ks=&work[ibase_work+ii];
    ii+=ydim;
    hfKsM2=&work[ibase_work+ii];
    ii+=ydim;
    hfKsM1_ast=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_Ks=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_Ks_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_Ks_Minus=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_KsM1_Plus=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_KsM1_Minus=&work[ibase_work+ii];
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
      
      /* See (3.29) and a memo on p. A1804 in [Abdulle:2013]. */
      /* Calculations for K_{s-2} and K_{s}  */
      ffunc(yn,fn); func_ev_num[ii_par]++;
      tmp1=alpha*step*Mu[0];
      for(ii=0; ii<ydim; ii++) {
	ksM2[ii]=yn[ii];
	ksM1[ii]=yn[ii]+tmp1*fn[ii];
	hfKsM2[ii]=step*fn[ii]; /* completed if ss=2 */
      }
      for(jj=1; jj<ss; jj++) {
	ffunc(ksM1,fn); func_ev_num[ii_par]++;
	if (jj==sM2) {
	  for(ii=0; ii<ydim; ii++) {
	    hfKsM2[ii]=step*fn[ii]; /* completed if ss>2 */
	  }
	}
	tmp1=alpha*step*Mu[jj];
	tmp2=1+Ka[jj-1];
	tmp3=-Ka[jj-1];
	for (ii=0; ii<ydim; ii++) {
	  ks[ii]=tmp1*fn[ii]+tmp2*ksM1[ii]+tmp3*ksM2[ii];
	  if (jj<sM1) {
	    ksM2[ii]=ksM1[ii];
	    ksM1[ii]=ks[ii];
	  }
	}
      }

      gfunc_diag(ks,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_Ks[ii]=sqstep*gn_diag[ii]; /* completed */
      }

      tmp1=2*tau_alpha;
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1:
	  tmp2=sqhg_diag_Ks[ii];
	  break;
	case -1:
	  tmp2=-sqhg_diag_Ks[ii];
	  break;
	default:
	  tmp2=0;
	}
	hfKsM1_ast[ii]=ksM2[ii]+tmp1*hfKsM2[ii]+SQ3*tmp2;
      }
      ffunc(hfKsM1_ast,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	hfKsM1_ast[ii]=step*fn[ii]; /* completed */
      }

      /* For details, see (3.2) and (3.29) in [Abdulle:2013]. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[ii] */
	  tmp1=2.0*sqhg_diag_Ks[ii];
	  break;
	case -1: /* -1==wj[ii] */
	  tmp1=2.0*sqhg_diag_Ks[ii];
	  break;
	default: /* 0==wj[ii] */
	  tmp1=-sqhg_diag_Ks[ii];
	}
	g_diag_Ks_Plus[ii]=ks[ii]+sqstep*tmp1/2.0;
	g_diag_Ks_Minus[ii]=ks[ii]-sqstep*tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_Ks_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_Ks_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_Ks_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_Ks_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=sqhg_diag_Ks[ii];
	} else {
	  tmp1=-sqhg_diag_Ks[ii];
	}
	sqhg_diag_KsM1_Plus[ii]=ksM1[ii]+tmp1/SQ2;
	sqhg_diag_KsM1_Minus[ii]=ksM1[ii]-tmp1/SQ2;
      }
      gfunc_diag(sqhg_diag_KsM1_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_KsM1_Plus[ii]=sqstep*gn_diag[ii]; /* completed */
      }
      gfunc_diag(sqhg_diag_KsM1_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_KsM1_Minus[ii]=sqstep*gn_diag[ii]; /* completed */
      }

      tmp1=2*sigma_alpha-1.0/2.0;
      for (ii=0; ii<ydim; ii++) {
	tmp2=g_diag_Ks_Plus[ii]-g_diag_Ks_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp3=(sqhg_diag_KsM1_Plus[ii]+sqhg_diag_KsM1_Minus[ii]);
	  break;
	case -1:
	  tmp3=-(sqhg_diag_KsM1_Plus[ii]+sqhg_diag_KsM1_Minus[ii]);
	  break;
	default:
	  tmp3=0;
	}
	yn1[ii]=ksM2[ii]+tmp1*hfKsM2[ii]+hfKsM1_ast[ii]/2.0
	  +tmp2/2.0+tmp3/2.0*SQ3;
      }
    } /* End of loop for itr */
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForAb; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
  }
  return 0;
}
