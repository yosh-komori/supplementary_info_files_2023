/* File name: splitting_srock2.c */
/* Split SROCK2 methods for
   Ito SDEs with noncommutative noise
   and
   Ito SDEs with diagonal noise */
/* This file was made to put on GitHub (21-Sep-2023). */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SQ2 1.4142135623730950 /* sqrt(2) */
#define SQ3 1.7320508075688773 /* sqrt(3) */

#define MaxCoreNum 8 /* 8, 1, 4 *//* Maximum number of multi-core. */

#define MaxStageNum 104 /* Maximum stage number for splitting SROCK2 method */

#define MaxStageNum_Ab 200 /* Maximum stage number for splitting-like
			    SROCK2 methods using Abdulle's parameter values */


extern int GetSROCK2Val_from_recp(int ss, double Mu[], double Ka[],
				  double *sig, double *tau, double *alpha);

extern int set_Al_for_revA_srock2(int ss, double *alpha) {
  /*
    3<=ss<=200: parameter values will be set for splitting-like SROCK2
    method (73) on Page No. 10 in Note '16. For details, see
    Math8_Research _E _for _ 2 nd_Extended _Splitting _ROCK2 _for _ItoSDE.nb
    (Ver. 0) in the folder "2016/Splitting_ROCK2_for_ItoSDE_MultiWin".
    Main stages are ss=3, 4, ..., 22, 24, 26, 28, 30, 32, 35, 38,
    41, 45, 49, 53, 58, 63, 68, 74, 80, 87, 95, 104, 114, 125,
    137, 150, 165, 182 and 200.
   */
  switch (ss) {
  case 3:
    *alpha=2.63;
    return 0;
  case 4:
    *alpha=2.73;
    return 0;
  case 5:
    *alpha=2.74;
    return 0;
  case 6:
    *alpha=2.73;
    return 0;
  case 7:
    *alpha=2.73;
    return 0;
  case 8:
    *alpha=2.72;
    return 0;
  case 9:
    *alpha=2.72;
    return 0;
  case 10:
    *alpha=2.72;
    return 0;
  case 11:
    *alpha=2.71;
    return 0;
  case 12:
    *alpha=2.71;
    return 0;
  case 13:
    *alpha=2.71;
    return 0;
  case 14:
    *alpha=2.71;
    return 0;
  case 15:
    *alpha=2.71;
    return 0;
  case 16:
    *alpha=2.71;
    return 0;
  case 17:
    *alpha=2.71;
    return 0;
  case 18:
    *alpha=2.71;
    return 0;
  case 19:
    *alpha=2.71;
    return 0;
  case 20:
    *alpha=2.71;
    return 0;
  case 21:
    *alpha=2.71;
    return 0;
  case 22:
    *alpha=2.72;
    return 0;
  case 24:
    *alpha=2.72;
    return 0;
  case 26:
    *alpha=2.72;
    return 0;
  case 28:
    *alpha=2.72;
    return 0;
  case 30:
    *alpha=2.72;
    return 0;
  case 32:
    *alpha=2.72;
    return 0;
  case 35:
    *alpha=2.72;
    return 0;
  case 38:
    *alpha=2.72;
    return 0;
  case 41:
    *alpha=2.72;
    return 0;
  case 45:
    *alpha=2.72;
    return 0;
  case 49:
    *alpha=2.72;
    return 0;
  case 53:
    *alpha=2.72;
    return 0;
  case 58:
    *alpha=2.72;
    return 0;
  case 63:
    *alpha=2.72;
    return 0;
  case 68:
    *alpha=2.72;
    return 0;
  case 74:
    *alpha=2.72;
    return 0;
  case 80:
    *alpha=2.72;
    return 0;
  case 87:
    *alpha=2.72;
    return 0;
  case 95:
    *alpha=2.72;
    return 0;
  case 104:
    *alpha=2.72;
    return 0;
  case 114:
    *alpha=2.72;
    return 0;
  case 125:
    *alpha=2.72;
    return 0;
  case 137:
    *alpha=2.72;
    return 0;
  case 150:
    *alpha=2.72;
    return 0;
  case 165:
    *alpha=2.72;
    return 0;
  case 182:
    *alpha=2.72;
    return 0;
  case 200:
    *alpha=2.72;
    return 0;
  default:
    return 1;
  }
}

extern int
OMP_wo2_revA_srock2_avr_Al_Ab_val_for_NonCommSDEs_WinMulti (int ydim,
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
/* wo2_revA_srock2_Al_2_75_Ab_val_for_NonCommSDEs_WinMulti for Open MP */
/* This function performs the splitting-like SROCK2 method using the parameter
   values of Abdulle's code.
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
   work: workspace of length MaxCoreNum*(7+6*wdim)*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, sM2, ii_par, sM3;
  double Mu[MaxStageNum_Ab], Ka[MaxStageNum_Ab-1], sigma, tau, alpha,
    sigma_alpha, tau_alpha, sqstep, tmp, half_step;
  
  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);
  half_step = step/2.0;

  if (0==static_flag) {
    if (0!=traj%MaxCoreNum) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNum);
      exit(1);
    }
    traj_mini=traj/MaxCoreNum;
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
  errflag=set_Al_for_revA_srock2(ss,&alpha); /* Over written */
  if (1==errflag) {
    printf("Error in set_Al_for_revA_srock2\n");
  }
  sM2=ss-2;
  sM3=ss-3;
  tmp=1-alpha;
  sigma_alpha=tmp/2.0+alpha*sigma;
  tau_alpha=tmp*tmp/2.0+2*alpha*tmp*sigma+alpha*alpha*tau;

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNum; ii_par++) {
    unsigned long itr;
    int ii, jj, rr, qq, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    double *yn, *yn1, *fn, *phiM2, *phiM1, *phi, *hD2fPhsM2, *hD2fPhsM1_til,
      tmp1, tmp2, tmp3;
    double **gn, *dummy, **sqhgrPhsM2, **grPhsM2_Plus, **grPhsM2_Minus,
      **sqhgrPhsM2_Plus, **sqhgrPhsM2_Minus;
    char *wj, *wtj;

    if (NULL == (gn = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_spl_revA_srock2_Ab_val_for_NonCommSDEs_WinMulti");
      exit(0);
    }
    if (NULL == (sqhgrPhsM2 = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_spl_revA_srock2_Ab_val_for_NonCommSDEs_WinMulti");
      exit(0);
    }
    if (NULL == (grPhsM2_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_spl_revA_srock2_Ab_val_for_NonCommSDEs_WinMulti");
      exit(0);
    }
    if (NULL == (grPhsM2_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_spl_revA_srock2_Ab_val_for_NonCommSDEs_WinMulti");
      exit(0);
    }
    if (NULL == (sqhgrPhsM2_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_spl_revA_srock2_Ab_val_for_NonCommSDEs_WinMulti");
      exit(0);
    }
    if (NULL == (sqhgrPhsM2_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_spl_revA_srock2_Ab_val_for_NonCommSDEs_WinMulti");
      exit(0);
    }

    ibase_work_step=7+6*wdim;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fn=&work[ibase_work+ii];
    ii+=ydim;
    phiM2=&work[ibase_work+ii];
    ii+=ydim;
    phiM1=&work[ibase_work+ii];
    ii+=ydim;
    phi=&work[ibase_work+ii];
    ii+=ydim;
    hD2fPhsM2=&work[ibase_work+ii];
    ii+=ydim;
    hD2fPhsM1_til=&work[ibase_work+ii];
    ii+=ydim;
    for (jj=0; jj<wdim; jj++) {
      gn[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgrPhsM2[jj]=&work[ibase_work+ii];
      ii+=ydim;
      grPhsM2_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      grPhsM2_Minus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgrPhsM2_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgrPhsM2_Minus[jj]=&work[ibase_work+ii];
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
      
      /* See (73) in [Note '16, Page No. 10]. */
      /* Calculations for K_{s-2} */
      ffunc(yn,fn);
      tmp1=alpha*half_step*Mu[0];
      for(ii=0; ii<ydim; ii++) { /* Initialization */
	if (1<sM2) {
	  phiM2[ii]=yn[ii];
	  phiM1[ii]=yn[ii]+tmp1*fn[ii];
	} else { /* Case of 1==sM2 */
	  phiM1[ii]=yn[ii];
	  phi[ii]=yn[ii]+tmp1*fn[ii];
	}
      }
      for(jj=1; jj<sM2; jj++) { /* Main iteration in Case of 1<sM2 */
	ffunc(phiM1,fn);
	tmp1=alpha*half_step*Mu[jj];
	tmp2=1+Ka[jj-1];
	tmp3=-Ka[jj-1];
	for (ii=0; ii<ydim; ii++) {
	  phi[ii]=tmp1*fn[ii]+tmp2*phiM1[ii]+tmp3*phiM2[ii];
	  if (jj<sM3) {
	    phiM2[ii]=phiM1[ii];
	    phiM1[ii]=phi[ii];
	  }
	}
      }

      ffunc(phi,fn);
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM2[ii]=half_step*fn[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=phi[ii]+2*tau_alpha*hD2fPhsM2[ii];
      }
      ffunc(hD2fPhsM1_til,fn);
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=half_step*fn[ii]; /* completed */
      }

      tmp1=2*sigma_alpha-1.0/2.0;
      for (ii=0; ii<ydim; ii++) {
	yn1[ii]=phi[ii]+tmp1*hD2fPhsM2[ii]+hD2fPhsM1_til[ii]/2.0;
	phiM1[ii]=phi[ii]+2*sigma_alpha*hD2fPhsM2[ii];
	/* The first splitting has finished. */
      }
      /* In what follows, note that
	 phi <---> Phi_{s-2} in (73),
	 yn1 <---> \tilde{Phi}_{s} in (73),
	 phiM1 <---> \hat{Phi}_{s} in (73).
       */

      for (rr=0; rr<wdim; rr++) {
	gfunc_gene(phi,rr,gn[rr]);
	for (ii=0; ii<ydim; ii++) {
	  sqhgrPhsM2[rr][ii]=sqstep*gn[rr][ii]; /* completed */
	}
      }

      /* For details, see (3.2) in [Abdulle:2013]
	 and (73) in [Note '16, Page No. 10]. */
      for (ii=0; ii<ydim; ii++) {
	for (rr=0; rr<wdim; rr++) {
	  tmp1=0;
	  switch (wj[rr]) {
	  case 1: /* 1==wj[rr] */
	    for (qq=0; qq<rr; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(3.0+wtj[rr])*sqhgrPhsM2[qq][ii];
		break;
	      case -1:
		tmp1+=(-3.0+wtj[rr])*sqhgrPhsM2[qq][ii];
		break;
	      default:
		tmp1+=(wtj[rr])*sqhgrPhsM2[qq][ii];
	      }
	    }
	    tmp1+=2.0*sqhgrPhsM2[rr][ii];
	    for (qq=rr+1; qq<wdim; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(3.0-wtj[qq])*sqhgrPhsM2[qq][ii];
		break;
	      case -1:
		tmp1+=(-3.0-wtj[qq])*sqhgrPhsM2[qq][ii];
		break;
	      default:
		tmp1+=(-wtj[qq])*sqhgrPhsM2[qq][ii];
	      }
	    }
	    break;
	  case -1: /* -1==wj[rr] */
	    for (qq=0; qq<rr; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(-3.0+wtj[rr])*sqhgrPhsM2[qq][ii];
		break;
	      case -1:
		tmp1+=(3.0+wtj[rr])*sqhgrPhsM2[qq][ii];
		break;
	      default:
		tmp1+=(wtj[rr])*sqhgrPhsM2[qq][ii];
	      }
	    }
	    tmp1+=2.0*sqhgrPhsM2[rr][ii];
	    for (qq=rr+1; qq<wdim; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(-3.0-wtj[qq])*sqhgrPhsM2[qq][ii];
		break;
	      case -1:
		tmp1+=(3.0-wtj[qq])*sqhgrPhsM2[qq][ii];
		break;
	      default:
		tmp1+=(-wtj[qq])*sqhgrPhsM2[qq][ii];
	      }
	    }
	    break;
	  default: /* 0==wj[rr] */
	    /* When rr=0, the following loop will not be performed. */
	    for (qq=0; qq<rr; qq++) {
	      tmp1+=(wtj[rr])*sqhgrPhsM2[qq][ii];
	    }
	    tmp1+=-sqhgrPhsM2[rr][ii];
	    for (qq=rr+1; qq<wdim; qq++) {
	      tmp1+=(-wtj[qq])*sqhgrPhsM2[qq][ii];
	    }
	  } /* End of the switch for rr */
	  grPhsM2_Plus[rr][ii]=phi[ii]+sqstep*tmp1/2.0;
	  grPhsM2_Minus[rr][ii]=phi[ii]-sqstep*tmp1/2.0;
	} /* End of the loop for rr */
      } /* End of the loop for ii */
      for (rr=0; rr<wdim; rr++) {
	gfunc_gene(grPhsM2_Plus[rr],rr,gn[rr]);
	for (ii=0; ii<ydim; ii++) {
	  grPhsM2_Plus[rr][ii]=gn[rr][ii]; /* completed */
	}
	gfunc_gene(grPhsM2_Minus[rr],rr,gn[rr]);
	for (ii=0; ii<ydim; ii++) {
	  grPhsM2_Minus[rr][ii]=gn[rr][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	for (rr=0; rr<wdim; rr++) {
	  tmp1=0;
	  for (qq=0; qq<wdim; qq++) {
	    if (0<wtj[qq]) {
	      tmp1+=sqhgrPhsM2[qq][ii];
	    } else {
	      tmp1-=sqhgrPhsM2[qq][ii];
	    }
	  }
	  sqhgrPhsM2_Plus[rr][ii]=phiM1[ii]+tmp1/SQ2;
	  sqhgrPhsM2_Minus[rr][ii]=phiM1[ii]-tmp1/SQ2;
	}
      }
      for (rr=0; rr<wdim; rr++) {
	gfunc_gene(sqhgrPhsM2_Plus[rr],rr,gn[rr]);
	for (ii=0; ii<ydim; ii++) {
	  sqhgrPhsM2_Plus[rr][ii]=sqstep*gn[rr][ii]; /* completed */
	}
	gfunc_gene(sqhgrPhsM2_Minus[rr],rr,gn[rr]);
	for (ii=0; ii<ydim; ii++) {
	  sqhgrPhsM2_Minus[rr][ii]=sqstep*gn[rr][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=tmp2=0;
	for (rr=0; rr<wdim; rr++) {
	  tmp1+=grPhsM2_Plus[rr][ii]-grPhsM2_Minus[rr][ii];
	  switch (wj[rr]) {
	  case 1:
	    tmp2+=(sqhgrPhsM2_Plus[rr][ii]+sqhgrPhsM2_Minus[rr][ii]);
	    break;
	  case -1:
	    tmp2-=(sqhgrPhsM2_Plus[rr][ii]+sqhgrPhsM2_Minus[rr][ii]);
	    break;
	    /* default: */
	    /* Nothing to do. */
	  }
	}
	yn1[ii]=yn1[ii]+tmp1/2.0+tmp2/2.0*SQ3;
	/* the second splitting has finished. */
      }

      /* The final splitting starts from here. */
      /* See (73) in [Note '16, Page No. 10]. */
      ffunc(yn1,fn);
      tmp1=alpha*half_step*Mu[0];
      for(ii=0; ii<ydim; ii++) { /* Initialization */
	if (1<sM2) {
	  phiM2[ii]=yn1[ii];
	  phiM1[ii]=yn1[ii]+tmp1*fn[ii];
	} else { /* Case of 1==sM2 */
	  phiM1[ii]=yn1[ii];
	  phi[ii]=yn1[ii]+tmp1*fn[ii];
	}
      }
      for(jj=1; jj<sM2; jj++) { /* Main iteration in Case of 1<sM2 */
	ffunc(phiM1,fn);
	tmp1=alpha*half_step*Mu[jj];
	tmp2=1+Ka[jj-1];
	tmp3=-Ka[jj-1];
	for (ii=0; ii<ydim; ii++) {
	  phi[ii]=tmp1*fn[ii]+tmp2*phiM1[ii]+tmp3*phiM2[ii];
	  if (jj<sM3) {
	    phiM2[ii]=phiM1[ii];
	    phiM1[ii]=phi[ii];
	  }
	}
      }

      ffunc(phi,fn);
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM2[ii]=half_step*fn[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=phi[ii]+2*tau_alpha*hD2fPhsM2[ii];
      }
      ffunc(hD2fPhsM1_til,fn);
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=half_step*fn[ii]; /* completed */
      }

      tmp1=2*sigma_alpha-1.0/2.0;
      for (ii=0; ii<ydim; ii++) {
	yn1[ii]=phi[ii]+tmp1*hD2fPhsM2[ii]+hD2fPhsM1_til[ii]/2.0;
	/* The final splitting has finished. */
      }
    } /* End of loop for itr */
    free(gn);
    free(sqhgrPhsM2);
    free(grPhsM2_Plus);
    free(grPhsM2_Minus);
    free(sqhgrPhsM2_Plus);
    free(sqhgrPhsM2_Minus);
  }
  return 0;
}

extern int
OMP_wo2_revA_srock2_avr_Al_Ab_val_for_NonCommSDEs_WinMulti_withCnt (int ydim,
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
/* wo2_revA_srock2_Al_2_75_Ab_val_for_NonCommSDEs_WinMulti for Open MP */
/* This function performs the splitting-like SROCK2 method using the parameter
   values of Abdulle's code.
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
   work: workspace of length MaxCoreNum*(7+6*wdim)*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function and random number evaluations.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, sM2, ii_par, sM3;
  double Mu[MaxStageNum_Ab], Ka[MaxStageNum_Ab-1], sigma, tau, alpha,
    sigma_alpha, tau_alpha, sqstep, tmp, half_step;
  unsigned long long func_ev_num[MaxCoreNum];
  
  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);
  half_step = step/2.0;

  if (0==static_flag) {
    if (0!=traj%MaxCoreNum) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNum);
      exit(1);
    }
    traj_mini=traj/MaxCoreNum;
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
  errflag=set_Al_for_revA_srock2(ss,&alpha); /* Over written */
  if (1==errflag) {
    printf("Error in set_Al_for_revA_srock2\n");
  }
  sM2=ss-2;
  sM3=ss-3;
  tmp=1-alpha;
  sigma_alpha=tmp/2.0+alpha*sigma;
  tau_alpha=tmp*tmp/2.0+2*alpha*tmp*sigma+alpha*alpha*tau;

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNum; ii_par++) {
    unsigned long itr;
    int ii, jj, rr, qq, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    double *yn, *yn1, *fn, *phiM2, *phiM1, *phi, *hD2fPhsM2, *hD2fPhsM1_til,
      tmp1, tmp2, tmp3;
    double **gn, *dummy, **sqhgrPhsM2, **grPhsM2_Plus, **grPhsM2_Minus,
      **sqhgrPhsM2_Plus, **sqhgrPhsM2_Minus;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;

    if (NULL == (gn = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_spl_revA_srock2_Ab_val_for_NonCommSDEs_WinMulti");
      exit(0);
    }
    if (NULL == (sqhgrPhsM2 = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_spl_revA_srock2_Ab_val_for_NonCommSDEs_WinMulti");
      exit(0);
    }
    if (NULL == (grPhsM2_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_spl_revA_srock2_Ab_val_for_NonCommSDEs_WinMulti");
      exit(0);
    }
    if (NULL == (grPhsM2_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_spl_revA_srock2_Ab_val_for_NonCommSDEs_WinMulti");
      exit(0);
    }
    if (NULL == (sqhgrPhsM2_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_spl_revA_srock2_Ab_val_for_NonCommSDEs_WinMulti");
      exit(0);
    }
    if (NULL == (sqhgrPhsM2_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in OMP_wo2_spl_revA_srock2_Ab_val_for_NonCommSDEs_WinMulti");
      exit(0);
    }

    ibase_work_step=7+6*wdim;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fn=&work[ibase_work+ii];
    ii+=ydim;
    phiM2=&work[ibase_work+ii];
    ii+=ydim;
    phiM1=&work[ibase_work+ii];
    ii+=ydim;
    phi=&work[ibase_work+ii];
    ii+=ydim;
    hD2fPhsM2=&work[ibase_work+ii];
    ii+=ydim;
    hD2fPhsM1_til=&work[ibase_work+ii];
    ii+=ydim;
    for (jj=0; jj<wdim; jj++) {
      gn[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgrPhsM2[jj]=&work[ibase_work+ii];
      ii+=ydim;
      grPhsM2_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      grPhsM2_Minus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgrPhsM2_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgrPhsM2_Minus[jj]=&work[ibase_work+ii];
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
      
      /* See (73) in [Note '16, Page No. 10]. */
      /* Calculations for K_{s-2} */
      ffunc(yn,fn); func_ev_num[ii_par]++;
      tmp1=alpha*half_step*Mu[0];
      for(ii=0; ii<ydim; ii++) { /* Initialization */
	if (1<sM2) {
	  phiM2[ii]=yn[ii];
	  phiM1[ii]=yn[ii]+tmp1*fn[ii];
	} else { /* Case of 1==sM2 */
	  phiM1[ii]=yn[ii];
	  phi[ii]=yn[ii]+tmp1*fn[ii];
	}
      }
      for(jj=1; jj<sM2; jj++) { /* Main iteration in Case of 1<sM2 */
	ffunc(phiM1,fn); func_ev_num[ii_par]++;
	tmp1=alpha*half_step*Mu[jj];
	tmp2=1+Ka[jj-1];
	tmp3=-Ka[jj-1];
	for (ii=0; ii<ydim; ii++) {
	  phi[ii]=tmp1*fn[ii]+tmp2*phiM1[ii]+tmp3*phiM2[ii];
	  if (jj<sM3) {
	    phiM2[ii]=phiM1[ii];
	    phiM1[ii]=phi[ii];
	  }
	}
      }

      ffunc(phi,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM2[ii]=half_step*fn[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=phi[ii]+2*tau_alpha*hD2fPhsM2[ii];
      }
      ffunc(hD2fPhsM1_til,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=half_step*fn[ii]; /* completed */
      }

      tmp1=2*sigma_alpha-1.0/2.0;
      for (ii=0; ii<ydim; ii++) {
	yn1[ii]=phi[ii]+tmp1*hD2fPhsM2[ii]+hD2fPhsM1_til[ii]/2.0;
	phiM1[ii]=phi[ii]+2*sigma_alpha*hD2fPhsM2[ii];
	/* The first splitting has finished. */
      }
      /* In what follows, note that
	 phi <---> Phi_{s-2} in (73),
	 yn1 <---> \tilde{Phi}_{s} in (73),
	 phiM1 <---> \hat{Phi}_{s} in (73).
       */

      for (rr=0; rr<wdim; rr++) {
	gfunc_gene(phi,rr,gn[rr]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  sqhgrPhsM2[rr][ii]=sqstep*gn[rr][ii]; /* completed */
	}
      }

      /* For details, see (3.2) in [Abdulle:2013]
	 and (73) in [Note '16, Page No. 10]. */
      for (ii=0; ii<ydim; ii++) {
	for (rr=0; rr<wdim; rr++) {
	  tmp1=0;
	  switch (wj[rr]) {
	  case 1: /* 1==wj[rr] */
	    for (qq=0; qq<rr; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(3.0+wtj[rr])*sqhgrPhsM2[qq][ii];
		break;
	      case -1:
		tmp1+=(-3.0+wtj[rr])*sqhgrPhsM2[qq][ii];
		break;
	      default:
		tmp1+=(wtj[rr])*sqhgrPhsM2[qq][ii];
	      }
	    }
	    tmp1+=2.0*sqhgrPhsM2[rr][ii];
	    for (qq=rr+1; qq<wdim; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(3.0-wtj[qq])*sqhgrPhsM2[qq][ii];
		break;
	      case -1:
		tmp1+=(-3.0-wtj[qq])*sqhgrPhsM2[qq][ii];
		break;
	      default:
		tmp1+=(-wtj[qq])*sqhgrPhsM2[qq][ii];
	      }
	    }
	    break;
	  case -1: /* -1==wj[rr] */
	    for (qq=0; qq<rr; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(-3.0+wtj[rr])*sqhgrPhsM2[qq][ii];
		break;
	      case -1:
		tmp1+=(3.0+wtj[rr])*sqhgrPhsM2[qq][ii];
		break;
	      default:
		tmp1+=(wtj[rr])*sqhgrPhsM2[qq][ii];
	      }
	    }
	    tmp1+=2.0*sqhgrPhsM2[rr][ii];
	    for (qq=rr+1; qq<wdim; qq++) {
	      switch (wj[qq]) {
	      case 1:
		tmp1+=(-3.0-wtj[qq])*sqhgrPhsM2[qq][ii];
		break;
	      case -1:
		tmp1+=(3.0-wtj[qq])*sqhgrPhsM2[qq][ii];
		break;
	      default:
		tmp1+=(-wtj[qq])*sqhgrPhsM2[qq][ii];
	      }
	    }
	    break;
	  default: /* 0==wj[rr] */
	    /* When rr=0, the following loop will not be performed. */
	    for (qq=0; qq<rr; qq++) {
	      tmp1+=(wtj[rr])*sqhgrPhsM2[qq][ii];
	    }
	    tmp1+=-sqhgrPhsM2[rr][ii];
	    for (qq=rr+1; qq<wdim; qq++) {
	      tmp1+=(-wtj[qq])*sqhgrPhsM2[qq][ii];
	    }
	  } /* End of the switch for rr */
	  grPhsM2_Plus[rr][ii]=phi[ii]+sqstep*tmp1/2.0;
	  grPhsM2_Minus[rr][ii]=phi[ii]-sqstep*tmp1/2.0;
	} /* End of the loop for rr */
      } /* End of the loop for ii */
      for (rr=0; rr<wdim; rr++) {
	gfunc_gene(grPhsM2_Plus[rr],rr,gn[rr]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  grPhsM2_Plus[rr][ii]=gn[rr][ii]; /* completed */
	}
	gfunc_gene(grPhsM2_Minus[rr],rr,gn[rr]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  grPhsM2_Minus[rr][ii]=gn[rr][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	for (rr=0; rr<wdim; rr++) {
	  tmp1=0;
	  for (qq=0; qq<wdim; qq++) {
	    if (0<wtj[qq]) {
	      tmp1+=sqhgrPhsM2[qq][ii];
	    } else {
	      tmp1-=sqhgrPhsM2[qq][ii];
	    }
	  }
	  sqhgrPhsM2_Plus[rr][ii]=phiM1[ii]+tmp1/SQ2;
	  sqhgrPhsM2_Minus[rr][ii]=phiM1[ii]-tmp1/SQ2;
	}
      }
      for (rr=0; rr<wdim; rr++) {
	gfunc_gene(sqhgrPhsM2_Plus[rr],rr,gn[rr]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  sqhgrPhsM2_Plus[rr][ii]=sqstep*gn[rr][ii]; /* completed */
	}
	gfunc_gene(sqhgrPhsM2_Minus[rr],rr,gn[rr]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  sqhgrPhsM2_Minus[rr][ii]=sqstep*gn[rr][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=tmp2=0;
	for (rr=0; rr<wdim; rr++) {
	  tmp1+=grPhsM2_Plus[rr][ii]-grPhsM2_Minus[rr][ii];
	  switch (wj[rr]) {
	  case 1:
	    tmp2+=(sqhgrPhsM2_Plus[rr][ii]+sqhgrPhsM2_Minus[rr][ii]);
	    break;
	  case -1:
	    tmp2-=(sqhgrPhsM2_Plus[rr][ii]+sqhgrPhsM2_Minus[rr][ii]);
	    break;
	    /* default: */
	    /* Nothing to do. */
	  }
	}
	yn1[ii]=yn1[ii]+tmp1/2.0+tmp2/2.0*SQ3;
	/* the second splitting has finished. */
      }

      /* The final splitting starts from here. */
      /* See (73) in [Note '16, Page No. 10]. */
      ffunc(yn1,fn); func_ev_num[ii_par]++;
      tmp1=alpha*half_step*Mu[0];
      for(ii=0; ii<ydim; ii++) { /* Initialization */
	if (1<sM2) {
	  phiM2[ii]=yn1[ii];
	  phiM1[ii]=yn1[ii]+tmp1*fn[ii];
	} else { /* Case of 1==sM2 */
	  phiM1[ii]=yn1[ii];
	  phi[ii]=yn1[ii]+tmp1*fn[ii];
	}
      }
      for(jj=1; jj<sM2; jj++) { /* Main iteration in Case of 1<sM2 */
	ffunc(phiM1,fn); func_ev_num[ii_par]++;
	tmp1=alpha*half_step*Mu[jj];
	tmp2=1+Ka[jj-1];
	tmp3=-Ka[jj-1];
	for (ii=0; ii<ydim; ii++) {
	  phi[ii]=tmp1*fn[ii]+tmp2*phiM1[ii]+tmp3*phiM2[ii];
	  if (jj<sM3) {
	    phiM2[ii]=phiM1[ii];
	    phiM1[ii]=phi[ii];
	  }
	}
      }

      ffunc(phi,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM2[ii]=half_step*fn[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=phi[ii]+2*tau_alpha*hD2fPhsM2[ii];
      }
      ffunc(hD2fPhsM1_til,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=half_step*fn[ii]; /* completed */
      }

      tmp1=2*sigma_alpha-1.0/2.0;
      for (ii=0; ii<ydim; ii++) {
	yn1[ii]=phi[ii]+tmp1*hD2fPhsM2[ii]+hD2fPhsM1_til[ii]/2.0;
	/* The final splitting has finished. */
      }
    } /* End of loop for itr */
    free(gn);
    free(sqhgrPhsM2);
    free(grPhsM2_Plus);
    free(grPhsM2_Minus);
    free(sqhgrPhsM2_Plus);
    free(sqhgrPhsM2_Minus);
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNum; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
  }
  return 0;
}

extern int
OMP_wo2_revA_srock2_avr_Al_Ab_val_for_DNoiseSDEs_WinMulti (int ydim,
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
/* wo2_revA_srock2_Al_2_75_Ab_val_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs the splitting-like SROCK2 method using the parameter
   values of Abdulle's code.
   It gives all trajectries for one step concerning SDEs with diagonal noise.
   If an error occurs, it will return 1, otherwise 0.

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
   work: workspace of length MaxCoreNum*13*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, sM2, ii_par, sM3, wdim;
  double Mu[MaxStageNum_Ab], Ka[MaxStageNum_Ab-1], sigma, tau, alpha,
    sigma_alpha, tau_alpha, sqstep, tmp, half_step;
  
  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);
  half_step = step/2.0;

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNum) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNum);
      exit(1);
    }
    traj_mini=traj/MaxCoreNum;
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
  errflag=set_Al_for_revA_srock2(ss,&alpha); /* Over written */
  if (1==errflag) {
    printf("Error in set_Al_for_revA_srock2\n");
  }
  sM2=ss-2;
  sM3=ss-3;
  tmp=1-alpha;
  sigma_alpha=tmp/2.0+alpha*sigma;
  tau_alpha=tmp*tmp/2.0+2*alpha*tmp*sigma+alpha*alpha*tau;

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNum; ii_par++) {
    unsigned long itr;
    int ii, jj, rr, qq, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    double *yn, *yn1, *fn, *phiM2, *phiM1, *phi, *hD2fPhsM2, *hD2fPhsM1_til,
      tmp1, tmp2, tmp3,
      *gn_diag, *sqhg_diag_PhsM2, *g_diag_PhsM2_Plus, *g_diag_PhsM2_Minus,
      *sqhg_diag_PhsM2_Plus, *sqhg_diag_PhsM2_Minus;
    char *wj, *wtj;

    ibase_work_step=13;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fn=&work[ibase_work+ii];
    ii+=ydim;
    phiM2=&work[ibase_work+ii];
    ii+=ydim;
    phiM1=&work[ibase_work+ii];
    ii+=ydim;
    phi=&work[ibase_work+ii];
    ii+=ydim;
    hD2fPhsM2=&work[ibase_work+ii];
    ii+=ydim;
    hD2fPhsM1_til=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_PhsM2=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_PhsM2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_PhsM2_Minus=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_PhsM2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_PhsM2_Minus=&work[ibase_work+ii];
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
      
      /* See (73) in [Note '16, Page No. 10]. */
      /* Calculations for K_{s-2} */
      ffunc(yn,fn);
      tmp1=alpha*half_step*Mu[0];
      for(ii=0; ii<ydim; ii++) { /* Initialization */
	if (1<sM2) {
	  phiM2[ii]=yn[ii];
	  phiM1[ii]=yn[ii]+tmp1*fn[ii];
	} else { /* Case of 1==sM2 */
	  phiM1[ii]=yn[ii];
	  phi[ii]=yn[ii]+tmp1*fn[ii];
	}
      }
      for(jj=1; jj<sM2; jj++) { /* Main iteration in Case of 1<sM2 */
	ffunc(phiM1,fn);
	tmp1=alpha*half_step*Mu[jj];
	tmp2=1+Ka[jj-1];
	tmp3=-Ka[jj-1];
	for (ii=0; ii<ydim; ii++) {
	  phi[ii]=tmp1*fn[ii]+tmp2*phiM1[ii]+tmp3*phiM2[ii];
	  if (jj<sM3) {
	    phiM2[ii]=phiM1[ii];
	    phiM1[ii]=phi[ii];
	  }
	}
      }

      ffunc(phi,fn);
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM2[ii]=half_step*fn[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=phi[ii]+2*tau_alpha*hD2fPhsM2[ii];
      }
      ffunc(hD2fPhsM1_til,fn);
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=half_step*fn[ii]; /* completed */
      }

      tmp1=2*sigma_alpha-1.0/2.0;
      for (ii=0; ii<ydim; ii++) {
	yn1[ii]=phi[ii]+tmp1*hD2fPhsM2[ii]+hD2fPhsM1_til[ii]/2.0;
	phiM1[ii]=phi[ii]+2*sigma_alpha*hD2fPhsM2[ii];
	/* The first splitting has finished. */
      }
      /* In what follows, note that
	 phi <---> Phi_{s-2} in (148),
	 yn1 <---> \tilde{Phi}_{s} in (148),
	 phiM1 <---> \hat{Phi}_{s} in (148).
       */
      
      gfunc_diag(phi,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_PhsM2[ii]=sqstep*gn_diag[ii]; /* completed */
      }

      /* For details, see (3.2) in [Abdulle:2013]
	 and (148) in [Note '16, Page No. 28]. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[ii] */
	  tmp1=2.0*sqhg_diag_PhsM2[ii];
	  break;
	case -1: /* -1==wj[ii] */
	  tmp1=2.0*sqhg_diag_PhsM2[ii];
	  break;
	default: /* 0==wj[ii] */
	  tmp1=-sqhg_diag_PhsM2[ii];
	}
	g_diag_PhsM2_Plus[ii]=phi[ii]+sqstep*tmp1/2.0;
	g_diag_PhsM2_Minus[ii]=phi[ii]-sqstep*tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_PhsM2_Plus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_PhsM2_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_PhsM2_Minus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_PhsM2_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=sqhg_diag_PhsM2[ii];
	} else {
	  tmp1=-sqhg_diag_PhsM2[ii];
	}
	sqhg_diag_PhsM2_Plus[ii]=phiM1[ii]+tmp1/SQ2;
	sqhg_diag_PhsM2_Minus[ii]=phiM1[ii]-tmp1/SQ2;
      }
      gfunc_diag(sqhg_diag_PhsM2_Plus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_PhsM2_Plus[ii]=sqstep*gn_diag[ii]; /* completed */
      }
      gfunc_diag(sqhg_diag_PhsM2_Minus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_PhsM2_Minus[ii]=sqstep*gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=g_diag_PhsM2_Plus[ii]-g_diag_PhsM2_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp2=(sqhg_diag_PhsM2_Plus[ii]+sqhg_diag_PhsM2_Minus[ii]);
	  break;
	case -1:
	  tmp2=-(sqhg_diag_PhsM2_Plus[ii]+sqhg_diag_PhsM2_Minus[ii]);
	  break;
	default:
	  tmp2=0;
	}
	yn1[ii]=yn1[ii]+tmp1/2.0+tmp2/2.0*SQ3;
	/* the second splitting has finished. */
      }

      /* The final splitting starts from here. */
      /* See (73) in [Note '16, Page No. 10]. */
      ffunc(yn1,fn);
      tmp1=alpha*half_step*Mu[0];
      for(ii=0; ii<ydim; ii++) { /* Initialization */
	if (1<sM2) {
	  phiM2[ii]=yn1[ii];
	  phiM1[ii]=yn1[ii]+tmp1*fn[ii];
	} else { /* Case of 1==sM2 */
	  phiM1[ii]=yn1[ii];
	  phi[ii]=yn1[ii]+tmp1*fn[ii];
	}
      }
      for(jj=1; jj<sM2; jj++) { /* Main iteration in Case of 1<sM2 */
	ffunc(phiM1,fn);
	tmp1=alpha*half_step*Mu[jj];
	tmp2=1+Ka[jj-1];
	tmp3=-Ka[jj-1];
	for (ii=0; ii<ydim; ii++) {
	  phi[ii]=tmp1*fn[ii]+tmp2*phiM1[ii]+tmp3*phiM2[ii];
	  if (jj<sM3) {
	    phiM2[ii]=phiM1[ii];
	    phiM1[ii]=phi[ii];
	  }
	}
      }

      ffunc(phi,fn);
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM2[ii]=half_step*fn[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=phi[ii]+2*tau_alpha*hD2fPhsM2[ii];
      }
      ffunc(hD2fPhsM1_til,fn);
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=half_step*fn[ii]; /* completed */
      }

      tmp1=2*sigma_alpha-1.0/2.0;
      for (ii=0; ii<ydim; ii++) {
	yn1[ii]=phi[ii]+tmp1*hD2fPhsM2[ii]+hD2fPhsM1_til[ii]/2.0;
	/* The final splitting has finished. */
      }
    } /* End of loop for itr */
  }
  return 0;
}

extern int
OMP_wo2_revA_srock2_avr_Al_Ab_val_for_DNoiseSDEs_WinMulti_withCnt (int ydim,
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
/* wo2_revA_srock2_Al_2_75_Ab_val_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs the splitting-like SROCK2 method using the parameter
   values of Abdulle's code.
   It gives all trajectries for one step concerning SDEs with diagonal noise.
   If an error occurs, it will return 1, otherwise 0.

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
   work: workspace of length MaxCoreNum*13*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function and random number evaluations.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, sM2, ii_par, sM3, wdim;
  double Mu[MaxStageNum_Ab], Ka[MaxStageNum_Ab-1], sigma, tau, alpha,
    sigma_alpha, tau_alpha, sqstep, tmp, half_step;
  unsigned long long func_ev_num[MaxCoreNum];
  
  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);
  half_step = step/2.0;

  wdim=ydim; /* Setting for diagonal noise */

  if (0==static_flag) {
    if (0!=traj%MaxCoreNum) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNum);
      exit(1);
    }
    traj_mini=traj/MaxCoreNum;
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
  errflag=set_Al_for_revA_srock2(ss,&alpha); /* Over written */
  if (1==errflag) {
    printf("Error in set_Al_for_revA_srock2\n");
  }
  sM2=ss-2;
  sM3=ss-3;
  tmp=1-alpha;
  sigma_alpha=tmp/2.0+alpha*sigma;
  tau_alpha=tmp*tmp/2.0+2*alpha*tmp*sigma+alpha*alpha*tau;

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNum; ii_par++) {
    unsigned long itr;
    int ii, jj, rr, qq, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    double *yn, *yn1, *fn, *phiM2, *phiM1, *phi, *hD2fPhsM2, *hD2fPhsM1_til,
      tmp1, tmp2, tmp3,
      *gn_diag, *sqhg_diag_PhsM2, *g_diag_PhsM2_Plus, *g_diag_PhsM2_Minus,
      *sqhg_diag_PhsM2_Plus, *sqhg_diag_PhsM2_Minus;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;

    ibase_work_step=13;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fn=&work[ibase_work+ii];
    ii+=ydim;
    phiM2=&work[ibase_work+ii];
    ii+=ydim;
    phiM1=&work[ibase_work+ii];
    ii+=ydim;
    phi=&work[ibase_work+ii];
    ii+=ydim;
    hD2fPhsM2=&work[ibase_work+ii];
    ii+=ydim;
    hD2fPhsM1_til=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_PhsM2=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_PhsM2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_PhsM2_Minus=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_PhsM2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    sqhg_diag_PhsM2_Minus=&work[ibase_work+ii];
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
      
      /* See (73) in [Note '16, Page No. 10]. */
      /* Calculations for K_{s-2} */
      ffunc(yn,fn); func_ev_num[ii_par]++;
      tmp1=alpha*half_step*Mu[0];
      for(ii=0; ii<ydim; ii++) { /* Initialization */
	if (1<sM2) {
	  phiM2[ii]=yn[ii];
	  phiM1[ii]=yn[ii]+tmp1*fn[ii];
	} else { /* Case of 1==sM2 */
	  phiM1[ii]=yn[ii];
	  phi[ii]=yn[ii]+tmp1*fn[ii];
	}
      }
      for(jj=1; jj<sM2; jj++) { /* Main iteration in Case of 1<sM2 */
	ffunc(phiM1,fn); func_ev_num[ii_par]++;
	tmp1=alpha*half_step*Mu[jj];
	tmp2=1+Ka[jj-1];
	tmp3=-Ka[jj-1];
	for (ii=0; ii<ydim; ii++) {
	  phi[ii]=tmp1*fn[ii]+tmp2*phiM1[ii]+tmp3*phiM2[ii];
	  if (jj<sM3) {
	    phiM2[ii]=phiM1[ii];
	    phiM1[ii]=phi[ii];
	  }
	}
      }

      ffunc(phi,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM2[ii]=half_step*fn[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=phi[ii]+2*tau_alpha*hD2fPhsM2[ii];
      }
      ffunc(hD2fPhsM1_til,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=half_step*fn[ii]; /* completed */
      }

      tmp1=2*sigma_alpha-1.0/2.0;
      for (ii=0; ii<ydim; ii++) {
	yn1[ii]=phi[ii]+tmp1*hD2fPhsM2[ii]+hD2fPhsM1_til[ii]/2.0;
	phiM1[ii]=phi[ii]+2*sigma_alpha*hD2fPhsM2[ii];
	/* The first splitting has finished. */
      }
      /* In what follows, note that
	 phi <---> Phi_{s-2} in (148),
	 yn1 <---> \tilde{Phi}_{s} in (148),
	 phiM1 <---> \hat{Phi}_{s} in (148).
       */
      
      gfunc_diag(phi,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_PhsM2[ii]=sqstep*gn_diag[ii]; /* completed */
      }

      /* For details, see (3.2) in [Abdulle:2013]
	 and (148) in [Note '16, Page No. 28]. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[ii] */
	  tmp1=2.0*sqhg_diag_PhsM2[ii];
	  break;
	case -1: /* -1==wj[ii] */
	  tmp1=2.0*sqhg_diag_PhsM2[ii];
	  break;
	default: /* 0==wj[ii] */
	  tmp1=-sqhg_diag_PhsM2[ii];
	}
	g_diag_PhsM2_Plus[ii]=phi[ii]+sqstep*tmp1/2.0;
	g_diag_PhsM2_Minus[ii]=phi[ii]-sqstep*tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_PhsM2_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_PhsM2_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_PhsM2_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_PhsM2_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=sqhg_diag_PhsM2[ii];
	} else {
	  tmp1=-sqhg_diag_PhsM2[ii];
	}
	sqhg_diag_PhsM2_Plus[ii]=phiM1[ii]+tmp1/SQ2;
	sqhg_diag_PhsM2_Minus[ii]=phiM1[ii]-tmp1/SQ2;
      }
      gfunc_diag(sqhg_diag_PhsM2_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_PhsM2_Plus[ii]=sqstep*gn_diag[ii]; /* completed */
      }
      gfunc_diag(sqhg_diag_PhsM2_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	sqhg_diag_PhsM2_Minus[ii]=sqstep*gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=g_diag_PhsM2_Plus[ii]-g_diag_PhsM2_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp2=(sqhg_diag_PhsM2_Plus[ii]+sqhg_diag_PhsM2_Minus[ii]);
	  break;
	case -1:
	  tmp2=-(sqhg_diag_PhsM2_Plus[ii]+sqhg_diag_PhsM2_Minus[ii]);
	  break;
	default:
	  tmp2=0;
	}
	yn1[ii]=yn1[ii]+tmp1/2.0+tmp2/2.0*SQ3;
	/* the second splitting has finished. */
      }

      /* The final splitting starts from here. */
      /* See (73) in [Note '16, Page No. 10]. */
      ffunc(yn1,fn); func_ev_num[ii_par]++;
      tmp1=alpha*half_step*Mu[0];
      for(ii=0; ii<ydim; ii++) { /* Initialization */
	if (1<sM2) {
	  phiM2[ii]=yn1[ii];
	  phiM1[ii]=yn1[ii]+tmp1*fn[ii];
	} else { /* Case of 1==sM2 */
	  phiM1[ii]=yn1[ii];
	  phi[ii]=yn1[ii]+tmp1*fn[ii];
	}
      }
      for(jj=1; jj<sM2; jj++) { /* Main iteration in Case of 1<sM2 */
	ffunc(phiM1,fn); func_ev_num[ii_par]++;
	tmp1=alpha*half_step*Mu[jj];
	tmp2=1+Ka[jj-1];
	tmp3=-Ka[jj-1];
	for (ii=0; ii<ydim; ii++) {
	  phi[ii]=tmp1*fn[ii]+tmp2*phiM1[ii]+tmp3*phiM2[ii];
	  if (jj<sM3) {
	    phiM2[ii]=phiM1[ii];
	    phiM1[ii]=phi[ii];
	  }
	}
      }

      ffunc(phi,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM2[ii]=half_step*fn[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=phi[ii]+2*tau_alpha*hD2fPhsM2[ii];
      }
      ffunc(hD2fPhsM1_til,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	hD2fPhsM1_til[ii]=half_step*fn[ii]; /* completed */
      }

      tmp1=2*sigma_alpha-1.0/2.0;
      for (ii=0; ii<ydim; ii++) {
	yn1[ii]=phi[ii]+tmp1*hD2fPhsM2[ii]+hD2fPhsM1_til[ii]/2.0;
	/* The final splitting has finished. */
      }
    } /* End of loop for itr */
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNum; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
  }
  return 0;
}
