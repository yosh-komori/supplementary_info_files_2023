/* Weak order stochastic exponential RK schemes for
   Ito SDEs with noncommutative noise */
/* This file was made to put on GitHub (26-Apr-2023). */
/*************************************************************/
     
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include "mkl_lapacke.h"

#define SQ2 1.4142135623730950 /* sqrt(2) */
#define SQ3 1.7320508075688773 /* sqrt(3) */
#define SQ6 2.4494897427831781 /* sqrt(6) */

#define MaxCoreNumForSERK 1 /* 8, 1, 4 *//* Maximum number of multi-core. */

int ludecomp(int matdim, double pivotmin, double _Complex matrix[],
	     int indexvec[]) {
  /* This gives an LU decomposition for a matrix. If an error occurs,
   it will return 1, otherwise 0.

   Input arguments
   ----------------
   matdim: dimension of a matrix,
   pivotmin: minimum value for a pivot,

   Input/output arguments
   ----------------
   matrix: matrix of complex values,

   Output arguments
   ----------------
   indexvec: index vector concerning permutation.
   */

  double rtemp;
  int ii, jj, kk, ll, ll1, itemp, itemp1;

  /* Initialization of index vector */
  for (kk=0; kk<matdim; kk++) {
    indexvec[kk]=kk;
  }

  for (kk=0; kk<matdim; kk++) {
    /* Selection of a pivot */
    itemp = kk;
    ll = indexvec[itemp]*matdim;
    rtemp = cabs(matrix[ll+kk]);
    /* The following is skipped if kk == matdim-1 */
    for (ii=kk+1; ii<matdim; ii++) {
      ll = indexvec[ii]*matdim;
      if (rtemp < cabs(matrix[ll+kk])) {
	itemp = ii;
	ll = indexvec[itemp]*matdim;
	rtemp = cabs(matrix[ll+kk]);
      }
    }

    /* The following is skipped if kk == matdim-1 */
    if (itemp != kk) {
      /* Replacement of rows  */
      itemp1 = indexvec[kk];
      indexvec[kk] = indexvec[itemp];
      indexvec[itemp] = itemp1;
    }

    /* Check for a pivot */
    ll = indexvec[kk]*matdim;
    if(cabs(matrix[ll+kk]) < pivotmin) {
      return 1;
    }

    /* The following is skipped if kk == matdim-1 */
    /* Calculation for the LU decomposition  */
    for (ii=kk+1; ii<matdim; ii++) {
      ll = indexvec[ii]*matdim;
      ll1 = indexvec[kk]*matdim;
      matrix[ll+kk] = matrix[ll+kk]/matrix[ll1+kk];
      for(jj=kk+1; jj<matdim; jj++)
	matrix[ll+jj] = matrix[ll+jj]-matrix[ll+kk]*matrix[ll1+jj];
    }
  }
  return 0;
}

void solve(int matdim, double _Complex matrix[], double bvector[],
	   int indexvec[], double _Complex solution[]) {
  /* This gives a solution of a system of linear equations.
     
     Input arguments
   ----------------
   matdim: dimension of a matrix,
   matrix: LU decomposed and complex-valued matrix,
   bvector: a vector in the right-hand side of the system,
   indexvec: index vector concerning permutation,

   Output arguments
   ----------------
   solution: complex-valued solution vector.
  */

  int ii, jj, ll;
  
  for (ii=0; ii<matdim; ii++) {
    solution[ii] = bvector[indexvec[ii]];
    for(jj=0; jj<=ii-1; jj++) {
      ll=indexvec[ii]*matdim;
      solution[ii] = solution[ii]-matrix[ll+jj]*solution[jj];
    }
  }

  for(ii=matdim-1; 0<=ii; ii--) {
    ll=indexvec[ii]*matdim;
    for(jj=ii+1; jj<matdim; jj++) {
      solution[ii] = solution[ii] - matrix[ll+jj]*solution[jj];
    }
    solution[ii] = solution[ii]/matrix[ll+ii];
  }
}

extern int OMP_wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti(int ydim,
							   unsigned long traj,
							   double *yvec,
							   double step,
							   char *ran2pFull,
							   char *ran3p,
							   double A_mat[],
							   void (*ffunc)(),
							   void (*gfunc_gene)(),
							   int wdim,
							   double work[],
							   double work_A[],
							   double *ynew)
/* wo2_AstabExpRK3_for_NonCommSDEs_WinMulti for Open MP */
/* This function performs an Efficient, A-stable and weak second order
   exponential RK scheme.
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
   A_mat: array for the matrix A in the drift coefficient,
   ffunc: nonlinear part in the drift coefficient,
   gfunc_gene: diffusion coefficients,
   wdim: dimension of Wiener process.

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK*(7+7*wdim)*ydim.
   work_A: workspace of length 6*ydim*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par;
  static double static_step=0,
    b1=(6+SQ6)/10.0,
    b2=(3-2*SQ6)/5.0;

  int ii, jj, kk, ll, ll1, ll2, ydimPow2=ydim*ydim;
  double sqstep, glTmp1, glTmp2;

  int  ydimDummy=ydim /* to avoid an error of dimension */;
  double *exMat, *hPh1Mat, *hPh2Mat, *ex_05Mat, *hPh1_05Mat, *hPh2_05Mat;
  
  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  ii=0;
  exMat=&work_A[ii];
  ii+=ydimPow2;
  hPh1Mat=&work_A[ii];
  ii+=ydimPow2;
  hPh2Mat=&work_A[ii];
  ii+=ydimPow2;
  ex_05Mat=&work_A[ii];
  ii+=ydimPow2;
  hPh1_05Mat=&work_A[ii];
  ii+=ydimPow2;
  hPh2_05Mat=&work_A[ii];
  ii+=ydimPow2;

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK;
    static_flag=1;
  }
  
  if (static_step!=step) {
    if(1==ydim) {
      exMat[0]=exp(A_mat[0]*step);
      hPh1Mat[0]=(exp(A_mat[0]*step)-1)/A_mat[0];
      glTmp1=A_mat[0]*step;
      glTmp2=exp(glTmp1)-1;
      hPh2Mat[0]=(glTmp2-glTmp1)/(glTmp1*A_mat[0]);
      ex_05Mat[0]=exp(A_mat[0]/2.0*step);
      hPh1_05Mat[0]=(exp(A_mat[0]/2.0*step)-1)/(A_mat[0]/2.0);
      glTmp1=A_mat[0]/2.0*step;
      glTmp2=exp(glTmp1)-1;
      hPh2_05Mat[0]=(glTmp2-glTmp1)/(glTmp1*A_mat[0]/2.0);
    }
    if(1<ydim) {
      int *indexvec;
      double *tmpMat, *bvector;
      
      double _Complex dummy, *diag, *matT, *InvMatT, *tmpMatC, *tmpDiag,
	*solution, *tmpExMat;
      
      double *eigenVal_real, *eigenVal_imag, *eigenVecs, *vecDummy;

      if (NULL == (indexvec = (int *)malloc(sizeof(int)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMat = (double *)malloc(sizeof(double)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (bvector = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (diag = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (matT = (double _Complex *)malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (InvMatT = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMatC = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpDiag = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (solution = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpExMat = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (eigenVal_real = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (eigenVal_imag = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (eigenVecs = (double *)
		   malloc(sizeof(double)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (vecDummy = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      /* Copy matrix A */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0;jj<ydim;jj++) {
	  tmpMat[ll+jj]=A_mat[ll+jj];
	}
      }
      
      if(0 != LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', ydim, tmpMat, ydim,
			    eigenVal_real, eigenVal_imag,
			    vecDummy, ydimDummy, eigenVecs, ydim)) {
	printf("Error in LAPACKE_dgeev!\n");
	exit(0);
      }

      /* Eigenvalues */
      for (ii=0; ii<ydim; ii++) {
	diag[ii]=eigenVal_real[ii]+eigenVal_imag[ii]*I;
      }

      /* Set matrix T */
      for (ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	jj = 0;
	while(jj<ydim) {
	  if (0.0 == eigenVal_imag[jj]) {
	    matT[ll+jj]=eigenVecs[ll+jj];
	    jj++;
	  } else {
	    matT[ll+jj]  =eigenVecs[ll+jj]+eigenVecs[ll+(jj+1)]*I;
	    matT[ll+jj+1]=eigenVecs[ll+jj]-eigenVecs[ll+(jj+1)]*I;
	    jj+=2;
	  }
	}
      }

      /* Calculation for inverse matrix T */
      for (ii=0; ii<ydimPow2; ii++) {
	tmpMatC[ii]=matT[ii];
      }
      if (1==ludecomp(ydim, 1.0e-14, tmpMatC, indexvec)) {
	printf("Error in ludecomp!\n");
	exit(0);
      }
      for (ii=0; ii<ydim; ii++) {
	for (jj=0; jj<ii; jj++) {
	  bvector[jj]=0;
	}
	bvector[ii]=1;
	for (jj=ii+1; jj<ydim; jj++) {
	  bvector[jj]=0;
	}
	solve(ydim, tmpMatC, bvector, indexvec, solution);
	
	for (jj=0; jj<ydim; jj++) {
	  InvMatT[jj*ydim+ii]=solution[jj];
	}
      }

      /* Calculation for diagonal elements in exMat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=cexp(diag[ii]*step);
      }

      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  exMat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in hPh1Mat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=(cexp(diag[ii]*step)-1)/diag[ii];
      }
      
      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }
      
      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  hPh1Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in hPh2Mat */
      for(ii=0;ii<ydim;ii++) {
	glTmp1=diag[ii]*step;
	glTmp2=cexp(glTmp1)-1;
	tmpDiag[ii]=(glTmp2-glTmp1)/(glTmp1*diag[ii]);
      }
      
      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }
      
      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  hPh2Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in ex_05Mat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=cexp(diag[ii]/2.0*step);
      }

      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  ex_05Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in hPh1_05Mat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=(cexp(diag[ii]/2.0*step)-1)/(diag[ii]/2.0);
      }
      
      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }
      
      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  hPh1_05Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in hPh2_05Mat */
      for(ii=0;ii<ydim;ii++) {
	glTmp1=diag[ii]/2.0*step;
	glTmp2=cexp(glTmp1)-1;
	tmpDiag[ii]=(glTmp2-glTmp1)/(glTmp1*diag[ii]/2.0);
      }
      
      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }
      
      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  hPh2_05Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

    } /* End of if(1<ydim) */
    static_step=step;
  } /* End of if (static_step!=step) */
  
#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    double *yn, *yn1, *fn, *Y1, *Y2, *fY2add_fY4add_Minus_fn,
      *sum_sqhgjY2_gzai, *sumH, tmp1, tmp2;
    double **gn, *dummy, **sqhgjY2, **hgjY2, **gjY2_Plus, **gjY2_Minus,
      **sqhgjY2_Plus, **sqhgjY2_Minus;
    char *wj, *wtj;

    if (NULL == (gn = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgjY2 = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (hgjY2 = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjY2_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjY2_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgjY2_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgjY2_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }

    ibase_work_step=7+7*wdim;
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
    for (jj=0; jj<wdim; jj++) {
      gn[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgjY2[jj]=&work[ibase_work+ii];
      ii+=ydim;
      hgjY2[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjY2_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjY2_Minus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgjY2_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgjY2_Minus[jj]=&work[ibase_work+ii];
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
      
      /* See No. 69 in Note '12/'13. */
      ffunc(yn,fn);
      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=tmp2=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=hPh1Mat[ll+jj]*fn[jj];
	  tmp2+=hPh1_05Mat[ll+jj]*fn[jj];
	}
	Y1[ii]=tmp1;
	Y2[ii]=tmp2/2.0;
	tmp1=tmp2=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=exMat[ll+jj]*yn[jj];
	  tmp2+=ex_05Mat[ll+jj]*yn[jj];
	}
	Y1[ii]+=tmp1; /* completed */
	Y2[ii]+=tmp2; /* completed */
      }

      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(Y2,jj,gn[jj]);
	for (ii=0; ii<ydim; ii++) {
	  sqhgjY2[jj][ii]=sqstep*gn[jj][ii]; /* completed */
	  hgjY2[jj][ii]=step*gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=0;
	for (jj=0; jj<wdim; jj++) {
	  switch (wj[jj]) {
	  case 1:
	    tmp1+=sqhgjY2[jj][ii];
	    break;
	  case -1:
	    tmp1-=sqhgjY2[jj][ii];
	    break;
	    /* default: */
	    /* Nothing to do */
	  }
	}
	sum_sqhgjY2_gzai[ii]=SQ3*tmp1;
      }
      
      for (ii=0; ii<ydim; ii++) {
	sumH[ii]=fn[ii];
      }
      ffunc(Y2,fn);
      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=(hPh2_05Mat[ll+jj]+hPh2Mat[ll+jj])
	    *(fn[jj]-sumH[jj]);
	}
	/* The next is for (Y4 + an addi. term). */
	fY2add_fY4add_Minus_fn[ii]=Y1[ii]+tmp1*2.0+b2*sum_sqhgjY2_gzai[ii];
      }
      ffunc(fY2add_fY4add_Minus_fn,fn);
      for (ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]=fn[ii]-5*sumH[ii];
      }
      for (ii=0; ii<ydim; ii++) {
	sumH[ii]=Y2[ii]+b1*sum_sqhgjY2_gzai[ii];
      }
      ffunc(sumH,fn);
      for (ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]+=4*fn[ii]; /* completed */
      }

      /* For details, see (337) and No. 69 in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	for (jj=0; jj<wdim; jj++) {
	  tmp1=0;
	  switch (wj[jj]) {
	  case 1: /* 1==wj[jj] */
	    for (kk=0; kk<jj; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(3.0+wtj[jj])*hgjY2[kk][ii];
		break;
	      case -1:
		tmp1+=(-3.0+wtj[jj])*hgjY2[kk][ii];
		break;
	      default:
		tmp1+=(wtj[jj])*hgjY2[kk][ii];
	      }
	    }
	    tmp1+=2.0*hgjY2[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(3.0-wtj[kk])*hgjY2[kk][ii];
		break;
	      case -1:
		tmp1+=(-3.0-wtj[kk])*hgjY2[kk][ii];
		break;
	      default:
		tmp1+=(-wtj[kk])*hgjY2[kk][ii];
	      }
	    }
	    break;
	  case -1: /* -1==wj[jj] */
	    for (kk=0; kk<jj; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(-3.0+wtj[jj])*hgjY2[kk][ii];
		break;
	      case -1:
		tmp1+=(3.0+wtj[jj])*hgjY2[kk][ii];
		break;
	      default:
		tmp1+=(wtj[jj])*hgjY2[kk][ii];
	      }
	    }
	    tmp1+=2.0*hgjY2[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(-3.0-wtj[kk])*hgjY2[kk][ii];
		break;
	      case -1:
		tmp1+=(3.0-wtj[kk])*hgjY2[kk][ii];
		break;
	      default:
		tmp1+=(-wtj[kk])*hgjY2[kk][ii];
	      }
	    }
	    break;
	  default: /* 0==wj[jj] */
	    /* When jj=0, the following loop will not be performed. */
	    for (kk=0; kk<jj; kk++) {
	      tmp1+=(wtj[jj])*hgjY2[kk][ii];
	    }
	    tmp1+=-hgjY2[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      tmp1+=(-wtj[kk])*hgjY2[kk][ii];
	    }
	  } /* End of the switch for jj */
	  gjY2_Plus[jj][ii]=Y2[ii]+tmp1/2.0;
	  gjY2_Minus[jj][ii]=Y2[ii]-tmp1/2.0;
	} /* End of the loop for jj */
      } /* End of the loop for ii */
      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(gjY2_Plus[jj],jj,gn[jj]);
	for (ii=0; ii<ydim; ii++) {
	  gjY2_Plus[jj][ii]=gn[jj][ii]; /* completed */
	}
	gfunc_gene(gjY2_Minus[jj],jj,gn[jj]);
	for (ii=0; ii<ydim; ii++) {
	  gjY2_Minus[jj][ii]=gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=0;
	for (kk=0; kk<wdim; kk++) {
	  if (0<wtj[kk]) {
	    tmp1+=sqhgjY2[kk][ii];
	  } else {
	    tmp1-=sqhgjY2[kk][ii];
	  }
	}
	for (jj=0; jj<wdim; jj++) {
	  sqhgjY2_Plus[jj][ii]=Y2[ii]+tmp1/SQ2;
	  sqhgjY2_Minus[jj][ii]=Y2[ii]-tmp1/SQ2;
	}
      }
      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(sqhgjY2_Plus[jj],jj,gn[jj]);
	for (ii=0; ii<ydim; ii++) {
	  sqhgjY2_Plus[jj][ii]=sqstep*gn[jj][ii]; /* completed */
	}
	gfunc_gene(sqhgjY2_Minus[jj],jj,gn[jj]);
	for (ii=0; ii<ydim; ii++) {
	  sqhgjY2_Minus[jj][ii]=sqstep*gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=tmp2=0;
	for (jj=0; jj<wdim; jj++) {
	  tmp1+=gjY2_Plus[jj][ii]-gjY2_Minus[jj][ii];
	  switch (wj[jj]) {
	  case 1:
	    tmp2+=(sqhgjY2_Plus[jj][ii]+sqhgjY2_Minus[jj][ii]);
	    break;
	  case -1:
	    tmp2-=(sqhgjY2_Plus[jj][ii]+sqhgjY2_Minus[jj][ii]);
	    break;
	    /* default: */
	    /* Nothing to do. */
	  }
	}
	sumH[ii]=tmp1/2.0+tmp2/2.0*SQ3; /* completed */
      }

      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=tmp2=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=hPh2Mat[ll+jj]*fY2add_fY4add_Minus_fn[jj];
	  tmp2+=ex_05Mat[ll+jj]*sumH[jj];
	}
	yn1[ii]=Y1[ii]+tmp1/3.0+tmp2;
      }
      
    } /* End of loop for itr */
    free(gn);
    free(sqhgjY2);
    free(hgjY2);
    free(gjY2_Plus);
    free(gjY2_Minus);
    free(sqhgjY2_Plus);
    free(sqhgjY2_Minus);
  }
  return 0;
}

extern int
OMP_wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti_withCnt(int ydim,
							unsigned long traj,
							double *yvec,
							double step,
							char *ran2pFull,
							char *ran3p,
							double A_mat[],
							void (*ffunc)(),
							void (*gfunc_gene)(),
							int wdim,
							double work[],
							double work_A[],
							double *ynew,
							unsigned long long *ev_cnt)
/* wo2_AstabExpRK3_for_NonCommSDEs_WinMulti for Open MP */
/* This function performs an Efficient, A-stable and weak second order
   exponential RK scheme.
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
   A_mat: array for the matrix A in the drift coefficient,
   ffunc: nonlinear part in the drift coefficient,
   gfunc_gene: diffusion coefficients,
   wdim: dimension of Wiener process.

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK*(7+7*wdim)*ydim.
   work_A: workspace of length 6*ydim*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function evaluations.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par;
  static double static_step=0,
    b1=(6+SQ6)/10.0,
    b2=(3-2*SQ6)/5.0;

  int ii, jj, kk, ll, ll1, ll2, ydimPow2=ydim*ydim;
  double sqstep, glTmp1, glTmp2;
  unsigned long long func_ev_num[MaxCoreNumForSERK];

  int  ydimDummy=ydim /* to avoid an error of dimension */;
  double *exMat, *hPh1Mat, *hPh2Mat, *ex_05Mat, *hPh1_05Mat, *hPh2_05Mat;
  
  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  ii=0;
  exMat=&work_A[ii];
  ii+=ydimPow2;
  hPh1Mat=&work_A[ii];
  ii+=ydimPow2;
  hPh2Mat=&work_A[ii];
  ii+=ydimPow2;
  ex_05Mat=&work_A[ii];
  ii+=ydimPow2;
  hPh1_05Mat=&work_A[ii];
  ii+=ydimPow2;
  hPh2_05Mat=&work_A[ii];
  ii+=ydimPow2;

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK;
    static_flag=1;
  }
  
  if (static_step!=step) {
    if(1==ydim) {
      exMat[0]=exp(A_mat[0]*step);
      hPh1Mat[0]=(exp(A_mat[0]*step)-1)/A_mat[0];
      glTmp1=A_mat[0]*step;
      glTmp2=exp(glTmp1)-1;
      hPh2Mat[0]=(glTmp2-glTmp1)/(glTmp1*A_mat[0]);
      ex_05Mat[0]=exp(A_mat[0]/2.0*step);
      hPh1_05Mat[0]=(exp(A_mat[0]/2.0*step)-1)/(A_mat[0]/2.0);
      glTmp1=A_mat[0]/2.0*step;
      glTmp2=exp(glTmp1)-1;
      hPh2_05Mat[0]=(glTmp2-glTmp1)/(glTmp1*A_mat[0]/2.0);
    }
    if(1<ydim) {
      int *indexvec;
      double *tmpMat, *bvector;
      
      double _Complex dummy, *diag, *matT, *InvMatT, *tmpMatC, *tmpDiag,
	*solution, *tmpExMat;
      
      double *eigenVal_real, *eigenVal_imag, *eigenVecs, *vecDummy;

      if (NULL == (indexvec = (int *)malloc(sizeof(int)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMat = (double *)malloc(sizeof(double)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (bvector = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (diag = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (matT = (double _Complex *)malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (InvMatT = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMatC = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpDiag = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (solution = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpExMat = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (eigenVal_real = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (eigenVal_imag = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (eigenVecs = (double *)
		   malloc(sizeof(double)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (vecDummy = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      /* Copy matrix A */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0;jj<ydim;jj++) {
	  tmpMat[ll+jj]=A_mat[ll+jj];
	}
      }
      
      if(0 != LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', ydim, tmpMat, ydim,
			    eigenVal_real, eigenVal_imag,
			    vecDummy, ydimDummy, eigenVecs, ydim)) {
	printf("Error in LAPACKE_dgeev!\n");
	exit(0);
      }

      /* Eigenvalues */
      for (ii=0; ii<ydim; ii++) {
	diag[ii]=eigenVal_real[ii]+eigenVal_imag[ii]*I;
      }

      /* Set matrix T */
      for (ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	jj = 0;
	while(jj<ydim) {
	  if (0.0 == eigenVal_imag[jj]) {
	    matT[ll+jj]=eigenVecs[ll+jj];
	    jj++;
	  } else {
	    matT[ll+jj]  =eigenVecs[ll+jj]+eigenVecs[ll+(jj+1)]*I;
	    matT[ll+jj+1]=eigenVecs[ll+jj]-eigenVecs[ll+(jj+1)]*I;
	    jj+=2;
	  }
	}
      }

      /* Calculation for inverse matrix T */
      for (ii=0; ii<ydimPow2; ii++) {
	tmpMatC[ii]=matT[ii];
      }
      if (1==ludecomp(ydim, 1.0e-14, tmpMatC, indexvec)) {
	printf("Error in ludecomp!\n");
	exit(0);
      }
      for (ii=0; ii<ydim; ii++) {
	for (jj=0; jj<ii; jj++) {
	  bvector[jj]=0;
	}
	bvector[ii]=1;
	for (jj=ii+1; jj<ydim; jj++) {
	  bvector[jj]=0;
	}
	solve(ydim, tmpMatC, bvector, indexvec, solution);
	
	for (jj=0; jj<ydim; jj++) {
	  InvMatT[jj*ydim+ii]=solution[jj];
	}
      }

      /* Calculation for diagonal elements in exMat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=cexp(diag[ii]*step);
      }

      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  exMat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in hPh1Mat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=(cexp(diag[ii]*step)-1)/diag[ii];
      }
      
      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }
      
      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  hPh1Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in hPh2Mat */
      for(ii=0;ii<ydim;ii++) {
	glTmp1=diag[ii]*step;
	glTmp2=cexp(glTmp1)-1;
	tmpDiag[ii]=(glTmp2-glTmp1)/(glTmp1*diag[ii]);
      }
      
      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }
      
      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  hPh2Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in ex_05Mat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=cexp(diag[ii]/2.0*step);
      }

      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  ex_05Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in hPh1_05Mat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=(cexp(diag[ii]/2.0*step)-1)/(diag[ii]/2.0);
      }
      
      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }
      
      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  hPh1_05Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in hPh2_05Mat */
      for(ii=0;ii<ydim;ii++) {
	glTmp1=diag[ii]/2.0*step;
	glTmp2=cexp(glTmp1)-1;
	tmpDiag[ii]=(glTmp2-glTmp1)/(glTmp1*diag[ii]/2.0);
      }
      
      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }
      
      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  hPh2_05Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

    } /* End of if(1<ydim) */
    static_step=step;
  } /* End of if (static_step!=step) */
  
#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    double *yn, *yn1, *fn, *Y1, *Y2, *fY2add_fY4add_Minus_fn,
      *sum_sqhgjY2_gzai, *sumH, tmp1, tmp2;
    double **gn, *dummy, **sqhgjY2, **hgjY2, **gjY2_Plus, **gjY2_Minus,
      **sqhgjY2_Plus, **sqhgjY2_Minus;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;

    if (NULL == (gn = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgjY2 = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (hgjY2 = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjY2_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjY2_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgjY2_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgjY2_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }

    ibase_work_step=7+7*wdim;
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
    for (jj=0; jj<wdim; jj++) {
      gn[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgjY2[jj]=&work[ibase_work+ii];
      ii+=ydim;
      hgjY2[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjY2_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjY2_Minus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgjY2_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgjY2_Minus[jj]=&work[ibase_work+ii];
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
      
      /* See No. 69 in Note '12/'13. */
      ffunc(yn,fn); func_ev_num[ii_par]++;
      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=tmp2=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=hPh1Mat[ll+jj]*fn[jj];
	  tmp2+=hPh1_05Mat[ll+jj]*fn[jj];
	}
	Y1[ii]=tmp1;
	Y2[ii]=tmp2/2.0;
	tmp1=tmp2=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=exMat[ll+jj]*yn[jj];
	  tmp2+=ex_05Mat[ll+jj]*yn[jj];
	}
	Y1[ii]+=tmp1; /* completed */
	Y2[ii]+=tmp2; /* completed */
      }

      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(Y2,jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  sqhgjY2[jj][ii]=sqstep*gn[jj][ii]; /* completed */
	  hgjY2[jj][ii]=step*gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=0;
	for (jj=0; jj<wdim; jj++) {
	  switch (wj[jj]) {
	  case 1:
	    tmp1+=sqhgjY2[jj][ii];
	    break;
	  case -1:
	    tmp1-=sqhgjY2[jj][ii];
	    break;
	    /* default: */
	    /* Nothing to do */
	  }
	}
	sum_sqhgjY2_gzai[ii]=SQ3*tmp1;
      }
      
      for (ii=0; ii<ydim; ii++) {
	sumH[ii]=fn[ii];
      }
      ffunc(Y2,fn); func_ev_num[ii_par]++;
      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=(hPh2_05Mat[ll+jj]+hPh2Mat[ll+jj])
	    *(fn[jj]-sumH[jj]);
	}
	/* The next is for (Y4 + an addi. term). */
	fY2add_fY4add_Minus_fn[ii]=Y1[ii]+tmp1*2.0+b2*sum_sqhgjY2_gzai[ii];
      }
      ffunc(fY2add_fY4add_Minus_fn,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]=fn[ii]-5*sumH[ii];
      }
      for (ii=0; ii<ydim; ii++) {
	sumH[ii]=Y2[ii]+b1*sum_sqhgjY2_gzai[ii];
      }
      ffunc(sumH,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]+=4*fn[ii]; /* completed */
      }

      /* For details, see (337) and No. 69 in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	for (jj=0; jj<wdim; jj++) {
	  tmp1=0;
	  switch (wj[jj]) {
	  case 1: /* 1==wj[jj] */
	    for (kk=0; kk<jj; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(3.0+wtj[jj])*hgjY2[kk][ii];
		break;
	      case -1:
		tmp1+=(-3.0+wtj[jj])*hgjY2[kk][ii];
		break;
	      default:
		tmp1+=(wtj[jj])*hgjY2[kk][ii];
	      }
	    }
	    tmp1+=2.0*hgjY2[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(3.0-wtj[kk])*hgjY2[kk][ii];
		break;
	      case -1:
		tmp1+=(-3.0-wtj[kk])*hgjY2[kk][ii];
		break;
	      default:
		tmp1+=(-wtj[kk])*hgjY2[kk][ii];
	      }
	    }
	    break;
	  case -1: /* -1==wj[jj] */
	    for (kk=0; kk<jj; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(-3.0+wtj[jj])*hgjY2[kk][ii];
		break;
	      case -1:
		tmp1+=(3.0+wtj[jj])*hgjY2[kk][ii];
		break;
	      default:
		tmp1+=(wtj[jj])*hgjY2[kk][ii];
	      }
	    }
	    tmp1+=2.0*hgjY2[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(-3.0-wtj[kk])*hgjY2[kk][ii];
		break;
	      case -1:
		tmp1+=(3.0-wtj[kk])*hgjY2[kk][ii];
		break;
	      default:
		tmp1+=(-wtj[kk])*hgjY2[kk][ii];
	      }
	    }
	    break;
	  default: /* 0==wj[jj] */
	    /* When jj=0, the following loop will not be performed. */
	    for (kk=0; kk<jj; kk++) {
	      tmp1+=(wtj[jj])*hgjY2[kk][ii];
	    }
	    tmp1+=-hgjY2[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      tmp1+=(-wtj[kk])*hgjY2[kk][ii];
	    }
	  } /* End of the switch for jj */
	  gjY2_Plus[jj][ii]=Y2[ii]+tmp1/2.0;
	  gjY2_Minus[jj][ii]=Y2[ii]-tmp1/2.0;
	} /* End of the loop for jj */
      } /* End of the loop for ii */
      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(gjY2_Plus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  gjY2_Plus[jj][ii]=gn[jj][ii]; /* completed */
	}
	gfunc_gene(gjY2_Minus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  gjY2_Minus[jj][ii]=gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=0;
	for (kk=0; kk<wdim; kk++) {
	  if (0<wtj[kk]) {
	    tmp1+=sqhgjY2[kk][ii];
	  } else {
	    tmp1-=sqhgjY2[kk][ii];
	  }
	}
	for (jj=0; jj<wdim; jj++) {
	  sqhgjY2_Plus[jj][ii]=Y2[ii]+tmp1/SQ2;
	  sqhgjY2_Minus[jj][ii]=Y2[ii]-tmp1/SQ2;
	}
      }
      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(sqhgjY2_Plus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  sqhgjY2_Plus[jj][ii]=sqstep*gn[jj][ii]; /* completed */
	}
	gfunc_gene(sqhgjY2_Minus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  sqhgjY2_Minus[jj][ii]=sqstep*gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=tmp2=0;
	for (jj=0; jj<wdim; jj++) {
	  tmp1+=gjY2_Plus[jj][ii]-gjY2_Minus[jj][ii];
	  switch (wj[jj]) {
	  case 1:
	    tmp2+=(sqhgjY2_Plus[jj][ii]+sqhgjY2_Minus[jj][ii]);
	    break;
	  case -1:
	    tmp2-=(sqhgjY2_Plus[jj][ii]+sqhgjY2_Minus[jj][ii]);
	    break;
	    /* default: */
	    /* Nothing to do. */
	  }
	}
	sumH[ii]=tmp1/2.0+tmp2/2.0*SQ3; /* completed */
      }

      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=tmp2=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=hPh2Mat[ll+jj]*fY2add_fY4add_Minus_fn[jj];
	  tmp2+=ex_05Mat[ll+jj]*sumH[jj];
	}
	yn1[ii]=Y1[ii]+tmp1/3.0+tmp2;
      }
      
    } /* End of loop for itr */
    free(gn);
    free(sqhgjY2);
    free(hgjY2);
    free(gjY2_Plus);
    free(gjY2_Minus);
    free(sqhgjY2_Plus);
    free(sqhgjY2_Minus);
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSERK; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
  }
  return 0;
}

extern int
OMP_wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti_withCntMatProc(int ydim,
							unsigned long traj,
							double *yvec,
							double step,
							char *ran2pFull,
							char *ran3p,
							double A_mat[],
							void (*ffunc)(),
							void (*gfunc_gene)(),
							int wdim,
							double work[],
							double work_A[],
							double *ynew,
							unsigned long long *ev_cnt,
							unsigned long long *mat_proc_cnt)
/* wo2_AstabExpRK3_for_NonCommSDEs_WinMulti for Open MP */
/* This function performs an Efficient, A-stable and weak second order
   exponential RK scheme.
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
   A_mat: array for the matrix A in the drift coefficient,
   ffunc: nonlinear part in the drift coefficient,
   gfunc_gene: diffusion coefficients,
   wdim: dimension of Wiener process.

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK*(7+7*wdim)*ydim.
   work_A: workspace of length 6*ydim*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function evaluations.
   mat_proc_cnt: the number of matrix products.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par;
  static double static_step=0,
    b1=(6+SQ6)/10.0,
    b2=(3-2*SQ6)/5.0;

  int ii, jj, kk, ll, ll1, ll2, ydimPow2=ydim*ydim;
  double sqstep, glTmp1, glTmp2;
  unsigned long long func_ev_num[MaxCoreNumForSERK], mat_proc_num[MaxCoreNumForSERK];

  int  ydimDummy=ydim /* to avoid an error of dimension */;
  double *exMat, *hPh1Mat, *hPh2Mat, *ex_05Mat, *hPh1_05Mat, *hPh2_05Mat;
  
  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  ii=0;
  exMat=&work_A[ii];
  ii+=ydimPow2;
  hPh1Mat=&work_A[ii];
  ii+=ydimPow2;
  hPh2Mat=&work_A[ii];
  ii+=ydimPow2;
  ex_05Mat=&work_A[ii];
  ii+=ydimPow2;
  hPh1_05Mat=&work_A[ii];
  ii+=ydimPow2;
  hPh2_05Mat=&work_A[ii];
  ii+=ydimPow2;

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK;
    static_flag=1;
  }
  
  if (static_step!=step) {
    if(1==ydim) {
      exMat[0]=exp(A_mat[0]*step);
      hPh1Mat[0]=(exp(A_mat[0]*step)-1)/A_mat[0];
      glTmp1=A_mat[0]*step;
      glTmp2=exp(glTmp1)-1;
      hPh2Mat[0]=(glTmp2-glTmp1)/(glTmp1*A_mat[0]);
      ex_05Mat[0]=exp(A_mat[0]/2.0*step);
      hPh1_05Mat[0]=(exp(A_mat[0]/2.0*step)-1)/(A_mat[0]/2.0);
      glTmp1=A_mat[0]/2.0*step;
      glTmp2=exp(glTmp1)-1;
      hPh2_05Mat[0]=(glTmp2-glTmp1)/(glTmp1*A_mat[0]/2.0);
    }
    if(1<ydim) {
      int *indexvec;
      double *tmpMat, *bvector;
      
      double _Complex dummy, *diag, *matT, *InvMatT, *tmpMatC, *tmpDiag,
	*solution, *tmpExMat;
      
      double *eigenVal_real, *eigenVal_imag, *eigenVecs, *vecDummy;

      if (NULL == (indexvec = (int *)malloc(sizeof(int)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMat = (double *)malloc(sizeof(double)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (bvector = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (diag = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (matT = (double _Complex *)malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (InvMatT = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMatC = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpDiag = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (solution = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpExMat = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (eigenVal_real = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (eigenVal_imag = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (eigenVecs = (double *)
		   malloc(sizeof(double)*ydimPow2))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (vecDummy = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      /* Copy matrix A */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0;jj<ydim;jj++) {
	  tmpMat[ll+jj]=A_mat[ll+jj];
	}
      }
      
      if(0 != LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', ydim, tmpMat, ydim,
			    eigenVal_real, eigenVal_imag,
			    vecDummy, ydimDummy, eigenVecs, ydim)) {
	printf("Error in LAPACKE_dgeev!\n");
	exit(0);
      }

      /* Eigenvalues */
      for (ii=0; ii<ydim; ii++) {
	diag[ii]=eigenVal_real[ii]+eigenVal_imag[ii]*I;
      }

      /* Set matrix T */
      for (ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	jj = 0;
	while(jj<ydim) {
	  if (0.0 == eigenVal_imag[jj]) {
	    matT[ll+jj]=eigenVecs[ll+jj];
	    jj++;
	  } else {
	    matT[ll+jj]  =eigenVecs[ll+jj]+eigenVecs[ll+(jj+1)]*I;
	    matT[ll+jj+1]=eigenVecs[ll+jj]-eigenVecs[ll+(jj+1)]*I;
	    jj+=2;
	  }
	}
      }

      /* Calculation for inverse matrix T */
      for (ii=0; ii<ydimPow2; ii++) {
	tmpMatC[ii]=matT[ii];
      }
      if (1==ludecomp(ydim, 1.0e-14, tmpMatC, indexvec)) {
	printf("Error in ludecomp!\n");
	exit(0);
      }
      for (ii=0; ii<ydim; ii++) {
	for (jj=0; jj<ii; jj++) {
	  bvector[jj]=0;
	}
	bvector[ii]=1;
	for (jj=ii+1; jj<ydim; jj++) {
	  bvector[jj]=0;
	}
	solve(ydim, tmpMatC, bvector, indexvec, solution);
	
	for (jj=0; jj<ydim; jj++) {
	  InvMatT[jj*ydim+ii]=solution[jj];
	}
      }

      /* Calculation for diagonal elements in exMat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=cexp(diag[ii]*step);
      }

      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  exMat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in hPh1Mat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=(cexp(diag[ii]*step)-1)/diag[ii];
      }
      
      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }
      
      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  hPh1Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in hPh2Mat */
      for(ii=0;ii<ydim;ii++) {
	glTmp1=diag[ii]*step;
	glTmp2=cexp(glTmp1)-1;
	tmpDiag[ii]=(glTmp2-glTmp1)/(glTmp1*diag[ii]);
      }
      
      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }
      
      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  hPh2Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in ex_05Mat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=cexp(diag[ii]/2.0*step);
      }

      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  ex_05Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in hPh1_05Mat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=(cexp(diag[ii]/2.0*step)-1)/(diag[ii]/2.0);
      }
      
      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }
      
      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  hPh1_05Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

      /* Calculation for diagonal elements in hPh2_05Mat */
      for(ii=0;ii<ydim;ii++) {
	glTmp1=diag[ii]/2.0*step;
	glTmp2=cexp(glTmp1)-1;
	tmpDiag[ii]=(glTmp2-glTmp1)/(glTmp1*diag[ii]/2.0);
      }
      
      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }
      
      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  hPh2_05Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

    } /* End of if(1<ydim) */
    static_step=step;
  } /* End of if (static_step!=step) */
  
#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    double *yn, *yn1, *fn, *Y1, *Y2, *fY2add_fY4add_Minus_fn,
      *sum_sqhgjY2_gzai, *sumH, tmp1, tmp2;
    double **gn, *dummy, **sqhgjY2, **hgjY2, **gjY2_Plus, **gjY2_Minus,
      **sqhgjY2_Plus, **sqhgjY2_Minus;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;
    mat_proc_num[ii_par]=0;

    if (NULL == (gn = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgjY2 = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (hgjY2 = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjY2_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjY2_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgjY2_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (sqhgjY2_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }

    ibase_work_step=7+7*wdim;
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
    for (jj=0; jj<wdim; jj++) {
      gn[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgjY2[jj]=&work[ibase_work+ii];
      ii+=ydim;
      hgjY2[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjY2_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjY2_Minus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgjY2_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      sqhgjY2_Minus[jj]=&work[ibase_work+ii];
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
      
      /* See No. 69 in Note '12/'13. */
      ffunc(yn,fn); func_ev_num[ii_par]++;
      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=tmp2=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=hPh1Mat[ll+jj]*fn[jj];
	  tmp2+=hPh1_05Mat[ll+jj]*fn[jj];
	}
	Y1[ii]=tmp1;
	Y2[ii]=tmp2/2.0;
	tmp1=tmp2=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=exMat[ll+jj]*yn[jj];
	  tmp2+=ex_05Mat[ll+jj]*yn[jj];
	}
	Y1[ii]+=tmp1; /* completed */
	Y2[ii]+=tmp2; /* completed */
      }
      mat_proc_num[ii_par]++; /* For hPh1Mat[ll+jj]*fn[jj] */
      mat_proc_num[ii_par]++; /* For hPh1_05Mat[ll+jj]*fn[jj] */
      mat_proc_num[ii_par]++; /* For exMat[ll+jj]*yn[jj] */
      mat_proc_num[ii_par]++; /* For ex_05Mat[ll+jj]*yn[jj] */

      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(Y2,jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  sqhgjY2[jj][ii]=sqstep*gn[jj][ii]; /* completed */
	  hgjY2[jj][ii]=step*gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=0;
	for (jj=0; jj<wdim; jj++) {
	  switch (wj[jj]) {
	  case 1:
	    tmp1+=sqhgjY2[jj][ii];
	    break;
	  case -1:
	    tmp1-=sqhgjY2[jj][ii];
	    break;
	    /* default: */
	    /* Nothing to do */
	  }
	}
	sum_sqhgjY2_gzai[ii]=SQ3*tmp1;
      }
      
      for (ii=0; ii<ydim; ii++) {
	sumH[ii]=fn[ii];
      }
      ffunc(Y2,fn); func_ev_num[ii_par]++;
      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=(hPh2_05Mat[ll+jj]+hPh2Mat[ll+jj])
	    *(fn[jj]-sumH[jj]);
	}
	/* The next is for (Y4 + an addi. term). */
	fY2add_fY4add_Minus_fn[ii]=Y1[ii]+tmp1*2.0+b2*sum_sqhgjY2_gzai[ii];
      }
      ffunc(fY2add_fY4add_Minus_fn,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]=fn[ii]-5*sumH[ii];
      }
      for (ii=0; ii<ydim; ii++) {
	sumH[ii]=Y2[ii]+b1*sum_sqhgjY2_gzai[ii];
      }
      ffunc(sumH,fn); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	fY2add_fY4add_Minus_fn[ii]+=4*fn[ii]; /* completed */
      }
      mat_proc_num[ii_par]++; /* For (hPh2_05Mat[ll+jj]+hPh2Mat[ll+jj])
			       *(fn[jj]-sumH[jj]) */

      /* For details, see (337) and No. 69 in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	for (jj=0; jj<wdim; jj++) {
	  tmp1=0;
	  switch (wj[jj]) {
	  case 1: /* 1==wj[jj] */
	    for (kk=0; kk<jj; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(3.0+wtj[jj])*hgjY2[kk][ii];
		break;
	      case -1:
		tmp1+=(-3.0+wtj[jj])*hgjY2[kk][ii];
		break;
	      default:
		tmp1+=(wtj[jj])*hgjY2[kk][ii];
	      }
	    }
	    tmp1+=2.0*hgjY2[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(3.0-wtj[kk])*hgjY2[kk][ii];
		break;
	      case -1:
		tmp1+=(-3.0-wtj[kk])*hgjY2[kk][ii];
		break;
	      default:
		tmp1+=(-wtj[kk])*hgjY2[kk][ii];
	      }
	    }
	    break;
	  case -1: /* -1==wj[jj] */
	    for (kk=0; kk<jj; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(-3.0+wtj[jj])*hgjY2[kk][ii];
		break;
	      case -1:
		tmp1+=(3.0+wtj[jj])*hgjY2[kk][ii];
		break;
	      default:
		tmp1+=(wtj[jj])*hgjY2[kk][ii];
	      }
	    }
	    tmp1+=2.0*hgjY2[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(-3.0-wtj[kk])*hgjY2[kk][ii];
		break;
	      case -1:
		tmp1+=(3.0-wtj[kk])*hgjY2[kk][ii];
		break;
	      default:
		tmp1+=(-wtj[kk])*hgjY2[kk][ii];
	      }
	    }
	    break;
	  default: /* 0==wj[jj] */
	    /* When jj=0, the following loop will not be performed. */
	    for (kk=0; kk<jj; kk++) {
	      tmp1+=(wtj[jj])*hgjY2[kk][ii];
	    }
	    tmp1+=-hgjY2[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      tmp1+=(-wtj[kk])*hgjY2[kk][ii];
	    }
	  } /* End of the switch for jj */
	  gjY2_Plus[jj][ii]=Y2[ii]+tmp1/2.0;
	  gjY2_Minus[jj][ii]=Y2[ii]-tmp1/2.0;
	} /* End of the loop for jj */
      } /* End of the loop for ii */
      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(gjY2_Plus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  gjY2_Plus[jj][ii]=gn[jj][ii]; /* completed */
	}
	gfunc_gene(gjY2_Minus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  gjY2_Minus[jj][ii]=gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=0;
	for (kk=0; kk<wdim; kk++) {
	  if (0<wtj[kk]) {
	    tmp1+=sqhgjY2[kk][ii];
	  } else {
	    tmp1-=sqhgjY2[kk][ii];
	  }
	}
	for (jj=0; jj<wdim; jj++) {
	  sqhgjY2_Plus[jj][ii]=Y2[ii]+tmp1/SQ2;
	  sqhgjY2_Minus[jj][ii]=Y2[ii]-tmp1/SQ2;
	}
      }
      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(sqhgjY2_Plus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  sqhgjY2_Plus[jj][ii]=sqstep*gn[jj][ii]; /* completed */
	}
	gfunc_gene(sqhgjY2_Minus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  sqhgjY2_Minus[jj][ii]=sqstep*gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=tmp2=0;
	for (jj=0; jj<wdim; jj++) {
	  tmp1+=gjY2_Plus[jj][ii]-gjY2_Minus[jj][ii];
	  switch (wj[jj]) {
	  case 1:
	    tmp2+=(sqhgjY2_Plus[jj][ii]+sqhgjY2_Minus[jj][ii]);
	    break;
	  case -1:
	    tmp2-=(sqhgjY2_Plus[jj][ii]+sqhgjY2_Minus[jj][ii]);
	    break;
	    /* default: */
	    /* Nothing to do. */
	  }
	}
	sumH[ii]=tmp1/2.0+tmp2/2.0*SQ3; /* completed */
      }

      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=tmp2=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=hPh2Mat[ll+jj]*fY2add_fY4add_Minus_fn[jj];
	  tmp2+=ex_05Mat[ll+jj]*sumH[jj];
	}
	yn1[ii]=Y1[ii]+tmp1/3.0+tmp2;
      }
      mat_proc_num[ii_par]++; /* For hPh2Mat[ll+jj]*fY2add_fY4add_Minus_fn[jj] */
      mat_proc_num[ii_par]++; /* For ex_05Mat[ll+jj]*sumH[jj] */
      
    } /* End of loop for itr */
    free(gn);
    free(sqhgjY2);
    free(hgjY2);
    free(gjY2_Plus);
    free(gjY2_Minus);
    free(sqhgjY2_Plus);
    free(sqhgjY2_Minus);
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSERK; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
    *mat_proc_cnt+=mat_proc_num[ii_par];
  }
  return 0;
}

extern int OMP_wo2_SSDFMT_for_NonCommSDEs_WinMulti(int ydim,
						   unsigned long traj,
						   double *yvec,
						   double step,
						   char *ran2pFull,
						   char *ran3p,
						   double A_mat[],
						   void (*ffunc)(),
						   void (*gfunc_gene)(),
						   int wdim,
						   double work[],
						   double work_A[],
						   double *ynew)
/* wo2_SSDFMT_for_NonCommSDEs_WinMulti for Open MP */
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
   A_mat: array for the matrix A in the drift coefficient,
   ffunc: drift coefficient,
   gfunc_gene: diffusion coefficients,
   wdim: dimension of Wiener process.

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK*(5+6*wdim)*ydim.
   work_A: workspace of length 1*ydim*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par;
  static double static_step=0;

  int ii, jj, kk, ll, ll1, ll2, ydimPow2=ydim*ydim;
  double sqstep;

  int  ydimDummy=ydim /* to avoid an error of dimension */;
  double *ex_05Mat;

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  ii=0;
  ex_05Mat=&work_A[ii];
  ii+=ydimPow2;

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK;
    static_flag=1;
  }

  if (static_step!=step) {
    if(1==ydim) {
      ex_05Mat[0]=exp(A_mat[0]/2.0*step);
    }
    if(1<ydim) {
      int *indexvec;
      double *tmpMat, *bvector;
      
      double _Complex dummy, *diag, *matT, *InvMatT, *tmpMatC, *tmpDiag,
	*solution, *tmpExMat;
      
      double *eigenVal_real, *eigenVal_imag, *eigenVecs, *vecDummy;

      if (NULL == (indexvec = (int *)malloc(sizeof(int)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMat = (double *)malloc(sizeof(double)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (bvector = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (diag = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (matT = (double _Complex *)malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (InvMatT = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMatC = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpDiag = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (solution = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpExMat = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (eigenVal_real = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (eigenVal_imag = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (eigenVecs = (double *)
		   malloc(sizeof(double)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (vecDummy = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      /* Copy matrix A */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0;jj<ydim;jj++) {
	  tmpMat[ll+jj]=A_mat[ll+jj];
	}
      }
      
      if(0 != LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', ydim, tmpMat, ydim,
			    eigenVal_real, eigenVal_imag,
			    vecDummy, ydimDummy, eigenVecs, ydim)) {
	printf("Error in LAPACKE_dgeev!\n");
	exit(0);
      }

      /* Eigenvalues */
      for (ii=0; ii<ydim; ii++) {
	diag[ii]=eigenVal_real[ii]+eigenVal_imag[ii]*I;
      }

      /* Set matrix T */
      for (ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	jj = 0;
	while(jj<ydim) {
	  if (0.0 == eigenVal_imag[jj]) {
	    matT[ll+jj]=eigenVecs[ll+jj];
	    jj++;
	  } else {
	    matT[ll+jj]  =eigenVecs[ll+jj]+eigenVecs[ll+(jj+1)]*I;
	    matT[ll+jj+1]=eigenVecs[ll+jj]-eigenVecs[ll+(jj+1)]*I;
	    jj+=2;
	  }
	}
      }

      /* Calculation for inverse matrix T */
      for (ii=0; ii<ydimPow2; ii++) {
	tmpMatC[ii]=matT[ii];
      }
      if (1==ludecomp(ydim, 1.0e-14, tmpMatC, indexvec)) {
	printf("Error in ludecomp!\n");
	exit(0);
      }
      for (ii=0; ii<ydim; ii++) {
	for (jj=0; jj<ii; jj++) {
	  bvector[jj]=0;
	}
	bvector[ii]=1;
	for (jj=ii+1; jj<ydim; jj++) {
	  bvector[jj]=0;
	}
	solve(ydim, tmpMatC, bvector, indexvec, solution);
	
	for (jj=0; jj<ydim; jj++) {
	  InvMatT[jj*ydim+ii]=solution[jj];
	}
      }

      /* Calculation for diagonal elements in ex_05Mat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=cexp(diag[ii]/2.0*step);
      }

      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  ex_05Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

    } /* End of if(1<ydim) */
    static_step=step;
  } /* End of if (static_step!=step) */
  
#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    double *yn, *yn1, *fyn, *K1, *fK2, *exYn, tmp1, tmp2;
    double **gn, *dummy, **gjyn, **gjyn_Plus, **gjyn_Minus,
      **gj_yn_K1_dev2_Plus, **gj_yn_K1_dev2_Minus;
    char *wj, *wtj;

    if (NULL == (gn = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjyn = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjyn_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjyn_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL ==
	(gj_yn_K1_dev2_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL ==
	(gj_yn_K1_dev2_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }

    ibase_work_step=5+6*wdim;
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
    for (jj=0; jj<wdim; jj++) {
      gn[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjyn[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjyn_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjyn_Minus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gj_yn_K1_dev2_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gj_yn_K1_dev2_Minus[jj]=&work[ibase_work+ii];
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
      
      /* See No. 42 and No. 56 in Note '12/'13. */
      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=ex_05Mat[ll+jj]*yn[jj];
	}
	exYn[ii]=tmp1;
      }
      ffunc(exYn,fyn); /* completed */
      for(ii=0; ii<ydim; ii++) {
	K1[ii]=exYn[ii]+step*fyn[ii]; /* completed */
      }

      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(exYn,jj,gjyn[jj]); /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=0;
	for (jj=0; jj<wdim; jj++) {
	  switch (wj[jj]) {
	  case 1:
	    tmp1+=gjyn[jj][ii];
	    break;
	  case -1:
	    tmp1-=gjyn[jj][ii];
	    break;
	    /* default: */
	    /* Nothing to do */
	  }
	}
	yn1[ii]=K1[ii]+sqstep*SQ3*tmp1;
      }
      ffunc(yn1,fK2); /* completed */

      /* For details, see (337) and (338) in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	for (jj=0; jj<wdim; jj++) {
	  tmp1=0;
	  switch (wj[jj]) {
	  case 1: /* 1==wj[jj] */
	    for (kk=0; kk<jj; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(3.0+wtj[jj])*gjyn[kk][ii];
		break;
	      case -1:
		tmp1+=(-3.0+wtj[jj])*gjyn[kk][ii];
		break;
	      default:
		tmp1+=(wtj[jj])*gjyn[kk][ii];
	      }
	    }
	    tmp1+=2.0*gjyn[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(3.0-wtj[kk])*gjyn[kk][ii];
		break;
	      case -1:
		tmp1+=(-3.0-wtj[kk])*gjyn[kk][ii];
		break;
	      default:
		tmp1+=(-wtj[kk])*gjyn[kk][ii];
	      }
	    }
	    break;
	  case -1: /* -1==wj[jj] */
	    for (kk=0; kk<jj; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(-3.0+wtj[jj])*gjyn[kk][ii];
		break;
	      case -1:
		tmp1+=(3.0+wtj[jj])*gjyn[kk][ii];
		break;
	      default:
		tmp1+=(wtj[jj])*gjyn[kk][ii];
	      }
	    }
	    tmp1+=2.0*gjyn[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(-3.0-wtj[kk])*gjyn[kk][ii];
		break;
	      case -1:
		tmp1+=(3.0-wtj[kk])*gjyn[kk][ii];
		break;
	      default:
		tmp1+=(-wtj[kk])*gjyn[kk][ii];
	      }
	    }
	    break;
	  default: /* 0==wj[jj] */
	    /* When jj=0, the following loop will not be performed. */
	    for (kk=0; kk<jj; kk++) {
	      tmp1+=(wtj[jj])*gjyn[kk][ii];
	    }
	    tmp1+=-gjyn[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      tmp1+=(-wtj[kk])*gjyn[kk][ii];
	    }
	  } /* End of the switch for jj */
	  gjyn_Plus[jj][ii]=exYn[ii]+step*tmp1/2.0;
	  gjyn_Minus[jj][ii]=exYn[ii]-step*tmp1/2.0;
	} /* End of the loop for jj */
      } /* End of the loop for ii */
      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(gjyn_Plus[jj],jj,gn[jj]);
	for (ii=0; ii<ydim; ii++) {
	  gjyn_Plus[jj][ii]=gn[jj][ii]; /* completed */
	}
	gfunc_gene(gjyn_Minus[jj],jj,gn[jj]);
	for (ii=0; ii<ydim; ii++) {
	  gjyn_Minus[jj][ii]=gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=0;
	for (kk=0; kk<wdim; kk++) {
	  if (0<wtj[kk]) {
	    tmp1+=gjyn[kk][ii];
	  } else {
	    tmp1-=gjyn[kk][ii];
	  }
	}
	for (jj=0; jj<wdim; jj++) {
	  gj_yn_K1_dev2_Plus[jj][ii]=(exYn[ii]+K1[ii])/2.0+sqstep*tmp1/SQ2;
	  gj_yn_K1_dev2_Minus[jj][ii]=(exYn[ii]+K1[ii])/2.0-sqstep*tmp1/SQ2;
	}
      }
      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(gj_yn_K1_dev2_Plus[jj],jj,gn[jj]);
	for (ii=0; ii<ydim; ii++) {
	  gj_yn_K1_dev2_Plus[jj][ii]=gn[jj][ii]; /* completed */
	}
	gfunc_gene(gj_yn_K1_dev2_Minus[jj],jj,gn[jj]);
	for (ii=0; ii<ydim; ii++) {
	  gj_yn_K1_dev2_Minus[jj][ii]=gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=tmp2=0;
	for (jj=0; jj<wdim; jj++) {
	  tmp1+=gjyn_Plus[jj][ii]-gjyn_Minus[jj][ii];
	  switch (wj[jj]) {
	  case 1:
	    tmp2+=(gj_yn_K1_dev2_Plus[jj][ii]+gj_yn_K1_dev2_Minus[jj][ii]);
	    break;
	  case -1:
	    tmp2-=(gj_yn_K1_dev2_Plus[jj][ii]+gj_yn_K1_dev2_Minus[jj][ii]);
	    break;
	    /* default: */
	    /* Nothing to do. */
	  }
	}
	exYn[ii]+=step*(fyn[ii]+fK2[ii])/2.0+tmp1/2.0+sqstep*tmp2/2.0*SQ3;
      }
      
      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=ex_05Mat[ll+jj]*exYn[jj];
	}
	yn1[ii]=tmp1; /* completed */
      }
    } /* End of loop for itr */
    free(gn);
    free(gjyn);
    free(gjyn_Plus);
    free(gjyn_Minus);
    free(gj_yn_K1_dev2_Plus);
    free(gj_yn_K1_dev2_Minus);
  }
  return 0;
}

extern int OMP_wo2_SSDFMT_for_NonCommSDEs_WinMulti_withCnt(int ydim,
							   unsigned long traj,
							   double *yvec,
							   double step,
							   char *ran2pFull,
							   char *ran3p,
							   double A_mat[],
							   void (*ffunc)(),
							   void (*gfunc_gene)(),
							   int wdim,
							   double work[],
							   double work_A[],
							   double *ynew,
							   unsigned long long *ev_cnt)
/* wo2_SSDFMT_for_NonCommSDEs_WinMulti for Open MP */
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
   A_mat: array for the matrix A in the drift coefficient,
   ffunc: drift coefficient,
   gfunc_gene: diffusion coefficients,
   wdim: dimension of Wiener process.

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK*(5+6*wdim)*ydim.
   work_A: workspace of length 1*ydim*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function evaluations.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par;
  static double static_step=0;

  int ii, jj, kk, ll, ll1, ll2, ydimPow2=ydim*ydim;
  double sqstep;

  int  ydimDummy=ydim /* to avoid an error of dimension */;
  double *ex_05Mat;
  unsigned long long func_ev_num[MaxCoreNumForSERK];

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  ii=0;
  ex_05Mat=&work_A[ii];
  ii+=ydimPow2;

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK;
    static_flag=1;
  }

  if (static_step!=step) {
    if(1==ydim) {
      ex_05Mat[0]=exp(A_mat[0]/2.0*step);
    }
    if(1<ydim) {
      int *indexvec;
      double *tmpMat, *bvector;
      
      double _Complex dummy, *diag, *matT, *InvMatT, *tmpMatC, *tmpDiag,
	*solution, *tmpExMat;
      
      double *eigenVal_real, *eigenVal_imag, *eigenVecs, *vecDummy;

      if (NULL == (indexvec = (int *)malloc(sizeof(int)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMat = (double *)malloc(sizeof(double)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (bvector = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (diag = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (matT = (double _Complex *)malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (InvMatT = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMatC = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpDiag = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (solution = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpExMat = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (eigenVal_real = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (eigenVal_imag = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (eigenVecs = (double *)
		   malloc(sizeof(double)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (vecDummy = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      /* Copy matrix A */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0;jj<ydim;jj++) {
	  tmpMat[ll+jj]=A_mat[ll+jj];
	}
      }
      
      if(0 != LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', ydim, tmpMat, ydim,
			    eigenVal_real, eigenVal_imag,
			    vecDummy, ydimDummy, eigenVecs, ydim)) {
	printf("Error in LAPACKE_dgeev!\n");
	exit(0);
      }

      /* Eigenvalues */
      for (ii=0; ii<ydim; ii++) {
	diag[ii]=eigenVal_real[ii]+eigenVal_imag[ii]*I;
      }

      /* Set matrix T */
      for (ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	jj = 0;
	while(jj<ydim) {
	  if (0.0 == eigenVal_imag[jj]) {
	    matT[ll+jj]=eigenVecs[ll+jj];
	    jj++;
	  } else {
	    matT[ll+jj]  =eigenVecs[ll+jj]+eigenVecs[ll+(jj+1)]*I;
	    matT[ll+jj+1]=eigenVecs[ll+jj]-eigenVecs[ll+(jj+1)]*I;
	    jj+=2;
	  }
	}
      }

      /* Calculation for inverse matrix T */
      for (ii=0; ii<ydimPow2; ii++) {
	tmpMatC[ii]=matT[ii];
      }
      if (1==ludecomp(ydim, 1.0e-14, tmpMatC, indexvec)) {
	printf("Error in ludecomp!\n");
	exit(0);
      }
      for (ii=0; ii<ydim; ii++) {
	for (jj=0; jj<ii; jj++) {
	  bvector[jj]=0;
	}
	bvector[ii]=1;
	for (jj=ii+1; jj<ydim; jj++) {
	  bvector[jj]=0;
	}
	solve(ydim, tmpMatC, bvector, indexvec, solution);
	
	for (jj=0; jj<ydim; jj++) {
	  InvMatT[jj*ydim+ii]=solution[jj];
	}
      }

      /* Calculation for diagonal elements in ex_05Mat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=cexp(diag[ii]/2.0*step);
      }

      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  ex_05Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

    } /* End of if(1<ydim) */
    static_step=step;
  } /* End of if (static_step!=step) */
  
#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    double *yn, *yn1, *fyn, *K1, *fK2, *exYn, tmp1, tmp2;
    double **gn, *dummy, **gjyn, **gjyn_Plus, **gjyn_Minus,
      **gj_yn_K1_dev2_Plus, **gj_yn_K1_dev2_Minus;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;

    if (NULL == (gn = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjyn = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjyn_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjyn_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL ==
	(gj_yn_K1_dev2_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL ==
	(gj_yn_K1_dev2_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }

    ibase_work_step=5+6*wdim;
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
    for (jj=0; jj<wdim; jj++) {
      gn[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjyn[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjyn_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjyn_Minus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gj_yn_K1_dev2_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gj_yn_K1_dev2_Minus[jj]=&work[ibase_work+ii];
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
      
      /* See No. 42 and No. 56 in Note '12/'13. */
      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=ex_05Mat[ll+jj]*yn[jj];
	}
	exYn[ii]=tmp1;
      }
      ffunc(exYn,fyn); /* completed */
      func_ev_num[ii_par]++;
      for(ii=0; ii<ydim; ii++) {
	K1[ii]=exYn[ii]+step*fyn[ii]; /* completed */
      }

      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(exYn,jj,gjyn[jj]); /* completed */
	func_ev_num[ii_par]++;
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=0;
	for (jj=0; jj<wdim; jj++) {
	  switch (wj[jj]) {
	  case 1:
	    tmp1+=gjyn[jj][ii];
	    break;
	  case -1:
	    tmp1-=gjyn[jj][ii];
	    break;
	    /* default: */
	    /* Nothing to do */
	  }
	}
	yn1[ii]=K1[ii]+sqstep*SQ3*tmp1;
      }
      ffunc(yn1,fK2); /* completed */
      func_ev_num[ii_par]++;

      /* For details, see (337) and (338) in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	for (jj=0; jj<wdim; jj++) {
	  tmp1=0;
	  switch (wj[jj]) {
	  case 1: /* 1==wj[jj] */
	    for (kk=0; kk<jj; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(3.0+wtj[jj])*gjyn[kk][ii];
		break;
	      case -1:
		tmp1+=(-3.0+wtj[jj])*gjyn[kk][ii];
		break;
	      default:
		tmp1+=(wtj[jj])*gjyn[kk][ii];
	      }
	    }
	    tmp1+=2.0*gjyn[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(3.0-wtj[kk])*gjyn[kk][ii];
		break;
	      case -1:
		tmp1+=(-3.0-wtj[kk])*gjyn[kk][ii];
		break;
	      default:
		tmp1+=(-wtj[kk])*gjyn[kk][ii];
	      }
	    }
	    break;
	  case -1: /* -1==wj[jj] */
	    for (kk=0; kk<jj; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(-3.0+wtj[jj])*gjyn[kk][ii];
		break;
	      case -1:
		tmp1+=(3.0+wtj[jj])*gjyn[kk][ii];
		break;
	      default:
		tmp1+=(wtj[jj])*gjyn[kk][ii];
	      }
	    }
	    tmp1+=2.0*gjyn[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(-3.0-wtj[kk])*gjyn[kk][ii];
		break;
	      case -1:
		tmp1+=(3.0-wtj[kk])*gjyn[kk][ii];
		break;
	      default:
		tmp1+=(-wtj[kk])*gjyn[kk][ii];
	      }
	    }
	    break;
	  default: /* 0==wj[jj] */
	    /* When jj=0, the following loop will not be performed. */
	    for (kk=0; kk<jj; kk++) {
	      tmp1+=(wtj[jj])*gjyn[kk][ii];
	    }
	    tmp1+=-gjyn[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      tmp1+=(-wtj[kk])*gjyn[kk][ii];
	    }
	  } /* End of the switch for jj */
	  gjyn_Plus[jj][ii]=exYn[ii]+step*tmp1/2.0;
	  gjyn_Minus[jj][ii]=exYn[ii]-step*tmp1/2.0;
	} /* End of the loop for jj */
      } /* End of the loop for ii */
      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(gjyn_Plus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  gjyn_Plus[jj][ii]=gn[jj][ii]; /* completed */
	}
	gfunc_gene(gjyn_Minus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  gjyn_Minus[jj][ii]=gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=0;
	for (kk=0; kk<wdim; kk++) {
	  if (0<wtj[kk]) {
	    tmp1+=gjyn[kk][ii];
	  } else {
	    tmp1-=gjyn[kk][ii];
	  }
	}
	for (jj=0; jj<wdim; jj++) {
	  gj_yn_K1_dev2_Plus[jj][ii]=(exYn[ii]+K1[ii])/2.0+sqstep*tmp1/SQ2;
	  gj_yn_K1_dev2_Minus[jj][ii]=(exYn[ii]+K1[ii])/2.0-sqstep*tmp1/SQ2;
	}
      }
      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(gj_yn_K1_dev2_Plus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  gj_yn_K1_dev2_Plus[jj][ii]=gn[jj][ii]; /* completed */
	}
	gfunc_gene(gj_yn_K1_dev2_Minus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  gj_yn_K1_dev2_Minus[jj][ii]=gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=tmp2=0;
	for (jj=0; jj<wdim; jj++) {
	  tmp1+=gjyn_Plus[jj][ii]-gjyn_Minus[jj][ii];
	  switch (wj[jj]) {
	  case 1:
	    tmp2+=(gj_yn_K1_dev2_Plus[jj][ii]+gj_yn_K1_dev2_Minus[jj][ii]);
	    break;
	  case -1:
	    tmp2-=(gj_yn_K1_dev2_Plus[jj][ii]+gj_yn_K1_dev2_Minus[jj][ii]);
	    break;
	    /* default: */
	    /* Nothing to do. */
	  }
	}
	exYn[ii]+=step*(fyn[ii]+fK2[ii])/2.0+tmp1/2.0+sqstep*tmp2/2.0*SQ3;
      }
      
      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=ex_05Mat[ll+jj]*exYn[jj];
	}
	yn1[ii]=tmp1; /* completed */
      }
    } /* End of loop for itr */
    free(gn);
    free(gjyn);
    free(gjyn_Plus);
    free(gjyn_Minus);
    free(gj_yn_K1_dev2_Plus);
    free(gj_yn_K1_dev2_Minus);
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSERK; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
  }
  return 0;
}

extern int
OMP_wo2_SSDFMT_for_NonCommSDEs_WinMulti_withCntMatProc(int ydim,
						       unsigned long traj,
						       double *yvec,
						       double step,
						       char *ran2pFull,
						       char *ran3p,
						       double A_mat[],
						       void (*ffunc)(),
						       void (*gfunc_gene)(),
						       int wdim,
						       double work[],
						       double work_A[],
						       double *ynew,
						       unsigned long long *ev_cnt,
						       unsigned long long *mat_proc_cnt)
/* wo2_SSDFMT_for_NonCommSDEs_WinMulti for Open MP */
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
   A_mat: array for the matrix A in the drift coefficient,
   ffunc: drift coefficient,
   gfunc_gene: diffusion coefficients,
   wdim: dimension of Wiener process.

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK*(5+6*wdim)*ydim.
   work_A: workspace of length 1*ydim*ydim.

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function evaluations.
   mat_proc_cnt: the number of matrix products.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par;
  static double static_step=0;

  int ii, jj, kk, ll, ll1, ll2, ydimPow2=ydim*ydim;
  double sqstep;

  int  ydimDummy=ydim /* to avoid an error of dimension */;
  double *ex_05Mat;
  unsigned long long func_ev_num[MaxCoreNumForSERK], mat_proc_num[MaxCoreNumForSERK];

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  ii=0;
  ex_05Mat=&work_A[ii];
  ii+=ydimPow2;

  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK;
    static_flag=1;
  }

  if (static_step!=step) {
    if(1==ydim) {
      ex_05Mat[0]=exp(A_mat[0]/2.0*step);
    }
    if(1<ydim) {
      int *indexvec;
      double *tmpMat, *bvector;
      
      double _Complex dummy, *diag, *matT, *InvMatT, *tmpMatC, *tmpDiag,
	*solution, *tmpExMat;
      
      double *eigenVal_real, *eigenVal_imag, *eigenVecs, *vecDummy;

      if (NULL == (indexvec = (int *)malloc(sizeof(int)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMat = (double *)malloc(sizeof(double)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (bvector = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (diag = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (matT = (double _Complex *)malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (InvMatT = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMatC = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpDiag = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (solution = (double _Complex *)malloc(sizeof(dummy)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpExMat = (double _Complex *)
		   malloc(sizeof(dummy)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (eigenVal_real = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (eigenVal_imag = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (eigenVecs = (double *)
		   malloc(sizeof(double)*ydimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (vecDummy = (double *)malloc(sizeof(double)*ydim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      /* Copy matrix A */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0;jj<ydim;jj++) {
	  tmpMat[ll+jj]=A_mat[ll+jj];
	}
      }
      
      if(0 != LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', ydim, tmpMat, ydim,
			    eigenVal_real, eigenVal_imag,
			    vecDummy, ydimDummy, eigenVecs, ydim)) {
	printf("Error in LAPACKE_dgeev!\n");
	exit(0);
      }

      /* Eigenvalues */
      for (ii=0; ii<ydim; ii++) {
	diag[ii]=eigenVal_real[ii]+eigenVal_imag[ii]*I;
      }

      /* Set matrix T */
      for (ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	jj = 0;
	while(jj<ydim) {
	  if (0.0 == eigenVal_imag[jj]) {
	    matT[ll+jj]=eigenVecs[ll+jj];
	    jj++;
	  } else {
	    matT[ll+jj]  =eigenVecs[ll+jj]+eigenVecs[ll+(jj+1)]*I;
	    matT[ll+jj+1]=eigenVecs[ll+jj]-eigenVecs[ll+(jj+1)]*I;
	    jj+=2;
	  }
	}
      }

      /* Calculation for inverse matrix T */
      for (ii=0; ii<ydimPow2; ii++) {
	tmpMatC[ii]=matT[ii];
      }
      if (1==ludecomp(ydim, 1.0e-14, tmpMatC, indexvec)) {
	printf("Error in ludecomp!\n");
	exit(0);
      }
      for (ii=0; ii<ydim; ii++) {
	for (jj=0; jj<ii; jj++) {
	  bvector[jj]=0;
	}
	bvector[ii]=1;
	for (jj=ii+1; jj<ydim; jj++) {
	  bvector[jj]=0;
	}
	solve(ydim, tmpMatC, bvector, indexvec, solution);
	
	for (jj=0; jj<ydim; jj++) {
	  InvMatT[jj*ydim+ii]=solution[jj];
	}
      }

      /* Calculation for diagonal elements in ex_05Mat */
      for(ii=0;ii<ydim;ii++) {
	tmpDiag[ii]=cexp(diag[ii]/2.0*step);
      }

      /* diag*InvMatT */
      for(ii=0;ii<ydim;ii++) {
	ll=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpMatC[ll+jj]=tmpDiag[ii]*InvMatT[ll+jj];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<ydim; ii++) {
	ll1=ii*ydim;
	for(jj=0; jj<ydim; jj++) {
	  tmpExMat[ll1+jj]=0;
	  for(kk=0; kk<ydim; kk++) {
	    ll2=kk*ydim;
	    tmpExMat[ll1+jj]+=matT[ll1+kk]*tmpMatC[ll2+jj];
	  }
	  ex_05Mat[ll1+jj]=creal(tmpExMat[ll1+jj]);
	}
      }

    } /* End of if(1<ydim) */
    static_step=step;
  } /* End of if (static_step!=step) */
  
#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step;
    double *yn, *yn1, *fyn, *K1, *fK2, *exYn, tmp1, tmp2;
    double **gn, *dummy, **gjyn, **gjyn_Plus, **gjyn_Minus,
      **gj_yn_K1_dev2_Plus, **gj_yn_K1_dev2_Minus;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;
    mat_proc_num[ii_par]=0;

    if (NULL == (gn = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjyn = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjyn_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL == (gjyn_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL ==
	(gj_yn_K1_dev2_Plus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }
    if (NULL ==
	(gj_yn_K1_dev2_Minus = (double **)malloc(sizeof(dummy)*wdim))) {
      printf("malloc error in wo2_DFMT_for_NonCommSDEs_WinMulti\n");
      exit(0);
    }

    ibase_work_step=5+6*wdim;
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
    for (jj=0; jj<wdim; jj++) {
      gn[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjyn[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjyn_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gjyn_Minus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gj_yn_K1_dev2_Plus[jj]=&work[ibase_work+ii];
      ii+=ydim;
      gj_yn_K1_dev2_Minus[jj]=&work[ibase_work+ii];
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
      
      /* See No. 42 and No. 56 in Note '12/'13. */
      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=ex_05Mat[ll+jj]*yn[jj];
	}
	exYn[ii]=tmp1;
      }
      mat_proc_num[ii_par]++; /* For ex_05Mat[ll+jj]*yn[jj] */
      
      ffunc(exYn,fyn); /* completed */
      func_ev_num[ii_par]++;
      for(ii=0; ii<ydim; ii++) {
	K1[ii]=exYn[ii]+step*fyn[ii]; /* completed */
      }

      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(exYn,jj,gjyn[jj]); /* completed */
	func_ev_num[ii_par]++;
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=0;
	for (jj=0; jj<wdim; jj++) {
	  switch (wj[jj]) {
	  case 1:
	    tmp1+=gjyn[jj][ii];
	    break;
	  case -1:
	    tmp1-=gjyn[jj][ii];
	    break;
	    /* default: */
	    /* Nothing to do */
	  }
	}
	yn1[ii]=K1[ii]+sqstep*SQ3*tmp1;
      }
      ffunc(yn1,fK2); /* completed */
      func_ev_num[ii_par]++;

      /* For details, see (337) and (338) in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	for (jj=0; jj<wdim; jj++) {
	  tmp1=0;
	  switch (wj[jj]) {
	  case 1: /* 1==wj[jj] */
	    for (kk=0; kk<jj; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(3.0+wtj[jj])*gjyn[kk][ii];
		break;
	      case -1:
		tmp1+=(-3.0+wtj[jj])*gjyn[kk][ii];
		break;
	      default:
		tmp1+=(wtj[jj])*gjyn[kk][ii];
	      }
	    }
	    tmp1+=2.0*gjyn[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(3.0-wtj[kk])*gjyn[kk][ii];
		break;
	      case -1:
		tmp1+=(-3.0-wtj[kk])*gjyn[kk][ii];
		break;
	      default:
		tmp1+=(-wtj[kk])*gjyn[kk][ii];
	      }
	    }
	    break;
	  case -1: /* -1==wj[jj] */
	    for (kk=0; kk<jj; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(-3.0+wtj[jj])*gjyn[kk][ii];
		break;
	      case -1:
		tmp1+=(3.0+wtj[jj])*gjyn[kk][ii];
		break;
	      default:
		tmp1+=(wtj[jj])*gjyn[kk][ii];
	      }
	    }
	    tmp1+=2.0*gjyn[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      switch (wj[kk]) {
	      case 1:
		tmp1+=(-3.0-wtj[kk])*gjyn[kk][ii];
		break;
	      case -1:
		tmp1+=(3.0-wtj[kk])*gjyn[kk][ii];
		break;
	      default:
		tmp1+=(-wtj[kk])*gjyn[kk][ii];
	      }
	    }
	    break;
	  default: /* 0==wj[jj] */
	    /* When jj=0, the following loop will not be performed. */
	    for (kk=0; kk<jj; kk++) {
	      tmp1+=(wtj[jj])*gjyn[kk][ii];
	    }
	    tmp1+=-gjyn[jj][ii];
	    for (kk=jj+1; kk<wdim; kk++) {
	      tmp1+=(-wtj[kk])*gjyn[kk][ii];
	    }
	  } /* End of the switch for jj */
	  gjyn_Plus[jj][ii]=exYn[ii]+step*tmp1/2.0;
	  gjyn_Minus[jj][ii]=exYn[ii]-step*tmp1/2.0;
	} /* End of the loop for jj */
      } /* End of the loop for ii */
      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(gjyn_Plus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  gjyn_Plus[jj][ii]=gn[jj][ii]; /* completed */
	}
	gfunc_gene(gjyn_Minus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  gjyn_Minus[jj][ii]=gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=0;
	for (kk=0; kk<wdim; kk++) {
	  if (0<wtj[kk]) {
	    tmp1+=gjyn[kk][ii];
	  } else {
	    tmp1-=gjyn[kk][ii];
	  }
	}
	for (jj=0; jj<wdim; jj++) {
	  gj_yn_K1_dev2_Plus[jj][ii]=(exYn[ii]+K1[ii])/2.0+sqstep*tmp1/SQ2;
	  gj_yn_K1_dev2_Minus[jj][ii]=(exYn[ii]+K1[ii])/2.0-sqstep*tmp1/SQ2;
	}
      }
      for (jj=0; jj<wdim; jj++) {
	gfunc_gene(gj_yn_K1_dev2_Plus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  gj_yn_K1_dev2_Plus[jj][ii]=gn[jj][ii]; /* completed */
	}
	gfunc_gene(gj_yn_K1_dev2_Minus[jj],jj,gn[jj]); func_ev_num[ii_par]++;
	for (ii=0; ii<ydim; ii++) {
	  gj_yn_K1_dev2_Minus[jj][ii]=gn[jj][ii]; /* completed */
	}
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=tmp2=0;
	for (jj=0; jj<wdim; jj++) {
	  tmp1+=gjyn_Plus[jj][ii]-gjyn_Minus[jj][ii];
	  switch (wj[jj]) {
	  case 1:
	    tmp2+=(gj_yn_K1_dev2_Plus[jj][ii]+gj_yn_K1_dev2_Minus[jj][ii]);
	    break;
	  case -1:
	    tmp2-=(gj_yn_K1_dev2_Plus[jj][ii]+gj_yn_K1_dev2_Minus[jj][ii]);
	    break;
	    /* default: */
	    /* Nothing to do. */
	  }
	}
	exYn[ii]+=step*(fyn[ii]+fK2[ii])/2.0+tmp1/2.0+sqstep*tmp2/2.0*SQ3;
      }
      for(ii=0; ii<ydim; ii++) {
	ll=ii*ydim;
	tmp1=0;
	for(jj=0; jj<ydim; jj++) {
	  tmp1+=ex_05Mat[ll+jj]*exYn[jj];
	}
	yn1[ii]=tmp1; /* completed */
      }
      mat_proc_num[ii_par]++; /* For ex_05Mat[ll+jj]*exYn[jj] */
    } /* End of loop for itr */
    free(gn);
    free(gjyn);
    free(gjyn_Plus);
    free(gjyn_Minus);
    free(gj_yn_K1_dev2_Plus);
    free(gj_yn_K1_dev2_Minus);
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSERK; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
    *mat_proc_cnt+=mat_proc_num[ii_par];
  }
  return 0;
}
