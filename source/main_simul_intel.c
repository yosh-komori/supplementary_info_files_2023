/******************************************************/
/* filename: main_simul_intel.c                       */
/*                                                    */
/* Ver. 0.1 (1-Sep-2023)                              */
/* 1) OMP_wo1_skrock_eta2_for_DNoiseSDEs_WinMulti
      and
      OMP_wo1_skrock_eta2_for_DNoiseSDEs_WinMulti_withCnt
      were added.
   2) ran_gene_2p_Only_using_genrand_int32 was added.
 */
/******************************************************/
/* Ver. 0                                             */
/* This program solves Ito SDEs                       */
/*    dy=ffunc dt + sum_i=1^m gfunc_i dw_i            */
/* by an SRK.                                         */
/* This file was made to put on GitHub (26-Apr-2023). */
/******************************************************/

#include <stdio.h>
#include <math.h>
#include <direct.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <time.h>
#include "mkl_lapacke.h"

/* The following flag is used to assign an srk method. */
#define SRK_TYPE 41 /* 1, 2, 3 (for noncommutative noise),
		      20, 21, 22, 23, 41 (for diagonal noise). */

#if (21 == SRK_TYPE) || (22 == SRK_TYPE) || (23 == SRK_TYPE)
#   include "For_Kry_sub_tech.h"
#endif

#if   1 == SRK_TYPE
#  define SRK    OMP_wo2_srock2_Ab_values_for_NonCommSDEs_WinMulti
#  define SRKcnt OMP_wo2_srock2_Ab_values_for_NonCommSDEs_WinMulti_withCnt
#elif 2 == SRK_TYPE
#  define SRK    OMP_wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti
#  define SRKcnt OMP_wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti_withCnt
#  define SRKcntmat OMP_wo2_Ef_AstabExpRK3_for_NonCommSDEs_WinMulti_withCntMatProc
#elif 3 == SRK_TYPE
#  define SRK    OMP_wo2_SSDFMT_for_NonCommSDEs_WinMulti
#  define SRKcnt OMP_wo2_SSDFMT_for_NonCommSDEs_WinMulti_withCnt
#  define SRKcntmat OMP_wo2_SSDFMT_for_NonCommSDEs_WinMulti_withCntMatProc
#elif 20 == SRK_TYPE
#  define SRK    OMP_wo2_srock2_Ab_values_for_DNoiseSDEs_WinMulti
#  define SRKcnt OMP_wo2_srock2_Ab_values_for_DNoiseSDEs_WinMulti_withCnt
#elif 21 == SRK_TYPE
#  define SRK    OMP_wo2_Ef_AstERK3_Kry_sym_for_DNoiseSDEs_WinMulti
#  define SRKcnt OMP_wo2_Ef_AstERK3_Kry_sym_for_DNoiseSDEs_WinMulti_withCnt
#  define SRKcntmat OMP_wo2_Ef_AstERK3_Kry_sym_for_DNoiseSDEs_WinMulti_withCntMatProd
#elif 22 == SRK_TYPE
#  define SRK    OMP_wo2_SSDFMT_Kry_sym_for_DNoiseSDEs_WinMulti
#  define SRKcnt OMP_wo2_SSDFMT_Kry_sym_for_DNoiseSDEs_WinMulti_withCnt
#  define SRKcntmat OMP_wo2_SSDFMT_Kry_sym_for_DNoiseSDEs_WinMulti_withCntMatProd
#elif 23 == SRK_TYPE
#  define SRK    OMP_wo2_SSDFMT_Taylor_for_DNoiseSDEs_WinMulti
#  define SRKcnt OMP_wo2_SSDFMT_Taylor_for_DNoiseSDEs_WinMulti_withCnt
#  define SRKcntmat OMP_wo2_SSDFMT_Taylor_for_DNoiseSDEs_WinMulti_withCntMatProd
#elif 41 == SRK_TYPE
#  define SRK    OMP_wo1_skrock_eta2_for_DNoiseSDEs_WinMulti
#  define SRKcnt OMP_wo1_skrock_eta2_for_DNoiseSDEs_WinMulti_withCnt
#else
#  define SRK    /* nothing */
#  define SRKcnt /* nothing */
#endif

/* If CMP is defined, the number of matrix products is counted in ExpRK
   methods.
 */
#define CMP

/* If Cal_Start_Set is set at an integer x > 1, then the calculations of SRK
   will be performed from Set_x to Set_NSet only, keeping to use the same
   pseudo-random values as those used when x=1.*/
#define Cal_Start_Set 1 /* 1 or x, where x is an integer > 1. */

/* When the following is set, dispersion will be outputed. */
#define  DISPER_DATA

/* Maximum of an independent variable x. */
#define  XRANGE               1.0/2.0 /* 1.0, 1.0/2.0 */

/* Minimum of a step size. */
#define  STEPLENGCONST    1.0/64.0 /*1.0/4.0, 1.0/8.0, 1.0/2048.0 */
#define  END_STEPLENG     1.0/64.0 /*1.0/4.0, 1.0/8.0, 1.0/2048.0 */

#define  NN        127
#define  YDIM_2    NN*NN

#define  YDIM      2*YDIM_2 /* 2 (for testfunc11, testfunc12),
			2*YDIM_2 (for testfunc13). */
/* A type of SDEs. */
#define TESTFUNC    13 /* 11, 12, 13 */

#if (13 == TESTFUNC)
#   define  FDIM      2
#else
#   define  FDIM      YDIM
#endif

#define  EX_MASTERNAME    "expectfile"
#define  MOM4_MASTERNAME   "2momentfile"
#define  TIME_MASTERNAME    "timefile"
#define  COST_MASTERNAME    "costfile"

/* SDE in experiment 1 */
#if (11 == TESTFUNC)
#define  OM0      (2.0)
#define  OM1      (0.5) /*(0.5), (0)*/
#define  OM2      (0.2)
#define  BT0      (-0.5)
#define  BT1      (-0.2)
#define  BT2      (-0.1)
#endif

/* SDE in experiment 2 */
#if (12 == TESTFUNC)
#define  A11      (-100.0) /* (-100.0) */
#define  A12      (0.0)    /* 0.0, (-99.0) */
#define  A21      (0.0)    /* 0.0, (-99.0) */
#define  A22      (-1.0)   /* (-1.0), (-100.0) */ 
#define  B11      0.0
#define  B12      (1.0/2.0)
#define  B21      (1.0/2.0)
#define  B22      0.0
#endif

/* SDE in experiment 3 */
#if (13 == TESTFUNC)
#define  PI  3.1415926535897932 /* pi */
#define  GAMMA    1.0/10
#define  BETA1    1.0/(NN+1)
#define  BETA2    1.0/(NN+1)
#define  COEF1    GAMMA*(NN+1)*(NN+1)
#define  COEF2    ((NN+1)/2.0)
#endif

#if (12 == TESTFUNC)
#define  WDIM         1
#elif (11 == TESTFUNC)
#define  WDIM         2
#elif (13 == TESTFUNC)
#define  WDIM        YDIM
#endif

#define  TRAJECT    8UL/*1000UL /* 10000UL, 40000UL, 1000UL */
#define  NWinMax    WDIM*TRAJECT
#define  NWin2Max   WDIM*WDIM*TRAJECT
#define  NArrayMax  YDIM*TRAJECT
/* Total number of trajectries is TRAJECT*BATCH_NUM. */
#define  BATCH_NUM  1 /* 1, 6400 */
#define  NSet       1 /* 1, 16 */

/* ffunc is a drift coefficient.  */
static void ffunc(double ynvec[],double foutput[]);
static void makeMatA(int dim, double A_mat[]);
static void setRMAT(int dim);

/* gfunc_gene is a diffusion coefficient. */
static void gfunc_gene(double ynvec[],int i_th,double goutput[]);

/* gfunc_diag is a diagonal diffusion coefficient. */
static void gfunc_diag(double ynvec[],double goutput[]);

/* setPVec sets a vector for DSEXPMVTRAY2 */
static void setPVec();

/* For 1 == SRK_TYPE */
extern int
OMP_wo2_srock2_Ab_values_for_NonCommSDEs_WinMulti(int ydim,
						  unsigned long traj,
						  double *yvec,
						  double step,
						  char *ran2pFull,
						  char *ran3p,
						  void (*ffunc)(),
						  void (*gfunc_gene)(),
						  int wdim, int ss,
						  double work[],
						  double *ynew);
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
							  unsigned long long *ev_cnt);

/* For 2 == SRK_TYPE */
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
							   double *ynew);
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
							unsigned long long *ev_cnt);
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
							unsigned long long *mat_proc_cnt);

/* For 20 == SRK_TYPE */
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
							     double *ynew);
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
							  unsigned long long *ev_cnt);

/* For 21 == SRK_TYPE */
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
							   double *ynew);
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
							   unsigned long long *ev_cnt);
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
							   unsigned long long *Kry_mat_prod_cnt);

/* For 3 == SRK_TYPE */
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
						   double *ynew);
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
							   unsigned long long *ev_cnt);
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
						       unsigned long long *mat_proc_cnt);

/* For 22 == SRK_TYPE */
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
						   double *ynew);
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
							   unsigned long long *ev_cnt);
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
							   unsigned long long *Kry_mat_prod_cnt);

/* For 23 == SRK_TYPE */
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
						   double *ynew);
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
							  unsigned long long *ev_cnt);
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
							  unsigned long long *Taylor_mat_prod_cnt);

/* For 41 == SRK_TYPE */
extern int OMP_wo1_skrock_eta2_for_DNoiseSDEs_WinMulti(int ydim,
						  unsigned long traj,
						  double *yvec,
						  double step,
						  char *ran2p,
						  void (*ffunc)(),
						  void (*gfunc_diag)(),
						  int ss,
						  double work[],
						  double *ynew);
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
						  unsigned long long *ev_cnt);

void init_genrand(unsigned long s);
int ran_gene_2p_Only_using_genrand_int32(int traject, int wdim, char ran2p[]);
int ran_gene_full_using_genrand_int32(int traject, int wdim,
				      char ran2p[], char ran3p[]);

void makename(char mastername[], int num, char outname[]) {
  char buffer[50];
    
  sprintf_s(buffer,50,"%d",num);
  strcpy(outname,mastername);
  strcat(outname,buffer);
}

#if (21 == SRK_TYPE) || (22 == SRK_TYPE) || (23 == SRK_TYPE)
   /* The following global variables are for COMMON in FORTRAN.
      The order of the definition of the variables has to be
      the same as that in RMAT of dgmatv.f.
   */
#  pragma pack(2)
struct commonFort {
  double a[NZMAX];
  int ia[NZMAX], ja[NZMAX];
  int nz, n;
} RMAT;

struct commonFort_for_p {
  double p[170];
 } PVEC;
#  pragma pack()
#endif

int main(void) {
  /* For large memory area, the following variables are allocated
     in static area. */
  static double yvec[NArrayMax], ynew[NArrayMax];
  static char ran2p[NWinMax], ran3p[NWinMax];
  static double batch_expect[BATCH_NUM][YDIM];
  
  unsigned long seed = 5489UL; /* 5489L, 0UL */
  int traject=TRAJECT,ydim=YDIM, wdim=WDIM, i, j, ibase, i_batch, i_set,
    stagenum, fdim, jbase, jbaseInit, jbaseEnd, k;
  double stepleng,
    expect[YDIM], xpoint, eps, yinit[YDIM], 
    set_expect[NSet][YDIM],
    tmp0a, tmp0b;

#if (2 == SRK_TYPE)
  static double A_mat[YDIM*YDIM];
  static double work_A[6*YDIM*YDIM];
#elif (21 == SRK_TYPE) || (22 == SRK_TYPE)
  static double work_K[8*LWSP], workA_K[8*LWSP];
  static int iWork_K[8*NMAX];
#elif (3 == SRK_TYPE)
  static double A_mat[YDIM*YDIM];
  static double work_A[1*YDIM*YDIM];
#elif (23 == SRK_TYPE)
  static double work_T[8*LWSP], workAux[8*2*YDIM];
#else
  /* Nothing */
#endif

#if (1 == SRK_TYPE)
  static double work[8*(7+6*WDIM)*YDIM];
#elif (2 == SRK_TYPE)
  double work[8*(7+7*WDIM)*YDIM];
#elif (20 == SRK_TYPE)
  static double work[8*13*YDIM];
#elif (21 == SRK_TYPE)
  static double work[8*16*YDIM];
#elif (3 == SRK_TYPE)
  double work[8*(5+6*WDIM)*YDIM];
#elif (22 == SRK_TYPE) || (23 == SRK_TYPE)
  static double work[8*11*YDIM];
#elif (41 == SRK_TYPE)
  static double work[8*6*YDIM];
#else
  /* nothing */
#endif
  char ex_mastername[]=EX_MASTERNAME, exfname[FDIM][15], dirname[15],
    timefname[]=TIME_MASTERNAME, costfname[]=COST_MASTERNAME;
  FILE   *exfp[FDIM], *timefp, *costfp;
#ifdef DISPER_DATA
  static double batch_fourthm[BATCH_NUM][YDIM];
  double fourthm[YDIM], 
    set_fourthm[NSet][YDIM],
    tmp, tmp1;
  char mom4_mastername[]=MOM4_MASTERNAME, mom4fname[FDIM][15];
  FILE   *mom4fp[FDIM];
#endif
  unsigned long long evf_cnt, evr_cnt, evm_cnt;
  time_t start_t, finish_t;
  double elapsed_time;

#if (21 == SRK_TYPE) || (22 == SRK_TYPE)
  int mMax, lwsp, liwsp, itrace, lwspA, errChkFlag;
  double anorm, tol;

  lwsp=LWSP;
  liwsp=NMAX;
  lwspA=LWSP;
  
  mMax=30; /* 30, 100, 150, 70, 50, 30, 20 */
  itrace=0;
  tol=0.0;
  errChkFlag=0;
#elif (23 == SRK_TYPE)
  int mMin, lwsp;

  lwsp=LWSP;
  mMin=10; /* 1, 10 */
#endif

  stagenum=15; /* 80, 53, 10, 7, 5, 4 (for srock) */

#if YDIM == 2
#   if 12 == TESTFUNC
  yinit[0] = 1.0;
  yinit[1] = 1.0;
#   endif
#   if (11 == TESTFUNC)
  yinit[0] = 1.0/2;
  yinit[1] = 1.0/2;
#   endif
#endif

#if (13 == TESTFUNC)
  for (ibase=0; ibase<YDIM_2; ibase+=NN) {
    tmp0a=(double)((ibase/NN)+1)/(NN+1);
    tmp0a=4*tmp0a*(1-tmp0a);
    for (i=0; i<NN; i++) {
      yinit[i+ibase]=tmp0a*sin(PI*(i+1)/(NN+1));
    }
  }
  for (ibase=YDIM_2; ibase<YDIM; ibase+=NN) {
    tmp0a=(double)((ibase/NN)-NN+1)/(NN+1);
    tmp0a=sin(2*PI*tmp0a);
    tmp0a=tmp0a*tmp0a;
    for (i=0; i<NN; i++) {
      yinit[i+ibase]=sin(PI*(i+1)/(NN+1))*tmp0a;
    }
  }
#endif

  if (NArrayMax < YDIM*TRAJECT) {
    printf("Error: NArrayMax is too small!\n");
    exit(0);
  }
  if (NWinMax < WDIM*TRAJECT) {
    printf("Error: NWinMax is too small!\n");
    exit(0);
  }

#if (20 == SRK_TYPE) || (21 == SRK_TYPE) || (22 == SRK_TYPE) || (23 == SRK_TYPE) || (41 == SRK_TYPE)
  if (YDIM != WDIM) {
    printf("Error: YDIM != WDIM in SDEs with diagonal noise!\n");
    exit(0);
  }
#endif

  /* The following sets a seed. */
  init_genrand(seed);
    
  for(i_set=1;i_set<=NSet;i_set++) {
    makename("Set_",i_set,dirname);
    _mkdir(dirname);
    _chdir(dirname);


#if (13 == TESTFUNC)
    fdim=FDIM;
#else
    fdim=ydim;
#endif    

    for(i=0;i<fdim;i++) {
      /* Making an exfname[i] file. */
      makename(ex_mastername,i,exfname[i]);
      if(0 != fopen_s(&exfp[i],exfname[i],"w"))
	{printf("Can not open %s file\n",exfname[i]);return 1;}
      fclose(exfp[i]);
#ifdef DISPER_DATA
      /* Making a mom4fname[i] file. */
      makename(mom4_mastername,i,mom4fname[i]);
      if(0 != fopen_s(&mom4fp[i] ,mom4fname[i],"w"))
	{printf("Can not open %s file\n",mom4fname[i]);return 1;}
      fclose(mom4fp[i]);
#endif
    }
    /* Making a timefname file. */
    if(0 != fopen_s(&timefp,timefname,"w"))
      {printf("Can not open %s file\n",timefname);return 1;}
    fprintf(timefp,"{");
    fclose(timefp);

    /* Making a costfname file. */
    if(0 != fopen_s(&costfp,costfname,"w"))
      {printf("Can not open %s file\n",costfname);return 1;}
    fprintf(costfp,"{");
    fclose(costfp);

#if (2 == SRK_TYPE) || (3 == SRK_TYPE)
    makeMatA(ydim, A_mat);
#elif (21 == SRK_TYPE) || (22 == SRK_TYPE)
    setRMAT(ydim);
    for (i=0;i<RMAT.n;i++) {
      work_K[i]=0.0;
    }
    for (i=0;i<RMAT.nz;i++) {
      work_K[RMAT.ia[i]-1]=work_K[RMAT.ia[i]-1]+fabs(RMAT.a[i]);
    }
    anorm=work_K[0];
    for (i=1;i<RMAT.n;i++) {
      if (anorm<work_K[i]) {
	anorm=work_K[i];
      }
    }
#elif (23 == SRK_TYPE)
    setRMAT(ydim);
    setPVec();
#endif

    /* Begining of a loop for stepleng. */
    for(stepleng=STEPLENGCONST;stepleng >= END_STEPLENG;stepleng/=2.0) {
      eps = 0.1*stepleng;

      time(&start_t);
      /* Begining of a loop for i_batch */
      for(i_batch=evf_cnt=evr_cnt=evm_cnt=0;i_batch<BATCH_NUM;i_batch++) {
	for(i=ibase=0;i<traject;i++) {/* Initialization for yvec. */
	  for(j=0;j<ydim;j++) {
	    yvec[ibase+j] = yinit[j];
	  }
	  ibase+=ydim;
	}
	
	xpoint = 0.0; /* Initialization for xpoint. */

	/* xpoint keeps changing until it reaches XRANGE. */
	while(eps<fabs(XRANGE-xpoint)) {
	  /* A loop for i to generate random numbers for one step. */
#if (41 == SRK_TYPE)
	  if(0!=ran_gene_2p_Only_using_genrand_int32(traject, wdim, ran2p)) {
	    printf("Error in ran_gene_2p_Only_using_genrand_init32!");
	    exit(1);
	  }
#else
	  if(0!=ran_gene_full_using_genrand_int32(traject, wdim, ran2p,
						  ran3p)) {
	    printf("Error in ran_gene_full_using_genrand_init32!");
	    exit(1);
	  }
#endif
	  
	  if(0 == i_batch) {
	    evr_cnt+=(2*wdim)*traject;
	  }
	  
	  if (Cal_Start_Set <= i_set) {
	    if(0 == i_batch) {
#if (2 == SRK_TYPE) || (3 == SRK_TYPE)
#   ifdef CMP
	      if(0 != SRKcntmat(ydim, traject, yvec, stepleng, ran2p, ran3p,
				A_mat, ffunc, gfunc_gene, wdim, work, work_A,
				ynew, &evf_cnt, &evm_cnt)) {
#   else
	      if(0 != SRKcnt(ydim, traject, yvec, stepleng, ran2p, ran3p,
			     A_mat, ffunc, gfunc_gene, wdim, work, work_A,
			     ynew, &evf_cnt)) {
#   endif
#elif (1 == SRK_TYPE)
	      if(0 != SRKcnt(ydim, traject, yvec, stepleng, ran2p, ran3p,
			     ffunc, gfunc_gene, wdim, stagenum, work, ynew, &evf_cnt)) {
#elif (20 == SRK_TYPE)
	      if(0 != SRKcnt(ydim, traject, yvec, stepleng, ran2p, ran3p,
	      ffunc, gfunc_diag, stagenum, work, ynew, &evf_cnt)) {
#elif (21 == SRK_TYPE) || (22 == SRK_TYPE)
#   ifdef CMP
	      if(0 != SRKcntmat(ydim, traject, yvec, stepleng,
	                        ran2p, ran3p,
				ffunc, gfunc_diag, mMax, anorm, lwsp,
			        lwspA, liwsp, itrace, errChkFlag, tol, work,
			        work_K, iWork_K, workA_K,
			        ynew, &evf_cnt, &evm_cnt)) {
#   else
	      if(0 != SRKcnt(ydim, traject, yvec, stepleng,
			     ran2p, ran3p,
			     ffunc, gfunc_diag, mMax, anorm, lwsp,
			     lwspA, liwsp, itrace, errChkFlag, tol, work,
			     work_K, iWork_K, workA_K,
			     ynew, &evf_cnt)) {
#   endif
#elif (23 == SRK_TYPE)
#   ifdef CMP
	      if(0 != SRKcntmat(ydim, traject, yvec, stepleng,
	                        ran2p, ran3p,
				ffunc, gfunc_diag, mMin, lwsp, work,
			        work_T, workAux,
			        ynew, &evf_cnt, &evm_cnt)) {
#   else
	      if(0 != SRKcnt(ydim, traject, yvec, stepleng,
	                        ran2p, ran3p,
				ffunc, gfunc_diag, mMin, lwsp, work,
			        work_T, workAux,
			        ynew, &evf_cnt)) {
#   endif
#elif (41 == SRK_TYPE)
	      if(0 != SRKcnt(ydim, traject, yvec, stepleng, ran2p,
			     ffunc, gfunc_diag, stagenum, work,
			     ynew, &evf_cnt)) {
#else
	      if(0 != SRKcnt(ydim, traject, yvec, stepleng, ran2p,
			     ffunc, gfunc_gene, wdim, work, ynew, &evf_cnt)) {
#endif
		printf("error in SRKcnt\n");
		return 1;
	      }
	    } else {
#if (2 == SRK_TYPE) || (3 == SRK_TYPE)
	      if(0 != SRK(ydim, traject, yvec, stepleng, ran2p, ran3p,
			  A_mat, ffunc, gfunc_gene, wdim, work, work_A,
			  ynew)) {
#elif (1 == SRK_TYPE)
	     if(0 != SRK(ydim, traject, yvec, stepleng, ran2p, ran3p,
			  ffunc, gfunc_gene, wdim, stagenum, work, ynew)) {
#elif (20 == SRK_TYPE)
	     if(0 != SRK(ydim, traject, yvec, stepleng, ran2p, ran3p,
			  ffunc, gfunc_diag, stagenum, work, ynew)) {
#elif (21 == SRK_TYPE) || (22 == SRK_TYPE)
		if(0 != SRK(ydim, traject, yvec, stepleng, ran2p, ran3p,
			    ffunc, gfunc_diag, mMax, anorm, lwsp,
			    lwspA, liwsp, itrace, errChkFlag, tol, work,
			    work_K, iWork_K, workA_K,
			    ynew)) {
#elif (23 == SRK_TYPE)
		if(0 != SRK(ydim, traject, yvec, stepleng, ran2p, ran3p,
			    ffunc, gfunc_diag, mMin, lwsp, work,
			    work_T, workAux,
			    ynew)) {
#elif (41 == SRK_TYPE)
	      if(0 != SRK(ydim, traject, yvec, stepleng, ran2p,
			  ffunc, gfunc_diag, stagenum, work, ynew)) {
#else
		  /* nothing */
#endif
		printf("error in SRK\n");
		return 1;
	      }
	    }
	      
	    for(i=ibase=0;i<traject;i++) {/* Update on yvec. */
	      for(j=0;j<ydim;j++) {
		yvec[ibase+j] = ynew[ibase+j];
	      }
	      ibase+=ydim;
	    }

	  } /* End of the 1st if (Cal_Start_Set <= i_set) */
	  xpoint += stepleng;           /* Update on xpoint. */
	}

	/* Calculations fo expectation in i_batch. */
	for(i=0;i<ydim;i++)
	  expect[i] = 0.0;
	for(i=ibase=0;i<traject;i++) {
	  for(j=0;j<ydim;j++) {
	    expect[j] += yvec[ibase+j];
	  }
	  ibase+=ydim;
	}
	for(i=0;i<ydim;i++)
	  expect[i]/=traject;
#ifdef DISPER_DATA
	/* Calculations for 2nd moment and so on in i_batch. */
	for(i=0;i<ydim;i++)
	  fourthm[i] = 0.0; 
	for(i=ibase=0;i<traject;i++) {
	  for(j=0;j<ydim;j++) {
	    fourthm[j] += yvec[ibase+j]*yvec[ibase+j];
	  }
	  ibase+=ydim;
	}
	for(i=0;i<ydim;i++)
	  fourthm[i]/=traject;
#endif
	for(i=0;i<ydim;i++)
	  batch_expect[i_batch][i]=expect[i];
#ifdef DISPER_DATA
	for(i=0;i<ydim;i++) {
	  batch_fourthm[i_batch][i]=fourthm[i];
	}
#endif
      } /* End of the loop for i_batch. */
      time(&finish_t);
      elapsed_time = difftime(finish_t, start_t);
      
      /* Calculations for expectation. */
      for(i=0;i<ydim;i++) {
	expect[i] = 0.0;
	for(i_batch=0;i_batch<BATCH_NUM;i_batch++) {
	  expect[i] += batch_expect[i_batch][i];
	}
	expect[i]/=BATCH_NUM;
      }
      
#ifdef DISPER_DATA
      /* Calculations for 2nd moment or others. */
      for(i=0;i<ydim;i++) {
	fourthm[i] = 0.0;
	for(i_batch=0;i_batch<BATCH_NUM;i_batch++) {
	  fourthm[i] += batch_fourthm[i_batch][i];
	}
	fourthm[i]/=BATCH_NUM;
      }
#endif

      for(i=0;i<fdim;i++) {
	if(0 != fopen_s(&exfp[i],exfname[i],"a"))
	  {printf("Can not open %s file\n",exfname[i]);return 1;}
#   if (13 == TESTFUNC)
	if (0==i) {
	  jbaseInit=0; jbaseEnd=YDIM_2;
	} else {
	  jbaseInit=YDIM_2; jbaseEnd=YDIM;
	}
	for (jbase = jbaseInit; jbase<jbaseEnd; jbase+=NN) {
	  for(j=jbase+0; j<jbase+NN; j++) {
	    fprintf(exfp[i],"%16.15le\t", expect[j]);
	  }
	  fprintf(exfp[i],"\n");
	}
#   else
	fprintf(exfp[i],"%lf\t%16.15le\t%lf\t%lf\t%lf\n",log(stepleng)/log(2.0),
		expect[i],
		((double)evf_cnt)*BATCH_NUM,((double)evr_cnt)*BATCH_NUM,
		((double)evm_cnt)*BATCH_NUM);
#   endif
	fclose(exfp[i]);
      }
#ifdef DISPER_DATA
      for(i=0;i<fdim;i++)	{
	if(0 != fopen_s(&mom4fp[i],mom4fname[i],"a"))
	  {printf("Can not open %s file\n",mom4fname[i]);return 1;}
#   if (13 == TESTFUNC)
	if (0==i) {
	  jbaseInit=0; jbaseEnd=YDIM_2;
	} else {
	  jbaseInit=YDIM_2; jbaseEnd=YDIM;
	}
	for (jbase = jbaseInit; jbase<jbaseEnd; jbase+=NN) {
	  for(j=jbase+0; j<jbase+NN; j++) {
	    fprintf(mom4fp[i],"%16.15le\t", fourthm[j]);
	  }
	  fprintf(mom4fp[i],"\n");
	}
#   else
	fprintf(mom4fp[i],"%lf\t%16.15le\t%lf\t%lf\t%lf\n",log(stepleng)/log(2.0),
		fourthm[i],((double)evf_cnt)*BATCH_NUM,
		((double)evr_cnt)*BATCH_NUM, ((double)evm_cnt)*BATCH_NUM);
#   endif
	fclose(mom4fp[i]);
      }
#endif

      if(0 != fopen_s(&timefp,timefname,"a"))
	{printf("Can not open %s file\n",timefname);return 1;}
      fprintf(timefp,"{%lf,%lf},",log(stepleng)/log(2.0),
	      elapsed_time);
#if (22 == SRK_TYPE) || (23 == SRK_TYPE)
      if(0 != fopen_s(&costfp,costfname,"a"))
	{printf("Can not open %s file\n",costfname);return 1;}
      if (1 == BATCH_NUM) {
	fprintf(costfp,"{%lf, (evf_cnt) %lf, (evm_cnt) %lf},",
		log(stepleng)/log(2.0),
		((double)evf_cnt),
		((double)evm_cnt));
      } else {
	fprintf(costfp,"{%lf, (evf_cnt) %lf},",
		log(stepleng)/log(2.0),
		((double)evf_cnt)*BATCH_NUM);
      }
#endif
      fclose(timefp);
      fclose(costfp);

    } /* End of the loop for stepleng. */
    _chdir("..");
  } /* End of the loop for i_set */
  return 0;
}

#if (11 == TESTFUNC)
#   if (2 == SRK_TYPE) || (3 == SRK_TYPE)
static void ffunc(double ynvec[],double foutput[])
{
  foutput[0] =-pow(ynvec[0]+ynvec[1],5.0)/5.0*ynvec[1];
  foutput[1] = pow(ynvec[0]+ynvec[1],5.0)/5.0*ynvec[0];
}

static void makeMatA(int dim, double A_mat[]) {
   int ii, ll;
   
   ii=0;
   ll=ii*dim;
   A_mat[ll+0]=0;
   A_mat[ll+1]=-OM0;
   
   ii=1;
   ll=ii*dim;
   A_mat[ll+0]=OM0;
   A_mat[ll+1]=OM0*BT0;
}
#   else
static void ffunc(double ynvec[],double foutput[])
{
  foutput[0] =-OM0*ynvec[1]-pow(ynvec[0]+ynvec[1],5.0)/5.0*ynvec[1];
  foutput[1] = OM0*(ynvec[0]+BT0*ynvec[1])
    +pow(ynvec[0]+ynvec[1],5.0)/5.0*ynvec[0];
}
#   endif

static void gfunc_gene(double ynvec[],int i_th,double goutput[])
{
  switch(i_th) {
  case 0:
    goutput[0] =-OM1*ynvec[1];
    goutput[1] = OM1*(ynvec[0]+BT1*ynvec[1]);
    break;
  case 1:
    goutput[0] =-OM2*ynvec[1]-pow(ynvec[0]+ynvec[1],3.0)/3.0*ynvec[1];;
    goutput[1] = OM2*(ynvec[0]+BT2*ynvec[1])
      +pow(ynvec[0]+ynvec[1],3.0)/3.0*ynvec[0];
    break;
  default:
    printf("Error_in_gfunc_gene: invalid diffusion coefficient was selected!\n");
    exit(0);
  }
}
#endif /* (11 == TESTFUNC) */

 #if (12 == TESTFUNC)
#   if (2 == SRK_TYPE) || (3 == SRK_TYPE)
static void ffunc(double ynvec[],double foutput[])
{
  foutput[0] = 0;
  foutput[1] = 0;
}

static void makeMatA(int dim, double A_mat[]) {
   int ii, ll;
   
   ii=0;
   ll=ii*dim;
   A_mat[ll+0]=A11;
   A_mat[ll+1]=A12;
   
   ii=1;
   ll=ii*dim;
   A_mat[ll+0]=A21;
   A_mat[ll+1]=A22;
}
#   else
static void ffunc(double ynvec[],double foutput[])
{
  foutput[0] = A11*ynvec[0]+A12*ynvec[1];
  foutput[1] = A21*ynvec[0]+A22*ynvec[1];
}
#   endif

static void gfunc_gene(double ynvec[],int i_th,double goutput[])
{
  switch(i_th) {
  case 0:
    goutput[0] = B11*ynvec[0]+B12*ynvec[1];
    goutput[1] = B21*ynvec[0]+B22*ynvec[1];
    break;
  default:
    printf("Error_in_gfunc_gene: invalid diffusion coefficient was selected!\n");
    exit(0);
  }
}
#endif /* (12 == TESTFUNC) */

#if (13 == TESTFUNC)
/* For this TESTFUNC, 2==SRK_TYPE is not used. */
#   if (21 == SRK_TYPE) || (22 == SRK_TYPE) || (23 == SRK_TYPE)
static void ffunc(double ynvec[],double foutput[])
{
  int ii, ii_base;
  double tmp;

  /* Note that the following are for nonlinear parts only */
  /***/
  ii_base=0;
  ii=ii_base+0;
  foutput[ii] =COEF2*(ynvec[ii]*ynvec[ii+1]+ynvec[ii+YDIM_2]*ynvec[ii+NN]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] =COEF2*(ynvec[ii]*(ynvec[ii+1]-ynvec[ii-1])
			+ynvec[ii+YDIM_2]*ynvec[ii+NN]);
  }
  ii=ii_base+NN-1;
  foutput[ii] =COEF2*(ynvec[ii]*(-ynvec[ii-1])
		      +ynvec[ii+YDIM_2]*ynvec[ii+NN]);
  for (ii_base=NN; ii_base<(NN-1)*NN; ii_base+=NN) {
    ii=ii_base+0;
    foutput[ii] =COEF2*(ynvec[ii]*ynvec[ii+1]
			+ynvec[ii+YDIM_2]*(ynvec[ii+NN]-ynvec[ii-NN]));
    for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
      foutput[ii] =COEF2*(ynvec[ii]*(ynvec[ii+1]-ynvec[ii-1])
			  +ynvec[ii+YDIM_2]*(ynvec[ii+NN]-ynvec[ii-NN]));
    }
    ii=ii_base+NN-1;
    foutput[ii] =COEF2*(ynvec[ii]*(-ynvec[ii-1])
			+ynvec[ii+YDIM_2]*(ynvec[ii+NN]-ynvec[ii-NN]));
  }
  ii_base=(NN-1)*NN;
  ii=ii_base+0;
  foutput[ii] =COEF2*(ynvec[ii]*ynvec[ii+1]
		      +ynvec[ii+YDIM_2]*(-ynvec[ii-NN]));
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] =COEF2*(ynvec[ii]*(ynvec[ii+1]-ynvec[ii-1])
			+ynvec[ii+YDIM_2]*(-ynvec[ii-NN]));
  }
  ii=ii_base+NN-1;
  foutput[ii] =COEF2*(ynvec[ii]*(-ynvec[ii-1])
		      +ynvec[ii+YDIM_2]*(-ynvec[ii-NN]));
  /***/
  ii_base=YDIM_2;
  ii=ii_base+0;
  foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*ynvec[ii+1]
		      +ynvec[ii]*ynvec[ii+NN]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*(ynvec[ii+1]-ynvec[ii-1])
			+ynvec[ii]*ynvec[ii+NN]);
  }
  ii=ii_base+NN-1;
  foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*(-ynvec[ii-1])
		      +ynvec[ii]*ynvec[ii+NN]);
  for (ii_base=YDIM_2+NN; ii_base<YDIM_2+(NN-1)*NN; ii_base+=NN) {
    ii=ii_base+0;
    foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*ynvec[ii+1]
			+ynvec[ii]*(ynvec[ii+NN]-ynvec[ii-NN]));
    for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
      foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*(ynvec[ii+1]-ynvec[ii-1])
			  +ynvec[ii]*(ynvec[ii+NN]-ynvec[ii-NN]));
    }
    ii=ii_base+NN-1;
    foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*(-ynvec[ii-1])
			+ynvec[ii]*(ynvec[ii+NN]-ynvec[ii-NN]));
  }
  ii_base=YDIM_2+(NN-1)*NN;
  ii=ii_base+0;
  foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*ynvec[ii+1]
		      +ynvec[ii]*(-ynvec[ii-NN]));
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*(ynvec[ii+1]-ynvec[ii-1])
			+ynvec[ii]*(-ynvec[ii-NN]));
  }
  ii=ii_base+NN-1;
  foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*(-ynvec[ii-1])
		      +ynvec[ii]*(-ynvec[ii-NN]));
}

static void setRMAT(int dim) {
  int i, j, inz, iBase;
  double diagA;

  /* The following were mainly copied from
     sample6_main_for_exp_sym.c (Ver. 0)
     in the folder
     "Example_2019_for_Matrix_exponential\Sample_for_C_and_Fort".
  */
  
  if (1>=dim) {
    printf("Error: ydim is too small in makeMatA!\n");
    exit(0);
  }
  
  RMAT.n=dim;
  diagA=-4.0*COEF1;

  inz=0;
  /*** for y ***/
  /* The 1st n lines */
  iBase=0;
  i=iBase;
  j=i;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=diagA;
  inz++;
  j=i+1;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;
  j=i+NN;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;

  for (i=iBase+1;i<iBase+NN-1;i++) {
    j=i-1;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=diagA;
    inz++;
    j=i+1;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i+NN;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
  }
  
  i=iBase+NN-1;
  j=i-1;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;
  j=i;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=diagA;
  inz++;
  j=i+NN;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;

  /* From the 2nd n lines upto the (n-1)st n lines */
  for (iBase=NN;iBase<(NN-1)*NN;iBase=iBase+NN) {
    i=iBase;
    j=i-NN;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=diagA;
    inz++;
    j=i+1;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i+NN;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    
    for (i=iBase+1;i<iBase+NN-1;i++) {
      j=i-NN;
      RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
      RMAT.a[inz]=COEF1;
      inz++;
      j=i-1;
      RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
      RMAT.a[inz]=COEF1;
      inz++;
      j=i;
      RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
      RMAT.a[inz]=diagA;
      inz++;
      j=i+1;
      RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
      RMAT.a[inz]=COEF1;
      inz++;
      j=i+NN;
      RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
      RMAT.a[inz]=COEF1;
      inz++;
    }
    
    i=iBase+NN-1;
    j=i-NN;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i-1;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=diagA;
    inz++;
    j=i+NN;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
  }

  /* The nth n lines */
  iBase=(NN-1)*NN;
  i=iBase;
  j=i-NN;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;
  j=i;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=diagA;
  inz++;
  j=i+1;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;
  
  for (i=iBase+1;i<iBase+NN-1;i++) {
    j=i-NN;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i-1;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=diagA;
    inz++;
    j=i+1;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
  }
  
  i=iBase+NN-1;
  j=i-NN;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;
  j=i-1;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;
  j=i;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=diagA;
  inz++;

  /*** for z ***/
  /* The (n+1)th n lines */
  iBase=YDIM_2;
  i=iBase;
  j=i;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=diagA;
  inz++;
  j=i+1;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;
  j=i+NN;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;

  for (i=iBase+1;i<iBase+NN-1;i++) {
    j=i-1;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=diagA;
    inz++;
    j=i+1;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i+NN;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
  }
  
  i=iBase+NN-1;
  j=i-1;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;
  j=i;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=diagA;
  inz++;
  j=i+NN;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;

  /* From the (n+2)th n lines upto the (n+n-1)st n lines */
  for (iBase=YDIM_2+NN;iBase<YDIM_2+(NN-1)*NN;iBase=iBase+NN) {
    i=iBase;
    j=i-NN;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=diagA;
    inz++;
    j=i+1;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i+NN;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    
    for (i=iBase+1;i<iBase+NN-1;i++) {
      j=i-NN;
      RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
      RMAT.a[inz]=COEF1;
      inz++;
      j=i-1;
      RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
      RMAT.a[inz]=COEF1;
      inz++;
      j=i;
      RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
      RMAT.a[inz]=diagA;
      inz++;
      j=i+1;
      RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
      RMAT.a[inz]=COEF1;
      inz++;
      j=i+NN;
      RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
      RMAT.a[inz]=COEF1;
      inz++;
    }
    
    i=iBase+NN-1;
    j=i-NN;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i-1;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=diagA;
    inz++;
    j=i+NN;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
  }

  /* The (n+n)th n lines */
  iBase=YDIM_2+(NN-1)*NN;
  i=iBase;
  j=i-NN;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;
  j=i;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=diagA;
  inz++;
  j=i+1;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;
  
  for (i=iBase+1;i<iBase+NN-1;i++) {
    j=i-NN;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i-1;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
    j=i;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=diagA;
    inz++;
    j=i+1;
    RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
    RMAT.a[inz]=COEF1;
    inz++;
  }
  
  i=iBase+NN-1;
  j=i-NN;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;
  j=i-1;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=COEF1;
  inz++;
  j=i;
  RMAT.ia[inz]=i+1; RMAT.ja[inz]=j+1;
  RMAT.a[inz]=diagA;
  inz++;

  RMAT.nz=inz;

}

#   else
static void ffunc(double ynvec[],double foutput[])
{
  int ii, ii_base;
  double tmp;

  ii_base=0;
  ii=ii_base+0;
  foutput[ii] = COEF1*(-4*ynvec[ii]+ynvec[ii+1]+ynvec[ii+NN]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] = COEF1*(ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+1]+ynvec[ii+NN]);
  }
  ii=ii_base+NN-1;
  foutput[ii] = COEF1*(ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+NN]);
  for (ii_base=NN; ii_base<(NN-1)*NN; ii_base+=NN) {
    ii=ii_base+0;
    foutput[ii] = COEF1*(ynvec[ii-NN]-4*ynvec[ii]+ynvec[ii+1]+ynvec[ii+NN]);
    for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
      foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+1]
			   +ynvec[ii+NN]);
    }
    ii=ii_base+NN-1;
    foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+NN]);
  }
  ii_base=(NN-1)*NN;
  ii=ii_base+0;
  foutput[ii] = COEF1*(ynvec[ii-NN]-4*ynvec[ii]+ynvec[ii+1]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+1]);
  }
  ii=ii_base+NN-1;
  foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]);
  /***/
  ii_base=YDIM_2;
  ii=ii_base+0;
  foutput[ii] = COEF1*(-4*ynvec[ii]+ynvec[ii+1]+ynvec[ii+NN]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] = COEF1*(ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+1]+ynvec[ii+NN]);
  }
  ii=ii_base+NN-1;
  foutput[ii] = COEF1*(ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+NN]);
  for (ii_base=YDIM_2+NN; ii_base<YDIM_2+(NN-1)*NN; ii_base+=NN) {
    ii=ii_base+0;
    foutput[ii] = COEF1*(ynvec[ii-NN]-4*ynvec[ii]+ynvec[ii+1]+ynvec[ii+NN]);
    for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
      foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+1]
			   +ynvec[ii+NN]);
    }
    ii=ii_base+NN-1;
    foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+NN]);
  }
  ii_base=YDIM_2+(NN-1)*NN;
  ii=ii_base+0;
  foutput[ii] = COEF1*(ynvec[ii-NN]-4*ynvec[ii]+ynvec[ii+1]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+1]);
  }
  ii=ii_base+NN-1;
  foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]);
  /***/
  ii_base=0;
  ii=ii_base+0;
  foutput[ii] +=COEF2*(ynvec[ii]*ynvec[ii+1]+ynvec[ii+YDIM_2]*ynvec[ii+NN]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] +=COEF2*(ynvec[ii]*(ynvec[ii+1]-ynvec[ii-1])
			 +ynvec[ii+YDIM_2]*ynvec[ii+NN]);
  }
  ii=ii_base+NN-1;
  foutput[ii] +=COEF2*(ynvec[ii]*(-ynvec[ii-1])
		       +ynvec[ii+YDIM_2]*ynvec[ii+NN]);
  for (ii_base=NN; ii_base<(NN-1)*NN; ii_base+=NN) {
    ii=ii_base+0;
    foutput[ii] +=COEF2*(ynvec[ii]*ynvec[ii+1]
			 +ynvec[ii+YDIM_2]*(ynvec[ii+NN]-ynvec[ii-NN]));
    for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
      foutput[ii] +=COEF2*(ynvec[ii]*(ynvec[ii+1]-ynvec[ii-1])
			   +ynvec[ii+YDIM_2]*(ynvec[ii+NN]-ynvec[ii-NN]));
    }
    ii=ii_base+NN-1;
    foutput[ii] +=COEF2*(ynvec[ii]*(-ynvec[ii-1])
			 +ynvec[ii+YDIM_2]*(ynvec[ii+NN]-ynvec[ii-NN]));
  }
  ii_base=(NN-1)*NN;
  ii=ii_base+0;
  foutput[ii] +=COEF2*(ynvec[ii]*ynvec[ii+1]
		       +ynvec[ii+YDIM_2]*(-ynvec[ii-NN]));
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] +=COEF2*(ynvec[ii]*(ynvec[ii+1]-ynvec[ii-1])
			 +ynvec[ii+YDIM_2]*(-ynvec[ii-NN]));
  }
  ii=ii_base+NN-1;
  foutput[ii] +=COEF2*(ynvec[ii]*(-ynvec[ii-1])
		       +ynvec[ii+YDIM_2]*(-ynvec[ii-NN]));
  /***/
  ii_base=YDIM_2;
  ii=ii_base+0;
  foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*ynvec[ii+1]
		       +ynvec[ii]*ynvec[ii+NN]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*(ynvec[ii+1]-ynvec[ii-1])
			 +ynvec[ii]*ynvec[ii+NN]);
  }
  ii=ii_base+NN-1;
  foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*(-ynvec[ii-1])
		       +ynvec[ii]*ynvec[ii+NN]);
  for (ii_base=YDIM_2+NN; ii_base<YDIM_2+(NN-1)*NN; ii_base+=NN) {
    ii=ii_base+0;
    foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*ynvec[ii+1]
			 +ynvec[ii]*(ynvec[ii+NN]-ynvec[ii-NN]));
    for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
      foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*(ynvec[ii+1]-ynvec[ii-1])
			   +ynvec[ii]*(ynvec[ii+NN]-ynvec[ii-NN]));
    }
    ii=ii_base+NN-1;
    foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*(-ynvec[ii-1])
			 +ynvec[ii]*(ynvec[ii+NN]-ynvec[ii-NN]));
  }
  ii_base=YDIM_2+(NN-1)*NN;
  ii=ii_base+0;
  foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*ynvec[ii+1]
		       +ynvec[ii]*(-ynvec[ii-NN]));
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*(ynvec[ii+1]-ynvec[ii-1])
			 +ynvec[ii]*(-ynvec[ii-NN]));
  }
  ii=ii_base+NN-1;
  foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*(-ynvec[ii-1])
		       +ynvec[ii]*(-ynvec[ii-NN]));
}
#   endif

static void gfunc_diag(double ynvec[],double goutput[])
{
  int ii, ii_base;
  double tmp1, tmp2;

  tmp1=BETA1*(NN+1);
  tmp2=BETA2*(NN+1);
  
  for (ii_base=0; ii_base<YDIM_2; ii_base+=NN) {
    for (ii=ii_base; ii<ii_base+NN; ii++) {
      goutput[ii] =tmp1;
    }
  }
  for (ii_base=YDIM_2; ii_base<YDIM; ii_base+=NN) {
    for (ii=ii_base; ii<ii_base+NN; ii++) {
      goutput[ii] =tmp2;
    }
  }
}
#endif /* (13 == TESTFUNC) */

#if (23 == SRK_TYPE)
static void setPVec() {
  int i;
  double tmpArray[]={
    1.000000000000000e0, 1.000000000000000e0,
    2.000000000000000e0, 6.000000000000000e0,
    24.00000000000000e0, 120.0000000000000e0,
    720.0000000000000e0, 5040.000000000000e0,
    40320.00000000000e0, 362880.0000000000e0,
    3.628800000000000e6, 3.991680000000000e7,
    4.790016000000000e8, 6.227020800000000e9,
    8.717829120000000e10, 1.307674368000000e12,
    2.092278988800000e13, 3.556874280960000e14,
    6.402373705728000e15, 1.216451004088320e17,
    2.432902008176640e18, 5.109094217170944e19,
    1.124000727777608e21, 2.585201673888498e22,
    6.204484017332394e23, 1.551121004333099e25,
    4.032914611266056e26, 1.088886945041835e28,
    3.048883446117139e29, 8.841761993739702e30,
    2.652528598121911e32, 8.222838654177923e33,
    2.631308369336935e35, 8.683317618811886e36,
    2.952327990396041e38, 1.033314796638614e40,
    3.719933267899012e41, 1.376375309122635e43,
    5.230226174666011e44, 2.039788208119744e46,
    8.159152832478977e47, 3.345252661316381e49,
    1.405006117752880e51, 6.041526306337384e52,
    2.658271574788449e54, 1.196222208654802e56,
    5.502622159812089e57, 2.586232415111682e59,
    1.241391559253607e61, 6.082818640342676e62,
    3.041409320171338e64, 1.551118753287382e66,
    8.065817517094388e67, 4.274883284060026e69,
    2.308436973392414e71, 1.269640335365828e73,
    7.109985878048635e74, 4.052691950487722e76,
    2.350561331282879e78, 1.386831185456898e80,
    8.320987112741390e81, 5.075802138772248e83,
    3.146997326038794e85, 1.982608315404440e87,
    1.268869321858842e89, 8.247650592082471e90,
    5.443449390774431e92, 3.647111091818869e94,
    2.480035542436831e96, 1.711224524281413e98,
    1.197857166996989e100, 8.504785885678623e101,
    6.123445837688609e103, 4.470115461512684e105,
    3.307885441519386e107, 2.480914081139540e109,
    1.885494701666050e111, 1.451830920282859e113,
    1.132428117820630e115, 8.946182130782975e116,
    7.156945704626380e118, 5.797126020747368e120,
    4.753643337012842e122, 3.945523969720659e124,
    3.314240134565353e126, 2.817104114380550e128,
    2.422709538367273e130, 2.107757298379528e132,
    1.854826422573984e134, 1.650795516090846e136,
    1.485715964481761e138, 1.352001527678403e140,
    1.243841405464131e142, 1.156772507081642e144,
    1.087366156656743e146, 1.032997848823906e148,
    9.916779348709497e149, 9.619275968248212e151,
    9.426890448883248e153, 9.332621544394415e155,
    9.332621544394415e157, 9.425947759838359e159,
    9.614466715035127e161, 9.902900716486180e163,
    1.029901674514563e166, 1.081396758240291e168,
    1.146280563734708e170, 1.226520203196138e172,
    1.324641819451829e174, 1.443859583202494e176,
    1.588245541522743e178, 1.762952551090245e180,
    1.974506857221074e182, 2.231192748659814e184,
    2.543559733472188e186, 2.925093693493016e188,
    3.393108684451898e190, 3.969937160808721e192,
    4.684525849754291e194, 5.574585761207606e196,
    6.689502913449127e198, 8.094298525273444e200,
    9.875044200833601e202, 1.214630436702533e205,
    1.506141741511141e207, 1.882677176888926e209,
    2.372173242880047e211, 3.012660018457660e213,
    3.856204823625804e215, 4.974504222477287e217,
    6.466855489220474e219, 8.471580690878821e221,
    1.118248651196004e224, 1.487270706090686e226,
    1.992942746161519e228, 2.690472707318050e230,
    3.659042881952549e232, 5.012888748274992e234,
    6.917786472619488e236, 9.615723196941089e238,
    1.346201247571752e241, 1.898143759076171e243,
    2.695364137888163e245, 3.854370717180073e247,
    5.550293832739305e249, 8.047926057471992e251,
    1.174997204390911e254, 1.727245890454639e256,
    2.556323917872866e258, 3.808922637630570e260,
    5.713383956445855e262, 8.627209774233240e264,
    1.311335885683453e267, 2.006343905095682e269,
    3.089769613847351e271, 4.789142901463394e273,
    7.471062926282894e275, 1.172956879426414e278,
    1.853271869493735e280, 2.946702272495038e282,
    4.714723635992061e284, 7.590705053947219e286,
    1.229694218739449e289, 2.004401576545303e291,
    3.287218585534296e293, 5.423910666131589e295,
    9.003691705778437e297, 1.503616514864999e300,
    2.526075744973198e302, 4.269068009004705e304
  };

  for (i=0; i<170; i++) {
    PVEC.p[i] = tmpArray[i];
  }
}
#endif
