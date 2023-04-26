/* GetSROCK2Val_from_rectp.c */
/*****/
/* Ver. 0 (23-Jul-2015) */

#include <math.h>

extern void COPY_AB_VALUES(int cp_ms[], double cp_fp1[], double cp_fp2[],
			   double cp_recalph[], double cp_recf[],
			   double cp_recf2[]);

int GetSROCK2Val_from_recp(int ss, double Mu[], double Ka[], double *sig,
			   double *tau, double *alpha) {
  /*
    3<=ss<=200: parameter values will be set for damping 0.95.
    Main stages are ss=3, 4, ..., 22, 24, 26, 28, 30, 32, 35, 38,
    41, 45, 49, 53, 58, 63, 68, 74, 80, 87, 95, 104, 114, 125,
    137, 150, 165, 182 and 200.
   */

  int sM2, cp_ms[46], ii, i_sM2, i_Mu, i_base;
  double cp_fp1[46], cp_fp2[46], cp_recalph[46], cp_recf[4476], cp_recf2[184];

  sM2=ss-2;

  COPY_AB_VALUES(cp_ms,cp_fp1,cp_fp2,cp_recalph,cp_recf,cp_recf2);

  /* Find the indexes for sM2 and Mu[0]. */
  i_Mu=0;
  for (ii=0;ii<46;ii++) {
    if (sM2==cp_ms[ii]) {
      i_sM2=ii;
      break;
    }
    i_Mu+=cp_ms[ii]*2-1;
  }

  if(46<=ii) {
    return 1;
  }

  Mu[0]=cp_recf[i_Mu];
  i_base=i_Mu;
  for (ii=1;ii<sM2;ii++) {
    Mu[ii]=cp_recf[i_base+1];
    Ka[ii-1]=cp_recf[i_base+2];
    i_base+=2;
  }

  i_base=i_sM2*4;
  Mu[sM2]=cp_recf2[i_base];
  Ka[sM2-1]=cp_recf2[i_base+1];
  Mu[sM2+1]=cp_recf2[i_base+2];
  Ka[sM2]=cp_recf2[i_base+3];

  *sig=cp_fp1[i_sM2];
  *tau=cp_fp1[i_sM2]*(cp_fp1[i_sM2]+cp_fp2[i_sM2]);
  *alpha=cp_recalph[i_sM2];

  return 0;
}
