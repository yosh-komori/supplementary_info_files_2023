/******************************************************/
/* filename: RanGene_no_intel.c                       */
/* Ver. 0.1 (1-Sep-2023)
   1) ran_gene_2p_Only_using_genrand_int32 was added.
 */
/******************************************************/
/* Ver. 0                                             */
/* This file was made to put on GitHub (26-Apr-2023). */
/******************************************************/

#include <stdio.h>

#define RMAX    4294967291UL /* (2^32-4)-1 */
#define HALF    2147483645UL /* (2^32-4)/2-1 */
#define ONE6TH   715827881UL /* (2^32-4)/6-1 */
#define FIVE6TH 3579139409UL /* (2^32-4)5/6-1 */

unsigned long genrand_int32(void);

int ran_gene_2p_Only_using_genrand_int32(int traject, int wdim, char ran2p[]) {
  /* When an error occurs, this function will return a non-zero value. */
  
  unsigned long rand;
  int i, j, ibase;

  for (i=ibase=0;i<traject;i++) {
    for (j=0; j<wdim; j++) {
      rand = genrand_int32();
      while (RMAX<rand) {
	rand = genrand_int32();
      }
      if (HALF>=rand) {
	ran2p[ibase+j]=-1;
      } else {
	ran2p[ibase+j]=1;
      }
    }
    ibase+=wdim;
  } /* End of the loop for i. */

  return 0;
}

int ran_gene_full_using_genrand_int32(int traject, int wdim,
				      char ran2p[], char ran3p[]) {
  /* When an error occurs, this function will return a non-zero value. */
  
  unsigned long rand;
  int i, j, ibase, ibase2p, j2p;

  for (i=ibase=ibase2p=0;i<traject;i++) {
    for (j2p=0; j2p<wdim; j2p++) {
      rand = genrand_int32();
      while (RMAX<rand) {
	rand = genrand_int32();
      }
      if (HALF>=rand) {
	ran2p[ibase2p+j2p]=-1;
      } else {
	ran2p[ibase2p+j2p]=1;
      }
    }
    ibase2p+=wdim;
    for (j=0; j<wdim; j++) {
      rand = genrand_int32();
      while (RMAX<rand) {
	rand = genrand_int32();
      }
      if (ONE6TH>=rand) {
	ran3p[ibase+j]=-1;
      } else if (FIVE6TH>=rand) {
	ran3p[ibase+j]=0;
      } else {
	ran3p[ibase+j]=1;
      }
    }
    ibase+=wdim;
  } /* End of the loop for i. */

  return 0;
}
