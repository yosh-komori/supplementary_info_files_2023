/******************************************************/
/* File name: For_Kry_sub_tech.h                      */
/* The following comments were added to put files on GitHub
   (26-Apr-2023): */
/******************************************************/

/* The following are for Krylov subspace techniques */
/* NMAX has to be greater than or equal to m+4 for dgphi2v. */
#  define NMAX   33000L
   /* Fortran does not have unsigned integer constants. */
   /* NZMAX has to be set as the same number as that in dgcoov.f for RMAT.
      It has to be greater than or equal to 6+1=7 for PADE. */
#  define NZMAX  600000L

#  define MMAX   100
#  define LWSP   (NMAX*(MMAX+4)+5*(MMAX+4)*(MMAX+4)+7)
