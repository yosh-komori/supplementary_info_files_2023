* File name: dsexpmvtay2.f
*   The following comments were added to put files on GitHub
*   (26-Apr-2023):
*   Copied from dsexpv.f in Expokit, and changed for dsexpmvtay2.
******************************
*----------------------------------------------------------------------|
      subroutine DSEXPMVTRAY2( n, mMin, t, v, w, wsp, lwsp, w_aux,
     .     m, npmv, matvec, iflag )

      implicit none
      integer n, mMin, lwsp, m, npmv, iflag
      double precision t, v(n), w(n), wsp(lwsp), w_aux(2*n)
      external matvec

*-----Purpose----------------------------------------------------------|
*
*---  DSEXPMVTRAY2 computes w = exp(t*A)*v - for a Symmetric matrix A.
*
*     It does not compute the matrix exponential in isolation but
*     instead, it computes directly the action of the exponential
*     operator on the operand vector.
*
*     The method used is based on [Ibanez:2022] and the matrix under
*     onsideration interacts only via the external routine `matvec'
*     performing the matrix-vector product (matrix-free method).
*
*-----Arguments--------------------------------------------------------|
*
*     n      : (input) order of the principal matrix A.
*                      
*     mMin   : (input) minimum order of polynomial.
*
*     t      : (input) time at wich the solution is needed.
*
*     v(n)   : (input) given operand vector.
*                      
*     w(n)   : (output) computed approximation of exp(t*A)*v.
*
*   wsp(lwsp): (workspace) lwsp .ge. n*(m+1)
*
*   w_aux(2*n) : (workspace)
*
*     m   : (output) order of polynomial for approximation.
*
*     npmv   : (output) number of matrix-vector products
*
*     matvec : external subroutine for matrix-vector multiplication.
*              synopsis: matvec( x, y )
*                        double precision x(*), y(*)
*              computes: y(1:n) <- A*x(1:n)
*                        where A is the principal matrix.
*     iflag  : (output) exit flag.
*               0 - no problem
*               1 - m reached mmax
*               2 - s reached smax
*     
*-----Accounts on the computation--------------------------------------|
*     Upon exit, an interested user may retrieve accounts on the 
*     computations. They are located in wsp as indicated below:
*
*     location  mnemonic                 description
*     -----------------------------------------------------------------|
*     wsp(1)  = s, scaling parameter.
*
*----------------------------------------------------------------------|
*-----The following parameters may also be adjusted herein-------------|
*
      integer smax, mmax
*      parameter( smax   = 45, mmax = 60)
*     The following line is the maximum for mmax.
*      parameter( smax   = 120, mmax = 168)
      parameter( smax   = 120, mmax = 100)
*      double precision p(72)
      double precision p(170)
      common /PVEC/ p

*     smax  : maximum allowable number of scaling parameter.
* 
*     mmax: maximum allowable number of the order of polynomial.
*
      integer np, i, j, k, j1v, f, np1
      double precision tol, p1, p2, p3, eps, vnorm, s, s1, tmp

      intrinsic ABS, CEILING

*      print *,'p(170)=',p(170)

*---  initialisations ...
*
      p1 = 4.0d0/3.0d0
 1    p2 = p1 - 1.0d0
      p3 = p2 + p2 + p2
      eps = ABS( p3-1.0d0 )
      if ( eps.eq.0.0d0 ) go to 1
      tol = eps/2
      m = mMin
      iflag = 0

      call DCOPY( n, v,1, w,1 )
      call matvec( w(1), wsp(1) )
      npmv = 1
      do i = 1,n
         wsp(i) = t*wsp(i)
      enddo
      
      j1v = 1
      do 100 j = 2,m+1
         j1v = j1v + n
         call matvec( wsp(j1v-n), wsp(j1v) )
         npmv = npmv + 1
         do i = 1,n
            wsp(j1v+i-1) = t*wsp(j1v+i-1)
         enddo
 100  continue
      
      vnorm = 0
      do i = 1,n
         vnorm = vnorm + ABS( wsp(j1v+i-1) )
      enddo
      s = CEILING((vnorm/(p(m+2)*tol))**(1.0d0/(m+1)))
      np = m*s
      f = 0
      do while ( (f.eq.0).and.(m.lt.mmax) )
         m = m + 1
         j1v = j1v + n
         call matvec( wsp(j1v-n), wsp(j1v) )
         npmv = npmv + 1
         do i = 1,n
            wsp(j1v+i-1) = t*wsp(j1v+i-1)
         enddo

         vnorm = 0
         do i = 1,n
            vnorm = vnorm + ABS( wsp(j1v+i-1) )
         enddo
         s1 = CEILING((vnorm/(p(m+2)*tol))**(1.0d0/(m+1)))
         np1 = m*s1
         if ( np1.le.np ) then
            np = np1
            s = s1
         else
            f = 1
            m = m - 1
         endif
      end do
      if ( mmax.le.m ) then
         iflag = 1
         return
      endif
      if ( smax.le.s ) then
         wsp(1)  = s
         iflag = 2
         return
      endif
      np = m*s

      do i = 1,n
         w(i) = v(i)/p(1)
      enddo
      
      j1v = 1
      do 200 j = 1,m
         tmp = p(j+1)*s**j
         do i = 1,n
            w(i) = w(i) + wsp(j1v+i-1)/tmp
         enddo
         j1v = j1v + n
 200  continue

      tmp = t/s
      do 400 i = 2,s
         do k = 1,n
            w_aux(k) = w(k)
            w(k) = w_aux(k)/p(1)
         enddo
         do 300 j = 1,m
            call matvec( w_aux(1), w_aux(n+1) )
            npmv = npmv + 1
            do k = 1,n
               w_aux(k) = tmp*w_aux(n+k)
               w(k) = w(k) + w_aux(k)/p(j+1)
            enddo
 300     continue
         
 400  continue

      wsp(1)  = s
      END
*----------------------------------------------------------------------|
