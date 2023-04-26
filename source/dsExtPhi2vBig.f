* File name: dsExtPhi2vBig.f
*   The following comments were added to put files on GitHub
*   (26-Apr-2023):
*   Copied from dgphi2v.f in Expokit, and revised for an extended
*   version of DGPHI2V to calculate t^{2}*phi2(tA)u at t and t/2.
******************************
*----------------------------------------------------------------------|
      subroutine DSEXTPHI2VBIG( n, m, t, u, w, wht, tol, anorm,
     .     wsp, lwsp, wspA, lwspA, w_aux, iwsp, liwsp, 
     .     matvec, itrace, iflag, errChkFlag ) 

      implicit none
      integer n, m, lwsp, liwsp, itrace, iflag, iwsp(liwsp), lwspA,
     .        errChkFlag
      double precision t, tol, anorm, u(n), w(n), wht(n), 
     .                 wsp(lwsp), wspA(lwspA), w_aux(n)
      external matvec
*-----Purpose----------------------------------------------------------|
*
*---  DGPHI1V computes w = t*t*phi2(tA)u, where
*     phi2(z) = (exp(z)-1-z)/z^{2} and A is a Symmetric matrix.
*
*     The method used is based on Krylov subspace projection
*     techniques and the matrix under consideration interacts only
*     via the external routine `matvec' performing the matrix-vector 
*     product (matrix-free method).
*
*-----Arguments--------------------------------------------------------|
*
*     n      : (input) order of the principal matrix A.
*                      
*     m      : (input) maximum size for the Krylov basis.
*                      
*     t      : (input) time at wich the solution is needed (can be < 0).
*   
*     u(n)   : (input) operand vector with respect to the phi function
*              (forcing term of the ODE problem).
*
*     w(n)   : (output) computed approximation of t*t*phi2(tA)u 
* 
*     wht(n) : (output) computed approximation of (t/2)*(t/2)*phi2(tA/2)u 
* 
*     tol    : (input/output) the requested accuracy tolerance on w. 
*              If on input tol=0.0d0 or tol is too small (tol.le.eps)
*              the internal value sqrt(eps) is used, and tol is set to
*              sqrt(eps) on output (`eps' denotes the machine epsilon).
*              (`Happy breakdown' is assumed if h(j+1,j) .le. anorm*tol)
*
*     anorm  : (input) an approximation of some norm of A.
*
*     wsp(lwsp): (workspace) lwsp .ge. n*(m+1)+n+(m+4)^2+4*(m+4)^2+ideg+1
*                                     +---------+-------+---------------+
*                (actually, ideg=6)        V        H      wsp for PADE
*          
*     wspA(lwspA): (workspace) lwspA .ge. n*(m+1)+n+(m+3)^2+4*(m+3)^2+ideg+1
*                                        +---------+-------+---------------+
*                  (actually, ideg=6)           VA       HA    wspA for PADE
*                   
*     iwsp(liwsp): (workspace) liwsp .ge. m+4
*
*     matvec : external subroutine for matrix-vector multiplication.
*              synopsis: matvec( x, y )
*                        double precision x(*), y(*)
*              computes: y(1:n) <- A*x(1:n)
*                        where A is the principal matrix.
*
*     itrace : (input) running mode. 0=silent, 1=print step-by-step info
*
*     iflag  : (output) exit flag.
*              <0 - bad input arguments 
*               0 - no problem
*               1 - maximum number of steps reached without convergence
*               2 - requested tolerance was too high
*
*     errChkFlag: (input) err estimation flag.
*               0 no err estimation for Phi1W
*               1 err estimation will be done for Phi1 to change t_step
*      
*-----Accounts on the computation--------------------------------------|
*     Upon exit, an interested user may retrieve accounts on the 
*     computations. They are located in the workspace arrays wsp and 
*     iwsp as indicated below: 
*
*     location  mnemonic                 description
*     -----------------------------------------------------------------|
*     iwsp(1) = nmult, number of matrix-vector multiplications used
*     iwsp(2) = nexph, number of Hessenberg matrix exponential evaluated
*     iwsp(3) = nscale, number of repeated squaring involved in Pade
*     iwsp(4) = nstep, number of integration steps used up to completion 
*     iwsp(5) = nreject, number of rejected step-sizes
*     iwsp(6) = ibrkflag, set to 1 if `happy breakdown' and 0 otherwise
*     iwsp(7) = mbrkdwn, if `happy brkdown', basis-size when it occured
*     -----------------------------------------------------------------|
*     wsp(1)  = step_min, minimum step-size used during integration
*     wsp(2)  = step_max, maximum step-size used during integration
*     wsp(3)  = dummy
*     wsp(4)  = dummy
*     wsp(5)  = x_error, maximum among all local truncation errors
*     wsp(6)  = s_error, global sum of local truncation errors
*     wsp(7)  = tbrkdwn, if `happy breakdown', time when it occured
*     wsp(8)  = t_now, integration domain successfully covered
*
*----------------------------------------------------------------------|
*-----The following parameters may also be adjusted herein-------------|
*
      integer mxstep, mxreject, ideg
      double precision delta, gamma
      parameter( mxstep   = 500, 
     .           mxreject = 0,
     .           ideg     = 6, 
     .           delta    = 1.2d0,
     .           gamma    = 0.9d0 )

*     mxstep  : maximum allowable number of integration steps.
*               The value 0 means an infinite number of steps.
* 
*     mxreject: maximum allowable number of rejections at each step. 
*               The value 0 means an infinite number of rejections.
*
*     ideg    : the Pade approximation of type (ideg,ideg) is used as 
*               an approximation to exp(H).
*
*     delta   : local truncation error `safety factor'
*
*     gamma   : stepsize `shrinking factor'
*
*----------------------------------------------------------------------|
*     Roger B. Sidje (rbs@maths.uq.edu.au)
*     EXPOKIT: Software Package for Computing Matrix Exponentials.
*     ACM - Transactions On Mathematical Software, 24(1):130-156, 1998
*----------------------------------------------------------------------|
*
      integer i, j, k1, mh, mx, iv, ih, j1v, ns, ifree, lfree, iexph,
     .        ireject,ibrkflag,mbrkdwn, nmult, nreject, nexph, nscale,
     .        nstep, iphi1h, iphi2h2,
     .        k1A, mhA, j1vA, ifreeA, lfreeA, iexphA, ibrkflagA, 
     .        mbrkdwnA, nmultA,  nscaleA, iphi1hA, flagPhi1W,
     .        kk, kTmp1, kTmp2, kTmp3
      double precision sgn, t_out, tbrkdwn, step_min,step_max, err_loc,
     .                 s_error, x_error, t_now, t_new, t_step, t_old,
     .                 xm, beta, break_tol, p1, p2, p3, eps, rndoff,
     .                 avnorm, hj1j, hjj, SQR1,
     .                 loc_t_out,
     .                 err_locA, t_newA, xmA, tbrkdwnA, betaA, avnormA,
     .                 hj1jA
      real t_start, t_end

      intrinsic AINT,ABS,DBLE,LOG10,MAX,MIN,NINT,SIGN,SQRT
      double precision DNRM2

*---  check restrictions on input parameters ...
      iflag = 0
*      if ( lwsp.lt.n*(m+4)+5*(m+4)**2+ideg+1 ) iflag = -1
      if ( lwsp.lt.n*(m+2)+5*(m+4)**2+ideg+1 ) iflag = -1
      if ( lwspA.lt.n*(m+2)+5*(m+3)**2+ideg+1 ) iflag = -1
      if ( liwsp.lt.m+4 ) iflag = -2
*!      if ( m.ge.n .or. m.le.0 ) iflag = -3
      if ( iflag.ne.0 ) stop 'bad sizes (in input of DSPHI2V)'
*
*---  initialisations ...
*
      k1 = 4
*     +4 is for error estimation of phi2.
      mh = m + 4
      iv = 1 
*     See the explanation for wsp(lwsp) in the first part of comments.
      ih = iv + n*(m+1) + n
      ifree = ih + mh*mh
      lfree = lwsp - ifree + 1
*---  for phi1*w, not phi2*(Aw+v)
      k1A = 3
*     +3 is for error estimation of phi1.
      mhA = m + 3
      ifreeA = ih + mhA*mhA
      lfreeA = lwspA - ifreeA + 1

      ibrkflag = 0
      mbrkdwn  = m
      nmult    = 0
      nreject  = 0
      nexph    = 0
      nscale   = 0
*---  for phi1*w, not phi2*(Aw+v)
      ibrkflagA = 0
      mbrkdwnA  = m
      nmultA    = 0
      nscaleA   = 0
      flagPhi1W = 0

      t_out    = ABS( t )
      tbrkdwn  = 0.0d0
*      step_min = t_out
      step_max = 0.0d0
      nstep    = 0
      s_error  = 0.0d0
      x_error  = 0.0d0
      t_now    = 0.0d0
      t_new    = 0.0d0
*---  for phi1*w, not phi2*(Aw+v)
      tbrkdwnA  = 0.0d0
*---  for output at t_out/2
      loc_t_out    = t_out/2.0d0
      step_min = loc_t_out
      
      p1 = 4.0d0/3.0d0
 1    p2 = p1 - 1.0d0
      p3 = p2 + p2 + p2
      eps = ABS( p3-1.0d0 )
      if ( eps.eq.0.0d0 ) go to 1
      if ( tol.le.eps ) tol = SQRT( eps )
      rndoff = eps*anorm
 
      break_tol = 1.0d-7
*>>>  break_tol = tol
*>>>  break_tol = anorm*tol

*
*---  step-by-step integration ...
*
      sgn = SIGN( 1.0d0,t )
      SQR1 = SQRT( 0.1d0 )
*     The following initializes w. Note that the label 100 can be a
*     starting point for a loop, which means that the commands below
*     the lable 100 can be performed more than once.
      do i=1,n
         w(i)=0.0d0
      enddo
*

 100  if ( t_now.ge.loc_t_out ) goto 500

      if ( t_now .ne. 0.0d0  ) then
         call DSCAL( n, 1.0d0/t_now, w,1 )
      endif

      nmult =  nmult + 1
      call matvec( w, wsp(iv) )
      call DAXPY( n, 1.0d0, u,1, wsp(iv),1 )
      beta = DNRM2( n, wsp(iv),1 )
      if ( beta.eq.0.0d0 ) goto 500
      call DSCAL( n, 1.0d0/beta, wsp(iv),1 )
      do i = 1,mh*mh
         wsp(ih+i-1) = 0.0d0
      enddo
*---  preparation for phi1*w, not phi2*(Aw+v)
      call DCOPY( n, w,1, wspA(iv),1 )
      betaA = DNRM2( n, wspA(iv),1 )
      if ( betaA.ne.0.0d0 ) then
         call DSCAL( n, 1.0d0/betaA, wspA(iv),1 )
         do i = 1,mhA*mhA
            wspA(ih+i-1) = 0.0d0
         enddo
         flagPhi1W = 1
      endif
      
      if ( nstep.eq.0 ) then
*---     obtain the very first stepsize ...
         xm = 1.0d0/DBLE( m )
         p1 = tol*(((m+1)/2.72D0)**(m+1))*SQRT(2.0D0*3.14D0*(m+1))
         t_new = (1.0d0/anorm)*(p1/(4.0d0*beta*anorm))**xm
         p1 = 10.0d0**(NINT( LOG10( t_new )-SQR1 )-1)
         t_new = AINT( t_new/p1 + 0.55d0 ) * p1
      endif
      nstep = nstep + 1
      t_step = MIN( loc_t_out-t_now, t_new )
*
*---  Lanczos loop ...
*     --- Begin (Lanczos process) ---
*
      j1v = iv + n
*---  for phi1*w, not phi2*(Aw+v)
      j1vA = iv + n
      do 200 j = 1,m
         if (ibrkflag.eq.0) then
            nmult = nmult + 1
            call matvec( wsp(j1v-n), wsp(j1v) )
*            if ( j.gt.1 )
*     .        call DAXPY(n,-wsp(ih+(j-1)*mh+j-2),wsp(j1v-2*n),1,
*     .                   wsp(j1v),1)
            if ( j.gt.1 ) then
*---  The following lines are replacements with DAXPY.
*---  The replacements of DAXPY starts here.
               hjj = wsp(ih+(j-1)*mh+j-2)
               kTmp1 = j1v-2*n
               kTmp2 = j1v
               kTmp3 = mod(n,4)
               if (kTmp3 .gt. 0) then
                  do kk = 1,kTmp3
                     wsp(kTmp2) = wsp(kTmp2) +(-hjj)*wsp(kTmp1)
                     kTmp1 = kTmp1 + 1
                     kTmp2 = kTmp2 + 1
                  enddo
               endif
               if (n .ge. 4) then
                  do kk = kTmp3+1,n,4
                     wsp(kTmp2) = wsp(kTmp2) +(-hjj)*wsp(kTmp1)
                     wsp(kTmp2+1) = wsp(kTmp2+1) +(-hjj)*wsp(kTmp1+1)
                     wsp(kTmp2+2) = wsp(kTmp2+2) +(-hjj)*wsp(kTmp1+2)
                     wsp(kTmp2+3) = wsp(kTmp2+3) +(-hjj)*wsp(kTmp1+3)
                     kTmp1 = kTmp1 + 4
                     kTmp2 = kTmp2 + 4
                  enddo
               endif
*---  The replacements of DAXPY ends here.
            endif
*            hjj = DDOT( n, wsp(j1v-n),1, wsp(j1v),1 )
*---  The following lines are replacements with DDOT.
*---  The replacements of DDOT starts here.
            kTmp1 = j1v-n
            kTmp2 = j1v
            kTmp3 = mod(n,5)
            hjj = 0.0d0
            if (kTmp3 .gt. 0) then
               do kk = 1,kTmp3
                  hjj = hjj + wsp(kTmp1)*wsp(kTmp2)
                  kTmp1 = kTmp1 + 1
                  kTmp2 = kTmp2 + 1
               enddo
            endif
            if (n .ge. 5) then
               do kk = kTmp3+1,n,5
                  hjj = hjj + wsp(kTmp1)*wsp(kTmp2)
     .                 + wsp(kTmp1+1)*wsp(kTmp2+1)
     .                 + wsp(kTmp1+2)*wsp(kTmp2+2)
     .                 + wsp(kTmp1+3)*wsp(kTmp2+3)
     .                 + wsp(kTmp1+4)*wsp(kTmp2+4)
                  kTmp1 = kTmp1 + 5
                  kTmp2 = kTmp2 + 5
               enddo
            endif
*---  The replacements of DDOT ends here.
*            call DAXPY( n, -hjj, wsp(j1v-n),1, wsp(j1v),1 )
*---  The following lines are replacements with DAXPY.
*---  The replacements of DAXPY starts here.
            kTmp1 = j1v-n
            kTmp2 = j1v
            kTmp3 = mod(n,4)
            if (kTmp3 .gt. 0) then
               do kk = 1,kTmp3
                  wsp(kTmp2) = wsp(kTmp2) +(-hjj)*wsp(kTmp1)
                  kTmp1 = kTmp1 + 1
                  kTmp2 = kTmp2 + 1
               enddo
            endif
            if (n .ge. 4) then
               do kk = kTmp3+1,n,4
                  wsp(kTmp2) = wsp(kTmp2) +(-hjj)*wsp(kTmp1)
                  wsp(kTmp2+1) = wsp(kTmp2+1) +(-hjj)*wsp(kTmp1+1)
                  wsp(kTmp2+2) = wsp(kTmp2+2) +(-hjj)*wsp(kTmp1+2)
                  wsp(kTmp2+3) = wsp(kTmp2+3) +(-hjj)*wsp(kTmp1+3)
                  kTmp1 = kTmp1 + 4
                  kTmp2 = kTmp2 + 4
               enddo
            endif
*---  The replacements of DAXPY ends here.
            hj1j = DNRM2( n, wsp(j1v),1 )
*           Note that the following is rewritten as wsp(ih+(j-1)*mh+j-1) = hjj.
            wsp(ih+(j-1)*(mh+1)) = hjj
         endif
*---     for phi1*w, not phi2*(Aw+v)
         if ( (ibrkflagA.eq.0) .and. (flagPhi1W.eq.1) ) then
            nmultA = nmultA + 1
            call matvec( wspA(j1vA-n), wspA(j1vA) )
*            if ( j.gt.1 )
*     .        call DAXPY(n,-wspA(ih+(j-1)*mhA+j-2),wspA(j1vA-2*n),1,
*     .                   wspA(j1vA),1)
            if ( j.gt.1 ) then
*---  The following lines are replacements with DAXPY.
*---  The replacements of DAXPY starts here.
               hjj = wspA(ih+(j-1)*mhA+j-2)
               kTmp1 = j1vA-2*n
               kTmp2 = j1vA
               kTmp3 = mod(n,4)
               if (kTmp3 .gt. 0) then
                  do kk = 1,kTmp3
                     wspA(kTmp2) = wspA(kTmp2) +(-hjj)*wspA(kTmp1)
                     kTmp1 = kTmp1 + 1
                     kTmp2 = kTmp2 + 1
                  enddo
               endif
               if (n .ge. 4) then
                  do kk = kTmp3+1,n,4
                     wspA(kTmp2) = wspA(kTmp2) +(-hjj)*wspA(kTmp1)
                     wspA(kTmp2+1) = wspA(kTmp2+1) +(-hjj)*wspA(kTmp1+1)
                     wspA(kTmp2+2) = wspA(kTmp2+2) +(-hjj)*wspA(kTmp1+2)
                     wspA(kTmp2+3) = wspA(kTmp2+3) +(-hjj)*wspA(kTmp1+3)
                     kTmp1 = kTmp1 + 4
                     kTmp2 = kTmp2 + 4
                  enddo
               endif
*---  The replacements of DAXPY ends here.              
            endif
*            hjj = DDOT( n, wspA(j1vA-n),1, wspA(j1vA),1 )
*---  The following lines are replacements with DDOT.
*---  The replacements of DDOT starts here.
            kTmp1 = j1vA-n
            kTmp2 = j1vA
            kTmp3 = mod(n,5)
            hjj = 0.0d0
            if (kTmp3 .gt. 0) then
               do kk = 1,kTmp3
                  hjj = hjj + wspA(kTmp1)*wspA(kTmp2)
                  kTmp1 = kTmp1 + 1
                  kTmp2 = kTmp2 + 1
               enddo
            endif
            if (n .ge. 5) then
               do kk = kTmp3+1,n,5
                  hjj = hjj + wspA(kTmp1)*wspA(kTmp2)
     .                 + wspA(kTmp1+1)*wspA(kTmp2+1)
     .                 + wspA(kTmp1+2)*wspA(kTmp2+2)
     .                 + wspA(kTmp1+3)*wspA(kTmp2+3)
     .                 + wspA(kTmp1+4)*wspA(kTmp2+4)
                  kTmp1 = kTmp1 + 5
                  kTmp2 = kTmp2 + 5
               enddo
            endif
*---  The replacements of DDOT ends here.
*            call DAXPY( n, -hjj, wspA(j1vA-n),1, wspA(j1vA),1 )
*---  The following lines are replacements with DAXPY.
*---  The replacements of DAXPY starts here.
            kTmp1 = j1vA-n
            kTmp2 = j1vA
            kTmp3 = mod(n,4)
            if (kTmp3 .gt. 0) then
               do kk = 1,kTmp3
                  wspA(kTmp2) = wspA(kTmp2) +(-hjj)*wspA(kTmp1)
                  kTmp1 = kTmp1 + 1
                  kTmp2 = kTmp2 + 1
               enddo
            endif
            if (n .ge. 4) then
               do kk = kTmp3+1,n,4
                  wspA(kTmp2) = wspA(kTmp2) +(-hjj)*wspA(kTmp1)
                  wspA(kTmp2+1) = wspA(kTmp2+1) +(-hjj)*wspA(kTmp1+1)
                  wspA(kTmp2+2) = wspA(kTmp2+2) +(-hjj)*wspA(kTmp1+2)
                  wspA(kTmp2+3) = wspA(kTmp2+3) +(-hjj)*wspA(kTmp1+3)
                  kTmp1 = kTmp1 + 4
                  kTmp2 = kTmp2 + 4
               enddo
            endif
*---  The replacements of DAXPY ends here.
            hj1jA = DNRM2( n, wspA(j1vA),1 )
*           Note that the following is rewritten
*           as wspA(ih+(j-1)*mhA+j-1) = hjj.
            wspA(ih+(j-1)*(mhA+1)) = hjj
         endif
         
         if (ibrkflag.eq.0) then
            if ( hj1j.le.break_tol ) then
*               print*,'happy breakdown: mbrkdwn =',j,' h =',hj1j
*     Only if a breakdown occures, then k1 and mbrkdwn change.
               k1 = 0
               ibrkflag = 1
               mbrkdwn = j
               tbrkdwn = t_now
               t_step = loc_t_out-t_now
*               goto 300
            else
               wsp(ih+(j-1)*mh+j) = hj1j
*              Note a memo in [Fujino:1996].
               wsp(ih+j*mh+j-1) = hj1j
               call DSCAL( n, 1.0d0/hj1j, wsp(j1v),1 )
               j1v = j1v + n
            endif
         endif
*---     for phi1*w, not phi2*(Aw+v)
         if ( (ibrkflagA.eq.0) .and. (flagPhi1W.eq.1) ) then
            if ( hj1jA.le.break_tol ) then
*               print*,'happy breakdown: mbrkdwnA =',j,' h =',hj1jA
               k1A = 0
               ibrkflagA = 1
               mbrkdwnA = j
               tbrkdwnA = t_now
*               t_step = loc_t_out-t_now
*               goto 300
            else
               wspA(ih+(j-1)*mhA+j) = hj1jA
*              Note a memo in [Fujino:1996].
               wspA(ih+j*mhA+j-1) = hj1jA
               call DSCAL( n, 1.0d0/hj1jA, wspA(j1vA),1 )
               j1vA = j1vA + n
            endif
         endif
*---     if `happy breakdown' go straightforward at the end ... 
         if ( (ibrkflag.eq.1) .and. (ibrkflagA.eq.1) ) then
            goto 300
         endif
 200  continue
*     --- End (Lanczos process) ---
      if (ibrkflag.eq.0) then
         nmult = nmult + 1
         call matvec( wsp(j1v-n), wsp(j1v) )
         avnorm = DNRM2( n, wsp(j1v),1 )
      endif
*---  for phi1*w, not phi2*(Aw+v)
      if ( (ibrkflagA.eq.0) .and. (flagPhi1W.eq.1) ) then
         nmultA = nmultA + 1
         call matvec( wspA(j1vA-n), wspA(j1vA) )
         avnormA = DNRM2( n, wspA(j1vA),1 )
      endif
*
*---  set 1's for the 4-extended scheme ...
*
 300  continue
*     The following is for \tilde{H}_{m+4} with c=e1 in Theorem 1
*     of [Sidje:1998]. If a happy breakdown has occurred, then k1=0.
*     (1,m+1) compornent of \tilde{H}_{m+4}.
      wsp(ih+mh*mbrkdwn) = 1.0d0
*     Note that we might have used wsp(ih+m*mh+m-1) temporarily in a loop
*     before Line 200.
      wsp(ih+m*mh+m-1)   = 0.0d0
*     Note that if a happy breakdown has occurred, then
*     wsp(ih+(mbrkdwn-1)*mh+mbrkdwn) is always zero.
      wsp(ih+(m-1)*mh+m) = 0.0d0
*     If a happy breakdown does not have occurred, then mbrkdwn=m.
      mx = mbrkdwn
      do i = 1,MAX(1,k1-1)
*     A component in each of rows from (m+1) to (m+4) of \tilde{H}_{m+4}.
         wsp(ih+(mx+i)*mh+mx+i-1) = 1.0d0
      enddo
*---  for phi1*w, not phi2*(Aw+v)
      if (flagPhi1W.eq.1) then
*        The following is for \tilde{H}_{m+3} with c=e1 in Theorem 1
*        of [Sidje:1998]. If a happy breakdown has occurred, then k1=0.
         wspA(ih+mhA*mbrkdwnA) = 1.0d0
*        Note that we might have used wspA(ih+m*mhA+m-1) temporarily in a loop
*        before Line 200.
         wspA(ih+m*mhA+m-1)   = 0.0d0
*        Note that if a happy breakdown has occurred, then
*        wsp(ih+(mbrkdwn-1)*mh+mbrkdwn) is always zero.
         wspA(ih+(m-1)*mhA+m) = 0.0d0
         do i = 1,k1A-1
            wspA(ih+(m+i)*mhA+m+i-1) = 1.0d0
         enddo
      endif
*
*---  loop while ireject<mxreject until the tolerance is reached ...
*
      ireject = 0
 401  continue
*
*---  compute w = beta*t_step*t_step*V*phi2(t_step*H)*e1
*
      nexph = nexph + 1
*---  irreducible rational Pade approximation ...
*     The following calculates exp(t*\tilde{H}_{m+4}) with c=e1 in Theorem 1
*     of [Sidje:1998]. If a happy breakdown have not occurred, then
*     mbrkdwn=m. Note a memo on Page 145 of [Sidje:1998].
      mx = mbrkdwn + MAX( 2,k1 )
      call DGPADM( ideg, mx, sgn*t_step, wsp(ih),mh,
     .             wsp(ifree),lfree, iwsp, iexph, ns, iflag )
      iexph = ifree + iexph - 1
      iphi1h = iexph + mbrkdwn*mx
      iphi2h2 = iphi1h + mx
      nscale = nscale + ns
*     The following are for the setting of err1 and err2 in Algorithm 2
*     of [Sidje:1998]. If a happy breakdown has not occurred, then
*     we also need the following wsp(iphi1h+mbrkdwn) to calculate phi1h
*     and wsp(iphi2h2+mbrkdwn) to calculate phi2h2
*     em^{T}tau^{3}phi3(tau*Hm)e1
      wsp(iphi2h2+mbrkdwn)   = hj1j*wsp(iphi2h2+mx+mbrkdwn-1)
*     em^{T}tau^{4}phi4(tau*Hm)e1
      wsp(iphi2h2+mbrkdwn+1) = hj1j*wsp(iphi2h2+2*mx+mbrkdwn-1)
*     em^{T}tau^{2}phi2(tau*Hm)e1
      wsp(iphi1h+mbrkdwn)   = hj1j*wsp(iphi1h+mx+mbrkdwn-1)
*---  for phi1*w, not phi2*(Aw+v)
      if (flagPhi1W.eq.1) then
         nexph = nexph + 1
*        The following calculates exp(t*\tilde{H}_{m+3}) with c=e1 in Theorem 1
*        of [Sidje:1998]. If a happy breakdown has occurred, then k1=0.
         mx = mbrkdwnA + MAX( 1,k1A )
         call DGPADM( ideg, mx, sgn*t_step, wspA(ih),mhA,
     .                wspA(ifreeA),lfreeA, iwsp, iexphA, ns, iflag )
         iexphA = ifreeA + iexphA - 1
         iphi1hA = iexphA + mbrkdwnA*mx
         nscaleA = nscaleA + ns
*        The following are for the setting of err1 and err2 in Algorithm 2
*        of [Sidje:1998]. If a happy breakdown has not occurred, then
*        we also need the following wsp(iphih+mbrkdwn) to calculate phih.
         wspA(iphi1hA+mbrkdwnA)   = hj1jA*wspA(iphi1hA+mx+mbrkdwnA-1)
         wspA(iphi1hA+mbrkdwnA+1) = hj1jA*wspA(iphi1hA+2*mx+mbrkdwnA-1)
      endif
      
 402  continue
*---  error estimate ...
*     The following calculations are for err1 and err2 in Algorithm 2
*     of [Sidje:1998]. If a happy breakdown has occurred, then k1=0.
      if ( k1.eq.0 ) then
         err_loc = tol
      else
         p1 = ABS( wsp(iphi2h2+m) )   * beta
         p2 = ABS( wsp(iphi2h2+m+1) ) * beta * avnorm 
         if ( p1.gt.10.0d0*p2 ) then
            err_loc = p2
            xm = 1.0d0/DBLE( m+2 )
         elseif ( p1.gt.p2 ) then
            err_loc = (p1*p2)/(p1-p2)
            xm = 1.0d0/DBLE( m+2 )
         else
            err_loc = p1
            xm = 1.0d0/DBLE( m+1 )
         endif
      endif
*---  for phi1*w, not phi2*(Aw+v)
*     The following calculations are for err1 and err2 in Algorithm 2
*     of [Sidje:1998]. If a happy breakdown has occurred, then k1A=0.
      if ( (errChkFlag.eq.1) .and. (flagPhi1W.eq.1) ) then
         if ( k1A.eq.0 ) then
            err_locA = tol
         else
            p1 = ABS( wspA(iphi1hA+m) )   * betaA
            p2 = ABS( wspA(iphi1hA+m+1) ) * betaA * avnormA
            if ( p1.gt.10.0d0*p2 ) then
               err_locA = p2
               xmA = 1.0d0/DBLE( m+1 )
            elseif ( p1.gt.p2 ) then
               err_locA = (p1*p2)/(p1-p2)
               xmA = 1.0d0/DBLE( m+1 )
            else
               err_locA = p1
               xmA = 1.0d0/DBLE( m )
            endif
         endif
      endif
      
*---  reject the step-size if the error is not acceptable ...   
      if ( (k1.ne.0) .and. (err_loc.gt.delta*t_step*tol) .and. 
     .     (mxreject.eq.0 .or. ireject.lt.mxreject) ) then
         t_old = t_step
         t_step = gamma * t_step * (t_step*tol/err_loc)**xm
         p1 = 10.0d0**(NINT( LOG10( t_step )-SQR1 )-1)
         t_step = AINT( t_step/p1 + 0.55d0 ) * p1
         if ( itrace.ne.0 ) then
            print*,'t_step =',t_old
            print*,'err_loc =',err_loc
            print*,'err_required =',delta*t_old*tol
            print*,'stepsize rejected, stepping down to:',t_step
         endif 
         ireject = ireject + 1
         nreject = nreject + 1
         if ( mxreject.ne.0 .and. ireject.gt.mxreject ) then
            print*,"Failure in DSEXTPHI2V: ---"
            print*,"The requested tolerance is too high."
            Print*,"Rerun with a smaller value."
            iflag = 2
            return
         endif
         goto 401
      endif
*---  for phi1*w, not phi2*(Aw+v)
*---  reject the step-size if the error is not acceptable ...
      if ( (errChkFlag.eq.1) .and. (flagPhi1W.eq.1) ) then
         if ( (k1A.ne.0) .and. (err_locA.gt.delta*t_step*tol) .and. 
     .        (mxreject.eq.0 .or. ireject.lt.mxreject) ) then
            t_old = t_step
            t_step = gamma * t_step * (t_step*tol/err_locA)**xmA
            p1 = 10.0d0**(NINT( LOG10( t_step )-SQR1 )-1)
            t_step = AINT( t_step/p1 + 0.55d0 ) * p1
            if ( itrace.ne.0 ) then
               print*,'t_step =',t_old
               print*,'err_locA =',err_locA
               print*,'err_required =',delta*t_old*tol
               print*,'stepsize rejected, stepping down to:',t_step
            endif 
            ireject = ireject + 1
            nreject = nreject + 1
            if ( mxreject.ne.0 .and. ireject.gt.mxreject ) then
               print*,"Failure in DSEXTPHI2V: ---"
               print*,"The requested tolerance is too high."
               Print*,"Rerun with a smaller value."
               iflag = 2
               return
            endif
            goto 401
         endif
      endif
*
*     The following calculates w plus the expression in the right hand
*     of (20) on Page 145 of [Sidje:1998]. If a happy breakdown has
*     occurred, then k1=0 and note a memo on Page 145 of [Sidje:1998].
*     mx decides which parts are used in exp(tau*\tilde{H}_{m+p}).
      mx = mbrkdwn + MAX( 0,k1-3 )
*     Note that the label 100 can be a starting point for a loop, which
*     means that the following can be performed more than once.
*     In that case, the following calculates t*t*phi2(tA)u by
*     al*(t-al)phi1((t-al)A)(A*w(al)+u)+t*w(al)
*     +(t-al)*(t-al)phi2((t-al)A)(A(w(al)+u)-(t-al)phi1((t-al)*A)w(al),
*     where al<t. For details, see a memo on Page 146 of [Sidje:1998].
      if (nstep.eq.1) then
         call DGEMV( 'n', n,mx,beta,wsp(iv),n,wsp(iphi2h2),1,1.0d0,w,1 )
      else
         xm = (t_now + t_step)/t_now
         call DGEMV( 'n', n,mx,beta,wsp(iv),n,wsp(iphi1h),1,xm,w,1 )
         call DGEMV( 'n', n,mx,beta,wsp(iv),n,wsp(iphi2h2),1,
     .        t_now,w,1 )
*---     for phi1*w, not phi2*(Aw+v)
*        The following calculates w plus the expression in the right hand
*        of (20) on Page 145 of [Sidje:1998]. If a happy breakdown has
*        occurred, then k1=0 and note a memo on Page 145 of [Sidje:1998].
         mx = mbrkdwnA + MAX( 0,k1A-2 )
*        Note that the label 100 can be a starting point for a loop, which
*        means that the following can be performed more than once.
*        In that case, the following calculates t*phi(tA)u by
*        exp((t-al)A)(al*phi(al*A)u)+(t-al)phi((t-al)A)u,
*        where al<t. For details, see memos on Page 145 of [Sidje:1998].
*
         call DGEMV( 'n', n,mx,betaA,wspA(iv),n,wspA(iphi1hA),1,0.0d0,
     .        w_aux,1 )
*
         call DAXPY( n, -1.0d0, w_aux,1, w,1 )
      endif
*
*---  suggested value for the next stepsize ...
*
      t_new = gamma * t_step * (t_step*tol/err_loc)**xm
      p1 = 10.0d0**(NINT( LOG10( t_new )-SQR1 )-1)
      t_new = AINT( t_new/p1 + 0.55d0 ) * p1 

      err_loc = MAX( err_loc,rndoff )
*---  for phi1*w, not phi2*(Aw+v)
      if ( (errChkFlag.eq.1) .and. (flagPhi1W.eq.1) ) then
         t_newA = gamma * t_step * (t_step*tol/err_locA)**xmA
         p1 = 10.0d0**(NINT( LOG10( t_newA )-SQR1 )-1)
         t_newA = AINT( t_newA/p1 + 0.55d0 ) * p1 

         err_locA = MAX( err_locA,rndoff )
         t_new = MIN( t_new,t_newA )
      endif
*
*---  update the time covered ...
*
      t_now = t_now + t_step 
*
*---  display and keep some information ...
*
      if ( itrace.ne.0 ) then
         print*,'integration',nstep,'---------------------------------'
         print*,'scale-square =',ns
         print*,'step_size =',t_step
         print*,'err_loc   =',err_loc
         print*,'next_step =',t_new
      endif
 
      step_min = MIN( step_min, t_step ) 
      step_max = MAX( step_max, t_step )
      s_error = s_error + err_loc
      x_error = MAX( x_error, err_loc )

*---  for phi1*w, not phi2*(Aw+v)
      if ( (errChkFlag.eq.1) .and. (flagPhi1W.eq.1) ) then
         s_error = s_error + err_locA
         x_error = MAX( x_error, err_locA )
      endif
      
      if ( mxstep.eq.0 .or. nstep.lt.mxstep ) goto 100
      iflag = 1
 
 500  continue
      
      if (loc_t_out.lt.t_out) then
         do i = 1,n
            wht(i) = w(i);
         enddo
         loc_t_out = t_out
         if (ibrkflag.eq.1) then
            ibrkflag = 0
         endif
         if (ibrkflagA.eq.1) then
            ibrkflagA = 0
         endif
         goto 100
      endif

      iwsp(1) = nmult + nmultA
      iwsp(2) = nexph
      iwsp(3) = nscale + nscaleA
      iwsp(4) = nstep
      iwsp(5) = nreject
      iwsp(6) = ibrkflag
      iwsp(7) = mbrkdwn

      wsp(1)  = step_min
      wsp(2)  = step_max
      wsp(3)  = 0.0d0
      wsp(4)  = 0.0d0
      wsp(5)  = x_error
      wsp(6)  = s_error
      wsp(7)  = tbrkdwn
      wsp(8)  = sgn*t_now
      END
*----------------------------------------------------------------------|
