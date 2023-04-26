* File name: dsExtPhiv.f
*   The following comments were added to put files on GitHub
*   (26-Apr-2023):
*   Copied from dsphiv.f in Expokit, and revised for an extended
*   version of DsPHIV to calculate exp(t*A)v + t*phi(tA)u at t and t/2.
******************************
*----------------------------------------------------------------------|
      subroutine DSEXTPHIV( n, m, t, u, v, w, wht, tol, anorm,
     .                      wsp,lwsp, iwsp,liwsp, matvec, itrace,
     .                      iflag ) 
      implicit none
      integer n, m, lwsp, liwsp, itrace, iflag, iwsp(liwsp)
      double precision t, tol, anorm, u(n), v(n), w(n), wht(n),
     .                 wsp(lwsp)
      external matvec

*-----Purpose----------------------------------------------------------|
*
*---  DSPHIV computes w = exp(t*A)v + t*phi(tA)u which is the solution 
*     of the nonhomogeneous linear ODE problem w' = Aw + u, w(0) = v.
*     phi(z) = (exp(z)-1)/z and A is a Symmetric matrix.
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
*     v(n)   : (input) operand vector with respect to the exp function
*              (initial condition of the ODE problem).
*  
*     w(n)   : (output) computed approximation of exp(t*A)v + t*phi(tA)u 
* 
*     wht(n) : (output) computed approximation of exp(t/2*A)v + t/2*phi(tA/2)u
*
*     tol    : (input/output) the requested accuracy tolerance on w. 
*              If on input tol=0.0d0 or tol is too small (tol.le.eps)
*              the internal value sqrt(eps) is used, and tol is set to
*              sqrt(eps) on output (`eps' denotes the machine epsilon).
*              (`Happy breakdown' is assumed if h(j+1,j) .le. anorm*tol)
*
*     anorm  : (input) an approximation of some norm of A.
*
*   wsp(lwsp): (workspace) lwsp .ge. n*(m+1)+n+(m+3)^2+4*(m+3)^2+ideg+1
*                                   +---------+-------+---------------+
*              (actually, ideg=6)        V        H      wsp for PADE
*                   
* iwsp(liwsp): (workspace) liwsp .ge. m+3
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
     .        nstep, iphih, kk, kTmp1, kTmp2, kTmp3
      double precision sgn, t_out, tbrkdwn, step_min,step_max, err_loc,
     .                 s_error, x_error, t_now, t_new, t_step, t_old,
     .                 xm, beta, break_tol, p1, p2, p3, eps, rndoff,
     .                 avnorm, hj1j, hjj, SQR1,
     .                 loc_t_out

      intrinsic AINT,ABS,DBLE,LOG10,MAX,MIN,NINT,SIGN,SQRT
      double precision DNRM2

*---  check restrictions on input parameters ...
      iflag = 0
*      if ( lwsp.lt.n*(m+3)+5*(m+3)**2+ideg+1 ) iflag = -1
      if ( lwsp.lt.n*(m+2)+5*(m+3)**2+ideg+1 ) iflag = -1
      if ( liwsp.lt.m+3 ) iflag = -2
*!      if ( m.ge.n .or. m.le.0 ) iflag = -3
      if ( iflag.ne.0 ) stop 'bad sizes (in input of DSPHIV)'
*
*---  initialisations ...
*
      k1 = 3
*     +3 is for error estimation of phi1.
      mh = m + 3
      iv = 1 
      ih = iv + n*(m+1) + n
      ifree = ih + mh*mh
      lfree = lwsp - ifree + 1

      ibrkflag = 0
      mbrkdwn  = m
      nmult    = 0
      nreject  = 0
      nexph    = 0
      nscale   = 0

      t_out    = ABS( t )
      tbrkdwn  = 0.0d0
*      step_min = t_out
      step_max = 0.0d0
      nstep    = 0
      s_error  = 0.0d0
      x_error  = 0.0d0
      t_now    = 0.0d0
      t_new    = 0.0d0
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
*     Refer to Aogo. 1 [Sidje:1998, p. 136]
*
      sgn = SIGN( 1.0d0,t )
      SQR1 = SQRT( 0.1d0 )
* The following copies v to w.
      call DCOPY( n, v,1, w,1 )

 100  if ( t_now.ge.loc_t_out ) goto 500

      nmult =  nmult + 1
      call matvec( w, wsp(iv) )
* DAXPY is given in blas.f.
      call DAXPY( n, 1.0d0, u,1, wsp(iv),1 )
      beta = DNRM2( n, wsp(iv),1 )
      if ( beta.eq.0.0d0 ) goto 500
      call DSCAL( n, 1.0d0/beta, wsp(iv),1 )
      do i = 1,mh*mh
         wsp(ih+i-1) = 0.0d0
      enddo

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
      do 200 j = 1,m
         nmult = nmult + 1
         call matvec( wsp(j1v-n), wsp(j1v) )
*         if ( j.gt.1 )
*     .     call DAXPY(n,-wsp(ih+(j-1)*mh+j-2),wsp(j1v-2*n),1,wsp(j1v),1)
         if ( j.gt.1 ) then
*---  The following lines are replacements with DAXPY.
*---  The replacements of DAXPY starts here. DAXPY is given in blas.f.
*---  If j>1, this parts computes A*vj - h_{j-1,j}*v_{j-1}.
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
*---  End of if j>1.
         endif
*         hjj = DDOT( n, wsp(j1v-n),1, wsp(j1v),1 )
*---  The following lines are replacements with DDOT.
*---  The replacements of DDOT starts here. DDOT is given in blas.f.
*---  This part computes h_{j,j}=(vj,A*vj - h_{j-1,j}*v_{j-1}=(vj,A*vj).
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
     .              + wsp(kTmp1+1)*wsp(kTmp2+1)
     .              + wsp(kTmp1+2)*wsp(kTmp2+2)
     .              + wsp(kTmp1+3)*wsp(kTmp2+3)
     .              + wsp(kTmp1+4)*wsp(kTmp2+4)
               kTmp1 = kTmp1 + 5
               kTmp2 = kTmp2 + 5
            enddo
         endif
*---  The replacements of DDOT ends here.
*---  End of computation of h_{j,j}=(vj,A*vj - h_{j-1,j}*v_{j-1}=(vj,A*vj).
*         call DAXPY( n, -hjj, wsp(j1v-n),1, wsp(j1v),1 )
*---  The following lines are replacements with DAXPY.
*---  The replacements of DAXPY starts here.
*---  This part computes A*vj - h_{j-1,j}*v_{j-1} - h_{j,j}vj.
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
*---  End of computation of A*vj - h_{j-1,j}*v_{j-1} - h_{j,j}vj.
         hj1j = DNRM2( n, wsp(j1v),1 )
*        Note that the following is rewritten as wsp(ih+(j-1)*mh+j-1) = hjj.
*--- Save h_{j,j}.
         wsp(ih+(j-1)*(mh+1)) = hjj
*---     if `happy breakdown' go straightforward at the end ... 
         if ( hj1j.le.break_tol ) then
*            print*,'happy breakdown: mbrkdwn =',j,' h =',hj1j
*---  Only if a breakdown occurs, then k1 and mbrkdwn change.
            k1 = 0
            ibrkflag = 1
            mbrkdwn = j
            tbrkdwn = t_now
            t_step = loc_t_out-t_now
            goto 300
         endif
*---  Save h_{j,j+1}=h_{j+1,j}.
         wsp(ih+(j-1)*mh+j) = hj1j
*        Note a memo in [Fujino:1996, p. 28].
*---  Save h_{j+1,j}.
         wsp(ih+j*mh+j-1) = hj1j
         call DSCAL( n, 1.0d0/hj1j, wsp(j1v),1 )
         j1v = j1v + n
 200  continue
*     --- End (Lanczos process) ---
      nmult = nmult + 1
      call matvec( wsp(j1v-n), wsp(j1v) )
      avnorm = DNRM2( n, wsp(j1v),1 )
*
*---  set 1's for the 3-extended scheme ...
*
 300  continue
*     The following is for \tilde{H}_{m+3} with c=e1 in Theorem 1
*     of [Sidje:1998]. For NO breakdown, mbrkdwn = m and k1 = 3.
      wsp(ih+mh*mbrkdwn) = 1.0d0
*     Note that we might have used wsp(ih+m*mh+m-1) temporarily in a loop
*     before Line 200.
      wsp(ih+m*mh+m-1)   = 0.0d0
      wsp(ih+(m-1)*mh+m) = 0.0d0
      do i = 1,k1-1
         wsp(ih+(m+i)*mh+m+i-1) = 1.0d0
      enddo
*
*---  loop while ireject<mxreject until the tolerance is reached ...
*
      ireject = 0
 401  continue
*
*---  compute w = beta*t_step*V*phi(t_step*H)*e1 + w
*
      nexph = nexph + 1
*---  irreducible rational Pade approximation ...
*     The following calculates exp(t*\tilde{H}_{m+3}) with c=e1 in Theorem 1
*     of [Sidje:1998].
      mx = mbrkdwn + MAX( 1,k1 )
      call DGPADM( ideg, mx, sgn*t_step, wsp(ih),mh,
     .             wsp(ifree),lfree, iwsp, iexph, ns, iflag )
*     As DGPADM was called with wsp(ifree), exp(tH) starts
*     at wsp(ifree + iexph -1). See a comment in dgpadm.f.
      iexph = ifree + iexph - 1
*     Note that mx corresponds to the dimension of a column
*     vector in exp(t \tilde{H}_{m+p}).
      iphih = iexph + mbrkdwn*mx
      nscale = nscale + ns
*     The following are for a similar setting of err1 and err2
*     in Algorithm 2 of [Sidje:1998].
*     The following line corresponds to
*     |h_{m+1,m}t^{2}e_{m}^{ast}phi2(t H_{m})e1| because
*     we now estimate the error of phi1, not exp.
      wsp(iphih+mbrkdwn)   = hj1j*wsp(iphih+mx+mbrkdwn-1)
      wsp(iphih+mbrkdwn+1) = hj1j*wsp(iphih+2*mx+mbrkdwn-1)
 
 402  continue
*---  error estimate ...
*     The following calculations are for err1 and err2 in Algorithm 2
*     of [Sidje:1998].
      if ( k1.eq.0 ) then
         err_loc = tol
      else
         p1 = ABS( wsp(iphih+m) )   * beta
         p2 = ABS( wsp(iphih+m+1) ) * beta * avnorm 
         if ( p1.gt.10.0d0*p2 ) then
            err_loc = p2
            xm = 1.0d0/DBLE( m+1 )
         elseif ( p1.gt.p2 ) then
            err_loc = (p1*p2)/(p1-p2)
            xm = 1.0d0/DBLE( m+1 )
         else
            err_loc = p1
            xm = 1.0d0/DBLE( m )
         endif
      endif

*---  reject the step-size if the error is not acceptable ...   
      if ( (k1.ne.0) .and. (err_loc.gt.delta*t_step*tol) .and. 
     .     (mxreject.eq.0 .or. ireject.lt.mxreject) ) then
         t_old = t_step
*---  Update step size by the formula (10) in [Sidje:1998, p. 141].
         t_step = gamma * t_step * (t_step*tol/err_loc)**xm
*---  Round the step size to 2 significant digits [Sidje:1998, p. 141].
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
            print*,"Failure in DSPHIV: ---"
            print*,"The requested tolerance is too high."
            Print*,"Rerun with a smaller value."
            iflag = 2
            return
         endif
         goto 401
      endif
*
*     The following calculates w plus the expression in the right hand
*     of (20) on Page 145 of [Sidje:1998].
      mx = mbrkdwn + MAX( 0,k1-2 )
*---  DGEMV is given in blas.f.
      call DGEMV( 'n', n,mx,beta,wsp(iv),n,wsp(iphih),1,1.0d0,w,1 )
*
*---  suggested value for the next stepsize ...
*
      t_new = gamma * t_step * (t_step*tol/err_loc)**xm
      p1 = 10.0d0**(NINT( LOG10( t_new )-SQR1 )-1)
      t_new = AINT( t_new/p1 + 0.55d0 ) * p1 

      err_loc = MAX( err_loc,rndoff )
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
         goto 100
      endif

      iwsp(1) = nmult
      iwsp(2) = nexph
      iwsp(3) = nscale
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
