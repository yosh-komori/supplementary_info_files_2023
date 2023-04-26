! filename: Copy_Ab_values_for_C.f90
!   Ver. 0 (22-Jul-2015)
!
subroutine copy_Ab_values(cp_ms,cp_fp1,cp_fp2,cp_recalph,cp_recf,cp_recf2)
  implicit none
  integer, parameter :: dp=selected_real_kind(15)
  
  integer, dimension(46), intent(out) :: cp_ms
  real(kind=dp), dimension(46), intent(out) :: cp_fp1, cp_fp2, cp_recalph
  real(kind=dp), dimension(4476), intent(out) :: cp_recf
  real(kind=dp), dimension(184), intent(out) :: cp_recf2

  integer :: i

  integer, dimension(46) :: ms
  real(kind=dp), dimension(46) :: fp1, fp2, recalph
  real(kind=dp), dimension(4476) :: recf
  real(kind=dp), dimension(184) :: recf2

  include 'rectp.f'

  do i=1,46
     cp_ms(i)=ms(i)
     cp_fp1(i)=fp1(i)
     cp_fp2(i)=fp2(i)
     cp_recalph(i)=recalph(i)
  end do

  do i=1,4476
     cp_recf(i)=recf(i)
  end do

  do i=1,184
     cp_recf2(i)=recf2(i)
  end do

end subroutine copy_Ab_values
!---------------------------------------------------------------------------
