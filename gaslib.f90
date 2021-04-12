subroutine update_positions(xs, ys, pxs, pys, circle_box, box_size, delta_t, edge_collisions, N)
    !Shifts the positions in xs and ys by the momentum in
    !pxs and pys. Reflects particles on the boundary of the
    !box if edge_collisions is true. Said box is a circle if
    !circle_box is true, otherwise it is a sqare.
    use omp_lib
    implicit none
    integer,  parameter                     :: DP = kind(1.d0)
    integer,  intent(in)                    :: N
    real(DP), intent(inout), dimension(N)   :: xs, ys, pxs, pys
    !f2py intent(in,out)
    real(DP), intent(in)                    :: box_size, delta_t
    logical,  intent(in)                    :: circle_box, edge_collisions
    integer :: i, box_edge
    real(DP) :: r, dot, nx, ny

    if(edge_collisions) then
        box_edge = box_size
    else
        box_edge = -box_size
    end if

    !$OMP parallel do shared(xs,ys,pxs,pys) private(r,dot,nx,ny)
    do i = 1, N
        xs(i) = xs(i) + delta_t*pxs(i)
        ys(i) = ys(i) + delta_t*pys(i)

        if(circle_box) then
            r = sqrt(xs(i)**2.d0 + ys(i)**2.d0)
            if(r > box_size) then
                nx = -xs(i)/r
                ny = -ys(i)/r
                xs(i) = -box_edge*nx
                ys(i) = -box_edge*ny
                if(edge_collisions) then
                    dot = pxs(i)*nx + pys(i)*ny
                    pxs(i) = pxs(i) - 2.d0*dot*nx
                    pys(i) = pys(i) - 2.d0*dot*ny
                end if
            end if
        else
            if(xs(i) < -box_size) then
                xs(i) = -box_edge
                if(edge_collisions) pxs(i) = -pxs(i)
            else if(xs(i) > box_size) then
                xs(i) = box_edge
                if(edge_collisions) pxs(i) = -pxs(i)
            end if

            if(ys(i) < -box_size) then
                ys(i) = -box_edge
                if(edge_collisions) pys(i) = -pys(i)
            else if(ys(i) > box_size) then
                ys(i) = box_edge
                if(edge_collisions) pys(i) = -pys(i)
            end if            
        end if
    end do
    !$OMP end parallel do

end subroutine update_positions

real*8 function maxwell_boltzmann_cdf(v,T,m,kB)
    !Returns the value of the cumulative density function
    !of the Maxwell-Boltzmann distribution
    implicit none
    integer,  parameter     :: DP = kind(1.d0)
    real(DP), parameter     :: PI = 4.d0*atan(1.d0)
    real(DP), intent(in)    :: v
    real(DP), intent(in)    :: T, m, kB
    real(DP) :: a

    a = sqrt(kB*T/m)
    maxwell_boltzmann_cdf = 0.d0
    if(v > 0.d0) maxwell_boltzmann_cdf = erf(v/(a*sqrt(2.d0)))-exp(-v**2/(2*a**2))*sqrt(2/PI)*v/a

end function maxwell_boltzmann_cdf

real*8 function inverse_maxwell_boltzmann(p,T,m,kB,tol)
    implicit none
    integer, parameter              :: DP = kind(1.d0)
    real(DP), intent(in)            :: p
    real(DP), intent(in)            :: T, m, kB
    real(DP), intent(in)            :: tol
    real(DP), external :: maxwell_boltzmann_cdf
    real(DP) :: left, right, mid, px, fm, diff

    !Find the range of inputs where the result could be
    left = 0.d0
    right = 0.d0
    px = 0.d0
    px = maxwell_boltzmann_cdf(1.d0,T,m,kB)
    if(p > px) then
        left = 1.d0
        right = 1.d0
        do while(maxwell_boltzmann_cdf(right,T,m,kB) < p)
            right = 2.d0*right
        end do
    else
        right = 1.d0
        if(maxwell_boltzmann_cdf(0.d0,T,m,kB) < p) then
            left = 0.d0
        else
            left = -1.d0
            do while(maxwell_boltzmann_cdf(left,T,m,kB) > p)
                left = 2.d0*left
            end do
        end if
    end if

    !Perform a binary search on the function space
    mid = 0.d0
    fm = 0.d0
    do while(left <= right)
        mid = left + (right-left)/2.d0
        fm = maxwell_boltzmann_cdf(mid,T,m,kB)
        diff = fm - p
        if(abs(diff) < tol) then
            inverse_maxwell_boltzmann = mid
            return
        end if
        if(fm > p) then
            right = mid - tol
        else
            left = mid + tol
        end if
    end do
end function inverse_maxwell_boltzmann

subroutine draw_from_maxwell_boltzmann(pxs,pys,T,m,kB,tol,N)
    !Fills up the given momentum vector with
    !results drawn from the Maxwell-Boltzmann distribution.
    use omp_lib
    implicit none
    integer,  parameter                     :: DP = kind(1.d0)
    real(DP), parameter                     :: PI = 4.d0*atan(1.d0)
    integer,  intent(in)                    :: N
    real(DP), intent(out), dimension(N)     :: pxs, pys
    real(DP), intent(in)                    :: T, m, kB
    real(DP), intent(in), optional          :: tol
    integer :: i
    real(DP) :: temp, tol_
    real(DP), external :: inverse_maxwell_boltzmann
    
    tol_ = 1.d-8
    if(present(tol)) tol_ = tol

    !Use the x-coordinates to store the randomly generated
    !cumulative probabillity used to invert the distribution
    call RANDOM_NUMBER(pxs)

    !$OMP parallel do shared(pxs,pys)
    do i = 1, N
        !Use the y-coordinate to store the resulting velocity
        pys(i) = inverse_maxwell_boltzmann(pxs(i),T,m,kB,tol_)
    end do
    !$OMP end parallel do

    !Use x-coordinates again to store a random angle
    call RANDOM_NUMBER(pxs)
    pxs = pxs*2.d0*PI

    !The velocity is now in pys and angle in pxs
    do i = 1, N
        temp = pys(i)*cos(pxs(i))
        pys(i) = pys(i)*sin(pxs(i))
        pxs(i) = temp
    enddo

end subroutine