! program test
!     integer,parameter :: N=1
!     real*8 :: xs(N), ys(N), pxs(N), pys(N)
!     real*8 :: delta_t, box_size
!     logical :: circle_box, edge_collisions
!     integer :: i

!     call RANDOM_NUMBER(xs)
!     call RANDOM_NUMBER(ys)
!     call RANDOM_NUMBER(pxs)
!     call RANDOM_NUMBER(pys)

!     delta_t = 0.01
!     box_size = 1
!     circle_box = .true.
!     edge_collisions = .true.
    
!     write(*,*) xs, pxs
!     write(*,*) ys, pys
!     call update_positions(xs,ys,pxs,pys,circle_box,box_size,delta_t,edge_collisions,N)
!     write(*,*) xs
!     write(*,*) ys
! contains

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
    real(DP), intent(in)                    :: box_size, delta_t
    logical,  intent(in)                    :: circle_box, edge_collisions
    integer :: i, box_edge!torus_factor
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
    enddo
    !$OMP end parallel do

end subroutine update_positions
! end program test