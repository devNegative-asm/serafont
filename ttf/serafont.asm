; This isn't currently used anywhere, but it has equivalent semantics to the odin version and can be linked in if you use x64
default rel
global serafont_fractional_winding_line_segment
global serafont_fast_cuberoot
global serafont_distance_squared_to_parabola
global serafont_fractional_winding_bezier

section .data

epsconst:
    dd 0.01

nosignconst:
    dd $7FFFFFFF
signconst:
    dd $80000000

atan_a1:
    dd  0.99997726
atan_a3:
    dd -0.33262347
atan_a5:
    dd  0.19354346
atan_a7:
    dd -0.11643287
atan_a9:
    dd  0.05265332
atan_a11:
    dd -0.01172120
pi_const:
    dd 3.14159265358979323846
pi_over_2_const:
    dd 1.57079632679489661923
neg_pi_over_2_const:
    dd -1.57079632679489661923
neg_pi_const:
    dd -3.14159265358979323846
radians_to_winding:
    dd -0.15915494309189535
one_third:
    dd 0.3333333333333333333333
    dd 0.3333333333333333333333
    dd 0.3333333333333333333333
    dd 0.3333333333333333333333
    dd 0.3333333333333333333333
    dd 0.3333333333333333333333
    dd 0.3333333333333333333333
    dd 0.3333333333333333333333
one_half:
    dd 0.500
one:
    dd 1.000000
two:
    dd 2.000000
negative_two:
    dd -2.000000
negative_one_sixth:
    dd -0.166666666666666666666666666666666666666
six:
    dd 6.000000
cuberoot_bits:
    dd $2A510554
epsilon:
    dd 7.5e-37
cuberoot_of_one_half:
    dd 0.79370052598409973737585281
float_bias:
    dd $1
small_number:
    dd 1e-8
section .text

; RDI,  RSI,  RDX, RCX,       R8
; &pxs, &pys, &op, &windings, &dists
serafont_fractional_winding_line_segment:
    vmovups ymm0, [rdi] ; p.x
    vmovups ymm1, [rsi] ; p.y
    vbroadcastss ymm2, [rdx + 24] ; op.a.x
    vbroadcastss ymm3, [rdx + 28] ; op.a.y
    vbroadcastss ymm4, [rdx + 32] ; op.c.x   c also means b
    vbroadcastss ymm5, [rdx + 36] ; op.c.y   c also means b

    ; calc ab = op.c - op.a
    vsubps ymm6, ymm4, ymm2
    vsubps ymm7, ymm5, ymm3

    ; cal ap = p - op.a
    vsubps ymm11, ymm0, ymm2
    vsubps ymm12, ymm1, ymm3

    ; calc bp = p - op.c
    vsubps ymm13, ymm0, ymm4
    vsubps ymm14, ymm1, ymm5


    ; calc ymm8 = dist2
    vmulps ymm8, ymm6, ymm6
    vfmadd231ps ymm8, ymm7, ymm7

    ; calc cross = ab x ap
    vmulps ymm10, ymm7, ymm11
    vfmsub231ps ymm10, ymm6, ymm12

    ; ab x ap is useful, so we'll push it, then pop into ymm12 before returning
    sub rsp, 32
    vmovups [rsp], ymm10

    ; dist = cross x cross / dist2
    vmulps ymm10, ymm10, ymm10
    vdivps ymm10, ymm8

    ; winding is going to be a pita because of atan2. Let's calculate it when we're not so tight on registers
    ; cross, and dist2 can be dropped. that's in ymm8, so that is available again. as are ymm0-ymm5
    ; let's create a mask in ymm8 if dist needs to be clamped to an endpoint

    ; ab.bp. if it's positive, set the mask
    vmulps ymm8, ymm6, ymm13
    vfmadd231ps ymm8, ymm7, ymm14
    vxorps ymm0, ymm0, ymm0
    vcmpgtps ymm8, ymm0

    ; ab.ap. if it's negative, set the mask
    vmulps ymm15, ymm6, ymm11
    vfmadd231ps ymm15, ymm7, ymm12
    vcmpltps ymm15, ymm0

    ; or the masks
    vorps ymm8, ymm15

    ; if the mask in ymm8 is set, dist = min(linalg.dot(ap,ap), linalg.dot(bp,bp))
    vmulps ymm0, ymm11, ymm11
    vfmadd231ps ymm0, ymm12, ymm12
    vmulps ymm1, ymm13, ymm13
    vfmadd231ps ymm1, ymm14, ymm14
    vminps ymm0, ymm0, ymm1
    
    ; if ymm8 is set, blend from ymm0, else from ymm10
    vblendvps ymm0, ymm10, ymm0, ymm8
    vmovups [r8], ymm0

    ; distance is set. Now we need to calculate winding, which is unconditionally atan2(bp x ap, bp . ap)
    ; bp is in (ymm13, ymm14) and ap is in (ymm11, ymm12). All other registers can be used.
    ; ymm0 = cross, ymm1 = dot
    vmulps ymm0, ymm14, ymm11
    vfmsub231ps ymm0, ymm13, ymm12
    vmulps ymm1, ymm13, ymm11
    vfmadd231ps ymm1, ymm14, ymm12

    ; calc atan2(y = ymm0, x = ymm1)

    ; first compare the absolute values of x and y so we know which way to divide them to be <= 1. hold on to mask in ymm2. We'll need it later
    vbroadcastss ymm2, [nosignconst]
    vandps ymm3, ymm2, ymm0
    vandps ymm4, ymm2, ymm1
    vcmpltps ymm2, ymm4, ymm3

    ; the mask is set when |y| < |x|. So when it's set, use the default path, otherwise flip it. save (x,y) to (ymm13,ymm14)
    vmovups ymm14, ymm0
    vmovups ymm13, ymm1
    vdivps ymm3, ymm0, ymm1
    vdivps ymm4, ymm1, ymm0
    vblendvps ymm3, ymm3, ymm4, ymm2

    ; run the approximation on ymm3

    vmulps ymm4, ymm3, ymm3
    vbroadcastss ymm7, [atan_a1] ; broadcast during the mul to avoid dependency chain stalls
    vbroadcastss ymm8, [atan_a5]
    vbroadcastss ymm9, [atan_a9]
    vmulps ymm5, ymm4, ymm4
    vbroadcastss ymm10, [atan_a3]
    vbroadcastss ymm11, [atan_a7]
    vbroadcastss ymm12, [atan_a11]
    vmulps ymm6, ymm5, ymm5

    vfmadd231ps ymm7, ymm4, ymm10 ; P0
    vfmadd231ps ymm8, ymm4, ymm11 ; P1
    vfmadd231ps ymm9, ymm4, ymm12 ; P2

    ; we use ymm2-3, ymm5-9. We can broadcast our constants to other regs.
    vxorps ymm0, ymm0, ymm0
    vbroadcastss ymm1, [pi_over_2_const]
    vbroadcastss ymm4, [pi_const]

    ; calc P0 + P1*ymm5 + P2*ymm6
    vfmadd231ps ymm7, ymm8, ymm5
    vfmadd231ps ymm7, ymm9, ymm6

    vbroadcastss ymm6, [neg_pi_over_2_const]
    vbroadcastss ymm5, [neg_pi_const]

    ; times original
    vmulps ymm7, ymm3

    vmovups ymm12, [rsp]
    add rsp, 32

    ; now in use:
    ;   ymm0 = 0
    ;   ymm1 = pi/2
    ;   ymm2 = reciprocal mask
    ;   ymm3 = input
    ;   ymm4 = pi
    ;   ymm5 = -pi
    ;   ymm6 = -pi/2
    ;   ymm7 = result
    ;   ymm12 = cross (not used here, but rather by caller)
    ;   ymm13 = x
    ;   ymm14 = y

    ; now we need to recombine it with the following logic
    ; res = swap ? (atan_input >= 0.0f ? M_PI_2 : -M_PI_2) - res : res;

    ; ymm3 = atan_input >= 0
    vcmpgeps ymm3, ymm3, ymm0
    ; ymm3 = atan_input >= 0.0f ? M_PI_2 : -M_PI_2
    vblendvps ymm3, ymm6, ymm1, ymm3
    ; ymm3 = (atan_input >= 0.0f ? M_PI_2 : -M_PI_2) - res
    vsubps ymm3, ymm7
    ; ymm3 = swap ? (atan_input >= 0.0f ? M_PI_2 : -M_PI_2) - res : res;
    vblendvps ymm3, ymm7, ymm3, ymm2

    vbroadcastss ymm15, [radians_to_winding]
    ; available regs: ymm1-2, ymm6-12
    ;   ymm0 = 0
    ;   ymm3 = res
    ;   ymm4 = pi
    ;   ymm5 = -pi
    ;   ymm13 = x
    ;   ymm14 = y
    ;   ymm15 = radians_to_winding

    ; now quadrant shenanigans
    ; if (x < 0.0f && y >= 0.0f) { res =  M_PI + res; } // 2nd quadrant
    ; else if (x <  0.0f && y <  0.0f) { res = -M_PI + res; } // 3rd quadrant
    ; implement as res = res + (x < 0 ? y < 0 ? -pi : pi : 0)
    vcmpltps ymm2, ymm14, ymm0 ; y < 0
    vcmpltps ymm1, ymm13, ymm0 ; x < 0
    vblendvps ymm2, ymm4, ymm5, ymm2 ; y < 0 ? -pi : pi
    vblendvps ymm1, ymm0, ymm2, ymm1 ; (x < 0 ? y < 0 ? -pi : pi : 0)
    vaddps ymm3, ymm1
    vmulps ymm3, ymm15

    vmovups [rcx], ymm3

    ret

; RDI
; &xs

serafont_fast_cuberoot:
    vmovups ymm0, [rdi]
serafont_fast_cuberoot_skipload:
    vxorps ymm7, ymm7
    vbroadcastss ymm2, [two]
    vbroadcastss ymm5, [nosignconst]
    vbroadcastss ymm6, [epsilon]

    ; save it for the sign later 
    vmovups ymm10, ymm0
    vandps ymm0, ymm0, ymm5

    vcmpltps ymm9, ymm0, ymm6 ;close enough to 0 it should probably just be 0

    ; x^(1/4)
    vsqrtps ymm1, ymm0
    vsqrtps ymm1, ymm1

    ; x^(5/16)
    vmulps ymm1, ymm1, ymm0
    vsqrtps ymm1, ymm1
    vsqrtps ymm1, ymm1

    ; iterate to improve estimate a(a^3+2x)/(2a^3+x)
    
    vmulps ymm3, ymm1, ymm1
    vmulps ymm3, ymm3, ymm1 ; a^3
    vmovups ymm4, ymm3
    vfmadd231ps ymm4, ymm2, ymm0 ; a^3+2x
    vmulps ymm4, ymm1 ; a(a^3+2x)
    vfmadd213ps ymm3, ymm2, ymm0 ; (2a^3+x)
    vdivps ymm1, ymm4, ymm3

    ; second iteration
    vmulps ymm3, ymm1, ymm1
    vmulps ymm3, ymm3, ymm1
    vmovups ymm4, ymm3
    vfmadd231ps ymm4, ymm2, ymm0 
    vmulps ymm4, ymm1
    vfmadd213ps ymm3, ymm2, ymm0
    vdivps ymm1, ymm4, ymm3

    ; if the mask is set in ymm9, set the output to 0
    vblendvps ymm1, ymm1, ymm7, ymm9
    
    ; copy the sign bit from the input
    vandnps ymm5, ymm5, ymm10
    vorps ymm2, ymm1, ymm5

    vmovups [rdi], ymm2
    ret


; rdi = &x, rsi = &y, rdx=&xmin, rcx=&xmax, r8=&dists
; calculate the closest distance from (x,y) to the parabola y=x^2. Only consider the interval [xmin, xmax], so if the closest point falls outside that interval, instead take the min of the distances to the 2 endpoints
serafont_distance_squared_to_parabola:
    
    ; calc s. s might be NaN if y < 1/2.
    ; the routine does not use ymm11 or ymm10, so we can reserve those

    ; stack space for result aggregation
    sub rsp, 64

    vxorps ymm1, ymm1
    vmovups ymm3, [rsi] ; load y ; todo: can we reserve this?
    vbroadcastss ymm4, [negative_one_sixth]
    vfmadd231ps ymm4, ymm3, [one_third]
    vcmpgeps ymm10, ymm4, ymm1 ; mask is true when the value of s makes sense. if s is NaN, all intervals are equivalent and there is one root
    vsqrtps ymm11, ymm4 ; ymm11 = s

    ; if ymm10 is set, use the interval [-s, s], clamped by [xmin, xmax]. otherwise use [xmin, xmax]
    vbroadcastss ymm7, [rdx]
    vbroadcastss ymm8, [rcx]
    vsubps ymm14, ymm1, ymm11 ; -s
    vmaxps ymm14, ymm7
    vminps ymm14, ymm8
    vmaxps ymm15, ymm11, ymm7
    vminps ymm15, ymm15, ymm8

    ; if s is NaN, mask out the interval by replacing it with [xmin, xmax]
    vblendvps ymm14, ymm7, ymm14, ymm10
    vblendvps ymm15, ymm8, ymm15, ymm10

    call .routine
    vmovups [rsp], ymm0

    ; lower interval, [-inf to -s], clamped by [xmin, xmax]. so really, it's clamped below by xmin in all cases. 
    vxorps ymm1, ymm1
    vbroadcastss ymm14, [rdx]
    vbroadcastss ymm8, [rcx]
    vsubps ymm15, ymm1, ymm11 ; -s

    vmaxps ymm15, ymm14
    vminps ymm15, ymm8

    vblendvps ymm15, ymm8, ymm15, ymm10
    
    call .routine
    vmovups [rsp + 32], ymm0

    ; upper interval, [s, inf] clamped by [xmin, xmax]. so really, it's clamped above by xmax in all cases. 
    vbroadcastss ymm7, [rdx]
    vbroadcastss ymm15, [rcx]

    vmaxps ymm14, ymm11, ymm7
    vminps ymm14, ymm14, ymm15

    vblendvps ymm14, ymm7, ymm14, ymm10
    
    call .routine
    vminps ymm0, [rsp]
    vminps ymm0, [rsp + 32]

    ; min with the 2 endpoints

    vbroadcastss ymm1, [rdx]
    vbroadcastss ymm2, [rcx]
    vmovups ymm7, [rdi]; x
    vmovups ymm8, [rsi]; y
    vmovups ymm5, ymm1
    vmovups ymm6, ymm2
    vsubps ymm5, ymm7
    vsubps ymm6, ymm7
    vfmsub213ps ymm1, ymm1, ymm8
    vfmsub213ps ymm2, ymm2, ymm8
    vmulps ymm5, ymm5
    vmulps ymm6, ymm6
    vfmadd213ps ymm1, ymm1, ymm5
    vfmadd213ps ymm2, ymm2, ymm6
    vminps ymm0, ymm1
    vminps ymm0, ymm2

    add rsp, 64
    vmovups [r8], ymm0
    ret


.routine:
    ; load constants

    vbroadcastss ymm6, [six]
    vbroadcastss ymm7, [negative_two]
    vbroadcastss ymm8, [one]

    ; assumes ymm14 is the lower bound of the interval (interval is defined as the section of the parabola, also bounded by one of the intervals)  (-∞, -s) (-s, s) or (s, ∞). if y < 1/2, this is instead just the parabola section
    ; assumes ymm15 is the upper bound of that same interval
    ; initial guess is the midpoint.

    vbroadcastss ymm5, [one_half]
    vaddps ymm1, ymm14, ymm15
    vmulps ymm1, ymm5

    vmovups ymm0, [rdi]

    vxorps ymm5, ymm5
    vmovups ymm3, [rsi] ; load y


    ; ymm0 = x
    ; ymm1 = k
    ; ymm3 = y
    ; ymm6 = 6
    ; ymm7 = -2
    ; ymm8 = 1
    ; avail = ymm4,5,9,12-15
    ; iterative step k_next = clamp(k - (2k^3 + (1-2y)k - x)*(6k^2+1-2y) / ((6k^2+1-2y)^2 - (2k^3 + (1-2y)k - x)*6k)) between xmin and xmax
    ;working up the factors, we need
    ; 6k
    ; 6k^2
    ; 2k^3
    ; 1-2y
    ; 6k^2 + 1 - 2y
    ; (1-2y)k-x

    mov rax, 3 ; 3 iterations

    ; load the endpoints ahead of time
    vbroadcastss ymm2, [nosignconst]

    vmovups ymm9, ymm3 ; ymm9 = y
    vfmadd213ps ymm9, ymm7, ymm8 ; ymm9 = 1-2y
    vmovups ymm13, ymm9; ymm13 = (1-2y)

.iteration:
    vmulps ymm4, ymm1, ymm6 ; ymm4 = 6k
    vmulps ymm12, ymm1, ymm1 ; ymm12 = k^2
    vmulps ymm5, ymm4, ymm1 ; ymm5 = 6k^2
    vmulps ymm12, ymm1 ; ymm12 = k^3
    vmovups ymm9, ymm13
    vaddps ymm12, ymm12 ; ymm12 = 2k^3
    vfmsub213ps ymm9, ymm1, ymm0 ; ymm9 = (1-2y)k-x

    ; ymm0 = x
    ; ymm1 = k
    ; ymm2 = absolute value mask
    ; ymm3 = y
    ; ymm4 = 6k
    ; ymm5 = 6k^2
    ; ymm6 = 6
    ; ymm7 = -2
    ; ymm8 = 1
    ; ymm9 = (1-2y)k-x
    ; ymm10 = /reserved by outer loop
    ; ymm11 = /reserved by outer loop
    ; ymm12 = 2k^3
    ; ymm13 = 1-2y
    ; ymm14 = xmin
    ; ymm15 = xmax

    ;the larger terms are (2k^3 + (1-2y)k - x) & (6k^2+1-2y)
    vaddps ymm12, ymm9 ; (2k^3 + (1-2y)k - x)
    vxorps ymm9, ymm9
    vaddps ymm5, ymm13 ; (6k^2+1-2y)
    vmulps ymm4, ymm12 ; (2k^3 + (1-2y)k - x)*6k
    vmulps ymm12, ymm12, ymm5 ; ymm12 = numerator
    vfmsub231ps ymm4, ymm5, ymm5; ymm5 = denominator
    
    ; ymm5, ymm9 are usable again. Use this to assure denominator doesn't get too low.

    vbroadcastss ymm9, [small_number]
    vandps ymm5, ymm2, ymm4
    vmaxps ymm5, ymm9
    vandnps ymm9, ymm2, ymm4 ; get the sign bit
    vorps ymm4, ymm5, ymm9

    vdivps ymm12, ymm4
    vsubps ymm1, ymm12 ; k - numerator / denom
    
    vmaxps ymm1, ymm1, ymm14 ; max (result, xmin)
    vminps ymm1, ymm1, ymm15 ; min (result, xmax)
    
    dec rax
    jnz .iteration

    ; this a point. (k,k^2)
    
    ; calc k^2 - y
    vmovups ymm4, ymm1
    vfmsub213ps ymm4, ymm4, ymm3

    ; calc x - k
    vsubps ymm0, ymm1

    ; square both and add
    vmulps ymm0, ymm0             ; ymm0 = (x - k)^2
    vfmadd231ps ymm0, ymm4, ymm4  ; ymm0 = (k^2 - y) * (k^2 - y) + (x - k)^2

    ; return is in ymm0
    ret


; RDI,  RSI,  RDX, RCX,       R8
; &pxs, &pys, &op, &windings, &dists

serafont_fractional_winding_bezier:

    call serafont_fractional_winding_line_segment
    ; dists is going to go unused, but that's fine. The winding takes so much more compute anyway

    ; op starts with a 2x3 column-major affine matrix. We need to load the coefficients then multiply {x,y} by it.
    ; leftovers from the call:
    ; ymm12 = ab x ap = query_side

    vmovups ymm13, [rdi]
    vmovups ymm14, [rsi]
    vbroadcastss ymm2, [rdx + $00]
    vbroadcastss ymm5, [rdx + $04]
    vbroadcastss ymm3, [rdx + $08]
    vbroadcastss ymm6, [rdx + $0C]
    vbroadcastss ymm4, [rdx + $10]
    vbroadcastss ymm7, [rdx + $14]

    vbroadcastss ymm11, [rdx + $34] ; ymm11 = curvature_sign
    vbroadcastss ymm10, [epsconst] ; ymm10 = EPS

    vbroadcastss ymm15, [signconst]
    vbroadcastss ymm8, [nosignconst]

    vmulps ymm0, ymm2, ymm13
    vmulps ymm1, ymm5, ymm13

    vmovups ymm5, [rcx] ; ymm5 = winding

    vfmadd231ps ymm0, ymm14, ymm3
    vfmadd231ps ymm1, ymm14, ymm6

    vbroadcastss ymm6, [one]

    vaddps ymm0, ymm4
    vaddps ymm1, ymm7

    vmovups [rdi], ymm13
    vmovups [rsi], ymm14

    ; ymm0 = translated x
    ; ymm1 = translated x. we no longer care about the matrix or original x & y

    ; first calculate the winding

    ; abs(query_side) < epsconst ? copy_sign(winding, -op.curvature_sign)
    ; : translated_x * translated_x < translated_y && sgn(query_side) == op.curvature_sign ? winding - sgn(winding)
    ; : winding

    vmulps ymm2, ymm0, ymm0 ; ymm2 = x^2
    vandps ymm3, ymm8, ymm12 ; ymm3 = abs(query_side)

    ; this gets a bit weird. vblendvps uses the sign bit to choose, so just xoring query_side with op.curvature_sign gives us a valid mask, though it is inverted
    ; I don't care about 0s because query_side == 0 is covered by the first condition, and curvature_sign not being 0 is part of the contract of this function.
    
    vxorps ymm12, ymm11 ; ymm12 = sgn(query_side) != op.curvature_sign
    vcmpltps ymm4, ymm2, ymm1 ; ymm4 = x^2 < y
    vcmpltps ymm3, ymm3, ymm10 ; ymm3 = abs(query_side) < epsconst

    ; registers that matter:
    ; ymm0 = x
    ; ymm1 = y
    ; ymm3 = mask[abs(query_side) < epsconst]
    ; ymm4 = mask[x^2 < y]
    ; ymm5 = winding
    ; ymm6 = 1
    ; ymm8 = abs mask
    ; ymm11 = curvature_sign
    ; ymm12 = ~mask[sgn(query_side) == op.curvature_sign]
    ; ymm15 = sign mask

    ; calc winding - sgn(winding)
    vxorps ymm7, ymm7
    vsubps ymm7, ymm6 ; ymm6 = 1, ymm7 = -1
    vblendvps ymm7, ymm7, ymm6, ymm5 ; ymm7 = -sgn(winding)
    vaddps ymm7, ymm5 ; ymm7 = winding - sgn(winding)

    ; calc final clause x^2 < y && sgn(query_side) == op.curvature_sign
    vandnps ymm12, ymm4
    vblendvps ymm12, ymm5, ymm7, ymm12 ; ymm12 = x^2 < y && sgn(query_side) == op.curvature_sign ? winding - sgn(winding) : winding

    ; calc copy_sign(winding, -curvature_sign)
    
    vandps ymm5, ymm8 ; abs(winding)
    vandnps ymm7, ymm11, ymm15 ; sign bit of -curvature_sign
    vorps ymm5, ymm7

    ; select based on ymm3 mask
    vblendvps ymm3, ymm12, ymm5, ymm3
    vmovups [rcx], ymm3

    ;to calc distances, we need these set up, where xmin and xmax are scalar, rest vector
    ;rdi = &x, rsi = &y, rdx=&xmin, rcx=&xmax, r8=&dists

    sub rsp, 96
    vmovups [rsp + 32], ymm0
    lea rdi, [rsp + 32]
    vmovups [rsp + 0], ymm1
    lea rsi, [rsp + 0]

    ; the only thing we need from op after loading xmin and xmax is the scale factor, which is in position 10. min and max are in 11 and 12.

    vbroadcastss ymm0, [rdx + 40]
    lea rcx, [rdx + 48]
    add rdx, 44
    vmulps ymm0, ymm0; the scale factor needs to be squared because distances returned by serafont_distance_squared_to_parabola are squared.
    vmovups [rsp + 64], ymm0 ; scale factor

    call serafont_distance_squared_to_parabola
    ; distances are now in [r8], but also in ymm0.
    vmulps ymm0, [rsp + 64]
    add rsp, 96
    vmovups [r8], ymm0

    ret