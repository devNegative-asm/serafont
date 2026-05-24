package serafont_ttf

import "base:runtime"
import "core:log"
import "core:slice"
import "core:reflect"
import "core:mem"
import "core:time"
import "core:math/linalg"
import "core:math"
import "core:simd"
import "core:fmt"
import "../odin-cuda-bindings/cuda"
import "vendor:kb_text_shape"

_EPS :: 1e-2
_EPS2 :: _EPS * _EPS
math_settings :: runtime.Fast_Math_Flags{.Allow_Reassoc, .No_NaNs, .No_Signed_Zeros, .Allow_Reciprocal, .Allow_Contract, .Approx_Func}
_contained_in_range :: proc (y, x0, x1: f32, minima, maxima: [2]f32, leeway: f32) -> bool {
    return x0 <= maxima.x + leeway && y <= maxima.y + leeway &&
           x1 >= minima.x - leeway && y >= minima.y - leeway
}

@(fast_math=math_settings)
_distance_squared_to_parabola_vec_steps :: proc "c" (
    x, y: #simd[8]f32,
    xmin, xmax: #simd[8]f32,
) -> #simd[8]f32 {
    k := (xmin + xmax) / 2
    for i in 0..<3 { // 2 steps is probably fine, but 1 definitely isn't and 2 might not be either. You can lower this to 2 if you want the speed
        f6k := 6*k
        f2ksq := 2*k*k
        f1m2y := (1-2*y)
        f1m2ykma := f1m2y * k - x
        f2 := simd.fma(f6k,k,f1m2y)
        f1 := simd.fma(f2ksq, k, f1m2ykma)
        denom := simd.fused_mul_sub(f2,f2, f1 * f6k)
        num := f1*f2

        // todo: sanity check that this ever happens? I don't think it will, but it's taking up a significant part of the runtime
        // simd_eps :#simd[8]f32 = 0.00000001
        // denom = simd.copysign(simd.max(simd.abs(denom), simd_eps), denom)
        step := num / denom
        k = simd.clamp(k - step, xmin, xmax)
    }
    return k
}
// finds the squared distance from the given point to the closest point that lies on the parabola y=x^2 within the segment where xmin<=x<=xmax
@(fast_math=math_settings)
_distance_squared_to_parabola_vec :: proc "c" (
    px, py: #simd[8]f32,
    xmin, xmax: f32
) -> #simd[8]f32 {
    _ONE_THIRD :: 0.333333333333333333333333333333333333333333333333333333333333
    _NEG_ONE_SIXTH :: -0.166666666666666666666666666666666666666666666666666666666666
    third :#simd[8]f32= _ONE_THIRD
    neg_sixth :#simd[8]f32= _NEG_ONE_SIXTH
    xmin_vec :#simd[8]f32 = xmin
    xmax_vec :#simd[8]f32 = xmax
    s2 := simd.fma(py, third, neg_sixth)
    valid := simd.lanes_ge(s2, 0)
    s := simd.sqrt(s2)

    interval_low  := simd.select(valid, simd.clamp(-s, xmin_vec, xmax_vec), xmin_vec)
    interval_high := simd.select(valid, simd.clamp( s, xmin_vec, xmax_vec), xmax_vec)

    low := _distance_squared_to_parabola_vec_steps(px, py, xmin_vec, interval_low)
    mid := _distance_squared_to_parabola_vec_steps(px, py, interval_low, interval_high)
    high := _distance_squared_to_parabola_vec_steps(px, py, interval_high, xmax_vec)

    dx1 := low - px
    dx2 := mid - px
    dx3 := high - px
    dx4 := xmin_vec - px
    dx5 := xmax_vec - px

    dy1 := simd.fused_mul_sub(low, low, py)
    dy2 := simd.fused_mul_sub(mid, mid, py)
    dy3 := simd.fused_mul_sub(high, high, py)
    dy4 := simd.fused_mul_sub(xmin_vec, xmin_vec, py)
    dy5 := simd.fused_mul_sub(xmax_vec, xmax_vec, py)
    d1 := simd.fma(dy1, dy1, dx1 * dx1)
    d2 := simd.fma(dy2, dy2, dx2 * dx2)
    d3 := simd.fma(dy3, dy3, dx3 * dx3)
    d4 := simd.fma(dy4, dy4, dx4 * dx4)
    d5 := simd.fma(dy5, dy5, dx5 * dx5)

    return simd.min(
        simd.min(d1, d2),
        simd.min(simd.min(d3, d4), d5),
    )
}

// calculates the fractional winding of a line segment described in `op` around the point px,py, and returns that winding, as well as the shortest squared distance from the line to the point
@(fast_math=math_settings)
_fractional_winding_line_segment_vec :: proc "c" (
    px, py: #simd[8]f32,
    op: _GlyphOperation,
) -> (windings, dists: #simd[8]f32) {

    //constants for approximating atan2
    atan_a1  ::  0.99997726
    atan_a3  :: -0.33262347
    atan_a5  ::  0.19354346
    atan_a7  :: -0.11643287
    atan_a9  ::  0.05265332
    atan_a11 :: -0.01172120
    pi_const            ::  3.14159265358979323846
    pi_over_2_const     ::  1.57079632679489661923
    neg_pi_over_2_const :: -1.57079632679489661923
    neg_pi_const        :: -3.14159265358979323846
    radians_to_winding  :: -0.15915494309189535

    zero : #simd[8]f32 = 0
    pi   : #simd[8]f32 = pi_const
    pi_2 : #simd[8]f32 = pi_over_2_const

    a := op.a
    b := op.c
    ab := b - a

    ap_x := px - a.x
    ap_y := py - a.y
    bp_x := px - b.x
    bp_y := py - b.y

    dist2 := linalg.dot(ab, ab)
    cross := ab.x * ap_y - ab.y * ap_x
    dists  = cross * cross / dist2

    override_dist_mask := simd.bit_or(
        simd.lanes_gt(ab.x * bp_x + ab.y * bp_y, zero),
        simd.lanes_lt(ab.x * ap_x + ab.y * ap_y, zero)
    )

    dists = simd.select(
        override_dist_mask,
        simd.min(ap_x * ap_x + ap_y * ap_y, bp_x * bp_x + bp_y * bp_y),
        dists,
    )

    atan2_y := bp_x * ap_y - bp_y * ap_x
    atan2_x := bp_x * ap_x + bp_y * ap_y
    
    swap_mask := simd.lanes_lt(simd.abs(atan2_x), simd.abs(atan2_y))
    
    ratio := simd.select(
        swap_mask,
        atan2_x / atan2_y,
        atan2_y / atan2_x,
    )
    
    r2 := ratio * ratio
    r4 := r2 * r2
    r8 := r4 * r4
    
    p0 := simd.fma(r2, atan_a3,  atan_a1)
    p1 := simd.fma(r2, atan_a7,  atan_a5)
    p2 := simd.fma(r2, atan_a11, atan_a9)
    
    p := simd.fma(p1, r4, p0)
    p = simd.fma(p2, r8, p)
    p *= ratio
    
    p = simd.select(
        swap_mask,
        simd.select(simd.lanes_ge(ratio, 0), pi_2, -pi_over_2_const) - p,
        p,
    )
    
    p += simd.select(
        simd.lanes_lt(atan2_x, 0),
        simd.select(simd.lanes_lt(atan2_y, 0), -pi, pi_const),
        0,
    )
    
    windings = radians_to_winding * p
    return
}

// calculates the fractional winding of a quadratic bezier described in `op` around the point px,py, and returns that winding, as well as the shortest squared distance from the curve to the point
@(fast_math=math_settings)
_fractional_winding_bezier_vec :: proc(px :#simd[8]f32, py :#simd[8]f32, op: _GlyphOperation) -> (winding: #simd[8]f32, distance_squared: #simd[8]f32) {
    provisional_winding, distance_to_ac_squared := _fractional_winding_line_segment_vec(px, py, op)

    affine := op.parabolic_affine
    p_translated_x := px * affine[0,0] + py * affine[0,1] + affine[0,2]
    p_translated_y := px * affine[1,0] + py * affine[1,1] + affine[1,2]

    cma_x :#simd[8]f32 = op.c.x - op.a.x
    cma_y :#simd[8]f32 = op.c.y - op.a.y
    curvature_sign :#simd[8]f32 = op.curvature_sign
    
    query_side := cma_x * (py - op.a.y) - cma_y * (px - op.a.x)
    query_and_control_same_side := simd.lanes_eq(simd.sign_bit(query_side), simd.sign_bit(curvature_sign))
    winding = simd.select(
        simd.lanes_lt(simd.abs(query_side), _EPS),
        simd.copysign(provisional_winding, -curvature_sign),
        simd.select(simd.bit_and(query_and_control_same_side, simd.lanes_lt(p_translated_x*p_translated_x, p_translated_y)),
            provisional_winding - simd.signum(provisional_winding),
            provisional_winding,
        )
    )
    distance_squared = op.inv_ysq_coefficient * op.inv_ysq_coefficient * _distance_squared_to_parabola_vec(p_translated_x,p_translated_y,op.endpoints_min,op.endpoints_max)
    return
}

_affine_point :: proc(
    transform: matrix[2, 3]f32,
    x, y: i16,
) -> [2]f32 {
    return cast([2]f32)(transform * [3]f32{f32(x), f32(y), 1.0});
}

// converts a list of ttf control points to a compiled list of glyph operations, each representing either a line segment or bezier curve. Also returns the bounding box
_compile_glyph_operation_list :: proc(
    xs: []i16,
    ys: []i16,
    flags: []_SimpleGlyphFlags,
    ops: ^[dynamic]_GlyphOperation,
    transform: matrix[2, 3]f32,
) -> (minima:[2]f32, maxima:[2]f32, err: GlyphJitError) {
    err = .JIT_OK
    minima = {math.INF_F32, math.INF_F32}
    maxima = {math.NEG_INF_F32, math.NEG_INF_F32}

    n := len(xs)
    if n == 0 {
        return {},{},.JIT_OK
    }
    if n < 2 {
        return {},{},.JIT_NOT_ENOUGH_POINTS
    }

    first_on := -1
    for i in 0..<n {
        if .ON_CURVE_POINT in flags[i] {
            first_on = i
            break
        }
    }

    // No explicit on-curve points:
    // each point is a control point, with implied on-curve midpoints.
    if first_on == -1 {
        prev := _affine_point(transform, xs[n-1], ys[n-1])
        curr := _affine_point(transform, xs[0], ys[0])
        curr_pt := (prev + curr) * .5
        for i in 0..<n {
            next_i := i + 1
            if next_i == n {
                next_i = 0
            }
            next := _affine_point(transform, xs[next_i], ys[next_i])
            maxima.x = max(maxima.x, next.x)
            maxima.y = max(maxima.y, next.y)
            minima.x = min(minima.x, next.x)
            minima.y = min(minima.y, next.y)
            end_pt := (curr + next) * .5
            append(ops, compile_bezier(curr_pt, curr, end_pt))
            curr_pt = end_pt
            curr = next
        }
        return
    }

    start_pt := _affine_point(transform, xs[first_on], ys[first_on])
    minima = start_pt
    maxima = start_pt
    curr_pt := start_pt
    have_ctrl := false
    ctrl: [2]f32

    idx := first_on + 1
    if idx == n {
        idx = 0
    }

    for step in 1..<n {
        p := _affine_point(transform, xs[idx], ys[idx])
        maxima.x = max(maxima.x, p.x)
        maxima.y = max(maxima.y, p.y)
        minima.x = min(minima.x, p.x)
        minima.y = min(minima.y, p.y)

        is_on := .ON_CURVE_POINT in flags[idx]
        if is_on {
            if have_ctrl {
                append(ops, compile_bezier(curr_pt, ctrl, p))
                have_ctrl = false
            } else {
                d := p - curr_pt
                if linalg.dot(d,d) > _EPS {
                    append(ops, _GlyphOperation{inv_ysq_coefficient = 0, a = curr_pt, c = p})
                }
            }
            curr_pt = p
        } else {
            if have_ctrl {
                mid := (ctrl + p) * .5
                append(ops, compile_bezier(curr_pt, ctrl, mid))
                curr_pt = mid
            }
            ctrl = p
            have_ctrl = true
        }
        idx += 1
        if idx == n do idx = 0
    }

    // Close contour.
    if have_ctrl {
        append(ops, compile_bezier(curr_pt, ctrl, start_pt))
    } else {
        if linalg.length2(start_pt - curr_pt) > _EPS2 {
            append(ops, _GlyphOperation{
                inv_ysq_coefficient = 0,
                a = curr_pt,
                c = start_pt,
            })
        }
    }

    compile_bezier :: proc(a,b,c: [2]f32) -> (op: _GlyphOperation) {
        op.a = a
        op.c = c
        d := op.a + op.c - 2 * b
        dsq := linalg.dot(d,d)
        ab := b-op.a

        // if dsq is too small, this is practically just a line.
        if dsq < _EPS {
            op.inv_ysq_coefficient = 0
            return
        }

        d_modulus := math.sqrt(dsq)
        d_norm := d / d_modulus
        // s is the parameter within the bezier at which the base of the parabola lies
        s := -linalg.dot(d,ab) / dsq
        // find the base of the parabola by plugging in the parameter.
        base := (1-s)*(1-s)*op.a + 2*s*(1-s)*b + s*s*op.c
        rotation : matrix[2,2]f32 = {
            d_norm.y, -d_norm.x,
            d_norm.x, d_norm.y,
        }
        //find the coefficient of x^2
        ab_cross_d := linalg.vector_cross2(ab, d)

        //too linear to be stable
        if abs(ab_cross_d) < 1.1 {
            op.inv_ysq_coefficient = 0
            return
        }

        ysq_coefficient := d_modulus * dsq / (4 * ab_cross_d * ab_cross_d)
        curvature_sign := math.sign(linalg.cross(op.c-op.a, ab))

        op.curvature_sign = curvature_sign

        affine_offset := rotation * -base
        // parabolic_affine transforms the bezier to the parabola y=x^2.
        op.parabolic_affine =
        matrix[2,2]f32{
            ysq_coefficient, 0,
            0, ysq_coefficient,
        } * matrix[2,3]f32{
            d_norm.y, -d_norm.x, affine_offset[0],
            d_norm.x, d_norm.y, affine_offset[1],
        }
        // the x values for the a and c points after transformation. used for calculating the distance to the segment of the parabola
        ax := (op.parabolic_affine * [3]f32{op.a[0], op.a[1], 1})[0][0]
        cx := (op.parabolic_affine * [3]f32{op.c[0], op.c[1], 1})[0][0]
        op.endpoints_max = max(ax,cx)
        op.endpoints_min = min(ax,cx)
        op.inv_ysq_coefficient = 1 / (ysq_coefficient) // just flipping it so we can skip a divide later.
        return
    }

    return
}

// calculates the bounding box of a glyph
_glyph_minimax :: proc (glyphList: []_Glyph, generic_glyph: _Glyph, transform: matrix[2, 3]f32 = {1,0,0,  0,1,0}) -> (minima, maxima: [2]f32, err: GlyphJitError) {
    minima = {math.INF_F32, math.INF_F32}
    maxima = {math.NEG_INF_F32, math.NEG_INF_F32}
    err = .JIT_OK
    switch glyph in generic_glyph {
        case _CompiledGlyph:
            return {}, {}, .JIT_ALREADY_COMPILED
        case _SimpleGlyph:
            start := 0
            for i in 0..<len(glyph.xCoordinates) {
                pt := _affine_point(transform, glyph.xCoordinates[i], glyph.yCoordinates[i])
                minima = {min(minima.x, pt.x), min(minima.y, pt.y)}
                maxima = {max(maxima.x, pt.x), max(maxima.y, pt.y)}
            }
            return
        case _CompositeGlyph:
            for subcomponent in glyph.components {
                subglyph, exists := slice.get(glyphList, int(subcomponent.glyphIndex))
                if !exists {
                    return {}, {}, .JIT_REFERENCES_NONEXISTANT_GLYPH
                }
                subtransform: matrix[3,3]f32
                subtransform = {
                    subcomponent.transform[0,0], subcomponent.transform[0,1], subcomponent.transform[0,2],
                    subcomponent.transform[1,0], subcomponent.transform[1,1], subcomponent.transform[1,2],
                    0, 0, 1
                }
                smin, smax := _glyph_minimax(glyphList, subglyph, transform * subtransform) or_return
                minima = {min(minima.x, smin.x), min(minima.y, smin.y)}
                maxima = {max(maxima.x, smax.x), max(maxima.y, smax.y)}
            }
            return
    }
    return {}, {}, .JIT_UNRECOGNIZED_VARIANT
}

// compiles a glyph into a list of contours and operations. Also returns a bounding box. Each contour has a reference to which ops make it up, as well as their own bounding boxes.
_compile_glyph :: proc (contours: ^[dynamic]_GlyphContour, ops: ^[dynamic]_GlyphOperation, glyphList: []_Glyph, generic_glyph: _Glyph, transform: matrix[2, 3]f32 = {1,0,0,  0,1,0}) -> (minima, maxima: [2]f32, err: GlyphJitError) {
    minima = {math.INF_F32, math.INF_F32}
    maxima = {math.NEG_INF_F32, math.NEG_INF_F32}
    err = .JIT_OK
    switch glyph in generic_glyph {
        case _CompiledGlyph:
            // it's probably possible to compose compiled glyphs, but I don't know how to do it
            return {}, {}, .JIT_ALREADY_COMPILED
        case _SimpleGlyph:
            start := 0
            for endpoint in glyph.endPtsOfContours {
                contour := _GlyphContour{}
                flags := glyph.flags[start:endpoint + 1]
                xs := glyph.xCoordinates[start:endpoint + 1]
                ys := glyph.yCoordinates[start:endpoint + 1]
                contour.start_index = u64(len(ops))
                contour.minima, contour.maxima = _compile_glyph_operation_list(xs, ys, flags, ops, transform) or_return
                minima = {min(minima.x, contour.minima.x), min(minima.y, contour.minima.y)}
                maxima = {max(maxima.x, contour.maxima.x), max(maxima.y, contour.maxima.y)}
                start = int(endpoint + 1)
                contour.end_index = u64(len(ops))
                append(contours, contour)
            }
            return
        case _CompositeGlyph:
            for subcomponent in glyph.components {
                subglyph, exists := slice.get(glyphList, int(subcomponent.glyphIndex))
                if !exists {
                    return {}, {}, .JIT_REFERENCES_NONEXISTANT_GLYPH
                }
                subtransform: matrix[3,3]f32
                subtransform = {
                    subcomponent.transform[0,0], subcomponent.transform[0,1], subcomponent.transform[0,2],
                    subcomponent.transform[1,0], subcomponent.transform[1,1], subcomponent.transform[1,2],
                    0, 0, 1
                }
                smin, smax := _compile_glyph(contours, ops, glyphList, subglyph, transform * subtransform) or_return
                minima = {min(minima.x, smin.x), min(minima.y, smin.y)}
                maxima = {max(maxima.x, smax.x), max(maxima.y, smax.y)}
            }
            return
    }
    return {}, {}, .JIT_UNRECOGNIZED_VARIANT
}
@(fast_math=math_settings)
_sample_glyph_signed_distance_vec :: proc(
    x: ^[8]f32, y: ^[8]f32,
    instance: _GlyphInstance,
    glyph: _CompiledGlyph,
    antialiasing: f32,
) -> #simd[8]f32 {
    simdx := transmute(#simd[8]f32)x^
    simdy := transmute(#simd[8]f32)y^
    total_winding: #simd[8]f32
    min_distance: #simd[8]f32 = math.INF_F32
    sub_windings: #simd[8]f32
    sub_distances: #simd[8]f32 = math.INF_F32
    for contour in glyph.contours[instance.start_index : instance.end_index] {
        px := simdx - instance.point.x
        py := simdy - instance.point.y
        if _contained_in_range(y[0]-instance.point.y,x[0]-instance.point.x,x[7]-instance.point.x, contour.minima, contour.maxima, antialiasing) {
            for &op in glyph.operations[contour.start_index:contour.end_index] {
                if 0 == op.inv_ysq_coefficient {
                    sub_windings, sub_distances = _fractional_winding_line_segment_vec(px, py, op)
                } else {
                    sub_windings, sub_distances = _fractional_winding_bezier_vec(px, py, op)
                }
                total_winding += sub_windings
                min_distance = simd.min(sub_distances, min_distance)
            }
        }
    }
    zero : #simd[8]f32 = 0
    return simd.select(
        simd.lanes_ne(simd.nearest(transmute(#simd[8]f32)total_winding), zero),
        zero,
        simd.sqrt(transmute(#simd[8]f32)min_distance))
}


// RENDERING

OutputFormat :: enum(u32) {
    RGB,
    BGRA,
}
CpuBuffer :: struct {
    // pointer to the first pixel in a y-strided pixel buffer + max size
    data: []u8,
}
CudaBuffer :: struct {
    // pointer to the first pixel in a y-strided pixel buffer
    data: cuda.CUdeviceptr,
}
FontColor :: struct #packed {b,g,r,a: u8}
FontColorRGB :: struct #packed {r,g,b: u8}
RenderSettings :: struct {
    // pointer to the parsed font to render with
    font: ^Font,
    // selector for output pixel format
    output_format: OutputFormat,
    // scale factor, as calculated by `get_font_scale_factor_for_metrics`
    scale: f32,
    // list of colors to use to color subsequent glyphs in writing order. If this list is nil or not long enough to cover all glyphs, the rest will use the default color
    color_list: []FontColor,
    // default color to use when `color_list` runs out. Alpha blending with the background always uses this color's alpha, while alphas in the color_list only control blending between overlapping glyphs
    defaultColor: FontColor,
    // when `outline_thickness` is set, this controls the color of the outline. `outline_color`'s alpha is ignored
    outline_color: FontColor,
    // position to start drawing. Other libraries might set y to be the baseline, but here we set y to be the previous line's baseline, thus x=0,y=0 will render fully on-screen
    x, y: f32,
    // distance in bytes between the starts of consecutive horizontal rows in the output buffer
    stride: int,
    // size of the output buffer in pixels
    w,h: int,
    // rather than area coverage antialiasing, this uses blur antialiasing. this parameter affects the blur distance. for your convenience, this defaults to 1. `blur_pixel_distance = antialiasing_level == 0 ? 1 : max(0, antialiasing_level)`
    antialiasing_level: f32,
    // thickness of the outline. antialiasing_level applies both to the glyph and to its outline. 0 = no outline
    outline_thickness: f32,
    // settings related to the output buffer and timing metrics
    backend: union{
        CudaRenderSettings,
        CpuRenderSettings,
    }
}
GlyphRenderSettings :: struct {
    using settings: RenderSettings,
    glyphs: []_GlyphInstance,
    contours: []_GlyphContour,
    ops: []_GlyphOperation,
}
StringRenderSettings :: struct {
    using settings: RenderSettings,
    text: string,
}
CudaRenderSettings :: struct {
    output_buffer: CudaBuffer,
    timing_metrics: ^CudaTimingMetrics,
    stream: cuda.CUstream,
    render_kernel: cuda.CUfunction,
}
CpuRenderSettings :: struct {
    output_buffer: CpuBuffer,
    timing_metrics: ^CpuTimingMetrics
}
CudaTimingMetrics :: struct {
    start, stop: cuda.CUevent,
}
CpuTimingMetrics :: struct {
    start, stop: time.Time
}
GlyphJitError :: enum {
    JIT_OK,
    JIT_ALREADY_COMPILED,
    JIT_UNRECOGNIZED_VARIANT,
    JIT_REFERENCES_NONEXISTANT_GLYPH,
    JIT_NOT_ENOUGH_POINTS,
}
FontRenderError :: enum {
    RENDER_OK,
    RENDER_NO_DEFAULT_SETTINGS
}

RenderError :: union #shared_nil {
    cuda.CUresult,
    GlyphJitError,
    FontRenderError,
}

render_string :: proc(font: ^Font, font_data: []u8, text: string) -> RenderError {
    if font.default_settings == nil do return .RENDER_NO_DEFAULT_SETTINGS
    render_settings : StringRenderSettings = {
        settings = font.default_settings^,
        text = text,
        font = font,
    }
    return render_string_to_buffer(&render_settings, font_data)
}

calculate_string_bounding_box :: proc(font: ^Font, font_data: []u8, text: string, buffer_space: f32, scale:f32) -> (minimum, maximum: [2]f32, err: RenderError) {
    starting_min :: [2]f32{math.INF_F32, math.INF_F32}
    starting_max :: [2]f32{math.NEG_INF_F32, math.NEG_INF_F32}
    minimum = starting_min
    maximum = starting_max

    pos_x :f32= 0
    pos_y :f32= 0 + (f32(font.hhea.lineGap) + f32(font.hhea.ascender) - f32(font.hhea.descender))

    // idk, just make an allocator
    alloc: mem.Tracking_Allocator
    mem.tracking_allocator_init(&alloc, context.allocator)
    allocator := mem.tracking_allocator(&alloc)
    context.allocator = allocator
    defer {
        free_all(allocator)
        mem.tracking_allocator_destroy(&alloc)
    }
    kbts_allocator, kbts_allocator_data := kb_text_shape.AllocatorFromOdinAllocator(&allocator)

    shape_context := kb_text_shape.CreateShapeContext(kbts_allocator, kbts_allocator_data)
    kb_text_shape.ShapePushFontFromMemory(shape_context, font_data, 0)
    kb_text_shape.ShapeBegin(shape_context, kb_text_shape.direction.DONT_KNOW, kb_text_shape.language.ARABIC)
    kb_text_shape.ShapeUtf8(shape_context, text, .CODEPOINT_INDEX)
    kb_text_shape.ShapeEnd(shape_context)

    ops:= make([dynamic]_GlyphOperation)
    contours:= make([dynamic]_GlyphContour)
    CachedGlyph :: struct{
        start:u64,
        end:u64,
        minima, maxima: [2]f32
    }
    completed_compilations: map[u16]CachedGlyph

    for run in kb_text_shape.ShapeRun(shape_context) {
        run := run
        if .LINE_HARD in run.Flags {
            pos_y += (f32(font.hhea.lineGap) + f32(font.hhea.ascender) - f32(font.hhea.descender))
            pos_x = 0
        }
        for glyphptr in kb_text_shape.GlyphIteratorNext(&run.Glyphs) {
            glyph := font.glyf.glyphs[glyphptr.Id]
    
            if glyphptr.Codepoint == '\n' do continue
            gX := pos_x + f32(glyphptr.OffsetX)
            gY := pos_y + f32(-glyphptr.OffsetY)
            contour_count := u64(len(contours))
            glyphIndices: CachedGlyph
            if cached_contours, is_cached := completed_compilations[glyphptr.Id]; is_cached {
                glyphIndices = cached_contours
                // basically append_elems(&contours, contours[cached_contours.start:cached_contours.end])
                // ^unfortunately that doesn't work because in append_elems, if the list has to grow and move, the list's old memory is freed before the copy starts.
                // If you want to know why I'm not using reserve here instead, keep in mind that that will cause quadratic copies since reserve and resize both try to fit to the upper bound rather than expanding exponentially
                for i in cached_contours.start ..< cached_contours.end {
                    append(&contours, contours[i])
                }
            } else {
                // TODO: hinting needs to happen here.
                // We could precompile the glyphs to move this outside the render loop, but a hinted glyph's compiled version depends on hints,
                // Hints depend on the font size, which means the glyphs would need to recompile for a font size change.
                glyphIndices.minima, glyphIndices.maxima = _compile_glyph(&contours, &ops, font.glyf.glyphs, glyph) or_return
                completed_compilations[glyphptr.Id] = glyphIndices
            }
            i_minima := glyphIndices.minima + {gX, -gY}
            i_maxima := glyphIndices.maxima + {gX, -gY}
            minimum = {min(minimum.x, i_minima.x), min(minimum.y, i_minima.y)}
            maximum = {max(maximum.x, i_maxima.x), max(maximum.y, i_maxima.y)}

            pos_x += f32(glyphptr.AdvanceX)
            pos_y += f32(-glyphptr.AdvanceY)
        }
    }

    if minimum == starting_min && maximum == starting_max {
        return {}, {}, .RENDER_OK
    }

    return {
        (minimum.x * scale - buffer_space), (-buffer_space - maximum.y * scale)
    } , {
        (maximum.x * scale + buffer_space), (+buffer_space - minimum.y * scale)
    }, .RENDER_OK
}

render_string_to_buffer :: proc(render: ^StringRenderSettings, font_data: []u8) -> RenderError {
    pos_x := render.x / render.scale
    pos_y := (render.y - f32(render.h)) / render.scale + (f32(render.font.hhea.lineGap) + f32(render.font.hhea.ascender) - f32(render.font.hhea.descender))

    // idk, just make an allocator
    alloc: mem.Tracking_Allocator
    mem.tracking_allocator_init(&alloc, context.allocator)
    allocator := mem.tracking_allocator(&alloc)
    context.allocator = allocator
    defer {
        free_all(allocator)
        mem.tracking_allocator_destroy(&alloc)
    }
    kbts_allocator, kbts_allocator_data := kb_text_shape.AllocatorFromOdinAllocator(&allocator)

    shape_context := kb_text_shape.CreateShapeContext(kbts_allocator, kbts_allocator_data)
    kb_text_shape.ShapePushFontFromMemory(shape_context, font_data, 0)
    kb_text_shape.ShapeBegin(shape_context, kb_text_shape.direction.DONT_KNOW, kb_text_shape.language.ARABIC)
    kb_text_shape.ShapeUtf8(shape_context, render.text, .CODEPOINT_INDEX)
    kb_text_shape.ShapeEnd(shape_context)

    ops:= make([dynamic]_GlyphOperation)
    contours:= make([dynamic]_GlyphContour)
    glyphs:= make([dynamic]_GlyphInstance)
    CachedGlyph :: struct{
        start:u64,
        end:u64,
        minima, maxima: [2]f32
    }
    completed_compilations: map[u16]CachedGlyph


    for run in kb_text_shape.ShapeRun(shape_context) {
        run := run
        if .LINE_HARD in run.Flags {
            pos_y += (f32(render.font.hhea.lineGap) + f32(render.font.hhea.ascender) - f32(render.font.hhea.descender))
            pos_x = render.x / render.scale
        }
        for glyphptr in kb_text_shape.GlyphIteratorNext(&run.Glyphs) {
            glyph := render.font.glyf.glyphs[glyphptr.Id]
    
            if glyphptr.Codepoint == '\n' do continue //wtf is going on here? I'm getting a default / error glyph?
            gX := pos_x + f32(glyphptr.OffsetX)
            gY := pos_y + f32(-glyphptr.OffsetY)
            contour_count := u64(len(contours))
            glyphIndices: CachedGlyph
            if cached_contours, is_cached := completed_compilations[glyphptr.Id]; is_cached {
                glyphIndices = cached_contours
                // basically append_elems(&contours, contours[cached_contours.start:cached_contours.end])
                // ^unfortunately that doesn't work because in append_elems, if the list has to grow and move, the list's old memory is freed before the copy starts.
                // If you want to know why I'm not using reserve here instead, keep in mind that that will cause quadratic copies since reserve and resize both try to fit to the upper bound rather than expanding exponentially
                for i in cached_contours.start ..< cached_contours.end {
                    append(&contours, contours[i])
                }
            } else {
                // TODO: hinting needs to happen here.
                // We could precompile the glyphs to move this outside the render loop, but a hinted glyph's compiled version depends on hints,
                // Hints depend on the font size, which means the glyphs would need to recompile for a font size change.
                glyphIndices.minima, glyphIndices.maxima = _compile_glyph(&contours, &ops, render.font.glyf.glyphs, glyph) or_return
                glyphIndices.start = contour_count
                glyphIndices.end = u64(len(contours))
                completed_compilations[glyphptr.Id] = glyphIndices
            }
            glyphInstance := _GlyphInstance{start_index = glyphIndices.start, end_index = glyphIndices.end, color = slice.get(render.color_list, int(glyphptr.UserIdOrCodepointIndex)) or_else render.defaultColor, point = {gX, -gY}, minima = glyphIndices.minima, maxima = glyphIndices.maxima}
            index := int(glyphptr.UserIdOrCodepointIndex)

            // For RTL languages, glyph overlap has to go the other way. I want overlayed glyphs with transparency to overlay in writing order
            if index == len(glyphs) {
                append(&glyphs, glyphInstance)
            } else if index > len(glyphs) {
                resize_dynamic_array(&glyphs, index + 1)
                glyphs[index] = glyphInstance
            } else {
                glyphs[index] = glyphInstance
            }
            pos_x += f32(glyphptr.AdvanceX)
            pos_y += f32(-glyphptr.AdvanceY)
        }
    }
    renderSettings: GlyphRenderSettings = {
        settings = render.settings,
        ops = ops[:],
        contours = contours[:],
        glyphs = glyphs[:]
    }

    return _render_shaped_contours_to_buffer(&renderSettings)
}

_rgb :: proc {_rgb_fc, _rgb_rgbc}
_rgb_fc :: #force_inline proc(color: FontColor) -> [3]f32 {
    return (cast([3]f32)(transmute([4]u8)color).zyx) / 255
}
_rgb_rgbc :: #force_inline proc(color: FontColorRGB) -> [3]f32 {
    return (cast([3]f32)(transmute([3]u8)color).xyz) / 255
}


@(fast_math=math_settings)
_render_shaped_contours_to_buffer_vec :: proc(render: ^GlyphRenderSettings) -> RenderError {
    // sample_glyph_signed_distance_simd
    if render.scale == 0 {
        return .SUCCESS
    }
    if len(render.ops) == 0 || len(render.contours) == 0{
        return .SUCCESS
    }
    antialiasing: f32
    if render.antialiasing_level == 0 {
        antialiasing = 1 / render.scale
    } else if render.antialiasing_level > 0 {
        antialiasing = render.antialiasing_level / render.scale
    } else {
        antialiasing = 0
    }
    outline_thickness := render.outline_thickness / render.scale
    backend, ok := render.backend.(CpuRenderSettings); 
    if !ok do return .ERROR_NOT_SUPPORTED
    if backend.timing_metrics != nil {
        backend.timing_metrics.start = time.now()
    }
    leeway := antialiasing + outline_thickness

    simd_antialiasing : #simd[8]f32 = antialiasing
    simd_antialiasing_inv : #simd[8]f32 = 1 / antialiasing
    simd_leeway : #simd[8]f32 = leeway
    simd_outline_thickness : #simd[8]f32 = outline_thickness
    simd_one : #simd[8]f32 = 1
    simd_zero : #simd[8]f32 = 0
    outline_color := _rgb(render.outline_color)

    PixelData :: struct {
        r,g,b, net_alpha, transparency_fix: #simd[8]f16
    }
    pixelData := make([]PixelData, render.h * ((render.w + 7) >> 3))
    defer delete(pixelData)

    final_min_x := render.w - 1
    final_min_y := render.h - 1
    final_max_x := 0
    final_max_y := 0
    for glyph in render.glyphs {
        // empty glyphs can end up in the glyph list because newlines get skipped
        // it wouldn't render, but it fucks up the bounding box calculations, so skip it.
        if glyph.start_index == glyph.end_index do continue

        maxim_y := min(int(math.ceil(f32(render.h) - (glyph.minima.y - leeway + glyph.point.y) * render.scale)), render.h - 1)
        minim_y := max(int(math.floor(f32(render.h) - (glyph.maxima.y + leeway + glyph.point.y) * render.scale)), 0)
        minim_x := max(int(math.ceil((glyph.minima.x - leeway + glyph.point.x) * render.scale)), 0) & (~int(7))
        maxim_x := min(int(math.ceil((glyph.maxima.x + leeway + glyph.point.x) * render.scale)), render.w - 1)

        final_min_x = min(final_min_x, minim_x)
        final_min_y = min(final_min_y, minim_y)
        final_max_x = max(final_max_x, maxim_x)
        final_max_y = max(final_max_y, maxim_y)

        for y in minim_y..=maxim_y {
            ys := ([8]f32{} + f32(render.h) - f32(y)) / render.scale
            for x:=minim_x; x <=maxim_x; x+=8 {
                xs := ([8]f32{0,1,2,3,4,5,6,7} + f32(x)) / render.scale

                row := &pixelData[((y * render.w + 7 + x) >> 3)]
                contribution_color := [3]#simd[8]f32{ cast(#simd[8]f32)row.r, cast(#simd[8]f32)row.g, cast(#simd[8]f32)row.b }
                net_alpha := cast(#simd[8]f32)row.net_alpha
                transparency_fix := cast(#simd[8]f32)row.transparency_fix
                    
                compiled := _CompiledGlyph{
                    contours = render.contours,
                    operations = render.ops,
                }

                simd_min_distance := _sample_glyph_signed_distance_vec(&xs, &ys, glyph, compiled, leeway)

                simd_antialiasing_mask := simd.lanes_eq(simd_antialiasing, 0)
                simd_falloff := simd.clamp(simd.mul(simd_min_distance, simd_antialiasing_inv), simd_zero, simd_one)
                simd_inline_alpha := simd.select(simd_antialiasing_mask,
                    simd.select(
                        simd.lanes_le(simd_min_distance, 0),
                        simd_one,
                        simd_zero
                    ),
                    simd.sub(simd_one, simd_falloff)
                )
                simd_outline_alpha := simd.select(simd_antialiasing_mask,
                    simd.sub(simd.select(
                        simd.lanes_le(simd_min_distance, simd_outline_thickness),
                        simd_one,
                        simd_zero
                    ), simd_inline_alpha),
                    simd.sub(
                        simd_falloff, 
                        simd.clamp(simd.mul(simd_min_distance - simd_outline_thickness, simd_antialiasing_inv), simd_zero, simd_one)
                    )
                )
                simd_glyph_alpha : #simd[8]f32 = f32(glyph.color.a) / 255.0
                simd_glyph_color: [3]#simd[8]f32
                simd_glyph_color.r = f32(glyph.color.r) / 255.0
                simd_glyph_color.g = f32(glyph.color.g) / 255.0
                simd_glyph_color.b = f32(glyph.color.b) / 255.0
                simd_outline: [3]#simd[8]f32
                simd_outline.r = f32(outline_color.r) / 255.0
                simd_outline.g = f32(outline_color.g) / 255.0
                simd_outline.b = f32(outline_color.b) / 255.0
                simd_color_with_outline: [3]#simd[8]f32
                simd_color_with_outline.r = simd.fma(simd_glyph_color.r, simd_inline_alpha, simd_outline.r * simd_outline_alpha)
                simd_color_with_outline.g = simd.fma(simd_glyph_color.g, simd_inline_alpha, simd_outline.g * simd_outline_alpha)
                simd_color_with_outline.b = simd.fma(simd_glyph_color.b, simd_inline_alpha, simd_outline.b * simd_outline_alpha)

                simd_net_alpha_this_glyph := simd.add(simd_inline_alpha, simd_outline_alpha)
                // we have to mask out the results if min_distance > leeway

                mask := simd.lanes_gt(simd_min_distance, simd_leeway)
                simd_new_contribution_color: [3]#simd[8]f32

                simd_retention_alpha := simd.sub(simd_one, simd.mul(simd_net_alpha_this_glyph, simd_glyph_alpha))
                simd_new_contribution_color.r = simd.fma(simd_color_with_outline.r, simd_glyph_alpha, contribution_color.r * simd_retention_alpha)
                simd_new_contribution_color.g = simd.fma(simd_color_with_outline.g, simd_glyph_alpha, contribution_color.g * simd_retention_alpha)
                simd_new_contribution_color.b = simd.fma(simd_color_with_outline.b, simd_glyph_alpha, contribution_color.b * simd_retention_alpha)
                contribution_color.r = simd.select(mask, contribution_color.r, simd_new_contribution_color.r)
                contribution_color.g = simd.select(mask, contribution_color.g, simd_new_contribution_color.g)
                contribution_color.b = simd.select(mask, contribution_color.b, simd_new_contribution_color.b)

                //optimize: 1-(1-a)(1-b) = 1-(1-a-b+ab) = a+b-ab = a(1-b)+b
                //          (3 ops)        (4 ops)        (3 ops)  (2 ops)
                // llvm can probably figure that out, but just in case, I'll do it myself.
                simd_new_net_alpha := simd.fma(simd_net_alpha_this_glyph, simd.sub(simd_one, net_alpha), net_alpha)
                simd_new_transparency_fix := simd.fma(transparency_fix, simd.sub(simd_one, simd_glyph_alpha), simd_glyph_alpha)

                net_alpha = simd.select(mask, net_alpha, simd_new_net_alpha)
                transparency_fix = simd.select(mask, transparency_fix, simd_new_transparency_fix)

                row.transparency_fix = cast(#simd[8]f16)transparency_fix
                row.net_alpha = cast(#simd[8]f16)net_alpha
                row.r = cast(#simd[8]f16)contribution_color.r
                row.g = cast(#simd[8]f16)contribution_color.g
                row.b = cast(#simd[8]f16)contribution_color.b
            }
        }
    }
    for y in final_min_y..=final_max_y {
        for x := final_min_x; x <= final_max_x; x+=8 {
            row := pixelData[((y * render.w + 7 + x) >> 3)]
            line := backend.output_buffer.data[y * render.stride:]
            bgra_output: []FontColor
            rgb_output: []FontColorRGB
            if render.output_format == .BGRA {
                bgra_output = slice.reinterpret([]FontColor, line)[x:]
            } else {
                rgb_output = slice.reinterpret([]FontColorRGB, line)[x:]
            }
            net_alpha := cast(#simd[8]f32)row.net_alpha
            transparency_fix := cast(#simd[8]f32)row.transparency_fix
            contribution_r := cast(#simd[8]f32)row.r
            contribution_g := cast(#simd[8]f32)row.g
            contribution_b := cast(#simd[8]f32)row.b
            naf := transmute([8]f32)net_alpha
            simd_inline_alpha :#simd[8]f32 = f32(render.defaultColor.a) * (f32(1) / f32(255))
            simd_original_factor :#simd[8]f32 = simd.sub(simd_one, simd.mul(simd_inline_alpha, net_alpha))
            simd_contribution :[3][8]f32 = {
                transmute([8]f32)simd.div(simd_inline_alpha * 255 * contribution_r, transparency_fix),
                transmute([8]f32)simd.div(simd_inline_alpha * 255 * contribution_g, transparency_fix),
                transmute([8]f32)simd.div(simd_inline_alpha * 255 * contribution_b, transparency_fix),
            }
            tfix := transmute([8]f32)transparency_fix
            inline_alpha := transmute([8]f32)simd_inline_alpha
            original_factor := transmute([8]f32)simd_original_factor
            for lane in 0..<min(8, render.w - x) {
                if naf[lane] == 0 do continue
                original :[3]f32
                if render.output_format == .BGRA {
                    original = [3]f32{f32(bgra_output[lane].r),f32(bgra_output[lane].g),f32(bgra_output[lane].b)}
                } else {
                    original = [3]f32{f32(rgb_output[lane].r),f32(rgb_output[lane].g),f32(rgb_output[lane].b)}
                }
                new_pixel := [3]f32{simd_contribution.r[lane], simd_contribution.g[lane],simd_contribution.b[lane]} + original * original_factor[lane]
                if render.output_format == .RGB {
                    rgb_output[lane] = FontColorRGB{cast(u8)new_pixel.x, cast(u8)new_pixel.y, cast(u8)new_pixel.z}
                } else {
                    bgra_output[lane] = FontColor{cast(u8)new_pixel.x, cast(u8)new_pixel.y, cast(u8)new_pixel.z,  bgra_output[lane].a}
                }
            }

        }
    }

    if backend.timing_metrics != nil {
        backend.timing_metrics.stop = time.now()
    }
    return .RENDER_OK
}

@(fast_math=math_settings)
_render_shaped_contours_to_buffer :: proc(render: ^GlyphRenderSettings) -> RenderError {
    if render.scale == 0 {
        return .SUCCESS
    }
    if len(render.ops) == 0 || len(render.contours) == 0{
        return .SUCCESS
    }
    antialiasing: f32
    if render.antialiasing_level == 0 {
        antialiasing = 1 / render.scale
    } else if render.antialiasing_level > 0 {
        antialiasing = render.antialiasing_level / render.scale
    } else {
        antialiasing = 0
    }
    outline_thickness := render.outline_thickness / render.scale
    switch &backend in render.backend {
        case CpuRenderSettings:
            return _render_shaped_contours_to_buffer_vec(render)
        case CudaRenderSettings:
            number_of_contours := len(render.contours)
            number_of_glyphs := len(render.glyphs)
            ops_memory: cuda.CUdeviceptr
            glyphs: cuda.CUdeviceptr
            params : [14]rawptr = {
                &ops_memory,
                &number_of_contours,
                &render.w,
                &render.h,
                &backend.output_buffer.data,
                &render.stride,
                &render.output_format,
                &render.scale,
                &render.defaultColor,
                &render.outline_color,
                &antialiasing,
                &outline_thickness,
                &glyphs,
                &number_of_glyphs,
            }
            ops := render.ops
            contours := render.contours
            cuda.cuMemAlloc_v2(&ops_memory, size_of(_GlyphOperation) * len(ops) + size_of(_GlyphContour) * number_of_contours + size_of(_GlyphInstance) * number_of_glyphs) or_return
            defer cuda.cuMemFree_v2(ops_memory)
            cuda.cuMemcpyHtoD_v2(ops_memory, cast([^]u8)raw_data(contours[:]), size_of(_GlyphContour) * number_of_contours) or_return
            cuda.cuMemcpyHtoD_v2(ops_memory + cuda.CUdeviceptr(size_of(_GlyphContour) * number_of_contours), cast([^]u8)raw_data(ops[:]), size_of(_GlyphOperation) * len(ops)) or_return
            glyphs = ops_memory + cuda.CUdeviceptr(size_of(_GlyphContour) * number_of_contours + size_of(_GlyphOperation) * len(ops))
            cuda.cuMemcpyHtoD_v2(glyphs, cast([^]u8)raw_data(render.glyphs), size_of(_GlyphInstance) * number_of_glyphs) or_return
            gdimy := u32((render.w + 31) / 32) & 0xffff
            gdimz := u32((render.h + 31) / 32) & 0xffff
            if backend.timing_metrics != nil {
                if backend.timing_metrics.start == nil {
                    cuda.cuEventCreate(&backend.timing_metrics.start, {}) or_return
                }
                cuda.cuEventRecord(backend.timing_metrics.start, backend.stream) or_return
            }
            cuda.cuLaunchKernel(backend.render_kernel, 1, gdimy, gdimz, 32, 32, 1, 0, backend.stream, &params[0], nil) or_return
            if backend.timing_metrics != nil {
                if backend.timing_metrics.stop == nil {
                    cuda.cuEventCreate(&backend.timing_metrics.stop, {}) or_return
                }
                cuda.cuEventRecord(backend.timing_metrics.stop, backend.stream) or_return
            }
            // I tried making this whole function async and performance tanked, so we're not doing that (for now).
            cuda.cuStreamSynchronize(backend.stream)
            return .SUCCESS
    }
    return .ERROR_NOT_SUPPORTED
}

// This may or may not be useful to you. kb_text_shape does this better by parsing the string and providing you the glyph index directly
get_glyph_from_unicode_codepoint :: proc(#by_ptr font: Font, character: rune) -> (glyph: _Glyph, exists: bool, index: int){
    switch &table in font.cmap.subtable {
        case _SegmentedCoverageTable:
            index, found := slice.binary_search_by(table.groups, character, proc(seq: _SequentialMapGroup, key: rune) -> slice.Ordering {
                base := rune(u32(seq.startCharCode))
                end := rune(u32(seq.endCharCode))
                if base <= key && key <= end {
                    return .Equal
                }
                if end < key {
                    return .Less
                }
                return .Greater
            })
            if found {
                segment := table.groups[index]
                offset := int(character) - int(u32(segment.startCharCode))
                offset += int(u32(segment.startGlyphId))
                
                glyph, glyph_exists := slice.get(font.glyf.glyphs, offset)
                if !glyph_exists {
                    return slice.get(font.glyf.glyphs, 0), 0
                }
                return glyph, glyph_exists, offset
            }
            return slice.get(font.glyf.glyphs, 0), 0
        case _ByteEncodingTable:
            g, found := slice.get(table.glyphIdArray[:], int(character))
            if !found {
                return slice.get(font.glyf.glyphs, 0), 0
            }
            offset := int(g)
            glyph, glyph_exists := slice.get(font.glyf.glyphs, offset)
            if !glyph_exists {
                return slice.get(font.glyf.glyphs, 0), 0
            }
            return glyph, glyph_exists, offset
        case _SegmentMappingToDeltaValuesTable:
            index, found := slice.binary_search_by(table.endCode, character, proc(code: u16be, key: rune) -> slice.Ordering {
                end := rune(u32(code))
                if end == key {
                    return .Equal
                }
                if end < key {
                    return .Less
                }
                return .Greater
            })
            if index >= len(table.endCode) {
                return slice.get(font.glyf.glyphs, 0), 0
            }
            startCode := table.startCode[index]
            if rune(u32(startCode)) > character {
                return slice.get(font.glyf.glyphs, 0), 0
            }
            endCode := table.endCode[index]
            idDelta := table.idDelta[index]
            idRangeOffset := table.idRangeOffset[index]
            glyph_id: int
            if idRangeOffset != 0 {
                net_offset := int(idRangeOffset / 2) + index + int(character) - int(startCode) - len(table.idRangeOffset)
                glyphId, exists := slice.get(table.glyphIdArray, net_offset)
                if (!exists) || glyphId == 0 {
                    return slice.get(font.glyf.glyphs, 0), 0
                }
                glyph_id = int(glyphId + u16be(idDelta)) //defined as wrapping mod 2^16
            } else {
                glyph_id = int(u32(idDelta) + u32(character))
            }
            glyph, glyph_exists := slice.get(font.glyf.glyphs, glyph_id)
            if !glyph_exists {
                return slice.get(font.glyf.glyphs, 0), 0
            }
            return glyph, glyph_exists, glyph_id
    }
    glyph, exists = slice.get(font.glyf.glyphs, 0)
    log.error("unimplemented table type %v\n", reflect.union_variant_typeid(font.cmap.subtable))

    return glyph, false, 0
}

Millimeters :: distinct f32
Inches :: distinct f32
SizeMetrics :: struct #all_or_none {
    // the number of pixels across your screen is, eg: 1920 for full HD
    resolution: f32,
    // the point size of the text to render, ie: 12 means render at 12 points per em
    point_size: f32,
    // the width of your screen, either in mm or in. Use the `Millimeters` or `Inches` type to pick one
    size: union #no_nil {Millimeters, Inches},
}
// Takes your screen's horizontal size and horizontal pixel count, returns a scale factor from font units to pixels, which is the `scale` factor used in `RenderSettings`
get_font_scale_factor_for_metrics :: proc(font: ^Font, metrics: SizeMetrics) -> f32 {
    inches, is_inches := metrics.size.(Inches)
    inches = is_inches ? inches : Inches(metrics.size.(Millimeters) / 25.4)
    points := f32(inches * 72)
    pixels_per_point := metrics.resolution / points
    font_units_per_point := f32(font.head.unitsPerEm)
    return pixels_per_point / font_units_per_point * metrics.point_size
}
