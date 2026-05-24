#define INFINITY 3.4e38
#define int64_t long long
#define EPS 1e-2
#define EPS2 (EPS * EPS)
#define TAU 6.2831853071795864769252867665590057683943387987502116419498891846f
#define SQRT_3 1.7320508075688772935274463415058723669428052538103806280558069794f //my source is that I made it the fuck up
#define ONE_OVER_27 0.0370370370370370370370370370370370370370370370370370370370370370f
#define ONE_THIRD 0.333333333333333333333333333333333333333333333333333333333333f

struct Mat2x3 {
    float m00, m10, m01, m11, m02, m12;
    __host__ __device__ Mat2x3() {}
    __host__ __device__ Mat2x3(float _m00, float _m10, float _m01, float _m11, float _m02, float _m12)
        : m00(_m00), m10(_m10), m01(_m01), m11(_m11), m02(_m02), m12(_m12) {}

    __host__ __device__
    float2 operator*(const float2& p) const {
        float2 r;
        r.x = m00 * p.x + m01 * p.y + m02;
        r.y = m10 * p.x + m11 * p.y + m12;
        return r;
    }
};

__host__ __device__ __forceinline__ float2 operator+(const float2 a, const float2 b) {
    return {a.x + b.x, a.y + b.y};
}
__host__ __device__ __forceinline__ float2 operator-(const float2 a, const float2 b) {
    return {a.x - b.x, a.y - b.y};
}
__host__ __device__ __forceinline__ float2 operator*(const float2 a, const float b) {
    return {a.x * b, a.y * b};
}
__host__ __device__ __forceinline__ float dot(const float2 a, const float2 b) {
    return a.x * b.x + a.y * b.y;
}
__host__ __device__ __forceinline__ float cross(const float2 a, const float2 b) {
    return a.x*b.y-a.y*b.x;
}
__host__ __device__ __forceinline__ float distance(const float2 a, const float2 b) {
    float2 diff = a-b;
    return sqrtf(dot(diff,diff));
}
__host__ __device__ __forceinline__ float sign(const float x) {
    return (float)((0.0f < x) - (x < 0.0f));
}
__host__ __device__ __forceinline__ float dist2(const float2 p, const float x) {
    float2 d = {x,x*x};
    d = d - p;
    return dot(d,d);
}
__host__ __device__ __forceinline__ float clamp(const float x, const float minimum, const float maximum) {
    return fminf(fmaxf(x, minimum), maximum);
}
__host__ __device__ __forceinline__ float3 operator+(const float3 a, const float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__host__ __device__ __forceinline__ float3 operator*(const float3 a, const float f) {
    return {a.x * f, a.y * f, a.z * f};
}


// Solves 2x^3 + 2Bx + 2C = 0 (depressed cubic).
// The 2 coefficients are there just to avoid a multiply
// Returns number of real roots and fills output array.
__host__ __device__ int solve_cubic(const float B, const float C, float (*roots)[3]) {
    float p = B;
    float p_cubed = p*p*p;
    float p_cubed_over_27 = p_cubed * ONE_OVER_27;
    float q = C;
    float D = (q*q)*.25 + p_cubed_over_27;
    float nq_half = q * -.5;
    if (D > 0) {
        float sqrtfD = sqrtf(D);
        float u = cbrtf(nq_half + sqrtfD);
        float v = cbrtf(nq_half - sqrtfD);
        (*roots)[0] = u + v;
        return 1;
    }
    if(p >= 0) {
        (*roots)[0] = cbrtf(-q);
        return 1;
    }
    float r = sqrtf(-p_cubed_over_27);
    float phi = acosf(clamp(nq_half / r, -1, 1));
    float half_of_T = sqrtf(p * -ONE_THIRD);
    float theta = phi * ONE_THIRD;
    float s, c;
    sincosf(theta, &s, &c);

    (*roots)[0] = half_of_T * c * 2;
    (*roots)[1] = half_of_T * (- SQRT_3*s - c);
    (*roots)[2] = half_of_T * (+ SQRT_3*s - c);

    return 3;
}

//returns distance^2
__host__ __device__ float distance_squared_to_parabola(const float2 p, const float xmin, const float xmax) {
    // scale the point and x-range so the parabola is y = x^2
    // y = x^2
    float roots[3] = {0,0,0};
    int n = solve_cubic(0.5 - p.y, -0.5 * p.x, &roots);

    float best = INFINITY;

    for (int ind = 0; ind < n; ind++) {
        float root = roots[ind];
        if (xmin <= root && root <= xmax) {
            float d = dist2(p, root);
            best = min(best, d);
        }
    }
    best = min(best, min(dist2(p, xmin), dist2(p, xmax)));

    return best;
}

//bits of the index are scattered
// input is grouped into 32x32 chunks. blockDim.x == 32, and blockDim.y == 32
// blockIdx.x uses 30 bits, grouped into 15.
// together, that makes 20 bits for both X and Y directions, which maxes out at 2^20 aka, 1M x 1M pixels.
struct RGBPixel {unsigned char r,g,b;};


struct BGRAPixel {unsigned char b,g,r,a;};

struct GlyphContour {
    float2 minima, maxima;
    size_t start_index, end_index;
};

struct GlyphOp {
    Mat2x3 parabolic_affine; // 00 10 01 11 20 21
    float2 a, c;
    float inv_ysq_coefficient, endpoints_min, endpoints_max, curvature_sign;
};

struct GlyphInstance {
    float2 minima, maxima;
    size_t start_index, end_index;
    float2 baseline_point;
    struct BGRAPixel color;
};

#define OUTPUT_FORMAT_RGB 0
#define OUTPUT_FORMAT_BGRA 1

extern "C" {

__device__ float2 fractional_winding_line_segment(float2 p, float2 a, float2 b) {
    float2 ab = b-a;
    float2 ap = p-a;
    float2 bp = p-b;
    float dist_squared, winding;
    float dist2 = dot(ab,ab);

    // edge for both degenerate line segments and beziers: if a==b. Unfortunately this costs a lot of the runtime for some reason
    // you can uncomment it if you care about this.
    
    // if (dist2 < EPS) {
    //     return {0, distance(p, a)};
    // }
    float ab_x_ap = cross(ab,ap);
    dist_squared = ab_x_ap * ab_x_ap / dist2;
    winding = -1.0f/(TAU) * atan2f(cross(bp, ap), dot(bp,ap));

    //test if the closest point is an endpoint

    if (dot(ab, bp) > 0 || dot(ab, ap) < 0) {
        dist_squared = fminf(dot(ap,ap), dot(bp,bp));
    }

    return {winding, dist_squared};
}

__device__ float2 fractional_winding_bezier(float2 p, GlyphOp op) {
    // if p is outside triangle ABC, the bezier ABC can be continuously deformed to line segment AC without passing through p, thus the winding is equivalent.
    // The winding is not equivalent if p lies between AC and the bezier.
    // if p lies between AC and the bezier, wrapping goes the other way around, 1 unit off what it would otherwise be.
    // thus the winding is modified by winding_real = winding_ac + -sgn(winding_ac)
    float2 provisional_winding_distance_to_ac = fractional_winding_line_segment(p, op.a, op.c);
    float &provisional_winding = provisional_winding_distance_to_ac.x;
    // float &distance_to_ac = provisional_winding_distance_to_ac.y;
    float winding = 0;
    // here we would check if the bezier is too linear to do accurate math using its curvature. But the compilation step turns near-linear beziers into lines anyway, so we can skip that.
    // if (1 / op.inv_ysq_coefficient < EPS) {
    //     return {provisional_winding, distance_to_ac};
    // }

    // this transform rotates the parabola to the origin, makes it point upward, then scales both x and y such that the x and x scaling are equal, and the parabola shifts to the y = x^2 parabola.
    // because it is scaled, we need to do the inverse scale when we calculate distances
    float2 p_translated = op.parabolic_affine * p;

    float query_side = cross(op.c - op.a, p - op.a);
    bool query_and_control_same_side = (sign(query_side) == op.curvature_sign);

    if (fabsf(query_side) < EPS) {
        // There is a degenerate case: When AP and AC are approximately collinear, or exactly collinear, the sign of the linear winding becomes unstable. 
        // In that case, the absolute value is going to be really close to 0.5. what we need to do to determine the sign is see whether the bezier itself curves left or right
        winding = copysignf(provisional_winding,  -op.curvature_sign);
    } else if (p_translated.x * p_translated.x < p_translated.y && query_and_control_same_side) {
        // if x^2 < y, we are inside the parabola.
        // whether we're inside the parabola or not only matters if we are inside the control triangle, which is what query_and_control_same_side determines.
        winding = provisional_winding - sign(provisional_winding);
    } else {
        winding = provisional_winding;
    }
    
    // limit the area of the parabola we consider to the region between a and c's (transformed) positions
    // scale distance back to the original coordinate system by dividing by the original ysq_coefficient
    float distance_squared = distance_squared_to_parabola(p_translated, op.endpoints_min, op.endpoints_max) * op.inv_ysq_coefficient * op.inv_ysq_coefficient;
    return {winding, distance_squared};
}

__forceinline__ __device__ bool contained_in(float2 point, GlyphContour contour, float leeway) {
    return point.x <= contour.maxima.x + leeway && point.y <= contour.maxima.y + leeway &&
           point.x >= contour.minima.x - leeway && point.y >= contour.minima.y - leeway;
}

__global__ void render_compiled_ttf_string(
        const char * const contours_and_ops,
        size_t num_contours,
        int64_t w,
        int64_t h,
        char * __restrict__ output_pixels,
        int64_t output_byte_stride,
        int output_format,
        float scale,
        struct BGRAPixel defaultColor,
        struct BGRAPixel outlineColor,
        float antialiasing,
        float outline_thickness,
        struct GlyphInstance *glyphs,
        size_t num_glyphs
    ) {
    int64_t pixel_index_x = threadIdx.x + blockIdx.y * 32;
    int64_t pixel_index_y = threadIdx.y + blockIdx.z * 32;

    if(!(pixel_index_x < w && pixel_index_y < h)) {
        return;
    }

    float leeway = outline_thickness + antialiasing;

    output_pixels += output_byte_stride * pixel_index_y;
    output_pixels += (output_format == 0 ? 3 : 4) * pixel_index_x;
    struct RGBPixel *rgb_output = (struct RGBPixel*)output_pixels;
    struct BGRAPixel *bgra_output = (struct BGRAPixel*)output_pixels;

    float2 xy = {(float)pixel_index_x, (float)(h-pixel_index_y)};
    float3 outline_color = {((float)outlineColor.r) / 255.0f, ((float)outlineColor.g) / 255.0f, ((float)outlineColor.b) / 255.0f};

    const struct GlyphContour *contours = (const struct GlyphContour *)contours_and_ops;
    const struct GlyphOp *ops = (const struct GlyphOp *)(contours_and_ops + (num_contours * sizeof(GlyphContour)));

    float3 contribution_color = {0,0,0};
    float net_alpha = 0;
    float transparency_fix = 0;

    for(size_t glyph_index = 0; glyph_index < num_glyphs; glyph_index += 1) {
        const struct GlyphInstance glyph = glyphs[glyph_index];
        float2 sample_point = (xy * (1/scale) -glyph.baseline_point) + float2{0.5,0.5} ;
        if(contained_in(sample_point, *(struct GlyphContour *)&glyph, leeway)) {
            float total_winding = 0;
            float min_distance = INFINITY;
            for(size_t contour_index = glyph.start_index; contour_index < glyph.end_index; contour_index += 1) {
                const struct GlyphContour contour = contours[contour_index];
                if(contained_in(sample_point, contour, leeway))
                for(size_t net_op_index = contour.start_index; net_op_index < contour.end_index; net_op_index += 1) {
                    GlyphOp op = ops[net_op_index];
                    float2 subwinding_subdistance;
                    if (0 == op.inv_ysq_coefficient) {
                        subwinding_subdistance = fractional_winding_line_segment(sample_point, op.a, op.c);
                    } else {
                        subwinding_subdistance = fractional_winding_bezier(sample_point, op);
                    }
                    total_winding += subwinding_subdistance.x;
                    min_distance = min(subwinding_subdistance.y, min_distance);
                }
            }
            if (roundf(total_winding) != 0.0f) {
                min_distance = 0;
            } else {
                min_distance = sqrtf(min_distance);
            }
            if (min_distance <= leeway) {
                float outline_alpha, inline_alpha;
                float glyph_alpha = ((float)glyph.color.a) / 255.0f;
                if (antialiasing == 0) {
                    inline_alpha = (float)(min_distance <= 0);
                    outline_alpha = ((float)(min_distance <= outline_thickness)) - inline_alpha;
                } else {
                    float falloff;
                    falloff = clamp(min_distance / antialiasing, 0.0, 1.0);
                    inline_alpha = 1.0f - falloff;
                    outline_alpha = falloff - clamp((min_distance - outline_thickness) / antialiasing, 0.0, 1.0);
                }

                float3 this_glyphs_color = {((float)glyph.color.r) / 255.0f, ((float)glyph.color.g) / 255.0f, ((float)glyph.color.b) / 255.0f};
                float3 color_with_outline = this_glyphs_color * inline_alpha + outline_color * outline_alpha;
                float net_alpha_this_glyph = inline_alpha + outline_alpha;

                contribution_color = color_with_outline * glyph_alpha + (contribution_color * (1 - net_alpha_this_glyph * glyph_alpha));
                net_alpha = 1-(1-net_alpha)*(1-net_alpha_this_glyph);
                transparency_fix = 1-(1-transparency_fix)*(1-glyph_alpha);
            }
        }
    }

    if(net_alpha == 0) {return;}

    float inline_alpha = ((float)defaultColor.a) / 255;

    float3 original = {((float)rgb_output->r), ((float)rgb_output->g), ((float)rgb_output->b)};
    if(output_format == 1) {
        original = {((float)bgra_output->r), ((float)bgra_output->g), ((float)bgra_output->b)};
    }
    float3 new_pixel = (contribution_color * (inline_alpha / transparency_fix)) * 255.0 + original * (1 - inline_alpha * net_alpha);

    if(output_format == 0) {
        *rgb_output = {(unsigned char)new_pixel.x, (unsigned char)new_pixel.y, (unsigned char)new_pixel.z};
    } else {
        *bgra_output = {(unsigned char)new_pixel.x, (unsigned char)new_pixel.y, (unsigned char)new_pixel.z,  bgra_output->a};
    }
    
}

}