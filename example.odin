package odinttf_example

import "core:math"
import "core:slice"
import "core:time"
import "core:os"
import "core:log"
import "core:fmt"
import "odin-cuda-bindings/cuda"
import "odin-cuda-bindings/nvrtc"
import "ttf"

write_ppm :: proc(name: string, rgb: []u8,  #any_int w,h: int) {
    f, _ := os.open(name, {.Create, .Write})
    header := fmt.tprintf("P6\n%d %d\n255\n", w,h)
    os.write(f, transmute([]u8)header)
    os.write(f, rgb)
    os.close(f)
}

fromHsva :: proc(h, s, v, a: f32) -> ttf.FontColor {
    r,g,b: f32
    i := math.floor(h * 6)
    f := h * 6 - i;
    p := v * (1 - s);
    q := v * (1 - f * s);
    t := v * (1 - (1 - f) * s);

    switch(int(i) % 6) {
        case 0:
            r = v
            g = t
            b = p
        case 1:
            r = q
            g = v
            b = p
        case 2:
            r = p
            g = v
            b = t
        case 3:
            r = p
            g = q
            b = v
        case 4:
            r = t
            g = p
            b = v
        case 5:
            r = v
            g = p
            b = q
    }
    return {r = u8(r * 255), g = u8(g * 255), b = u8(b * 255), a = u8(a * 255)}
}

main_ram :: proc(testfile: []u8, font: ^ttf.Font, colors: []ttf.FontColor, scale: f32, text:string) {
    space_y := 400 * 2
    space_x := 750 * 4
    output_stride := 3 * space_x
    output_ram := make([]u8, output_stride * space_y)
    bitmap := make([]u8, space_x*space_y*3)
    timings: ttf.CpuTimingMetrics
    slice.fill(bitmap, 0xff)
    ttf.render_string_to_buffer(&ttf.StringRenderSettings{
        backend = ttf.CpuRenderSettings{
            output_buffer = {
                bitmap,
            },
            timing_metrics = &timings,
        },
        output_format = .RGB,
        defaultColor = {0,200,0,255},
        scale = scale,
        x = 0,
        y = 0,
        stride = output_stride,
        w = space_x,
        h = space_y,
        text = text,
        font = font,
        color_list = colors,
        antialiasing_level = 0,
        outline_thickness = 2,
    }, testfile)
    fmt.printfln("rendered in %v", time.diff(timings.start, timings.stop))
    write_ppm("data.ppm", bitmap, space_x, space_y)
}

main :: proc() {
    context.logger = log.create_console_logger()
    testfile, reade := os.read_entire_file_from_path("./fonts/AozoraMinchoRegular.ttf", context.allocator)
    
    if reade != nil {
        fmt.panicf("error reading font file %v", reade)
    }
    
    font, err := ttf.parse_ttf(testfile)

    // plug in your horizontal resolution
    scale := ttf.get_font_scale_factor_for_metrics(&font, ttf.SizeMetrics{
        size = ttf.Millimeters(620),
        resolution = 3840,
        point_size = 60
    })
    
    if err != .NONE {
        fmt.panicf("error parsing font file %v", err)
    }

    text := "斜め七十七度の並びで\n泣く泣く嘶くナナハン七台\n難なく並べて長眺め\nDoes it look good?"
    colors : [dynamic]ttf.FontColor
    for ccode, ind in text {
        append(&colors, fromHsva(f32(ind) / f32(19), 1, 1, 1))
    }

    if cuda.cuInit({}) != .SUCCESS {
        main_ram(testfile, &font, colors[:], scale, text)
        ttf.destroy_font(font)
        os.exit(0)
    }

    bbox_min, bbox_max := ttf.calculate_string_bounding_box(&font, testfile, text, 2, scale) or_else panic("render error while calcing bounding box")

    space_y := int(math.ceil(bbox_max.y)) - int(math.floor(bbox_min.y))
    space_x := int(math.ceil(bbox_max.x)) - int(math.floor(bbox_min.x))

    cuda_device: cuda.CUdevice
    cuda_ctx: cuda.CUcontext
    module: cuda.CUmodule
    stream: cuda.CUstream
    ttf_kernel: cuda.CUfunction
    
    cuda.cu_assert(cuda.cuDeviceGet(&cuda_device, 0))
    cuda.cu_assert(cuda.cuCtxCreate_v2(&cuda_ctx, {.SCHED_BLOCKING_SYNC}, cuda_device))
    
    //I can't assume you have nvcc set up, so you have to jit the kernel before calling into it.
    cuda_src := #load("./serafont.cu", cstring)
    prog: nvrtc.NvrtcProgram
    nvrtc.nvrtc_assert(nvrtc.nvrtcCreateProgram(&prog, cuda_src, "ttfkernel", 0, nil, nil))
    nvrtcCompileOptions := []cstring{"--std=c++11"}
    comp_err := nvrtc.nvrtcCompileProgram(prog, auto_cast len(nvrtcCompileOptions), raw_data(nvrtcCompileOptions))
    if comp_err != .SUCCESS {
        logsize: uint
        nvrtc.nvrtc_assert(nvrtc.nvrtcGetProgramLogSize(prog, &logsize))
        log := make([]u8, logsize)
        nvrtc.nvrtc_assert(nvrtc.nvrtcGetProgramLog(prog, raw_data(log)))
        fmt.panicf("compilation error %v. log:\n%s", comp_err, cstring(raw_data(log)))
    }
    ptx_size: uint
    nvrtc.nvrtc_assert(nvrtc.nvrtcGetPTXSize(prog, &ptx_size))
    ptx := make([]u8, ptx_size)
    nvrtc.nvrtc_assert(nvrtc.nvrtcGetPTX(prog, raw_data(ptx)))
    cuda.cu_assert(cuda.cuModuleLoadData(&module, raw_data(ptx)))
    cuda.cu_assert(cuda.cuModuleGetFunction(&ttf_kernel, module, "render_compiled_ttf_string"))
    
    output_memory : cuda.CUdeviceptr
    output_stride := 3 * space_x
    cuda.cu_assert(cuda.cuMemAlloc_v2(&output_memory, output_stride * space_y))
    cuda.cu_assert(cuda.cuMemsetD8_v2(output_memory,  0xff , uint(output_stride * space_y)))
    cuda.cu_assert(cuda.cuStreamCreate(&stream, .NON_BLOCKING))

    timings: ttf.CudaTimingMetrics
    t0 := time.now()

    //here's where the important stuff starts
    default_settings : ttf.StringRenderSettings = {
        backend = ttf.CudaRenderSettings{
            output_buffer = {
                output_memory,
            },
            stream = stream,
            render_kernel = ttf_kernel,
            timing_metrics = &timings,
        },
        output_format = .RGB,
        defaultColor = {0,200,0,255},
        scale = scale,
        x = -bbox_min.x,
        y = -bbox_min.y,
        stride = output_stride,
        w = space_x,
        h = space_y,
        antialiasing_level = 0,
        outline_thickness = 2,
    }
    font.default_settings = &default_settings
    
    default_settings.color_list = colors[:]

    cuda.cu_assert(ttf.render_string(&font, testfile, text))
    cuda.cu_assert(cuda.cuStreamSynchronize(default_settings.backend.(ttf.CudaRenderSettings).stream))

    cuda.cu_assert(cuda.cuEventSynchronize(timings.stop))
    t1 := time.now()
    msdiff : f32
    cuda.cu_assert(cuda.cuEventElapsedTime(&msdiff, timings.start, timings.stop))
    fmt.printfln("time spent by last kernel %v", cast(time.Duration) (msdiff * 1_000_000))
    fmt.printfln("time spent by render loop %v", time.diff(t0, t1))
    
    bitmap := make([]u8, space_x*space_y*3)
    slice.fill(bitmap, 0x18)
    cuda.cu_assert(cuda.cuMemcpyDtoH_v2(&bitmap[0], output_memory, 3*int(space_x)*int(space_y)))
    
    write_ppm("data.ppm", bitmap, space_x, space_y)
}
