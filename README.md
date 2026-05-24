This is a (hopefully) simple odin font renderer.

1. Parse the font binary. 
    ```odin
    font, err := ttf.parse_ttf(font_file)
    ```
2. Get the scale factor for your monitor and font size
    ```odin
    scale := ttf.get_font_scale_factor_for_metrics(&font, ttf.SizeMetrics{
        size = ttf.Millimeters(620),
        resolution = 3840,
        point_size = 60
    })
    ```
3. Pick your rendering settings. See `RenderSettings`. To reuse settings, set the `font.default_settings = settings`
4. Call `render_string_to_buffer(settings, font_file)` if using default settings, otherwise call `render_string(font, font_file, "string you want to render")`
5. `ttf.destroy_font(font)`

![](https://github.com/devNegative-asm/serafont/blob/master/data.png)