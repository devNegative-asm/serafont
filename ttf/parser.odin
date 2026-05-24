package serafont_ttf

import "base:runtime"
import "core:c"
import "core:log"
import "core:slice"

_fourcc :: [4]u8
FWORD  :: distinct i16be
UFWORD :: distinct u16be

TTFParseError :: enum {
    NONE,
    OUT_OF_BOUNDS,
    UNSUPPORTED_VERSION,
    MAXIMUM_EXCEEDED,
    NO_SUPPORTED_CMAP,
    CMAP_SUBTABLE_UNSUPPORTED,
    CMAP_SUBTABLE_OVERLAPPING_RANGES,
    CMAP_SUBTABLE_INVERTED_RANGES,
    CMAP_SUBTABLE_INVALID_HEADER,
    CMAP_SUBTABLE_NOT_SORTED,
    GLYF_DATA_TOO_SMALL,
    TABLE_TOO_SMALL_HEAD,
    TABLE_TOO_SMALL_CMAP,
    TABLE_TOO_SMALL_CMAP_SUBTABLE,
    TABLE_TOO_SMALL_GLYF,
    TABLE_TOO_SMALL_HHEA,
    TABLE_TOO_SMALL_HMTX,
    TABLE_TOO_SMALL_LOCA,
    TABLE_TOO_SMALL_MAXP,
    TABLE_TOO_SMALL_POST,
    TABLE_TOO_SMALL_OS2,
    REQUIRED_TABLE_MISSING_HEAD,
    REQUIRED_TABLE_MISSING_CMAP,
    REQUIRED_TABLE_MISSING_GLYF,
    REQUIRED_TABLE_MISSING_HHEA,
    REQUIRED_TABLE_MISSING_HMTX,
    REQUIRED_TABLE_MISSING_LOCA,
    REQUIRED_TABLE_MISSING_MAXP,
    REQUIRED_TABLE_MISSING_POST,
}
Font :: struct {
    base: _TableDirectoryBase,
    head: _FontHeaderTable,
    maxp: _MaximumProfileTable,
    hhea: _HorizontalHeaderTable,
    hmtx: _HorizontalMetricsTable,
    cmap: _CharacterToGlyphMap,
    loca: _IndexToLocationTable,
    glyf: _GlyphTable `fmt:"-"`,
    allocator: runtime.Allocator,
    // default render settings used when calling `render_string`.
    default_settings: ^RenderSettings,
}
_GlyphTable :: struct {
    glyphs: []_Glyph
}
_SimpleGlyphFlagsBits :: enum u8 {
    ON_CURVE_POINT,
    X_SHORT_VECTOR,
    Y_SHORT_VECTOR,
    REPEAT,
    X_IS_SAME_OR_POSITIVE_X_SHORT_VECTOR,
    Y_IS_SAME_OR_POSITIVE_Y_SHORT_VECTOR,
    OVERLAP_SIMPLE,
    _reserved,
}
_SimpleGlyphFlags :: bit_set[_SimpleGlyphFlagsBits; u8]
_Glyph :: union {
    _SimpleGlyph,
    _CompositeGlyph,
    _CompiledGlyph,
}
_SimpleGlyph :: struct {
    header: _GlyphHeader,
    endPtsOfContours: []u16be,
    instructions: []u8,
    flags: []_SimpleGlyphFlags,
    xCoordinates: []i16,
    yCoordinates: []i16,
}
_GlyphInstance :: struct {
    minima, maxima: [2]f32,
    start_index: u64,
    end_index: u64,
    point: [2]f32,
    color: FontColor,
}
_GlyphOperation :: struct {
    parabolic_affine: matrix[2,3]f32,
    a: [2]f32, //start point
    c: [2]f32, //end point
    inv_ysq_coefficient: f32, // 0 indicates line. Non-0 indicates bezier
    endpoints_min, endpoints_max: f32,
    curvature_sign: f32,
}
_GlyphContour :: struct {
    minima, maxima: [2]f32,
    start_index: u64,
    end_index: u64,
}
_CompiledGlyph :: struct {
    header: _GlyphHeader,
    point: [2]f32,
    contours: []_GlyphContour,
    operations: []_GlyphOperation
}
_CompositeGlyph :: struct {
    header: _GlyphHeader,
    components: []_ComponentGlyph
}
_ComponentGlyphFlagBits :: enum {
    ARG_1_AND_2_ARE_WORDS,
    ARGS_ARE_XY_VALUES,
    ROUND_XY_TO_GRID,
    WE_HAVE_A_SCALE,
    _reserved_4,
    MORE_COMPONENTS,
    WE_HAVE_AN_X_AND_Y_SCALE,
    WE_HAVE_A_TWO_BY_TWO,
    WE_HAVE_INSTRUCTIONS,
    USE_MY_METRICS,
    OVERLAP_COMPOUND,
    SCALED_COMPONENT_OFFSET,
    UNSCALED_COMPONENT_OFFSET,
}
_ComponentGlyphFlags :: bit_set[_ComponentGlyphFlagBits; u16be]
_ComponentGlyph :: struct {
    flags: _ComponentGlyphFlags,
    glyphIndex: u16be,
    argument1,
    argument2: struct #raw_union {
        unsigned: u16be,
        signed: i16be,
    },
    transform: matrix[2,3]f32,
}
_GlyphHeader :: struct {
    number_of_contours: i16be,
    x_min, y_min, x_max, y_max: i16be,
}
_TableRecord :: struct {
    tag: _fourcc `fmt:"s"`,
    checksum: u32be,
    offset: u32be,
    length: u32be,
}
_TableDirectoryBase :: struct {
    sfnVersion: u32be `fmt:"08x"`,
    numTables: u16be,
    searchRange: u16be,
    entrySelector: u16be,
    rangeShift: u16be,
}
_FontHeaderTableFlags :: enum {
    baseline_y_0,
    leftsidebearing_x_0,
    hinting_depends_on_point_size,
    force_ppem_to_int,
    nonlinear_advance,
    _legacy_vertical,
    _legacy_zero,
    _legacy_requires_layout,
    _legacy_metamorphasis_by_default,
    _legacy_contains_strong_rtl,
    _legacy_contains_indc_rearrangement,
    lossless,
    font_converted,
    cleartype_opt,
    last_resort, // incidation in cmap does not imply support
    _reserved_0,
}
_FontHeaderTableStyleFlags :: enum {
    bold,
    italic,
    underline,
    outline,
    shadow,
    condensed,
    extended,
}
_FontHeaderTable :: struct #packed {
    majorVersion: u16be,
    minorVersion: u16be,
    fontRevision: u32be `fmt:"08X"`,
    checksumAdjustment: u32be `fmt:"08X"`,
    magic: u32be `fmt:"08X"`,
    flags: bit_set[_FontHeaderTableFlags; u16be],
    unitsPerEm: u16be,
    created: i64be, //in seconds, epoch is jan 1, 1904
    modified: i64be, //in seconds, epoch is jan 1, 1904
    xMin, yMin, xMax, yMax: i16be,
    style: bit_set[_FontHeaderTableStyleFlags; u16be],
    lowestRecPPEM: u16be,
    directionHint: i16be,
    indexToLocFormat: i16be, // 0 == 16 bit offsets, 1 == 32 bit offsets in the loca table.
}
_MaximumProfileTableV0_5 :: struct #packed {
    majorVersion: u16be,
    minorVersion: u16be,
    numGlyphs: u16be
}
_MaximumProfileTable :: struct #packed {
    using header: _MaximumProfileTableV0_5,
    maxPoints,
    maxContours,
    maxCompositePoints,
    maxCompositeContours,
    maxZones,
    maxTwilightPoints,
    maxStorage,
    maxFunctionDefs,
    maxInstructionDefs,
    maxStackElements,
    maxSizeOfInstructions,
    maxComponentElements,
    maxComponentDepth: u16be,
}
_HorizontalHeaderTable :: struct #packed {
    majorVersion: u16be,
    minorVersion: u16be,
    ascender: FWORD,
    descender: FWORD,
    lineGap: FWORD,
    advanceWidthMax: UFWORD,
    minLeftSideBearing: FWORD,
    minRightSideBearing: FWORD,
    xMaxExtent: FWORD,
    caretSlopeRise,
    caretSlopeRun,
    caretOffset,
    _reserved_0,
    _reserved_1,
    _reserved_2,
    _reserved_3,
    metricDataFormat: i16be,
    numberOfHMetrics: u16be,
}
_LongHorMetric :: struct #packed {
    advanceWidth: UFWORD,
    left_side_bearing: FWORD,
}
_HorizontalMetricsTable :: struct {
    hmetrics: []_LongHorMetric `fmt:"-"`,
    leftSideBearings: []FWORD `fmt:"-"`,
}
_EncodingRecord :: struct #packed {
    platformId: u16be,
    encodingId: u16be,
    subtableOffset: u32be
}
_CharacterToGlyphMap :: struct {
    version: u16be,
    subtable: _CmapSubtable `fmt:"-"`,
}
_CmapSubtable :: union {
    _ByteEncodingTable,
    _SegmentedCoverageTable,
    _SegmentMappingToDeltaValuesTable,
}
_SegmentMappingToDeltaValuesTable :: struct {
    header: _SegmentMappingToDeltaValuesHeader,
    endCode,
    startCode: []u16be,
    idDelta: []i16be,
    idRangeOffset: []u16be,
    glyphIdArray: []u16be,
}
_SegmentMappingToDeltaValuesHeader :: struct {
    format,
    length,
    language,
    segCountX2,
    searchRange,
    entrySelector,
    rangeShift: u16be,
}
_ByteEncodingTable :: struct #packed {
    format: u16be,
    length: u16be,
    language: u16be,
    glyphIdArray: [256]u8
}
_SegmentedCoverageTableHeader :: struct #packed {
    format: u16be,
    reserved: u16be,
    length: u32be,
    language: u32be,
    numGroups: u32be,
}
_SegmentedCoverageTable :: struct #packed {
    using header: _SegmentedCoverageTableHeader,
    groups: []_SequentialMapGroup,
}
_SequentialMapGroup :: struct #packed {
    startCharCode: u32be,
    endCharCode: u32be,
    startGlyphId: u32be,
}
_IndexToLocationTable :: struct {
    offsets: []u32 `fmt:"-"`
}

_peek_type :: proc(slc: []u8, index: int, $type: typeid, err_type: TTFParseError = .OUT_OF_BOUNDS) -> (rv: type, status: TTFParseError) {
    return transmute(type)_peek_array(slc, index, size_of(type), err_type) or_return, .NONE
}
_peek_array :: proc(slc: []u8, index: int, $amount: int, err_type: TTFParseError = .OUT_OF_BOUNDS) -> (rv: [amount]u8 , status: TTFParseError) {
    if len(slc[index:]) >= amount {
        #no_bounds_check {
            copy(rv[:], slc[index:])
        }
        status = .NONE
        return
    }
    status = err_type
    return
}
_consume_type :: proc(slc: []u8, index: ^int, $type: typeid, err_type: TTFParseError = .OUT_OF_BOUNDS) -> (rv: type, status: TTFParseError) {
    return transmute(type)_consume_array(slc, index, size_of(type), err_type) or_return, .NONE
}
_consume_array :: proc(slc: []u8, index: ^int, $amount: int, err_type: TTFParseError = .OUT_OF_BOUNDS) -> (rv: [amount]u8 , status: TTFParseError) {
    if len(slc[index^:]) >= amount {
        #no_bounds_check {
            copy(rv[:], slc[index^:])
        }
        index^ += amount
        status = .NONE
        return
    }
    status = err_type
    return
}
_consume_slice :: proc(slc: []u8, index: ^int, $type: typeid, #any_int amount: int, err_type: TTFParseError = .OUT_OF_BOUNDS) -> (rv: []type , status: TTFParseError) {
    list := slice.reinterpret([]type, slc[index^:])
    if len(list) >= amount {
        rv = make([]type, amount)
        #no_bounds_check {
            copy(rv, list)
        }
        index^ += amount * size_of(type)
        status = .NONE
        return
    }
    status = err_type
    return
}

destroy_font :: proc(#by_ptr font: Font) {
    context.allocator = font.allocator
    switch table in font.cmap.subtable {
        case _SegmentedCoverageTable:
            delete(table.groups)
        case _SegmentMappingToDeltaValuesTable:
            delete(table.startCode)
            delete(table.endCode)
            delete(table.idDelta)
            delete(table.idRangeOffset)
            delete(table.glyphIdArray)
        case _ByteEncodingTable:
    }
    delete(font.hmtx.hmetrics)
    delete(font.hmtx.leftSideBearings)
    delete(font.loca.offsets)
    for generic_glyph in font.glyf.glyphs {
        switch glyph in generic_glyph {
            case _CompositeGlyph:
                delete(glyph.components)
            case _SimpleGlyph:
                delete(glyph.endPtsOfContours)
                delete(glyph.flags)
                delete(glyph.instructions)
                delete(glyph.xCoordinates)
                delete(glyph.yCoordinates)
            case _CompiledGlyph:
                delete(glyph.operations)
                delete(glyph.contours)
        }
    }
    delete(font.glyf.glyphs)
}

parse_ttf :: proc(data: []u8) -> (rv: Font, status: TTFParseError) {
    rv.allocator = context.allocator
    index := 0

    defer if status != .NONE {
        destroy_font(rv)
    }

    rv.base = _consume_type(data, &index, _TableDirectoryBase) or_return
    table_records := make([]_TableRecord, rv.base.numTables)
    defer delete(table_records)
    for &record in table_records {
        record = _consume_type(data, &index, _TableRecord) or_return
    }

    head_index := -1
    cmap_index := -1
    glyf_index := -1
    hhea_index := -1
    hmtx_index := -1
    loca_index := -1
    maxp_index := -1
    post_index := -1

    for tableRecord, ind in table_records {
        if i64(tableRecord.offset) + i64(tableRecord.length) > i64(len(data)) {
            return {}, .OUT_OF_BOUNDS
        }
        switch tableRecord.tag {
            case "head":
                head_index = ind
            case "cmap":
                cmap_index = ind
            case "glyf":
                glyf_index = ind
            case "hhea":
                hhea_index = ind
            case "hmtx":
                hmtx_index = ind
            case "loca":
                loca_index = ind
            case "maxp":
                maxp_index = ind
            case "post":
                post_index = ind
        }
    }
    if head_index == -1 do return {}, .REQUIRED_TABLE_MISSING_HEAD
    if cmap_index == -1 do return {}, .REQUIRED_TABLE_MISSING_CMAP
    if glyf_index == -1 do return {}, .REQUIRED_TABLE_MISSING_GLYF
    if hhea_index == -1 do return {}, .REQUIRED_TABLE_MISSING_HHEA
    if hmtx_index == -1 do return {}, .REQUIRED_TABLE_MISSING_HMTX
    if loca_index == -1 do return {}, .REQUIRED_TABLE_MISSING_LOCA
    if maxp_index == -1 do return {}, .REQUIRED_TABLE_MISSING_MAXP
    // if post_index == -1 do return {}, .REQUIRED_TABLE_MISSING_POST

    table: []u8
    //head
    record := table_records[head_index]
    table = data[record.offset:record.offset + record.length]
    if len(table) < size_of(_FontHeaderTable) {
        return {}, .TABLE_TOO_SMALL_HEAD
    }
    rv.head = slice.to_type(table, _FontHeaderTable)

    //maxp
    record = table_records[maxp_index]
    table = data[record.offset:record.offset + record.length]
    if len(table) < size_of(_MaximumProfileTableV0_5) {
        return {}, .TABLE_TOO_SMALL_MAXP
    }
    rv.maxp.header = slice.to_type(table, _MaximumProfileTableV0_5)
    if rv.maxp.majorVersion == 0 {

    } else if rv.maxp.majorVersion == 1 {
        if len(table) < size_of(_MaximumProfileTable) {
            return {}, .TABLE_TOO_SMALL_MAXP
        }
        rv.maxp = slice.to_type(table, _MaximumProfileTable)
    } else {
        return {}, .UNSUPPORTED_VERSION
    }

    //hhea
    record = table_records[hhea_index]
    table = data[record.offset:record.offset + record.length]
    if len(table) < size_of(_HorizontalHeaderTable) {
        return {}, .TABLE_TOO_SMALL_HHEA
    }
    rv.hhea = slice.to_type(table, _HorizontalHeaderTable)

    if rv.hhea.numberOfHMetrics > rv.maxp.numGlyphs {
        return {}, .MAXIMUM_EXCEEDED
    }

    //hmtx
    record = table_records[hmtx_index]
    table = data[record.offset:record.offset + record.length]
    numberOfHMetrics := rv.hhea.numberOfHMetrics

    hmtx_size := size_of(_LongHorMetric) * int(numberOfHMetrics) +
                 size_of(FWORD) * int(rv.maxp.numGlyphs - numberOfHMetrics)

    if len(table) < hmtx_size {
        return {}, .TABLE_TOO_SMALL_HMTX
    }
    rv.hmtx = {
        make([]_LongHorMetric, numberOfHMetrics),
        make([]FWORD, rv.maxp.numGlyphs - numberOfHMetrics),
    }
    copy(rv.hmtx.hmetrics, slice.reinterpret([]_LongHorMetric, table))
    copy(rv.hmtx.leftSideBearings, slice.reinterpret([]FWORD, table[size_of(_LongHorMetric) * int(numberOfHMetrics):]))

    //cmap
    record = table_records[cmap_index]
    table = data[record.offset:record.offset + record.length]
    cmap_reader := 0
    rv.cmap.version = _consume_type(table, &cmap_reader, u16be) or_return
    if rv.cmap.version != 0 do return {}, .UNSUPPORTED_VERSION
    cmap_num_tables := int(_consume_type(table, &cmap_reader, u16be) or_return)
    if len(table) - cmap_reader < cmap_num_tables * size_of(_EncodingRecord) {
        return {}, .TABLE_TOO_SMALL_CMAP
    }
    
    cmap_offset := -1
    cmap_priority := -1
    for encodingRecord in slice.reinterpret([]_EncodingRecord, table[cmap_reader:]) {
        //prioritize windows full unicode

        //windows full unicode
        if encodingRecord.platformId == 3 && encodingRecord.encodingId == 10 && cmap_priority < 99 {
            cmap_priority = 99
            cmap_offset = int(encodingRecord.subtableOffset)
        }
        //windows unicode BMP
        if encodingRecord.platformId == 3 && encodingRecord.encodingId == 1 && cmap_priority < 50 {
            cmap_priority = 50
            cmap_offset = int(encodingRecord.subtableOffset)
        }
        //unicode 2.0 BMP
        if encodingRecord.platformId == 0 && encodingRecord.encodingId == 3 && cmap_priority < 10 {
            cmap_priority = 10
            cmap_offset = int(encodingRecord.subtableOffset)
        }
        //unicode 2.0 full unicode
        if encodingRecord.platformId == 0 && encodingRecord.encodingId == 4 && cmap_priority < 70 {
            cmap_priority = 70
            cmap_offset = int(encodingRecord.subtableOffset)
        }
    }
    if cmap_offset == -1 do return {}, .NO_SUPPORTED_CMAP
    if cmap_offset > len(table) {
        log.warn("cmap subtable offset exceeds remaining space")
        return {}, .OUT_OF_BOUNDS
    }
    cmap_subtable := table[cmap_offset:]
    cmap_subtable_reader := 0
    format := _peek_type(cmap_subtable, cmap_subtable_reader, u16be) or_return
    cmap_subtable_reader = 0
    switch format {
        case 0:
            rv.cmap.subtable = _consume_type(cmap_subtable, &cmap_subtable_reader, _ByteEncodingTable) or_return
        case 12:
            rv.cmap.subtable = _SegmentedCoverageTable{}
            coverage_table := &rv.cmap.subtable.(_SegmentedCoverageTable)
            coverage_table.header = _consume_type(cmap_subtable, &cmap_subtable_reader, _SegmentedCoverageTableHeader) or_return
            if int(coverage_table.header.length) > len(cmap_subtable) {
                log.warn("segmented coverage table length exceeds remaining space")
                return {}, .TABLE_TOO_SMALL_CMAP_SUBTABLE
            }
            cmap_subtable = cmap_subtable[:coverage_table.header.length]
            groups := slice.reinterpret([]_SequentialMapGroup, cmap_subtable[cmap_subtable_reader:])
            entries_remaining := len(groups)
            if entries_remaining < int(coverage_table.header.numGroups) {
                log.warn("segmented coverage table numGroups exceeds remaining space")
                return {}, .TABLE_TOO_SMALL_CMAP_SUBTABLE
            }
            coverage_table.groups = make([]_SequentialMapGroup, coverage_table.numGroups)
            copy(coverage_table.groups, groups)
            //I don't trust the data to actually be sorted
            slice.sort_by_key(coverage_table.groups, proc(group: _SequentialMapGroup) -> u32be {
                return group.startCharCode
            })
            // segments should be ordered and disjoint
            max_id := -1
            for group in coverage_table.groups {
                if group.startCharCode > group.endCharCode {
                    log.warn("incorrect ordering for start and end codes:", group.startCharCode, ">", group.endCharCode)
                    return {}, .CMAP_SUBTABLE_INVERTED_RANGES
                }
                if int(group.startCharCode) <= max_id {
                    log.warn("overlapping range detected:", group.startCharCode, "<=", max_id)
                    return {}, .CMAP_SUBTABLE_OVERLAPPING_RANGES
                }
                max_id = int(group.endCharCode)
            }
        case 4:
            rv.cmap.subtable = _SegmentMappingToDeltaValuesTable{}
            delta_table := &rv.cmap.subtable.(_SegmentMappingToDeltaValuesTable)
            delta_table.header = _consume_type(cmap_subtable, &cmap_subtable_reader, _SegmentMappingToDeltaValuesHeader) or_return
            if (delta_table.header.segCountX2 & 1) != 0 {
                log.warn("segment mapping to delta values header segCountX2 should not be even:", delta_table.header.segCountX2)
                return {}, .CMAP_SUBTABLE_INVALID_HEADER
            }
            expected_space := 4 * int(delta_table.header.segCountX2) + 2
            segcount := delta_table.header.segCountX2 / 2
            if len(cmap_subtable[cmap_subtable_reader:]) < expected_space || expected_space > int(delta_table.header.length) || len(cmap_subtable) < int(delta_table.header.length) {
                log.warn("segment mapping to delta values table too small", len(cmap_subtable[cmap_subtable_reader:]), "<", expected_space)
                return {}, .TABLE_TOO_SMALL_CMAP_SUBTABLE
            }
            cmap_subtable = cmap_subtable[:delta_table.header.length]
            delta_table.endCode = make([]u16be, segcount)
            delta_table.startCode = make([]u16be, segcount)
            delta_table.idDelta = make([]i16be, segcount)
            delta_table.idRangeOffset = make([]u16be, segcount)

            cmap_subtable_reader += 2 * copy(delta_table.endCode, slice.reinterpret([]u16be, cmap_subtable[cmap_subtable_reader:]))
            //padding
            _consume_type(cmap_subtable, &cmap_subtable_reader, u16be) or_return
            cmap_subtable_reader += 2 * copy(delta_table.startCode, slice.reinterpret([]u16be, cmap_subtable[cmap_subtable_reader:]))
            cmap_subtable_reader += 2 * copy(delta_table.idDelta, slice.reinterpret([]i16be, cmap_subtable[cmap_subtable_reader:]))
            cmap_subtable_reader += 2 * copy(delta_table.idRangeOffset, slice.reinterpret([]u16be, cmap_subtable[cmap_subtable_reader:]))
            remainder := slice.reinterpret([]u16be, cmap_subtable[cmap_subtable_reader:])
            delta_table.glyphIdArray = make([]u16be, len(remainder))
            copy(delta_table.glyphIdArray, remainder)

            //don't trust the array to actually be valid
            start_code, end_code := -1, -1
            for i in 0..<segcount {
                start := int(delta_table.startCode[i])
                end := int(delta_table.endCode[i])
                if start > end {
                    log.warn("incorrect ordering for start and end codes:", start, ">", end)
                    return {}, .CMAP_SUBTABLE_INVERTED_RANGES
                }
                if start <= end_code {
                    log.warn("overlapping range detected:", start, "<= ", end_code)
                    return {}, .CMAP_SUBTABLE_INVERTED_RANGES
                }
                if start <= start_code || end <= end_code {
                    //idk how you get here, but ok
                    log.warn("not sorted:", start, "<=", start_code, "||", end, "<=", end_code)
                    return {}, .CMAP_SUBTABLE_NOT_SORTED
                }
                start_code = start
                end_code = end
            }
            

        case:
            log.warn("unsupported cmap subtable", format)
            return {}, .CMAP_SUBTABLE_UNSUPPORTED
    }
    //loca
    record = table_records[loca_index]
    table = data[record.offset:record.offset + record.length]
    rv.loca = _IndexToLocationTable{make([]u32, int(rv.maxp.numGlyphs) + 1)}
    if rv.head.indexToLocFormat == 0 {
        locations := slice.reinterpret([]u16be, table)
        if len(locations) < int(rv.maxp.numGlyphs) + 1 {
            return {}, .TABLE_TOO_SMALL_LOCA
        }
        for &tblEntry, ind in rv.loca.offsets {
            tblEntry = u32(locations[ind]) * 2
        }
    } else {
        locations := slice.reinterpret([]u32be, table)
        if len(locations) < int(rv.maxp.numGlyphs) + 1 {
            return {}, .TABLE_TOO_SMALL_LOCA
        }
        for &tblEntry, ind in rv.loca.offsets {
            tblEntry = u32(locations[ind])
        }
    }

    //glyf
    record = table_records[glyf_index]
    table = data[record.offset:record.offset + record.length]
    rv.glyf.glyphs = make([]_Glyph, rv.maxp.numGlyphs)
    for i in 0..<rv.maxp.numGlyphs {
        offset0 := rv.loca.offsets[i]
        offset1 := rv.loca.offsets[i + 1]
        glyph_size := int(offset1) - int(offset0)
        if glyph_size < 0 {
            log.warn("Glyph index", i, "of size", glyph_size, "has negative size")
        }
        if glyph_size == 0 {
            rv.glyf.glyphs[i] = _SimpleGlyph{}
            continue
        }
        if int(offset0) >= len(table) || int(offset1) > len(table) {
            log.warn("offsets point to data outside bounds of glyf table", offset0, offset1, "out of", len(table))
            return {}, .TABLE_TOO_SMALL_GLYF
        }
        glyph_data := table[offset0:offset1]
        within_glyph_offset := 0
        glif_header := _consume_type(glyph_data, &within_glyph_offset, _GlyphHeader, .GLYF_DATA_TOO_SMALL) or_return
        if glif_header.number_of_contours >= 0 {
            rv.glyf.glyphs[i] = _SimpleGlyph{header = glif_header}
            glif := &rv.glyf.glyphs[i].(_SimpleGlyph)
            //simple glyph
            glif.endPtsOfContours = _consume_slice(glyph_data, &within_glyph_offset, u16be, glif_header.number_of_contours, .GLYF_DATA_TOO_SMALL) or_return
            instruction_length := _consume_type(glyph_data, &within_glyph_offset, u16be, .GLYF_DATA_TOO_SMALL) or_return
            glif.instructions = _consume_slice(glyph_data, &within_glyph_offset, u8, instruction_length, .GLYF_DATA_TOO_SMALL) or_return

            if glif.header.number_of_contours > 0 {
                //flags
                number_of_points := int(glif.endPtsOfContours[glif.header.number_of_contours - 1]) + 1
                glif.flags = make([]_SimpleGlyphFlags, number_of_points)
                points_left := number_of_points
                for points_left > 0 {
                    flags := _consume_type(glyph_data, &within_glyph_offset, _SimpleGlyphFlags, .GLYF_DATA_TOO_SMALL) or_return
                    glif.flags[number_of_points - points_left] = flags
                    points_left -= 1
                    if .REPEAT in flags {
                        repeat_count := int(_consume_type(glyph_data, &within_glyph_offset, u8, .GLYF_DATA_TOO_SMALL) or_return)
                        if repeat_count > points_left {
                            log.warn("Glyph index", i, "of size", glyph_size, "attempts to repeat a flag for more space than remains in the flags array")
                            return {}, .OUT_OF_BOUNDS
                        }
                        slice.fill(glif.flags[number_of_points - points_left : number_of_points - points_left + repeat_count], flags)
                        points_left -= repeat_count
                    }
                }
                coord_sets: [2]struct {
                    coords: ^[]i16,
                    short_mask: _SimpleGlyphFlagsBits,
                    same_or_positive_mask: _SimpleGlyphFlagsBits,
                } = {{
                    &glif.xCoordinates,
                    .X_SHORT_VECTOR, .X_IS_SAME_OR_POSITIVE_X_SHORT_VECTOR
                }, {
                    &glif.yCoordinates,
                    .Y_SHORT_VECTOR, .Y_IS_SAME_OR_POSITIVE_Y_SHORT_VECTOR
                }}
                for coord_set in coord_sets {
                    short_mask := coord_set.short_mask
                    same_or_positive_mask := coord_set.same_or_positive_mask
                    coords := coord_set.coords
                    coords^ = make([]i16, number_of_points)
                    p0 :i16= 0
                    for &point, index in coords {
                        flags := glif.flags[index]
                        if short_mask in flags {
                            delta := i16(_consume_type(glyph_data, &within_glyph_offset, u8, .GLYF_DATA_TOO_SMALL) or_return)
                            p0 = p0 + (same_or_positive_mask in flags ? delta : -delta)
                        } else {
                            if same_or_positive_mask not_in flags {
                                p0 = p0 + i16(_consume_type(glyph_data, &within_glyph_offset, i16be, .GLYF_DATA_TOO_SMALL) or_return)
                            }
                        }
                        point = p0
                    }
                }
            }

        } else {
            //composite glyph
            rv.glyf.glyphs[i] = _CompositeGlyph{header = glif_header}
            composite := &rv.glyf.glyphs[i].(_CompositeGlyph)
            within_glyph_offset := 10
            components: [dynamic]_ComponentGlyph
            defer if status != .NONE {
                delete(components)
            }
            component := _ComponentGlyph{flags = {.MORE_COMPONENTS}}
            for within_glyph_offset < glyph_size && (.MORE_COMPONENTS in component.flags) {
                component = _ComponentGlyph{}
                if glyph_size - within_glyph_offset < 4 {
                    log.warn("Composite glyph index", i, "of size", glyph_size, "bytes has no room for flags or index")
                    return {}, .GLYF_DATA_TOO_SMALL
                }
                dats := slice.reinterpret([]u16be, glyph_data[within_glyph_offset:])
                component.flags = transmute(_ComponentGlyphFlags) dats[0]
                component.glyphIndex = dats[1]
                within_glyph_offset += 4
                if .ARG_1_AND_2_ARE_WORDS in component.flags {
                    dats := slice.reinterpret([]u16be, glyph_data[within_glyph_offset:])
                    if len(dats) < 2 {
                        log.warn("Composite glyph index", i, "of size", glyph_size, "bytes has no room for 16 bit arguments")
                        return {}, .GLYF_DATA_TOO_SMALL
                    }
                    component.argument1.unsigned = dats[0]
                    component.argument2.unsigned = dats[1]
                    within_glyph_offset += 4
                } else {
                    dats := glyph_data[within_glyph_offset:]
                    if len(dats) < 2 {
                        log.warn("Composite glyph index", i, "of size", glyph_size, "bytes has no room for 8 bit arguments")
                        return {}, .GLYF_DATA_TOO_SMALL
                    }
                    component.argument1.unsigned = cast(u16be) dats[0]
                    component.argument2.unsigned = cast(u16be) dats[1]
                    if .ARGS_ARE_XY_VALUES in component.flags {
                        //when this flag is set, these aren't indices, so we need to sign extend negative values
                        component.argument1.signed = cast(i16be) transmute(i8)(dats[0])
                        component.argument2.signed = cast(i16be) transmute(i8)(dats[1])
                    }
                    within_glyph_offset += 2
                }
                child_offset: [2]f32 = .ARGS_ARE_XY_VALUES in component.flags ? [2]f32{f32(component.argument1.signed), f32(component.argument2.signed)} : [2]f32{}
                t_coeff := slice.reinterpret([]i16be, glyph_data[within_glyph_offset:])
                component.transform = {
                    1,0, child_offset[0],
                    0,1, child_offset[1],
                }
                if .WE_HAVE_A_SCALE in component.flags {
                    if len(t_coeff) < 1 {
                        log.warn("Composite glyph index", i, "of size", glyph_size, "bytes has no room for XY scale factor")
                        return {}, .GLYF_DATA_TOO_SMALL
                    }
                    component.transform = {
                        f32(t_coeff[0]) / 16384, 0, child_offset[0],
                        0, f32(t_coeff[0]) / 16384, child_offset[1],
                    }
                    within_glyph_offset += 2
                } else if .WE_HAVE_AN_X_AND_Y_SCALE in component.flags {
                    if len(t_coeff) < 2 {
                        log.warn("Composite glyph index", i, "of size", glyph_size, "bytes has no room for X & Y scale factors")
                        return {}, .GLYF_DATA_TOO_SMALL
                    }
                    component.transform = {
                        f32(t_coeff[0]) / 16384, 0, child_offset[0],
                        0, f32(t_coeff[1]) / 16384, child_offset[1],
                    }
                    within_glyph_offset += 4
                } else if .WE_HAVE_A_TWO_BY_TWO in component.flags {
                    if len(t_coeff) < 4 {
                        log.warn("Composite glyph index", i, "of size", glyph_size, "bytes has no room for affine factors")
                        return {}, .GLYF_DATA_TOO_SMALL
                    }
                    component.transform = {
                        f32(t_coeff[0]) / 16384, f32(t_coeff[1]) / 16384, child_offset[0],
                        f32(t_coeff[2]) / 16384, f32(t_coeff[3]) / 16384, child_offset[1],
                    }
                    within_glyph_offset += 8
                }
                if .SCALED_COMPONENT_OFFSET in component.flags {
                    scaled_offset := component.transform * [3]f32{child_offset[0],child_offset[1],0}
                    component.transform[0,2] = scaled_offset[0,0]
                    component.transform[1,2] = scaled_offset[1,0]
                }
                if (component.flags & {.WE_HAVE_A_SCALE, .WE_HAVE_A_TWO_BY_TWO, .WE_HAVE_AN_X_AND_Y_SCALE}) != {} {
                    // log.debug("transform", component.transform)
                }
                append(&components, component)
            }
            composite.components = components[:]
        }
    }

    return
}

_get_glyph_header :: proc(glyph: _Glyph) -> _GlyphHeader {
    glyph:=glyph
    return (transmute(^_GlyphHeader)&glyph)^
}
