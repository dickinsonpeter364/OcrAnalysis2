# RenderElementsToPNG Bounds Mode Feature

## Overview
The `renderElementsToPNG` function now supports two modes for determining the rendering bounds:
- **USE_CROP_MARKS** (default): Uses crop marks to determine the content area
- **USE_LARGEST_RECTANGLE**: Uses the largest rectangle in the PDF to determine the content area

## API Changes

### New Enum
```cpp
enum class RenderBoundsMode {
  USE_CROP_MARKS,      ///< Use crop marks to determine rendering bounds
  USE_LARGEST_RECTANGLE ///< Use largest rectangle to determine rendering bounds
};
```

### Updated Function Signature
```cpp
PNGRenderResult renderElementsToPNG(
    const PDFElements &elements,
    const std::string &pdfPath,
    double dpi = 300.0,
    const std::string &outputDir = "images",
    RenderBoundsMode boundsMode = RenderBoundsMode::USE_CROP_MARKS);
```

## Usage Examples

### Using Crop Marks (Default)
```cpp
ocr::OCRAnalysis analyzer;
auto elements = analyzer.extractPDFElements("document.pdf");
auto result = analyzer.renderElementsToPNG(elements, "document.pdf", 300.0, "output");
// Uses crop marks by default
```

### Using Largest Rectangle
```cpp
ocr::OCRAnalysis analyzer;
auto elements = analyzer.extractPDFElements("document.pdf");
auto result = analyzer.renderElementsToPNG(
    elements, 
    "document.pdf", 
    300.0, 
    "output",
    ocr::OCRAnalysis::RenderBoundsMode::USE_LARGEST_RECTANGLE);
```

## Command Line Testing

### test_render_info
Tests with filtered elements (after bleed mark stripping):
```bash
# Use crop marks (default)
.\Debug\test_render_info.exe L20033877.pdf images 300 crop

# Use largest rectangle
.\Debug\test_render_info.exe L20033877.pdf images 300 rect
```

### test_bounds_modes
Tests with raw PDF elements (before bleed mark stripping):
```bash
# Use crop marks (default)
.\Debug\test_bounds_modes.exe L20033877.pdf images 300 crop

# Use largest rectangle
.\Debug\test_bounds_modes.exe L20033877.pdf images 300 rect
```

## Implementation Details

### USE_CROP_MARKS Mode
1. Checks if `elements.linesBoundingBoxWidth > 0 && elements.linesBoundingBoxHeight > 0`
2. Uses the pre-calculated bounding box from crop mark analysis
3. Falls back to calculating bounds from all elements if no crop marks found

### USE_LARGEST_RECTANGLE Mode
1. Iterates through all rectangles in `elements.rectangles`
2. Calculates area for each rectangle (width × height)
3. Selects the rectangle with the largest area
4. Uses that rectangle's bounds as the rendering area
5. Returns error if no rectangles are found

## Output Comparison

Example with L20033877.pdf:

**USE_CROP_MARKS:**
- Bounds: (229.645, 220.64) to (612.315, 368.37)
- Output size: 1594x615 pixels @ 300 DPI
- Captures full label content area

**USE_LARGEST_RECTANGLE:**
- Bounds: (577.165, 350.738) to (603.805, 368.358)
- Output size: 110x73 pixels @ 300 DPI
- Captures only the largest rectangle (area: 469.397 sq pts)
