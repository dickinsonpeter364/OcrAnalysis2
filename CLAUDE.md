# OcrAnalysis

C++ library for OCR and PDF analysis. Uses Poppler, OpenCV, Cairo, Tesseract, and ZXing.

## Build

CMake-generated Visual Studio solution. Build a specific target with:

```sh
powershell -Command "& 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\MSBuild\\Current\\Bin\\MSBuild.exe' c:\\OcrAnalysis\\OcrAnalysis.sln /p:Configuration=Debug /p:Platform=x64 /t:<TARGET> /m /nologo 2>&1 | Select-String 'error C|Build succeeded|FAILED' | Out-String"
```

Replace `<TARGET>` with the project name (e.g. `test_png_render`, `test_check_image`, `ocr_analysis`).

## Run tests

```sh
cd c:\OcrAnalysis && powershell -Command ".\debug\<TEST_EXE> <args> 2>&1 | Out-String"
```

## Project structure

- `include/OCRAnalysis.hpp` — Main header with all struct/function declarations
- `src/OCRAnalysis.cpp` — Core implementation (extractPDFElements, checkImage, etc.)
- `src/create_relative_map.cpp` — createRelativeMap implementation
- `src/render_elements_png.cpp` — renderElementsToPNG implementation
- `src/render_pdf.cpp` — PDF rendering utilities
- `src/test_*.cpp` — Test executables

## Coordinate systems

- **PDF bottom-left**: y increases upward. Used by `img.x`, `img.y`, `linesBoundingBoxX/Y`.
- **`text.boundingBox.y`**: Screen y of the TOP of the text box (y increases downward). Converted from Poppler's PDF coords via `y = pageRect.height() - bbox.y() - bbox.height()`.
- To convert back to PDF: `pdfBottom = pageHeight - boundingBox.y - boundingBox.height`.

## Key conventions

- C++20 standard
- Windows/MSVC build environment
- Render DPI is configurable (commonly 1200 for production, 96 for detection)
- Test PDFs are in the project root (e.g. `L20033877.pdf`)
