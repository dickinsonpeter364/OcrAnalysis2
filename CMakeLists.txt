cmake_minimum_required(VERSION 3.16)
project(OcrAnalysis VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard (C++20 required for Poppler's low-level API)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TESSERACT REQUIRED tesseract)
pkg_check_modules(LEPTONICA REQUIRED lept)
pkg_check_modules(POPPLER_CPP REQUIRED poppler-cpp)
pkg_check_modules(POPPLER REQUIRED poppler)  # Core library for low-level API

# Try to find Cairo (optional for PDF rendering)
pkg_check_modules(CAIRO cairo)
if(CAIRO_FOUND)
    message(STATUS "Cairo found - PDF rendering will be available")
else()
    message(STATUS "Cairo not found - PDF rendering will use text-based fallback")
endif()

# Print package information
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Tesseract include dirs: ${TESSERACT_INCLUDE_DIRS}")
message(STATUS "Leptonica include dirs: ${LEPTONICA_INCLUDE_DIRS}")
message(STATUS "Poppler include dirs: ${POPPLER_CPP_INCLUDE_DIRS}")

# Define the library
add_library(ocr_analysis
    src/OCRAnalysis.cpp
)

target_include_directories(ocr_analysis
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${TESSERACT_INCLUDE_DIRS}
        ${LEPTONICA_INCLUDE_DIRS}
        ${POPPLER_CPP_INCLUDE_DIRS}
        ${POPPLER_INCLUDE_DIRS}
)

if(CAIRO_FOUND)
    target_include_directories(ocr_analysis PUBLIC ${CAIRO_INCLUDE_DIRS})
    target_compile_definitions(ocr_analysis PUBLIC HAVE_CAIRO)
endif()

target_link_libraries(ocr_analysis
    PUBLIC
        ${OpenCV_LIBS}
        ${TESSERACT_LIBRARIES}
        ${LEPTONICA_LIBRARIES}
        ${POPPLER_CPP_LIBRARIES}
        ${POPPLER_LIBRARIES}
)

if(CAIRO_FOUND)
    target_link_libraries(ocr_analysis PUBLIC ${CAIRO_LIBRARIES})
endif()

target_link_directories(ocr_analysis
    PUBLIC
        ${TESSERACT_LIBRARY_DIRS}
        ${LEPTONICA_LIBRARY_DIRS}
        ${POPPLER_CPP_LIBRARY_DIRS}
        ${POPPLER_LIBRARY_DIRS}
)

# Define the main executable
add_executable(ocr_demo
    src/main.cpp
)

target_link_libraries(ocr_demo
    PRIVATE
        ocr_analysis
)

# Define the test executable
add_executable(test_ocr
    src/test_ocr.cpp
)

target_link_libraries(test_ocr
    PRIVATE
        ocr_analysis
)

# Define the orientation test executable
add_executable(test_orientation
    src/test_orientation.cpp
)

target_link_libraries(test_orientation
    PRIVATE
        ocr_analysis
)

# Define the identify regions test executable (visual output)
add_executable(test_identify_regions
    src/test_identify_regions.cpp
)

target_link_libraries(test_identify_regions
    PRIVATE
        ocr_analysis
)

# Define the PDF extraction test executable
add_executable(test_pdf
    src/test_pdf.cpp
)

target_link_libraries(test_pdf
    PRIVATE
        ocr_analysis
)

# Define the PDF graphics extraction test executable
add_executable(test_pdf_graphics
    src/test_pdf_graphics.cpp
)

target_link_libraries(test_pdf_graphics
    PRIVATE
        ocr_analysis
)

# Define the PDF embedded images extraction test executable
add_executable(test_pdf_embedded
    src/test_pdf_embedded.cpp
)

target_link_libraries(test_pdf_embedded
    PRIVATE
        ocr_analysis
)

# Define the PDF rectangles extraction test executable
add_executable(test_pdf_rectangles
    src/test_pdf_rectangles.cpp
)

target_link_libraries(test_pdf_rectangles
    PRIVATE
        ocr_analysis
)

# Define the PDF lines extraction test executable
add_executable(test_pdf_lines
    src/test_pdf_lines.cpp
)

target_link_libraries(test_pdf_lines
    PRIVATE
        ocr_analysis
)

# Define the PDF elements extraction test executable
add_executable(test_pdf_elements
    src/test_pdf_elements.cpp
)

target_link_libraries(test_pdf_elements
    PRIVATE
        ocr_analysis
)

# Define the PNG rendering test executable
add_executable(test_png_render
    src/test_png_render.cpp
)

target_link_libraries(test_png_render
    PRIVATE
        ocr_analysis
)

# Define the bleed mark stripping test executable
add_executable(test_strip_bleed_marks
    src/test_strip_bleed_marks.cpp
)

target_link_libraries(test_strip_bleed_marks
    PRIVATE
        ocr_analysis
)

# Define the render info test executable
add_executable(test_render_info
    src/test_render_info.cpp
)

target_link_libraries(test_render_info
    PRIVATE
        ocr_analysis
)

# Define the bounds modes test executable
add_executable(test_bounds_modes
    src/test_bounds_modes.cpp
)

target_link_libraries(test_bounds_modes
    PRIVATE
        ocr_analysis
)

# Optional: Install targets
install(TARGETS ocr_analysis ocr_demo
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)
