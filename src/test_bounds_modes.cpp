#include "OCRAnalysis.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <pdf_file> [output_dir] [dpi] [bounds_mode] [mark_to_file]"
              << std::endl;
    std::cerr << "  bounds_mode: 'crop' (default) or 'rect'" << std::endl;
    std::cerr
        << "  mark_to_file: optional image file to mark with element boxes"
        << std::endl;
    return 1;
  }

  std::string pdfPath = argv[1];
  std::string outputDir = (argc > 2) ? argv[2] : "images";
  double dpi = (argc > 3) ? std::stod(argv[3]) : 300.0;

  // Parse bounds mode
  ocr::OCRAnalysis::RenderBoundsMode boundsMode =
      ocr::OCRAnalysis::RenderBoundsMode::USE_CROP_MARKS;
  if (argc > 4) {
    std::string modeStr = argv[4];
    std::transform(modeStr.begin(), modeStr.end(), modeStr.begin(), ::tolower);
    if (modeStr == "rect" || modeStr == "rectangle") {
      boundsMode = ocr::OCRAnalysis::RenderBoundsMode::USE_LARGEST_RECTANGLE;
    }
  }

  // Optional mark to file parameter
  std::string markToFile = (argc > 5) ? argv[5] : "";

  try {
    ocr::OCRAnalysis analyzer;

    // Extract PDF elements (without stripping bleed marks)
    std::cout << "Extracting PDF elements from: " << pdfPath << "\n\n";
    auto elements = analyzer.extractPDFElements(pdfPath);

    if (!elements.success) {
      std::cerr << "Error: " << elements.errorMessage << std::endl;
      return 1;
    }

    std::cout << "Extracted elements:\n";
    std::cout << "  Text lines: " << elements.textLineCount << "\n";
    std::cout << "  Images: " << elements.imageCount << "\n";
    std::cout << "  Rectangles: " << elements.rectangleCount << "\n";
    std::cout << "  Lines: " << elements.graphicLineCount << "\n\n";

    // Render the elements to PNG
    std::cout << "Rendering with bounds mode: "
              << (boundsMode ==
                          ocr::OCRAnalysis::RenderBoundsMode::USE_CROP_MARKS
                      ? "USE_CROP_MARKS"
                      : "USE_LARGEST_RECTANGLE")
              << "\n\n";

    auto renderResult = analyzer.renderElementsToPNG(
        elements, pdfPath, dpi, outputDir, boundsMode, markToFile);

    if (!renderResult.success) {
      std::cerr << "Error rendering PNG: " << renderResult.errorMessage
                << std::endl;
      return 1;
    }

    std::cout << "=== PNG Rendering Complete ===\n";
    std::cout << "Output: " << renderResult.outputPath << "\n";
    std::cout << "Size: " << renderResult.imageWidth << "x"
              << renderResult.imageHeight << " pixels\n";
    std::cout << "Total rendered elements: " << renderResult.elements.size()
              << "\n";

    // Create OCR-aligned marked image (only if markToFile was provided)
    if (!markToFile.empty()) {
      std::cout << "\n=== Creating OCR-Aligned Marked Image ===\n";
      std::filesystem::path markPath(markToFile);
      std::string alignedPath = markPath.parent_path().string() + "/" +
                                markPath.stem().string() + "_aligned" +
                                markPath.extension().string();

      // Use rendered image for OCR, mark on original image
      if (analyzer.alignAndMarkElements(renderResult.outputPath, markToFile,
                                        renderResult, alignedPath)) {
        std::cout << "OCR-aligned marked image created: " << alignedPath
                  << "\n";
      } else {
        std::cerr << "Warning: Failed to create OCR-aligned marked image\n";
      }
    }

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
