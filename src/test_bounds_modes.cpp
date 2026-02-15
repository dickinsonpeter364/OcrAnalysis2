#include "OCRAnalysis.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <pdf_file> [dpi] [bounds_mode] [output_dir] [mark_to_file]"
              << std::endl;
    std::cerr << "  dpi: resolution (default: 300)" << std::endl;
    std::cerr << "  bounds_mode: 'crop' (default), 'rect', or 'relmap'"
              << std::endl;
    std::cerr << "  output_dir: directory for output (default: 'images')"
              << std::endl;
    std::cerr
        << "  mark_to_file: optional image file to mark with element boxes"
        << std::endl;
    std::cerr << "\nExamples:" << std::endl;
    std::cerr << "  " << argv[0] << " document.pdf" << std::endl;
    std::cerr << "  " << argv[0] << " document.pdf 1200 rect" << std::endl;
    std::cerr << "  " << argv[0] << " document.pdf 1200 relmap images photo.bmp"
              << std::endl;
    std::cerr << "  " << argv[0]
              << " document.pdf 1200 rect images rendered.png" << std::endl;
    return 1;
  }

  try {
    std::string pdfPath = argv[1];
    double dpi = 300.0;
    std::string outputDir = "images";
    ocr::OCRAnalysis::RenderBoundsMode boundsMode =
        ocr::OCRAnalysis::RenderBoundsMode::USE_CROP_MARKS;
    std::string markToFile = "";
    bool useRelativeMap = false;

    // Parse remaining arguments flexibly
    for (int i = 2; i < argc; i++) {
      std::string arg = argv[i];
      std::string argLower = arg;
      std::transform(argLower.begin(), argLower.end(), argLower.begin(),
                     ::tolower);

      // Check if it's a bounds mode
      if (argLower == "crop" || argLower == "rect" || argLower == "rectangle" ||
          argLower == "relmap") {
        if (argLower == "rect" || argLower == "rectangle") {
          boundsMode =
              ocr::OCRAnalysis::RenderBoundsMode::USE_LARGEST_RECTANGLE;
        } else if (argLower == "relmap") {
          useRelativeMap = true;
          // relmap defaults to USE_LARGEST_RECTANGLE unless 'crop' was also
          // specified
          boundsMode =
              ocr::OCRAnalysis::RenderBoundsMode::USE_LARGEST_RECTANGLE;
        }
      }
      // Check if it's a number (DPI)
      else if (arg.find_first_not_of("0123456789.") == std::string::npos) {
        try {
          dpi = std::stod(arg);
        } catch (...) {
          std::cerr << "Warning: Could not parse DPI value '" << arg
                    << "', using default 300" << std::endl;
        }
      }
      // Check if it's a file path (contains . or / or \)
      else if (arg.find('.') != std::string::npos ||
               arg.find('/') != std::string::npos ||
               arg.find('\\') != std::string::npos) {
        // If we haven't set markToFile yet and this looks like an image, use it
        if (markToFile.empty() &&
            (argLower.ends_with(".png") || argLower.ends_with(".jpg") ||
             argLower.ends_with(".jpeg") || argLower.ends_with(".bmp"))) {
          markToFile = arg;
        } else if (outputDir == "images") {
          // Otherwise it might be an output directory
          outputDir = arg;
        } else {
          markToFile = arg;
        }
      }
      // Otherwise assume it's an output directory
      else {
        outputDir = arg;
      }
    }

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

    if (useRelativeMap) {
      // Use createRelativeMap instead of renderElementsToPNG
      std::cout << "Creating relative map with bounds mode: "
                << (boundsMode ==
                            ocr::OCRAnalysis::RenderBoundsMode::USE_CROP_MARKS
                        ? "USE_CROP_MARKS"
                        : "USE_LARGEST_RECTANGLE")
                << "\n\n";

      auto relMapResult =
          analyzer.createRelativeMap(elements, boundsMode, dpi, markToFile);

      if (!relMapResult.success) {
        std::cerr << "Error creating relative map: "
                  << relMapResult.errorMessage << std::endl;
        return 1;
      }

      std::cout << "=== Relative Map Complete ===\n";
      std::cout << "Bounds: (" << relMapResult.boundsX << ", "
                << relMapResult.boundsY
                << ") size: " << relMapResult.boundsWidth << " x "
                << relMapResult.boundsHeight << " pt\n";
      std::cout << "Total elements: " << relMapResult.elements.size() << "\n\n";

      // Print each element
      for (size_t i = 0; i < relMapResult.elements.size(); i++) {
        const auto &elem = relMapResult.elements[i];
        std::cout << "Element " << i << ": ";
        if (elem.type == ocr::OCRAnalysis::RelativeElement::TEXT) {
          std::cout << "TEXT \"" << elem.text << "\"";
        } else if (elem.type == ocr::OCRAnalysis::RelativeElement::IMAGE) {
          std::cout << "IMAGE";
        }
        std::cout << " centre=(" << elem.relativeX << ", " << elem.relativeY
                  << ") size=(" << elem.relativeWidth << ", "
                  << elem.relativeHeight << ")\n";
      }

    } else {
      // Original renderElementsToPNG path
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
    }

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
