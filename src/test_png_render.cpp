#include "OCRAnalysis.hpp"
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

int main(int argc, char *argv[]) {
  std::cout << "=== PDF to PNG Rendering Test ===" << std::endl << std::endl;

  // Check for PDF file argument
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <pdf_file> [dpi] [output_dir] [bounds_mode] [overlay_image]"
              << std::endl;
    std::cerr << "  dpi: Resolution in dots per inch (default: 300)"
              << std::endl;
    std::cerr << "  output_dir: Directory to save PNG (default: images)"
              << std::endl;
    std::cerr << "  bounds_mode: 'crop' (default) or 'rect' for largest "
                 "rectangle"
              << std::endl;
    std::cerr << "  overlay_image: Path to an extra image to annotate with "
                 "element boxes (any resolution)"
              << std::endl;
    std::cerr << std::endl;
    std::cerr << "Example: " << argv[0] << " document.pdf 600 output rect"
              << std::endl;
    return 1;
  }

  std::string pdfPath = argv[1];
  double dpi = 300.0;
  std::string outputDir = "images";
  std::string overlayImagePath;

  // Auto-detect bounds mode from filename:
  // L2* files have crop marks; all others use largest rectangle
  std::string stem = std::filesystem::path(pdfPath).stem().string();
  ocr::OCRAnalysis::RenderBoundsMode boundsMode =
      (stem.size() >= 2 && stem[0] == 'L' && stem[1] == '2')
          ? ocr::OCRAnalysis::RenderBoundsMode::USE_CROP_MARKS
          : ocr::OCRAnalysis::RenderBoundsMode::USE_LARGEST_RECTANGLE;

  // Parse remaining arguments flexibly (order doesn't matter)
  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    std::string argLower = arg;
    std::transform(argLower.begin(), argLower.end(), argLower.begin(),
                   ::tolower);

    // Check if it's a bounds mode keyword
    if (argLower == "rect" || argLower == "rectangle") {
      boundsMode = ocr::OCRAnalysis::RenderBoundsMode::USE_LARGEST_RECTANGLE;
    } else if (argLower == "crop") {
      boundsMode = ocr::OCRAnalysis::RenderBoundsMode::USE_CROP_MARKS;
    }
    // Check if it's a number (DPI)
    else if (arg.find_first_not_of("0123456789.") == std::string::npos) {
      try {
        dpi = std::stod(arg);
      } catch (...) {
        std::cerr << "Warning: Could not parse DPI '" << arg
                  << "', using default 300" << std::endl;
      }
    }
    // Check if it looks like an image file (has a recognised image extension)
    else {
      std::string ext = std::filesystem::path(argLower).extension().string();
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
          ext == ".bmp" || ext == ".tif" || ext == ".tiff") {
        overlayImagePath = arg;
      } else {
        outputDir = arg;
      }
    }
  }

  std::string modeStr =
      (boundsMode == ocr::OCRAnalysis::RenderBoundsMode::USE_LARGEST_RECTANGLE)
          ? "USE_LARGEST_RECTANGLE"
          : "USE_CROP_MARKS";

  std::cout << "PDF file: " << pdfPath << std::endl;
  std::cout << "DPI: " << dpi << std::endl;
  std::cout << "Output directory: " << outputDir << std::endl;
  std::cout << "Bounds mode: " << modeStr << std::endl;
  std::cout << std::endl;

  // Create OCR analyzer
  ocr::OCRAnalysis analyzer;

  // Extract elements from PDF (and save images to output dir)
  std::cout << "Extracting elements from PDF..." << std::endl;
  ocr::OCRAnalysis::PDFElements elements =
      analyzer.extractPDFElements(pdfPath, 5.0, 5.0, outputDir);

  if (!elements.success) {
    std::cerr << "Failed to extract elements: " << elements.errorMessage
              << std::endl;
    return 1;
  }

  std::cout << "Extracted elements:" << std::endl;
  std::cout << "  Text lines: " << elements.textLineCount << std::endl;
  std::cout << "  Images: " << elements.imageCount << std::endl;
  std::cout << "  DataMatrix: " << elements.dataMatrixCount << std::endl;
  std::cout << "  Rectangles: " << elements.rectangleCount << std::endl;
  std::cout << "  Lines: " << elements.graphicLineCount << std::endl;
  std::cout << std::endl;

  // Display extracted images
  if (!elements.images.empty()) {
    std::cout << "Extracted images:" << std::endl;
    for (size_t i = 0; i < elements.images.size(); i++) {
      const auto &img = elements.images[i];
      std::cout << "  Image " << (i + 1) << ": x=" << img.x << ", y=" << img.y
                << ", w=" << img.displayWidth << ", h=" << img.displayHeight;
      if (!img.image.empty()) {
        std::cout << ", pixels=" << img.image.cols << "x" << img.image.rows
                  << ", channels=" << img.image.channels();
      } else {
        std::cout << ", (empty)";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  // Display DataMatrix barcodes
  if (!elements.dataMatrices.empty()) {
    std::cout << "Detected DataMatrix barcodes:" << std::endl;
    for (size_t i = 0; i < elements.dataMatrices.size(); i++) {
      const auto &dm = elements.dataMatrices[i];
      std::cout << "  DataMatrix " << (i + 1) << ": \"" << dm.text << "\""
                << ", x=" << dm.x << ", y=" << dm.y << ", w=" << dm.width
                << ", h=" << dm.height << ", source=";
      if (dm.sourceImageIndex >= 0) {
        std::cout << "image_" << (dm.sourceImageIndex + 1);
      } else {
        std::cout << "rasterised_page";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << "DEBUG: linesBoundingBox: x=" << elements.linesBoundingBoxX
            << ", y=" << elements.linesBoundingBoxY
            << ", w=" << elements.linesBoundingBoxWidth
            << ", h=" << elements.linesBoundingBoxHeight << std::endl;
  std::cout << std::endl;

  // Render to PNG
  std::cout << "Rendering to PNG..." << std::endl;
  ocr::OCRAnalysis::PNGRenderResult result = analyzer.renderElementsToPNG(
      elements, pdfPath, dpi, outputDir, boundsMode);

  if (!result.success) {
    std::cerr << "Failed to render PNG: " << result.errorMessage << std::endl;
    return 1;
  }

  // Display results
  std::cout << std::endl;
  std::cout << "+========================================+" << std::endl;
  std::cout << "|          RENDERING SUCCESSFUL          |" << std::endl;
  std::cout << "+========================================+" << std::endl;
  std::cout << std::endl;

  std::cout << "Output file: " << result.outputPath << std::endl;
  std::cout << "Image size: " << result.imageWidth << " x "
            << result.imageHeight << " pixels" << std::endl;
  std::cout << "Total elements rendered: " << result.elements.size()
            << std::endl;
  std::cout << std::endl;

  // Count elements by type
  int textCount = 0, imageCount = 0, rectCount = 0, lineCount = 0,
      dataMatrixCount = 0;
  for (const auto &elem : result.elements) {
    switch (elem.type) {
    case ocr::OCRAnalysis::RenderedElement::TEXT:
      textCount++;
      break;
    case ocr::OCRAnalysis::RenderedElement::IMAGE:
      imageCount++;
      break;
    case ocr::OCRAnalysis::RenderedElement::RECTANGLE:
      rectCount++;
      break;
    case ocr::OCRAnalysis::RenderedElement::LINE:
      lineCount++;
      break;
    case ocr::OCRAnalysis::RenderedElement::DATAMATRIX:
      dataMatrixCount++;
      break;
    }
  }

  std::cout << "Element breakdown:" << std::endl;
  std::cout << "  Text elements: " << textCount << std::endl;
  std::cout << "  Image elements: " << imageCount << std::endl;
  std::cout << "  Rectangle elements: " << rectCount << std::endl;
  std::cout << "  Line elements: " << lineCount << std::endl;
  std::cout << "  DataMatrix elements: " << dataMatrixCount << std::endl;
  std::cout << std::endl;

  // Display DataMatrix barcodes if any
  if (dataMatrixCount > 0) {
    std::cout << "DataMatrix barcodes detected:" << std::endl;
    int idx = 0;
    for (const auto &elem : result.elements) {
      if (elem.type == ocr::OCRAnalysis::RenderedElement::DATAMATRIX) {
        idx++;
        std::cout << "  " << idx << ". \"" << elem.barcodeText << "\"";
        std::cout << "  centre=(" << std::fixed << std::setprecision(4)
                  << elem.relativeX << ", " << elem.relativeY << ")";
        std::cout << "  size=(" << elem.relativeWidth << " x "
                  << elem.relativeHeight << ")";
        if (!elem.image.empty()) {
          std::cout << "  crop=" << elem.image.cols << "x" << elem.image.rows;
        }
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }

  // Display ALL elements with relative coordinates
  std::cout << "All elements (with relative coordinates):" << std::endl;
  std::cout
      << "+------+------------+----------+----------+---------+----------+"
         "----------------------+------------------+----------+"
      << std::endl;
  std::cout
      << "| #    | Type       | CX (rel) | CY (rel) | W (rel) | H (rel)  |"
         " Detail               | Font             | OCR Conf |"
      << std::endl;
  std::cout
      << "+------+------------+----------+----------+---------+----------+"
         "----------------------+------------------+----------+"
      << std::endl;

  for (size_t i = 0; i < result.elements.size(); i++) {
    const auto &elem = result.elements[i];

    // Type string
    std::string typeStr;
    switch (elem.type) {
    case ocr::OCRAnalysis::RenderedElement::TEXT:
      typeStr = "TEXT";
      break;
    case ocr::OCRAnalysis::RenderedElement::IMAGE:
      typeStr = "IMAGE";
      break;
    case ocr::OCRAnalysis::RenderedElement::RECTANGLE:
      typeStr = "RECT";
      break;
    case ocr::OCRAnalysis::RenderedElement::LINE:
      typeStr = "LINE";
      break;
    case ocr::OCRAnalysis::RenderedElement::DATAMATRIX:
      typeStr = "DATAMATRIX";
      break;
    }

    // Detail string depends on type
    std::string detail;
    if (elem.type == ocr::OCRAnalysis::RenderedElement::TEXT) {
      detail = elem.text;
    } else if (elem.type == ocr::OCRAnalysis::RenderedElement::DATAMATRIX) {
      detail = elem.barcodeText;
    } else if (elem.type == ocr::OCRAnalysis::RenderedElement::IMAGE) {
      detail = elem.image.empty()
                   ? "(empty)"
                   : std::to_string(elem.image.cols) + "x" +
                         std::to_string(elem.image.rows) + " ch" +
                         std::to_string(elem.image.channels());
    } else if (elem.type == ocr::OCRAnalysis::RenderedElement::LINE) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(4) << "→(" << elem.relativeX2
          << "," << elem.relativeY2 << ")";
      detail = oss.str();
    }
    if (detail.length() > 20) {
      detail = detail.substr(0, 17) + "...";
    }

    // Font column (TEXT elements only): "name size[B][I]", truncated to 16
    std::string fontStr;
    if (elem.type == ocr::OCRAnalysis::RenderedElement::TEXT) {
      std::ostringstream fs;
      fs << std::fixed << std::setprecision(1) << elem.fontSize;
      std::string attrs = (elem.isBold ? "B" : "") + std::string(elem.isItalic ? "I" : "");
      std::string candidate = elem.fontName + " " + fs.str() + attrs;
      if (candidate.length() > 16)
        candidate = candidate.substr(0, 13) + "...";
      fontStr = candidate;
    }

    std::cout << "| " << std::setw(4) << std::left << i + 1 << " | "
              << std::setw(10) << std::left << typeStr << " | " << std::setw(8)
              << std::right << std::fixed << std::setprecision(4)
              << elem.relativeX << " | " << std::setw(8) << elem.relativeY
              << " | " << std::setw(7) << elem.relativeWidth << " | "
              << std::setw(8) << elem.relativeHeight << " | " << std::setw(20)
              << std::left << detail << " | " << std::setw(16) << std::left
              << fontStr << " | ";
    // OCR confidence column
    if (elem.ocrConfidence >= 0.0f) {
      std::cout << std::setw(7) << std::right << std::fixed
                << std::setprecision(1) << elem.ocrConfidence << "%";
    } else {
      std::cout << std::setw(8) << " ";
    }
    std::cout << " |" << std::endl;
  }
  // Append ignored text lines (symbol fonts such as Wingdings) to the table
  for (size_t i = 0; i < elements.ignoredTextLines.size(); i++) {
    const auto &tr = elements.ignoredTextLines[i];
    std::ostringstream fs;
    fs << std::fixed << std::setprecision(1) << tr.fontSize;
    std::string attrs =
        (tr.isBold ? "B" : "") + std::string(tr.isItalic ? "I" : "");
    std::string fontStr = tr.fontName + " " + fs.str() + attrs;
    if (fontStr.length() > 16)
      fontStr = fontStr.substr(0, 13) + "...";
    std::string detail = tr.text;
    if (detail.length() > 20)
      detail = detail.substr(0, 17) + "...";
    std::cout << "| " << std::setw(4) << std::left
              << (result.elements.size() + i + 1) << " | " << std::setw(10)
              << std::left << "TEXT(ign)" << " | " << std::setw(8) << std::right
              << "     n/a" << " | " << std::setw(8) << "     n/a" << " | "
              << std::setw(7) << "    n/a" << " | " << std::setw(8) << "     n/a"
              << " | " << std::setw(20) << std::left << detail << " | "
              << std::setw(16) << std::left << fontStr << " |          |"
              << std::endl;
  }
  std::cout
      << "+------+------------+----------+----------+---------+----------+"
         "----------------------+------------------+----------+"
      << std::endl;
  std::cout << std::endl;

  // Display statistics
  std::cout << "Coordinate statistics (relative):" << std::endl;

  double minX = 1.0, minY = 1.0, maxX = 0.0, maxY = 0.0;
  for (const auto &elem : result.elements) {
    double left = elem.relativeX - elem.relativeWidth / 2.0;
    double top = elem.relativeY - elem.relativeHeight / 2.0;
    double right = elem.relativeX + elem.relativeWidth / 2.0;
    double bottom = elem.relativeY + elem.relativeHeight / 2.0;
    minX = std::min(minX, left);
    minY = std::min(minY, top);
    maxX = std::max(maxX, right);
    maxY = std::max(maxY, bottom);
  }

  std::cout << "  Bounding box: (" << minX << ", " << minY << ") to (" << maxX
            << ", " << maxY << ")" << std::endl;
  std::cout << "  Content area: " << (maxX - minX) << " x " << (maxY - minY)
            << " (relative)" << std::endl;
  std::cout << std::endl;

  // If an overlay image was supplied, draw the same element boxes onto it
  // (scales automatically because relative coordinates are used).
  if (!overlayImagePath.empty()) {
    std::cout << "Overlay image: " << overlayImagePath << std::endl;
    cv::Mat overlay = cv::imread(overlayImagePath, cv::IMREAD_COLOR);
    if (overlay.empty()) {
      std::cerr << "Warning: Could not load overlay image: " << overlayImagePath
                << std::endl;
    } else {
      cv::Mat annotated =
          ocr::OCRAnalysis::drawElementBoxes(overlay, result.elements);
      std::string overlayStem =
          std::filesystem::path(overlayImagePath).stem().string();
      std::string overlayOut = outputDir + "/" + overlayStem + "_annotated.png";
      if (cv::imwrite(overlayOut, annotated)) {
        std::cout << "Overlay annotated image saved: " << overlayOut
                  << std::endl;
      } else {
        std::cerr << "Warning: Failed to save overlay annotated image: "
                  << overlayOut << std::endl;
      }
    }
    std::cout << std::endl;
  }

  std::cout << "Test completed successfully!" << std::endl;

  return 0;
}
