#include "OCRAnalysis.hpp"
#include <iomanip>
#include <iostream>


int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <pdf_file> [output_dir] [dpi]"
              << std::endl;
    return 1;
  }

  std::string pdfPath = argv[1];
  std::string outputDir = (argc > 2) ? argv[2] : "images";
  double dpi = (argc > 3) ? std::stod(argv[3]) : 300.0;

  try {
    ocr::OCRAnalysis analyzer;

    // Strip bleed marks and get filtered PDF elements
    std::cout << "Processing PDF: " << pdfPath << "\n\n";
    auto filteredElements = analyzer.stripBleedMarks(pdfPath);

    if (!filteredElements.success) {
      std::cerr << "Error: " << filteredElements.errorMessage << std::endl;
      return 1;
    }

    // Render the filtered elements to PNG
    auto renderResult =
        analyzer.renderElementsToPNG(filteredElements, pdfPath, dpi, outputDir);

    if (!renderResult.success) {
      std::cerr << "Error rendering PNG: " << renderResult.errorMessage
                << std::endl;
      return 1;
    }

    std::cout << "=== PNG Rendering Complete ===\n";
    std::cout << "Output: " << renderResult.outputPath << "\n";
    std::cout << "Size: " << renderResult.imageWidth << "x"
              << renderResult.imageHeight << " pixels\n\n";

    std::cout << "=== Rendered Elements (" << renderResult.elements.size()
              << " total) ===\n\n";

    int textCount = 0, imageCount = 0, lineCount = 0, rectCount = 0;

    for (size_t i = 0; i < renderResult.elements.size(); i++) {
      const auto &elem = renderResult.elements[i];

      std::cout << "Element " << i << ": ";

      switch (elem.type) {
      case ocr::OCRAnalysis::RenderedElement::TEXT:
        std::cout << "TEXT\n";
        std::cout << "  Position: (" << elem.pixelX << ", " << elem.pixelY
                  << ")\n";
        std::cout << "  Size: " << elem.pixelWidth << "x" << elem.pixelHeight
                  << " px\n";
        std::cout << "  Content: \"" << elem.text << "\"\n";
        std::cout << "  Font: " << elem.fontName << " " << elem.fontSize
                  << "pt";
        if (elem.isBold)
          std::cout << " bold";
        if (elem.isItalic)
          std::cout << " italic";
        std::cout << "\n";
        textCount++;
        break;

      case ocr::OCRAnalysis::RenderedElement::IMAGE:
        std::cout << "IMAGE\n";
        std::cout << "  Position: (" << elem.pixelX << ", " << elem.pixelY
                  << ")\n";
        std::cout << "  Size: " << elem.pixelWidth << "x" << elem.pixelHeight
                  << " px\n";
        std::cout << "  Image data: " << elem.image.cols << "x"
                  << elem.image.rows << " (" << elem.image.channels()
                  << " channels)\n";
        std::cout << "  Rotation: " << (elem.rotationAngle * 180.0 / 3.14159265)
                  << " degrees\n";
        imageCount++;
        break;

      case ocr::OCRAnalysis::RenderedElement::LINE:
        std::cout << "LINE\n";
        std::cout << "  Start: (" << elem.pixelX << ", " << elem.pixelY
                  << ")\n";
        std::cout << "  End: (" << elem.pixelX2 << ", " << elem.pixelY2
                  << ")\n";
        std::cout << "  Bounding box: " << elem.pixelWidth << "x"
                  << elem.pixelHeight << " px\n";
        lineCount++;
        break;

      case ocr::OCRAnalysis::RenderedElement::RECTANGLE:
        std::cout << "RECTANGLE\n";
        std::cout << "  Position: (" << elem.pixelX << ", " << elem.pixelY
                  << ")\n";
        std::cout << "  Size: " << elem.pixelWidth << "x" << elem.pixelHeight
                  << " px\n";
        rectCount++;
        break;
      }
      std::cout << "\n";
    }

    std::cout << "=== Summary ===\n";
    std::cout << "Text elements: " << textCount << "\n";
    std::cout << "Image elements: " << imageCount << "\n";
    std::cout << "Line elements: " << lineCount << "\n";
    std::cout << "Rectangle elements: " << rectCount << "\n";

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
