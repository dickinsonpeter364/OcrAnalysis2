#include "OCRAnalysis.hpp"
#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
  std::cout << "=== PDF to PNG Rendering Test ===" << std::endl << std::endl;

  // Check for PDF file argument
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <pdf_file> [dpi] [output_dir]"
              << std::endl;
    std::cerr << "  dpi: Resolution in dots per inch (default: 300)"
              << std::endl;
    std::cerr << "  output_dir: Directory to save PNG (default: images)"
              << std::endl;
    std::cerr << std::endl;
    std::cerr << "Example: " << argv[0] << " document.pdf 600 output"
              << std::endl;
    return 1;
  }

  std::string pdfPath = argv[1];
  double dpi = (argc >= 3) ? std::stod(argv[2]) : 300.0;
  std::string outputDir = (argc >= 4) ? argv[3] : "images";

  std::cout << "PDF file: " << pdfPath << std::endl;
  std::cout << "DPI: " << dpi << std::endl;
  std::cout << "Output directory: " << outputDir << std::endl;
  std::cout << std::endl;

  // Create OCR analyzer
  ocr::OCRAnalysis analyzer;

  // Extract elements from PDF
  std::cout << "Extracting elements from PDF..." << std::endl;
  ocr::OCRAnalysis::PDFElements elements =
      analyzer.extractPDFElements(pdfPath, 5.0, 5.0);

  if (!elements.success) {
    std::cerr << "Failed to extract elements: " << elements.errorMessage
              << std::endl;
    return 1;
  }

  std::cout << "Extracted elements:" << std::endl;
  std::cout << "  Text lines: " << elements.textLineCount << std::endl;
  std::cout << "  Images: " << elements.imageCount << std::endl;
  std::cout << "  Rectangles: " << elements.rectangleCount << std::endl;
  std::cout << "  Lines: " << elements.graphicLineCount << std::endl;
  std::cout << std::endl;

  // Render to PNG
  std::cout << "Rendering to PNG..." << std::endl;
  ocr::OCRAnalysis::PNGRenderResult result =
      analyzer.renderElementsToPNG(elements, pdfPath, dpi, outputDir);

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
  int textCount = 0, imageCount = 0, rectCount = 0, lineCount = 0;
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
    }
  }

  std::cout << "Element breakdown:" << std::endl;
  std::cout << "  Text elements: " << textCount << std::endl;
  std::cout << "  Image elements: " << imageCount << std::endl;
  std::cout << "  Rectangle elements: " << rectCount << std::endl;
  std::cout << "  Line elements: " << lineCount << std::endl;
  std::cout << std::endl;

  // Display first few text elements with pixel coordinates
  std::cout << "First 10 text elements (with pixel coordinates):" << std::endl;
  std::cout
      << "+------+--------+--------+-------+--------+----------------------+"
      << std::endl;
  std::cout
      << "| #    | X (px) | Y (px) | W(px) | H (px) | Text                 |"
      << std::endl;
  std::cout
      << "+------+--------+--------+-------+--------+----------------------+"
      << std::endl;

  int count = 0;
  for (const auto &elem : result.elements) {
    if (elem.type == ocr::OCRAnalysis::RenderedElement::TEXT && count < 10) {
      std::string displayText = elem.text;
      if (displayText.length() > 20) {
        displayText = displayText.substr(0, 17) + "...";
      }

      std::cout << "| " << std::setw(4) << std::left << count + 1 << " | "
                << std::setw(6) << std::right << elem.pixelX << " | "
                << std::setw(6) << elem.pixelY << " | " << std::setw(5)
                << elem.pixelWidth << " | " << std::setw(6) << elem.pixelHeight
                << " | " << std::setw(20) << std::left << displayText << " |"
                << std::endl;
      count++;
    }
  }
  std::cout
      << "+------+--------+--------+-------+--------+----------------------+"
      << std::endl;
  std::cout << std::endl;

  // Display image elements
  if (imageCount > 0) {
    std::cout << "Image elements (with pixel coordinates):" << std::endl;
    std::cout << "+------+--------+--------+-------+--------+----------+"
              << std::endl;
    std::cout << "| #    | X (px) | Y (px) | W(px) | H (px) | Channels |"
              << std::endl;
    std::cout << "+------+--------+--------+-------+--------+----------+"
              << std::endl;

    count = 0;
    for (const auto &elem : result.elements) {
      if (elem.type == ocr::OCRAnalysis::RenderedElement::IMAGE) {
        int channels = elem.image.empty() ? 0 : elem.image.channels();
        std::cout << "| " << std::setw(4) << std::left << count + 1 << " | "
                  << std::setw(6) << std::right << elem.pixelX << " | "
                  << std::setw(6) << elem.pixelY << " | " << std::setw(5)
                  << elem.pixelWidth << " | " << std::setw(6)
                  << elem.pixelHeight << " | " << std::setw(8) << channels
                  << " |" << std::endl;
        count++;
      }
    }
    std::cout << "+------+--------+--------+-------+--------+----------+"
              << std::endl;
    std::cout << std::endl;
  }

  // Display statistics
  std::cout << "Coordinate statistics:" << std::endl;

  int minX = INT_MAX, minY = INT_MAX, maxX = 0, maxY = 0;
  for (const auto &elem : result.elements) {
    minX = std::min(minX, elem.pixelX);
    minY = std::min(minY, elem.pixelY);
    maxX = std::max(maxX, elem.pixelX + elem.pixelWidth);
    maxY = std::max(maxY, elem.pixelY + elem.pixelHeight);
  }

  std::cout << "  Bounding box: (" << minX << ", " << minY << ") to (" << maxX
            << ", " << maxY << ")" << std::endl;
  std::cout << "  Content area: " << (maxX - minX) << " x " << (maxY - minY)
            << " pixels" << std::endl;
  std::cout << std::endl;

  std::cout << "Test completed successfully!" << std::endl;

  return 0;
}
