#include "OCRAnalysis.hpp"
#include <filesystem>
#include <iostream>
#include <string>

/**
 * @brief Test program for stripBleedMarks functionality
 *
 * This program demonstrates how to use the stripBleedMarks method to remove
 * bleed marks from a PDF file and render the result to a PNG image.
 *
 * Usage: test_strip_bleed_marks <pdf_file> [output_dir] [dpi]
 */

void printUsage(const char *programName) {
  std::cout << "Usage: " << programName << " <pdf_file> [output_dir] [dpi]\n";
  std::cout << "\n";
  std::cout << "Arguments:\n";
  std::cout << "  pdf_file    - Path to the input PDF file\n";
  std::cout << "  output_dir  - Directory to save the output PNG (default: "
               "current directory)\n";
  std::cout << "  dpi         - Resolution in dots per inch (default: 300)\n";
  std::cout << "\n";
  std::cout << "Example:\n";
  std::cout << "  " << programName << " document.pdf\n";
  std::cout << "  " << programName << " document.pdf output 600\n";
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  std::string pdfPath = argv[1];
  std::string outputDir = (argc >= 3) ? argv[2] : ".";
  double dpi = (argc >= 4) ? std::stod(argv[3]) : 300.0;

  // Check if the PDF file exists
  if (!std::filesystem::exists(pdfPath)) {
    std::cerr << "Error: PDF file not found: " << pdfPath << std::endl;
    return 1;
  }

  std::cout << "=== Strip Bleed Marks Test ===\n";
  std::cout << "Input PDF: " << pdfPath << "\n";
  std::cout << "Output directory: " << outputDir << "\n";
  std::cout << "DPI: " << dpi << "\n";
  std::cout << "==============================\n\n";

  try {
    // Create OCRAnalysis instance
    ocr::OCRAnalysis analyzer;

    // Strip bleed marks and get filtered PDF elements
    std::cout << "Stripping bleed marks...\n";
    auto filteredElements = analyzer.stripBleedMarks(pdfPath);

    if (!filteredElements.success) {
      std::cerr << "Error: " << filteredElements.errorMessage << std::endl;
      return 1;
    }

    std::cout << "✓ Bleed marks stripped successfully\n";
    std::cout << "  Processing time: " << filteredElements.processingTimeMs
              << " ms\n";
    std::cout << "\nFiltered PDF Elements:\n";
    std::cout << "  Text lines: " << filteredElements.textLineCount << "\n";
    std::cout << "  Images: " << filteredElements.imageCount << "\n";
    std::cout << "  Rectangles: " << filteredElements.rectangleCount << "\n";
    std::cout << "  Lines: " << filteredElements.graphicLineCount << "\n";
    std::cout << "  Page size: " << filteredElements.pageWidth << " x "
              << filteredElements.pageHeight << " points\n\n";

    // Render the filtered elements to PNG
    std::cout << "Rendering filtered elements to PNG...\n";
    auto renderResult =
        analyzer.renderElementsToPNG(filteredElements, pdfPath, dpi, outputDir);

    if (!renderResult.success) {
      std::cerr << "Error rendering PNG: " << renderResult.errorMessage
                << std::endl;
      return 1;
    }

    std::cout << "✓ PNG rendered successfully\n";
    std::cout << "  Output file: " << renderResult.outputPath << "\n";
    std::cout << "  Image size: " << renderResult.imageWidth << " x "
              << renderResult.imageHeight << " pixels\n";
    std::cout << "  Rendered elements: " << renderResult.elements.size()
              << "\n\n";

    std::cout << "=== Test Completed Successfully ===\n";
    std::cout << "Output PNG: " << renderResult.outputPath << "\n";

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
