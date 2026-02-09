#include "OCRAnalysis.hpp"

#include <cstring>
#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
  std::cout << "=== PDF Text Extraction Test ===" << std::endl;
  std::cout << "Tesseract version: " << ocr::OCRAnalysis::getTesseractVersion()
            << std::endl
            << std::endl;

  // Check for PDF file argument
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <pdf_file> [--lines]" << std::endl;
    std::cerr << "  --lines  Extract at line level instead of word level"
              << std::endl;
    return 1;
  }

  std::string pdfPath = argv[1];

  // Check for --lines flag
  ocr::OCRAnalysis::PDFExtractionLevel level =
      ocr::OCRAnalysis::PDFExtractionLevel::Word;
  for (int i = 2; i < argc; i++) {
    if (std::strcmp(argv[i], "--lines") == 0) {
      level = ocr::OCRAnalysis::PDFExtractionLevel::Line;
    }
  }

  std::cout << "Loading PDF: " << pdfPath << std::endl;
  std::cout << "Extraction level: "
            << (level == ocr::OCRAnalysis::PDFExtractionLevel::Line ? "Line"
                                                                    : "Word")
            << std::endl
            << std::endl;

  // Create OCR analyzer (initialization not required for PDF extraction)
  ocr::OCRAnalysis analyzer;

  // Extract text from PDF
  std::cout << "Extracting text from PDF..." << std::endl;
  ocr::OCRResult result = analyzer.extractTextFromPDF(pdfPath, level);

  if (!result.success) {
    std::cerr << "Failed to extract text: " << result.errorMessage << std::endl;
    return 1;
  }

  std::cout << "Processing time: " << result.processingTimeMs << " ms"
            << std::endl;
  std::cout << "Text regions found: " << result.regions.size() << std::endl
            << std::endl;

  // Count by orientation
  int horizontalCount = 0;
  int verticalCount = 0;
  int unknownCount = 0;

  for (const auto &region : result.regions) {
    switch (region.orientation) {
    case ocr::TextOrientation::Horizontal:
      horizontalCount++;
      break;
    case ocr::TextOrientation::Vertical:
      verticalCount++;
      break;
    default:
      unknownCount++;
      break;
    }
  }

  std::cout << "=== Orientation Summary ===" << std::endl;
  std::cout << "Horizontal: " << horizontalCount << " regions" << std::endl;
  std::cout << "Vertical:   " << verticalCount << " regions" << std::endl;
  std::cout << "Unknown:    " << unknownCount << " regions" << std::endl
            << std::endl;

  // Print text regions with position information
  std::cout << "=== Text Regions with Position ===" << std::endl;
  std::cout << std::string(100, '-') << std::endl;
  std::cout << std::left << std::setw(6) << "Page" << std::setw(12)
            << "Orientation" << std::setw(30) << "Position (x,y w×h)"
            << "Text" << std::endl;
  std::cout << std::string(100, '-') << std::endl;

  int count = 0;
  for (const auto &region : result.regions) {
    count++;
    if (count > 50) {
      std::cout << "... (showing first 50 of " << result.regions.size()
                << " regions)" << std::endl;
      break;
    }

    std::string orientStr =
        (region.orientation == ocr::TextOrientation::Horizontal) ? "Horizontal"
        : (region.orientation == ocr::TextOrientation::Vertical) ? "Vertical"
                                                                 : "Unknown";

    // Format position string
    std::string posStr = "(" + std::to_string(region.boundingBox.x) + "," +
                         std::to_string(region.boundingBox.y) + " " +
                         std::to_string(region.boundingBox.width) + "×" +
                         std::to_string(region.boundingBox.height) + ")";

    // Truncate text for display
    std::string displayText = region.text;
    std::replace(displayText.begin(), displayText.end(), '\n', ' ');
    if (displayText.length() > 40) {
      displayText = displayText.substr(0, 37) + "...";
    }

    std::cout << std::left << std::setw(6) << region.level << std::setw(12)
              << orientStr << std::setw(30) << posStr << "\"" << displayText
              << "\"" << std::endl;
  }

  std::cout << std::string(100, '-') << std::endl;

  // Print full extracted text
  std::cout << std::endl << "=== Full Extracted Text ===" << std::endl;
  std::cout << std::string(60, '-') << std::endl;

  // Limit output if very long
  if (result.fullText.length() > 2000) {
    std::cout << result.fullText.substr(0, 2000) << std::endl;
    std::cout << "... (truncated, total " << result.fullText.length()
              << " characters)" << std::endl;
  } else {
    std::cout << result.fullText << std::endl;
  }

  std::cout << std::string(60, '-') << std::endl;

  std::cout << std::endl << "Extraction completed successfully!" << std::endl;

  return 0;
}
