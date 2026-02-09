#include "OCRAnalysis.hpp"

#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
  std::cout << "=== PDF Line Extraction Test ===" << std::endl << std::endl;

  // Check for PDF file argument
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <pdf_file> [min_length]"
              << std::endl;
    std::cerr << "  min_length  Minimum line length in points (default: 5)"
              << std::endl;
    return 1;
  }

  std::string pdfPath = argv[1];
  double minLength = (argc >= 3) ? std::stod(argv[2]) : 5.0;

  std::cout << "Loading PDF: " << pdfPath << std::endl;
  std::cout << "Minimum length: " << minLength << " points" << std::endl
            << std::endl;

  // Create OCR analyzer
  ocr::OCRAnalysis analyzer;

  // Extract lines from first page of PDF
  std::cout << "Extracting lines from first page of PDF..." << std::endl;
  ocr::OCRAnalysis::PDFLinesResult result =
      analyzer.extractLinesFromPDF(pdfPath, minLength);

  if (!result.success) {
    std::cerr << "Failed to extract lines: " << result.errorMessage
              << std::endl;
    return 1;
  }

  std::cout << "Processing time: " << std::fixed << std::setprecision(2)
            << result.processingTimeMs << " ms" << std::endl;
  std::cout << "Lines found: " << result.lines.size() << std::endl << std::endl;

  if (result.lines.empty()) {
    std::cout << "No lines found in this PDF." << std::endl;
    return 0;
  }

  // Count by orientation
  int horizontalCount = 0, verticalCount = 0, diagonalCount = 0;
  for (const auto &line : result.lines) {
    if (line.isHorizontal)
      horizontalCount++;
    else if (line.isVertical)
      verticalCount++;
    else
      diagonalCount++;
  }

  std::cout << "Orientation Summary:" << std::endl;
  std::cout << "  Horizontal: " << horizontalCount << std::endl;
  std::cout << "  Vertical:   " << verticalCount << std::endl;
  std::cout << "  Diagonal:   " << diagonalCount << std::endl << std::endl;

  // Display lines
  std::cout << "=== Lines (coordinates in points, origin bottom-left) ==="
            << std::endl;
  std::cout << std::string(110, '-') << std::endl;
  std::cout << std::left << std::setw(6) << "Page" << std::setw(30)
            << "From (x1,y1)" << std::setw(30) << "To (x2,y2)" << std::setw(12)
            << "Length" << std::setw(10) << "Width"
            << "Orientation" << std::endl;
  std::cout << std::string(110, '-') << std::endl;

  // Show first 30 lines to avoid overwhelming output
  int shown = 0;
  for (const auto &line : result.lines) {
    if (shown >= 30) {
      std::cout << "... and " << (result.lines.size() - 30) << " more lines"
                << std::endl;
      break;
    }

    std::ostringstream fromStr, toStr;
    fromStr << std::fixed << std::setprecision(1) << "(" << line.x1 << ", "
            << line.y1 << ")";
    toStr << std::fixed << std::setprecision(1) << "(" << line.x2 << ", "
          << line.y2 << ")";

    std::string orientation = line.isHorizontal
                                  ? "Horizontal"
                                  : (line.isVertical ? "Vertical" : "Diagonal");

    std::cout << std::left << std::setw(6) << line.pageNumber << std::setw(30)
              << fromStr.str() << std::setw(30) << toStr.str() << std::setw(12)
              << std::fixed << std::setprecision(1) << line.length
              << std::setw(10) << std::fixed << std::setprecision(2)
              << line.lineWidth << orientation << std::endl;
    shown++;
  }

  std::cout << std::string(110, '-') << std::endl;
  std::cout << std::endl << "Extraction completed successfully!" << std::endl;

  return 0;
}
