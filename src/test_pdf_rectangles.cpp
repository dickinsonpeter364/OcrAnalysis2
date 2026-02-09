#include "OCRAnalysis.hpp"

#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
  std::cout << "=== PDF Rectangle Extraction Test ===" << std::endl
            << std::endl;

  // Check for PDF file argument
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <pdf_file> [min_size]" << std::endl;
    std::cerr << "  min_size  Minimum rectangle size in points (default: 5)"
              << std::endl;
    return 1;
  }

  std::string pdfPath = argv[1];
  double minSize = (argc >= 3) ? std::stod(argv[2]) : 5.0;

  std::cout << "Loading PDF: " << pdfPath << std::endl;
  std::cout << "Minimum size: " << minSize << " points" << std::endl
            << std::endl;

  // Create OCR analyzer
  ocr::OCRAnalysis analyzer;

  // Extract rectangles from first page of PDF
  std::cout << "Extracting rectangles from first page of PDF..." << std::endl;
  ocr::OCRAnalysis::PDFRectanglesResult result =
      analyzer.extractRectanglesFromPDF(pdfPath, minSize);

  if (!result.success) {
    std::cerr << "Failed to extract rectangles: " << result.errorMessage
              << std::endl;
    return 1;
  }

  std::cout << "Processing time: " << std::fixed << std::setprecision(2)
            << result.processingTimeMs << " ms" << std::endl;
  std::cout << "Rectangles found: " << result.rectangles.size() << std::endl
            << std::endl;

  if (result.rectangles.empty()) {
    std::cout << "No rectangles found in this PDF." << std::endl;
    return 0;
  }

  // Display rectangles
  std::cout << "=== Rectangles (coordinates in points, origin bottom-left) ==="
            << std::endl;
  std::cout << std::string(100, '-') << std::endl;
  std::cout << std::left << std::setw(6) << "Page" << std::setw(22)
            << "Position (x,y)" << std::setw(20) << "Size (w×h)"
            << std::setw(12) << "Line Width" << std::setw(10) << "Filled"
            << "Stroked" << std::endl;
  std::cout << std::string(100, '-') << std::endl;

  for (const auto &rect : result.rectangles) {
    std::ostringstream posStr;
    posStr << std::fixed << std::setprecision(1) << "(" << rect.x << ", "
           << rect.y << ")";

    std::ostringstream sizeStr;
    sizeStr << std::fixed << std::setprecision(1) << rect.width << " × "
            << rect.height;

    std::cout << std::left << std::setw(6) << rect.pageNumber << std::setw(22)
              << posStr.str() << std::setw(20) << sizeStr.str() << std::setw(12)
              << std::fixed << std::setprecision(2) << rect.lineWidth
              << std::setw(10) << (rect.filled ? "Yes" : "No")
              << (rect.stroked ? "Yes" : "No") << std::endl;
  }

  std::cout << std::string(100, '-') << std::endl;
  std::cout << std::endl;

  // Summary by size
  int smallRects = 0, mediumRects = 0, largeRects = 0;
  for (const auto &rect : result.rectangles) {
    double area = rect.width * rect.height;
    if (area < 1000)
      smallRects++;
    else if (area < 10000)
      mediumRects++;
    else
      largeRects++;
  }

  std::cout << "Size Distribution:" << std::endl;
  std::cout << "  Small (<1000 sq pts):  " << smallRects << std::endl;
  std::cout << "  Medium (1000-10000):   " << mediumRects << std::endl;
  std::cout << "  Large (>10000):        " << largeRects << std::endl;

  std::cout << std::endl << "Extraction completed successfully!" << std::endl;

  return 0;
}
