#include "OCRAnalysis.hpp"

#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
  std::cout << "=== PDF Graphics Extraction Test ===" << std::endl << std::endl;

  // Check for PDF file argument
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <pdf_file> [dpi] [output_prefix]"
              << std::endl;
    std::cerr << "  dpi           Resolution for rendering (default: 150)"
              << std::endl;
    std::cerr
        << "  output_prefix Prefix for output image files (default: 'page_')"
        << std::endl;
    return 1;
  }

  std::string pdfPath = argv[1];
  double dpi = (argc >= 3) ? std::stod(argv[2]) : 1200.0;
  std::string outputPrefix = (argc >= 4) ? argv[3] : "page_";

  std::cout << "Loading PDF: " << pdfPath << std::endl;
  std::cout << "DPI: " << dpi << std::endl;
  std::cout << "Output prefix: " << outputPrefix << std::endl << std::endl;

  // Create OCR analyzer
  ocr::OCRAnalysis analyzer;

  // Extract graphics from PDF
  std::cout << "Extracting graphics from PDF..." << std::endl;
  ocr::OCRAnalysis::PDFGraphicsResult result =
      analyzer.extractGraphicsFromPDF(pdfPath, dpi);

  if (!result.success) {
    std::cerr << "Failed to extract graphics: " << result.errorMessage
              << std::endl;
    return 1;
  }

  std::cout << "Processing time: " << std::fixed << std::setprecision(2)
            << result.processingTimeMs << " ms" << std::endl;
  std::cout << "Pages rendered: " << result.pages.size() << std::endl
            << std::endl;

  // Save each page as an image
  std::cout << "=== Rendered Pages ===" << std::endl;
  std::cout << std::string(60, '-') << std::endl;
  std::cout << std::left << std::setw(8) << "Page" << std::setw(20)
            << "Dimensions" << std::setw(10) << "DPI"
            << "Output File" << std::endl;
  std::cout << std::string(60, '-') << std::endl;

  for (const auto &page : result.pages) {
    std::string dimensions =
        std::to_string(page.width) + "×" + std::to_string(page.height);

    std::string outputFile =
        outputPrefix + std::to_string(page.pageNumber) + ".png";

    // Save the image
    bool saved = cv::imwrite(outputFile, page.image);

    std::cout << std::left << std::setw(8) << page.pageNumber << std::setw(20)
              << dimensions << std::setw(10) << static_cast<int>(page.dpi)
              << (saved ? outputFile : "FAILED") << std::endl;
  }

  std::cout << std::string(60, '-') << std::endl;
  std::cout << std::endl << "Extraction completed successfully!" << std::endl;

  return 0;
}
