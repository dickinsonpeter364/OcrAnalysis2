#include "OCRAnalysis.hpp"

#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
  std::cout << "=== PDF Embedded Image Extraction Test ===" << std::endl
            << std::endl;

  // Check for PDF file argument
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <pdf_file> [output_prefix]"
              << std::endl;
    std::cerr << "  output_prefix  Prefix for output image files (default: "
                 "'embedded_')"
              << std::endl;
    return 1;
  }

  std::string pdfPath = argv[1];
  std::string outputPrefix = (argc >= 3) ? argv[2] : "embedded_";

  std::cout << "Loading PDF: " << pdfPath << std::endl;
  std::cout << "Output prefix: " << outputPrefix << std::endl << std::endl;

  // Create OCR analyzer
  ocr::OCRAnalysis analyzer;

  // Extract embedded images from PDF
  std::cout << "Extracting embedded images from PDF..." << std::endl;
  ocr::OCRAnalysis::PDFEmbeddedImagesResult result =
      analyzer.extractEmbeddedImagesFromPDF(pdfPath);

  if (!result.success) {
    std::cerr << "Failed to extract images: " << result.errorMessage
              << std::endl;
    return 1;
  }

  std::cout << "Processing time: " << std::fixed << std::setprecision(2)
            << result.processingTimeMs << " ms" << std::endl;
  std::cout << "Embedded images found: " << result.images.size() << std::endl
            << std::endl;

  if (result.images.empty()) {
    std::cout << "No embedded images found in this PDF." << std::endl;
    std::cout << "Note: This extracts actual embedded image objects, not "
                 "rendered pages."
              << std::endl;
    std::cout << "Use extractGraphicsFromPDF() to render pages as images."
              << std::endl;
    return 0;
  }

  // Display and save each image
  std::cout << "=== Embedded Images ===" << std::endl;
  std::cout << std::string(90, '-') << std::endl;
  std::cout << std::left << std::setw(8) << "Page" << std::setw(8) << "Index"
            << std::setw(20) << "Dimensions" << std::setw(25)
            << "Position (x,y)"
            << "Output File" << std::endl;
  std::cout << std::string(90, '-') << std::endl;

  for (const auto &img : result.images) {
    std::string dimensions =
        std::to_string(img.width) + "×" + std::to_string(img.height);
    std::string position = "(" + std::to_string(static_cast<int>(img.x)) +
                           ", " + std::to_string(static_cast<int>(img.y)) + ")";

    std::string outputFile = outputPrefix + "p" +
                             std::to_string(img.pageNumber) + "_" +
                             std::to_string(img.imageIndex) + ".png";

    // Save the image
    bool saved = false;
    if (!img.image.empty()) {
      saved = cv::imwrite(outputFile, img.image);
    }

    std::cout << std::left << std::setw(8) << img.pageNumber << std::setw(8)
              << img.imageIndex << std::setw(20) << dimensions << std::setw(25)
              << position << (saved ? outputFile : "FAILED (empty image)")
              << std::endl;
  }

  std::cout << std::string(90, '-') << std::endl;
  std::cout << std::endl << "Extraction completed successfully!" << std::endl;

  return 0;
}
