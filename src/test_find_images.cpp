#include "OCRAnalysis.hpp"

#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  std::cout << "=== PDF Embedded Image Finder ===" << std::endl;
  std::cout << std::endl;

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <pdf_path> [output_dir]"
              << std::endl;
    std::cerr << std::endl;
    std::cerr << "  pdf_path    Path to the PDF file" << std::endl;
    std::cerr << "  output_dir  Directory to save images (default: images)"
              << std::endl;
    return 1;
  }

  std::string pdfPath = argv[1];
  std::string outputDir = (argc >= 3) ? argv[2] : "images";

  std::cout << "PDF file:         " << pdfPath << std::endl;
  std::cout << "Output directory: " << outputDir << std::endl;
  std::cout << std::endl;

  ocr::OCRAnalysis analyzer;

  int result = analyzer.writeAllImages(pdfPath, outputDir);

  std::cout << std::endl;
  if (result >= 0) {
    std::cout << "Completed successfully. " << result << " image(s) saved."
              << std::endl;
    return 0;
  } else {
    std::cerr << "Failed to extract images." << std::endl;
    return 1;
  }
}
