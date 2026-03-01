#include "OCRAnalysis.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

static void printUsage(const char *prog) {
  std::cerr << "Usage: " << prog
            << " <input_image> <output_dir> <output_name>\n"
            << "\n"
            << "  input_image  Path to the image to clean up (BMP, PNG, JPEG, "
               "TIFF, …)\n"
            << "  output_dir   Directory where the cleaned image will be "
               "written (created if absent)\n"
            << "  output_name  Filename for the cleaned image (e.g. "
               "cleaned.png)\n"
            << "\n"
            << "The cleaned image is a binary (black text / white background) "
               "PNG optimised for Tesseract OCR.\n";
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printUsage(argv[0]);
    return 1;
  }

  const std::string inputPath  = argv[1];
  const std::string outputDir  = argv[2];
  const std::string outputName = argv[3];

  // ── Load input ────────────────────────────────────────────────────────────
  cv::Mat input = cv::imread(inputPath, cv::IMREAD_UNCHANGED);
  if (input.empty()) {
    std::cerr << "Error: could not read image: " << inputPath << "\n";
    return 1;
  }

  std::cout << "Loaded: " << inputPath << "  (" << input.cols << "x"
            << input.rows << ", " << input.channels() << " ch)\n";

  // ── Clean up ──────────────────────────────────────────────────────────────
  cv::Mat cleaned = ocr::OCRAnalysis::cleanupForOCR(input);

  if (cleaned.empty()) {
    std::cerr << "Error: cleanup produced an empty image\n";
    return 1;
  }

  // ── Write output ──────────────────────────────────────────────────────────
  fs::path outDir(outputDir);
  if (!fs::exists(outDir)) {
    std::error_code ec;
    fs::create_directories(outDir, ec);
    if (ec) {
      std::cerr << "Error: could not create output directory: " << outputDir
                << " – " << ec.message() << "\n";
      return 1;
    }
    std::cout << "Created output directory: " << outputDir << "\n";
  }

  fs::path outPath = outDir / outputName;
  if (!cv::imwrite(outPath.string(), cleaned)) {
    std::cerr << "Error: could not write image: " << outPath << "\n";
    return 1;
  }

  // Report ink coverage so the aggressiveness of the threshold is visible.
  int totalPixels = cleaned.rows * cleaned.cols;
  int inkPixels   = totalPixels - cv::countNonZero(cleaned); // black = 0
  double pct = 100.0 * inkPixels / totalPixels;
  std::cout << "Cleaned image written to: " << outPath.string() << "\n"
            << "Ink coverage: " << std::fixed << std::setprecision(2) << pct
            << "% (" << inkPixels << " / " << totalPixels << " px)\n";
  return 0;
}
