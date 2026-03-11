#include "OCRAnalysis.hpp"

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

static void printUsage(const char *prog) {
  std::cerr << "Usage: " << prog
            << " <input_image> <output_dir> <output_name> [--backing]\n"
            << "\n"
            << "  input_image  Path to the source image (PNG, JPEG, BMP, …)\n"
            << "  output_dir   Directory where the crop will be written\n"
            << "  output_name  Filename for the cropped image (e.g. crop.png)\n"
            << "  --backing    Return the full backing-paper region (step 1)\n"
            << "               instead of the default tight label crop (step 2).\n";
}

int main(int argc, char *argv[]) {
  if (argc < 4 || argc > 5) {
    printUsage(argv[0]);
    return 1;
  }

  const std::string inputPath  = argv[1];
  const std::string outputDir  = argv[2];
  const std::string outputName = argv[3];
  const bool tightLabel = !(argc == 5 && std::string(argv[4]) == "--backing");

  // ── Load ──────────────────────────────────────────────────────────────────
  cv::Mat input = cv::imread(inputPath, cv::IMREAD_UNCHANGED);
  if (input.empty()) {
    std::cerr << "Error: could not read image: " << inputPath << "\n";
    return 1;
  }
  std::cout << "Loaded: " << inputPath << "  ("
            << input.cols << "x" << input.rows
            << ", " << input.channels() << " ch)\n";

  // ── Crop ──────────────────────────────────────────────────────────────────
  cv::Mat cropped = ocr::OCRAnalysis::cropToLabel(input, 50, 40, tightLabel);
  if (cropped.empty()) {
    std::cerr << "Error: cropToLabel returned an empty image\n";
    return 1;
  }
  std::cout << "Crop size: " << cropped.cols << "x" << cropped.rows
            << "  (mode: " << (tightLabel ? "tight label" : "backing paper") << ")\n";

  // ── Write ─────────────────────────────────────────────────────────────────
  fs::path outDir(outputDir);
  if (!fs::exists(outDir)) {
    std::error_code ec;
    fs::create_directories(outDir, ec);
    if (ec) {
      std::cerr << "Error: could not create output directory: "
                << outputDir << " – " << ec.message() << "\n";
      return 1;
    }
    std::cout << "Created output directory: " << outputDir << "\n";
  }

  fs::path outPath = outDir / outputName;
  if (!cv::imwrite(outPath.string(), cropped)) {
    std::cerr << "Error: could not write image: " << outPath << "\n";
    return 1;
  }
  std::cout << "Crop written to: " << outPath.string() << "\n";
  return 0;
}
