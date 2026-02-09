#include "OCRAnalysis.hpp"

#include <iomanip>
#include <iostream>

void printUsage(const char *programName) {
  std::cout
      << "Usage: " << programName << " <image_path> [options]\n"
      << "\nOptions:\n"
      << "  -l, --language <lang>   Set OCR language (default: eng)\n"
      << "  -r, --regions           Show detected text regions\n"
      << "  -c, --confidence <val>  Minimum confidence threshold (0-100)\n"
      << "  -h, --help              Show this help message\n"
      << "\nExamples:\n"
      << "  " << programName << " document.png\n"
      << "  " << programName << " document.png -l eng+deu -r\n"
      << "  " << programName << " document.png --confidence 80\n";
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  std::string imagePath;
  ocr::OCRConfig config;
  bool showRegions = false;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      return 0;
    } else if (arg == "-l" || arg == "--language") {
      if (i + 1 < argc) {
        config.language = argv[++i];
      } else {
        std::cerr << "Error: --language requires an argument\n";
        return 1;
      }
    } else if (arg == "-r" || arg == "--regions") {
      showRegions = true;
    } else if (arg == "-c" || arg == "--confidence") {
      if (i + 1 < argc) {
        config.minConfidence = std::stoi(argv[++i]);
      } else {
        std::cerr << "Error: --confidence requires an argument\n";
        return 1;
      }
    } else if (arg[0] != '-') {
      imagePath = arg;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      printUsage(argv[0]);
      return 1;
    }
  }

  if (imagePath.empty()) {
    std::cerr << "Error: No image path provided\n";
    printUsage(argv[0]);
    return 1;
  }

  // Display version info
  std::cout << "=== OCR Analysis Demo ===\n"
            << "Tesseract version: " << ocr::OCRAnalysis::getTesseractVersion()
            << "\n"
            << "OpenCV version: " << CV_VERSION << "\n"
            << "Language: " << config.language << "\n"
            << "========================\n\n";

  // Create and initialize OCR analyzer
  ocr::OCRAnalysis analyzer(config);

  if (!analyzer.initialize()) {
    std::cerr
        << "Failed to initialize OCR engine.\n"
        << "Make sure Tesseract is installed and tessdata is available.\n";
    return 1;
  }

  // Show available languages
  std::cout << "Available languages: ";
  auto languages = analyzer.getAvailableLanguages();
  for (size_t i = 0; i < languages.size(); ++i) {
    std::cout << languages[i];
    if (i < languages.size() - 1)
      std::cout << ", ";
  }
  std::cout << "\n\n";

  // Analyze image
  std::cout << "Analyzing image: " << imagePath << "\n";
  std::cout << "-------------------------------------------\n";

  auto result = analyzer.analyzeImage(imagePath);

  if (!result.success) {
    std::cerr << "OCR failed: " << result.errorMessage << "\n";
    return 1;
  }

  // Display results
  std::cout << "\n[Extracted Text]\n";
  std::cout << "-------------------------------------------\n";
  std::cout << result.fullText << "\n";
  std::cout << "-------------------------------------------\n";

  if (showRegions && !result.regions.empty()) {
    std::cout << "\n[Detected Text Regions]\n";
    std::cout << std::setw(6) << "No." << std::setw(10) << "Conf%"
              << std::setw(30) << "Bounding Box"
              << "  Text\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t i = 0; i < result.regions.size(); ++i) {
      const auto &region = result.regions[i];
      std::ostringstream bbox;
      bbox << "(" << region.boundingBox.x << "," << region.boundingBox.y << ","
           << region.boundingBox.width << "," << region.boundingBox.height
           << ")";

      std::cout << std::setw(6) << (i + 1) << std::setw(10) << std::fixed
                << std::setprecision(1) << region.confidence << std::setw(30)
                << bbox.str() << "  " << region.text << "\n";
    }
  }

  std::cout << "\nProcessing time: " << std::fixed << std::setprecision(2)
            << result.processingTimeMs << " ms\n";
  std::cout << "Total regions detected: " << result.regions.size() << "\n";

  return 0;
}
