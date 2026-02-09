#include "OCRAnalysis.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
  // Create a simple test image with text
  cv::Mat testImage(200, 600, CV_8UC3, cv::Scalar(255, 255, 255));

  // Draw text on the image
  cv::putText(testImage, "OCR Analysis Test", cv::Point(50, 50),
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
  cv::putText(testImage, "Hello World!", cv::Point(50, 100),
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
  cv::putText(testImage, "Testing 123", cv::Point(50, 150),
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);

  // Save the test image
  cv::imwrite("test_generated.png", testImage);
  std::cout << "Created test image: test_generated.png\n\n";

  // Test OCR with preprocessing disabled (works better for clean digital
  // images)
  ocr::OCRConfig config;
  config.language = "eng";
  config.preprocessImage = false; // Disable for clean digital images

  ocr::OCRAnalysis analyzer(config);

  if (!analyzer.initialize()) {
    std::cerr << "Failed to initialize OCR engine\n";
    return 1;
  }

  std::cout << "Tesseract version: " << ocr::OCRAnalysis::getTesseractVersion()
            << "\n";
  std::cout << "Running OCR on generated image...\n\n";

  auto result = analyzer.analyzeImage(testImage);

  if (result.success) {
    std::cout << "=== Extracted Text ===\n";
    std::cout << result.fullText << "\n";
    std::cout << "======================\n\n";

    std::cout << "Detected " << result.regions.size() << " text regions:\n";
    for (size_t i = 0; i < result.regions.size(); ++i) {
      const auto &region = result.regions[i];
      std::cout << "  [" << (i + 1) << "] \"" << region.text
                << "\" (confidence: " << region.confidence << "%)\n";
    }

    std::cout << "\nProcessing time: " << result.processingTimeMs << " ms\n";
  } else {
    std::cerr << "OCR failed: " << result.errorMessage << "\n";
    return 1;
  }

  return 0;
}
