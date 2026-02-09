#include "OCRAnalysis.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>

/**
 * @brief Converts TextOrientation enum to a human-readable string
 */
std::string orientationToString(ocr::TextOrientation orientation) {
  switch (orientation) {
  case ocr::TextOrientation::Horizontal:
    return "Horizontal";
  case ocr::TextOrientation::Vertical:
    return "Vertical";
  case ocr::TextOrientation::Unknown:
  default:
    return "Unknown";
  }
}

/**
 * @brief Creates a test image with text in horizontal and vertical orientations
 */
cv::Mat createTestImage() {
  // Create a white background image
  int width = 800;
  int height = 600;
  cv::Mat image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

  // Font settings
  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = 1.5;
  int thickness = 2;
  cv::Scalar textColor(0, 0, 0); // Black text

  // Draw horizontal text at the top
  std::string horizText1 = "HORIZONTAL TEXT";
  cv::putText(image, horizText1, cv::Point(50, 80), fontFace, fontScale,
              textColor, thickness);

  std::string horizText2 = "READING LEFT TO RIGHT";
  cv::putText(image, horizText2, cv::Point(50, 150), fontFace, fontScale,
              textColor, thickness);

  // Draw vertical text by rotating
  // Create a temporary image for vertical text
  std::string vertText1 = "VERTICAL";
  std::string vertText2 = "TEXT";

  // Get text sizes
  int baseline = 0;
  cv::Size textSize1 =
      cv::getTextSize(vertText1, fontFace, fontScale, thickness, &baseline);
  cv::Size textSize2 =
      cv::getTextSize(vertText2, fontFace, fontScale, thickness, &baseline);

  // Create rotated text images
  cv::Mat textImg1(textSize1.height + baseline + 10, textSize1.width + 10,
                   CV_8UC3, cv::Scalar(255, 255, 255));
  cv::putText(textImg1, vertText1, cv::Point(5, textSize1.height + 5), fontFace,
              fontScale, textColor, thickness);

  cv::Mat textImg2(textSize2.height + baseline + 10, textSize2.width + 10,
                   CV_8UC3, cv::Scalar(255, 255, 255));
  cv::putText(textImg2, vertText2, cv::Point(5, textSize2.height + 5), fontFace,
              fontScale, textColor, thickness);

  // Rotate 90 degrees clockwise for vertical text
  cv::Mat rotatedText1, rotatedText2;
  cv::rotate(textImg1, rotatedText1, cv::ROTATE_90_CLOCKWISE);
  cv::rotate(textImg2, rotatedText2, cv::ROTATE_90_CLOCKWISE);

  // Copy rotated text onto main image
  int vertX1 = 650;
  int vertY1 = 100;
  if (vertX1 + rotatedText1.cols <= width &&
      vertY1 + rotatedText1.rows <= height) {
    rotatedText1.copyTo(
        image(cv::Rect(vertX1, vertY1, rotatedText1.cols, rotatedText1.rows)));
  }

  int vertX2 = 720;
  int vertY2 = 100;
  if (vertX2 + rotatedText2.cols <= width &&
      vertY2 + rotatedText2.rows <= height) {
    rotatedText2.copyTo(
        image(cv::Rect(vertX2, vertY2, rotatedText2.cols, rotatedText2.rows)));
  }

  // Add more horizontal text at the bottom
  std::string horizText3 = "ANOTHER LINE";
  cv::putText(image, horizText3, cv::Point(50, 500), fontFace, fontScale,
              textColor, thickness);

  // Add labels
  cv::putText(image, "(Horizontal)", cv::Point(50, 200), fontFace, 0.7,
              cv::Scalar(128, 128, 128), 1);
  cv::putText(image, "(Vertical)", cv::Point(620, 450), fontFace, 0.7,
              cv::Scalar(128, 128, 128), 1);

  return image;
}

int main() {
  std::cout << "=== OCR Orientation Detection Test ===" << std::endl;
  std::cout << "Tesseract version: " << ocr::OCRAnalysis::getTesseractVersion()
            << std::endl
            << std::endl;

  // Load the test image
  std::string imagePath = "1.bmp";
  std::cout << "Loading test image: " << imagePath << std::endl;
  cv::Mat testImage = cv::imread(imagePath);

  if (testImage.empty()) {
    std::cerr << "Failed to load image: " << imagePath << std::endl;
    return 1;
  }
  std::cout << "Image loaded: " << testImage.cols << "x" << testImage.rows
            << " pixels" << std::endl
            << std::endl;

  // Initialize OCR
  ocr::OCRConfig config;
  config.language = "eng";
  config.preprocessImage = false; // Use original image for better results
  config.minConfidence = 0;       // Show all results for testing

  ocr::OCRAnalysis analyzer(config);

  std::cout << "Initializing OCR engine..." << std::endl;
  if (!analyzer.initialize()) {
    std::cerr << "Failed to initialize OCR engine!" << std::endl;
    return 1;
  }
  std::cout << "OCR engine initialized successfully." << std::endl << std::endl;

  // Analyze the image
  std::cout << "Analyzing image for text orientation..." << std::endl;
  ocr::OCRResult result = analyzer.analyzeImage(testImage);

  if (!result.success) {
    std::cerr << "OCR analysis failed: " << result.errorMessage << std::endl;
    return 1;
  }

  // Display results
  std::cout << std::endl << "=== Results ===" << std::endl;
  std::cout << "Processing time: " << result.processingTimeMs << " ms"
            << std::endl;
  std::cout << "Regions found: " << result.regions.size() << std::endl
            << std::endl;

  // Count orientations
  int horizontalCount = 0;
  int verticalCount = 0;
  int unknownCount = 0;

  std::cout << "Detected text regions:" << std::endl;
  std::cout << std::string(70, '-') << std::endl;
  std::cout << "| " << std::left << std::setw(20) << "Text"
            << " | " << std::setw(12) << "Orientation"
            << " | " << std::setw(10) << "Confidence"
            << " | " << std::setw(15) << "Bounding Box"
            << " |" << std::endl;
  std::cout << std::string(70, '-') << std::endl;

  for (const auto &region : result.regions) {
    // Trim text for display
    std::string displayText = region.text;
    if (displayText.length() > 18) {
      displayText = displayText.substr(0, 15) + "...";
    }
    // Remove newlines
    displayText.erase(std::remove(displayText.begin(), displayText.end(), '\n'),
                      displayText.end());

    std::string orientStr = orientationToString(region.orientation);

    std::cout << "| " << std::left << std::setw(20) << displayText << " | "
              << std::setw(12) << orientStr << " | " << std::setw(10)
              << std::fixed << std::setprecision(1) << region.confidence
              << " | " << std::setw(15)
              << ("(" + std::to_string(region.boundingBox.x) + "," +
                  std::to_string(region.boundingBox.y) + ")")
              << " |" << std::endl;

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

  std::cout << std::string(70, '-') << std::endl;

  // Summary
  std::cout << std::endl << "=== Orientation Summary ===" << std::endl;
  std::cout << "Horizontal: " << horizontalCount << " regions" << std::endl;
  std::cout << "Vertical:   " << verticalCount << " regions" << std::endl;
  std::cout << "Unknown:    " << unknownCount << " regions" << std::endl;

  std::cout << std::endl << "Full extracted text:" << std::endl;
  std::cout << std::string(40, '-') << std::endl;
  std::cout << result.fullText << std::endl;
  std::cout << std::string(40, '-') << std::endl;

  std::cout << std::endl << "Test completed successfully!" << std::endl;

  return 0;
}
