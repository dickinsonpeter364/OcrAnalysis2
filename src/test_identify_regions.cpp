#include "OCRAnalysis.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>

/**
 * @brief Get color for a given text orientation
 * Green = Horizontal, Blue = Vertical, Red = Unknown
 */
cv::Scalar getOrientationColor(ocr::TextOrientation orientation) {
  switch (orientation) {
  case ocr::TextOrientation::Horizontal:
    return cv::Scalar(0, 255, 0); // Green (BGR)
  case ocr::TextOrientation::Vertical:
    return cv::Scalar(255, 0, 0); // Blue (BGR)
  case ocr::TextOrientation::Unknown:
  default:
    return cv::Scalar(0, 0, 255); // Red (BGR)
  }
}

/**
 * @brief Get label for a given text orientation
 */
std::string getOrientationLabel(ocr::TextOrientation orientation) {
  switch (orientation) {
  case ocr::TextOrientation::Horizontal:
    return "H";
  case ocr::TextOrientation::Vertical:
    return "V";
  case ocr::TextOrientation::Unknown:
  default:
    return "?";
  }
}

/**
 * @brief Draw legend on the image
 */
void drawLegend(cv::Mat &image, int x, int y) {
  int lineHeight = 30;
  int boxSize = 20;

  // Background
  cv::rectangle(image, cv::Point(x, y), cv::Point(x + 200, y + 100),
                cv::Scalar(255, 255, 255), cv::FILLED);
  cv::rectangle(image, cv::Point(x, y), cv::Point(x + 200, y + 100),
                cv::Scalar(0, 0, 0), 2);

  // Horizontal legend
  cv::rectangle(image, cv::Point(x + 10, y + 10),
                cv::Point(x + 10 + boxSize, y + 10 + boxSize),
                getOrientationColor(ocr::TextOrientation::Horizontal),
                cv::FILLED);
  cv::putText(image, "Horizontal", cv::Point(x + 40, y + 28),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

  // Vertical legend
  cv::rectangle(image, cv::Point(x + 10, y + 40),
                cv::Point(x + 10 + boxSize, y + 40 + boxSize),
                getOrientationColor(ocr::TextOrientation::Vertical),
                cv::FILLED);
  cv::putText(image, "Vertical", cv::Point(x + 40, y + 58),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

  // Unknown legend
  cv::rectangle(image, cv::Point(x + 10, y + 70),
                cv::Point(x + 10 + boxSize, y + 70 + boxSize),
                getOrientationColor(ocr::TextOrientation::Unknown), cv::FILLED);
  cv::putText(image, "Unknown", cv::Point(x + 40, y + 88),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
}

int main(int argc, char *argv[]) {
  std::cout << "=== Text Region Identification Test ===" << std::endl;
  std::cout << "Tesseract version: " << ocr::OCRAnalysis::getTesseractVersion()
            << std::endl
            << std::endl;

  // Determine input image
  std::string imagePath = "1.bmp";
  if (argc > 1) {
    imagePath = argv[1];
  }

  // Load the test image
  std::cout << "Loading image: " << imagePath << std::endl;
  cv::Mat inputImage = cv::imread(imagePath);

  if (inputImage.empty()) {
    std::cerr << "Failed to load image: " << imagePath << std::endl;
    return 1;
  }
  std::cout << "Image loaded: " << inputImage.cols << "x" << inputImage.rows
            << " pixels" << std::endl
            << std::endl;

  // Initialize OCR
  ocr::OCRConfig config;
  config.language = "eng";
  config.preprocessImage = false;

  ocr::OCRAnalysis analyzer(config);

  std::cout << "Initializing OCR engine..." << std::endl;
  if (!analyzer.initialize()) {
    std::cerr << "Failed to initialize OCR engine!" << std::endl;
    return 1;
  }
  std::cout << "OCR engine initialized successfully." << std::endl << std::endl;

  // Identify text regions
  std::cout << "Identifying text regions in all orientations..." << std::endl;
  auto startTime = std::chrono::high_resolution_clock::now();

  // First, mask any logos or graphics with white boxes
  std::cout << "Masking logos and graphics..." << std::endl;
  cv::Mat maskedImage = analyzer.maskNonTextRegions(inputImage);

  // Save the masked image for inspection
  cv::imwrite("masked_image.png", maskedImage);
  std::cout << "Masked image saved to: masked_image.png" << std::endl;

  std::vector<ocr::TextRegion> regions =
      analyzer.identifyTextRegions(maskedImage);

  auto endTime = std::chrono::high_resolution_clock::now();
  double processingTime =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();

  std::cout << "Processing time: " << processingTime << " ms" << std::endl;
  std::cout << "Regions found: " << regions.size() << std::endl << std::endl;

  // Count by orientation
  int horizontalCount = 0;
  int verticalCount = 0;
  int unknownCount = 0;

  for (const auto &region : regions) {
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

  std::cout << "=== Orientation Summary ===" << std::endl;
  std::cout << "Horizontal: " << horizontalCount << " regions" << std::endl;
  std::cout << "Vertical:   " << verticalCount << " regions" << std::endl;
  std::cout << "Unknown:    " << unknownCount << " regions" << std::endl
            << std::endl;

  // Create output image with bounding boxes
  cv::Mat outputImage = inputImage.clone();

  // Draw bounding boxes for each region
  for (const auto &region : regions) {
    cv::Scalar color = getOrientationColor(region.orientation);
    std::string label = getOrientationLabel(region.orientation);

    // Draw the bounding box
    cv::rectangle(outputImage, region.boundingBox, color, 3);

    // Draw orientation label
    int labelX = region.boundingBox.x;
    int labelY = region.boundingBox.y - 5;
    if (labelY < 20) {
      labelY = region.boundingBox.y + region.boundingBox.height + 20;
    }

    // Background for label
    cv::Size textSize =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
    cv::rectangle(outputImage, cv::Point(labelX, labelY - textSize.height - 2),
                  cv::Point(labelX + textSize.width + 4, labelY + 2), color,
                  cv::FILLED);

    // Label text
    cv::putText(outputImage, label, cv::Point(labelX + 2, labelY),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
  }

  // Draw legend
  drawLegend(outputImage, 20, 20);

  // Save output image
  std::string outputPath = "text_regions_output.png";
  cv::imwrite(outputPath, outputImage);
  std::cout << "Output image saved to: " << outputPath << std::endl;

  // Print detected regions
  std::cout << std::endl << "=== Detected Regions ===" << std::endl;
  std::cout << std::string(80, '-') << std::endl;

  int count = 0;
  for (const auto &region : regions) {
    count++;
    std::string orientStr =
        (region.orientation == ocr::TextOrientation::Horizontal) ? "Horizontal"
        : (region.orientation == ocr::TextOrientation::Vertical) ? "Vertical"
                                                                 : "Unknown";

    // Truncate text for display
    std::string displayText = region.text;
    // Remove newlines
    std::replace(displayText.begin(), displayText.end(), '\n', ' ');
    if (displayText.length() > 40) {
      displayText = displayText.substr(0, 37) + "...";
    }

    std::cout << count << ". [" << orientStr << "] "
              << "Box(" << region.boundingBox.x << "," << region.boundingBox.y
              << " " << region.boundingBox.width << "x"
              << region.boundingBox.height << ") "
              << "Conf: " << std::fixed << std::setprecision(1)
              << region.confidence << "% "
              << "\"" << displayText << "\"" << std::endl;
  }

  std::cout << std::string(80, '-') << std::endl;
  std::cout << std::endl << "Test completed successfully!" << std::endl;

  return 0;
}
