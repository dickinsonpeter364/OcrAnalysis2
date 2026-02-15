#include "OCRAnalysis.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>

namespace ocr {

OCRAnalysis::RelativeMapResult
OCRAnalysis::createRelativeMap(const PDFElements &elements,
                               RenderBoundsMode boundsMode, double dpi,
                               const std::string &markToFile) {
  OCRAnalysis::RelativeMapResult result;

  try {
    // Calculate bounds using the same logic as renderElementsToPNG
    double minX, minY, maxX, maxY;

    if (boundsMode == RenderBoundsMode::USE_LARGEST_RECTANGLE) {
      // Use the largest rectangle to determine bounds
      double largestArea = 0;
      const PDFRectangle *largestRect = nullptr;

      for (const auto &rect : elements.rectangles) {
        double area = rect.width * rect.height;
        if (area > largestArea) {
          largestArea = area;
          largestRect = &rect;
        }
      }

      if (largestRect == nullptr) {
        // No rectangles found - try to use largest image instead
        if (!elements.images.empty()) {
          double largestImageArea = 0.0;
          const PDFEmbeddedImage *largestImage = nullptr;

          for (const auto &img : elements.images) {
            double area = img.displayWidth * img.displayHeight;
            if (area > largestImageArea) {
              largestImageArea = area;
              largestImage = &img;
            }
          }

          if (largestImage != nullptr) {
            minX = largestImage->x;
            minY = largestImage->y;
            maxX = largestImage->x + largestImage->displayWidth;
            maxY = largestImage->y + largestImage->displayHeight;
          } else {
            result.errorMessage = "Could not find valid rectangle or image";
            return result;
          }
        } else {
          result.errorMessage =
              "No rectangles or images found for USE_LARGEST_RECTANGLE mode";
          return result;
        }
      } else {
        // Convert rectangle from top-left to bottom-left origin
        double rectTopLeftY = largestRect->y;
        double rectBottomLeftY = rectTopLeftY + largestRect->height;

        minX = largestRect->x;
        minY = elements.pageHeight - rectBottomLeftY;
        maxX = largestRect->x + largestRect->width;
        maxY = elements.pageHeight - rectTopLeftY;
      }

    } else if (elements.linesBoundingBoxWidth > 0 &&
               elements.linesBoundingBoxHeight > 0) {
      // Use the interior box calculated from crop marks/lines
      minX = elements.linesBoundingBoxX;
      minY = elements.linesBoundingBoxY;
      maxX = elements.linesBoundingBoxX + elements.linesBoundingBoxWidth;
      maxY = elements.linesBoundingBoxY + elements.linesBoundingBoxHeight;

    } else {
      // Fall back to calculating bounding box from all elements
      minX = std::numeric_limits<double>::max();
      minY = std::numeric_limits<double>::max();
      maxX = std::numeric_limits<double>::lowest();
      maxY = std::numeric_limits<double>::lowest();

      // Check all elements to find actual bounds
      for (const auto &text : elements.textLines) {
        double textX = text.boundingBox.x;
        double textY = text.boundingBox.y;
        double textWidth = text.boundingBox.width;
        double textHeight = text.boundingBox.height;

        minX = std::min(minX, textX);
        minY = std::min(minY, textY);
        maxX = std::max(maxX, textX + textWidth);
        maxY = std::max(maxY, textY + textHeight);
      }

      for (const auto &img : elements.images) {
        minX = std::min(minX, img.x);
        minY = std::min(minY, img.y);
        maxX = std::max(maxX, img.x + img.displayWidth);
        maxY = std::max(maxY, img.y + img.displayHeight);
      }

      if (minX == std::numeric_limits<double>::max()) {
        result.errorMessage = "No elements found to create relative map";
        return result;
      }
    }

    // Store the calculated bounds
    result.boundsX = minX;
    result.boundsY = minY;
    result.boundsWidth = maxX - minX;
    result.boundsHeight = maxY - minY;

    std::cerr << "Relative map bounds: (" << minX << ", " << minY << ") to ("
              << maxX << ", " << maxY << ")" << std::endl;
    std::cerr << "  Width: " << result.boundsWidth
              << " pt, Height: " << result.boundsHeight << " pt" << std::endl;

    // Convert text elements to relative coordinates (using centre point)
    for (const auto &text : elements.textLines) {
      RelativeElement elem;
      elem.type = RelativeElement::TEXT;

      double textX = text.boundingBox.x;
      double textY = text.boundingBox.y;
      double textWidth = text.boundingBox.width;
      double textHeight = text.boundingBox.height;

      // Convert from bottom-left to top-left origin
      double topLeftY = result.boundsHeight - (textY - minY + textHeight);

      // Calculate relative width and height
      elem.relativeWidth = textWidth / result.boundsWidth;
      elem.relativeHeight = textHeight / result.boundsHeight;

      // Calculate relative centre coordinates (0.0 to 1.0)
      elem.relativeX = (textX - minX + textWidth / 2.0) / result.boundsWidth;
      elem.relativeY = (topLeftY + textHeight / 2.0) / result.boundsHeight;

      // Copy text-specific fields
      elem.text = text.text;
      elem.fontName = text.fontName;
      elem.fontSize = text.fontSize;
      elem.isBold = text.isBold;
      elem.isItalic = text.isItalic;

      result.elements.push_back(elem);
    }

    // Convert image elements to relative coordinates (using centre point)
    for (const auto &img : elements.images) {
      RelativeElement elem;
      elem.type = RelativeElement::IMAGE;

      // Convert to top-left origin
      double topLeftY =
          result.boundsHeight - (img.y - minY + img.displayHeight);

      // Calculate relative width and height
      elem.relativeWidth = img.displayWidth / result.boundsWidth;
      elem.relativeHeight = img.displayHeight / result.boundsHeight;

      // Calculate relative centre coordinates (0.0 to 1.0)
      elem.relativeX =
          (img.x - minX + img.displayWidth / 2.0) / result.boundsWidth;
      elem.relativeY =
          (topLeftY + img.displayHeight / 2.0) / result.boundsHeight;

      result.elements.push_back(elem);
    }

    std::cerr << "Created relative map with " << result.elements.size()
              << " elements" << std::endl;
    std::cerr << "  " << elements.textLines.size() << " text elements"
              << std::endl;
    std::cerr << "  " << elements.images.size() << " image elements"
              << std::endl;

    // If markToFile is provided, draw bounding boxes on the image
    if (!markToFile.empty()) {
      cv::Mat targetImage = cv::imread(markToFile);
      if (targetImage.empty()) {
        std::cerr << "Warning: Could not load image for marking: " << markToFile
                  << std::endl;
      } else {
        int canvasWidth = targetImage.cols;
        int canvasHeight = targetImage.rows;

        std::cerr << "Marking image " << markToFile << " (" << canvasWidth
                  << "x" << canvasHeight << ")" << std::endl;

        cv::Mat markedImage = targetImage.clone();
        cv::Scalar blueColor(255, 0, 0);  // Blue in BGR (for text)
        cv::Scalar greenColor(0, 255, 0); // Green in BGR (for images)
        int drawnCount = 0;

        for (const auto &elem : result.elements) {
          // Convert from centre-based relative coords to pixel top-left
          int pixelWidth = static_cast<int>(elem.relativeWidth * canvasWidth);
          int pixelHeight =
              static_cast<int>(elem.relativeHeight * canvasHeight);
          int pixelX = static_cast<int>(
              (elem.relativeX - elem.relativeWidth / 2.0) * canvasWidth);
          int pixelY = static_cast<int>(
              (elem.relativeY - elem.relativeHeight / 2.0) * canvasHeight);

          // Clamp to image bounds
          int drawX1 = std::max(0, std::min(pixelX, canvasWidth));
          int drawY1 = std::max(0, std::min(pixelY, canvasHeight));
          int drawX2 = std::max(0, std::min(pixelX + pixelWidth, canvasWidth));
          int drawY2 =
              std::max(0, std::min(pixelY + pixelHeight, canvasHeight));

          if (drawX2 > drawX1 && drawY2 > drawY1) {
            cv::Scalar color =
                (elem.type == RelativeElement::TEXT) ? blueColor : greenColor;
            cv::rectangle(markedImage, cv::Point(drawX1, drawY1),
                          cv::Point(drawX2, drawY2), color, 2);
            drawnCount++;

            // Log element details
            if (elem.type == RelativeElement::TEXT) {
              std::cerr << "  Text \"" << elem.text << "\": (" << drawX1 << ","
                        << drawY1 << ") " << (drawX2 - drawX1) << "x"
                        << (drawY2 - drawY1) << std::endl;
            } else {
              std::cerr << "  Image: (" << drawX1 << "," << drawY1 << ") "
                        << (drawX2 - drawX1) << "x" << (drawY2 - drawY1)
                        << std::endl;
            }
          }
        }

        std::cerr << "Drew " << drawnCount << " boxes on image" << std::endl;

        // Save with _relmap suffix
        std::filesystem::path markPath(markToFile);
        std::string outputPath = markPath.parent_path().string() + "/" +
                                 markPath.stem().string() + "_relmap" +
                                 markPath.extension().string();

        if (cv::imwrite(outputPath, markedImage)) {
          std::cerr << "Relative map marked image saved: " << outputPath
                    << std::endl;
        } else {
          std::cerr << "ERROR: Failed to save marked image: " << outputPath
                    << std::endl;
        }
      }
    }

    result.success = true;
    return result;

  } catch (const std::exception &e) {
    result.errorMessage =
        std::string("Error creating relative map: ") + e.what();
    return result;
  }
}

} // namespace ocr
