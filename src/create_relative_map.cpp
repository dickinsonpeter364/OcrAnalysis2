#include "OCRAnalysis.hpp"
#include <algorithm>
#include <iostream>
#include <limits>

namespace ocr {

OCRAnalysis::RelativeMapResult
OCRAnalysis::createRelativeMap(const PDFElements &elements,
                               RenderBoundsMode boundsMode) {
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
        // Text uses bottom-left origin (already converted in
        // extractTextFromPDF)
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

      // Text uses bottom-left origin (already converted in extractTextFromPDF)
      // Convert to top-left origin for consistency
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

      // Images use bottom-left origin
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

    result.success = true;
    return result;

  } catch (const std::exception &e) {
    result.errorMessage =
        std::string("Error creating relative map: ") + e.what();
    return result;
  }
}

} // namespace ocr
