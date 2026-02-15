#include "OCRAnalysis.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>

namespace ocr {

/**
 * @brief Structure to hold an OCR-detected word with its pixel bounding box.
 */
struct OcrWord {
  std::string text;
  int x, y, width, height;
  float confidence;
};

/**
 * @brief Clean and normalise a string for fuzzy comparison.
 *        Removes whitespace, lowercases, strips underscores.
 */
static std::string normaliseForMatch(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    if (std::isspace(c) || c == '_')
      continue;
    out.push_back(static_cast<char>(std::tolower(c)));
  }
  return out;
}

/**
 * @brief Run Tesseract OCR on an image and return all detected words with their
 *        pixel bounding boxes.
 */
static std::vector<OcrWord> ocrDetectWords(const cv::Mat &image) {
  std::vector<OcrWord> words;

  tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
  if (ocr->Init("C:/tessdata/tessdata", "eng") != 0 &&
      ocr->Init(NULL, "eng") != 0) {
    std::cerr << "Warning: Could not initialize Tesseract for auto-crop"
              << std::endl;
    delete ocr;
    return words;
  }

  ocr->SetImage(image.data, image.cols, image.rows, image.channels(),
                static_cast<int>(image.step));
  ocr->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
  ocr->Recognize(0);

  tesseract::ResultIterator *ri = ocr->GetIterator();
  if (ri != nullptr) {
    do {
      const char *word = ri->GetUTF8Text(tesseract::RIL_WORD);
      float conf = ri->Confidence(tesseract::RIL_WORD);
      if (word != nullptr && *word != '\0' && conf > 30.0f) {
        int x1, y1, x2, y2;
        ri->BoundingBox(tesseract::RIL_WORD, &x1, &y1, &x2, &y2);
        OcrWord w;
        w.text = word;
        w.x = x1;
        w.y = y1;
        w.width = x2 - x1;
        w.height = y2 - y1;
        w.confidence = conf;
        words.push_back(w);
      }
      delete[] word;
    } while (ri->Next(tesseract::RIL_WORD));
  }

  ocr->End();
  delete ocr;
  return words;
}

/**
 * @brief A matched pair: a RelativeElement and its corresponding OCR word
 *        detected in the target image.
 */
struct MatchedPair {
  double relCentreX; // relative centre X from PDF map
  double relCentreY; // relative centre Y from PDF map
  double ocrCentreX; // pixel centre X from OCR detection
  double ocrCentreY; // pixel centre Y from OCR detection
};

/**
 * @brief Solve for the crop rectangle that maps relative coords to pixel
 * coords.
 *
 * Given matched pairs where:
 *   ocrCentreX = relCentreX * cropWidth + cropX
 *   ocrCentreY = relCentreY * cropHeight + cropY
 *
 * We solve two independent linear systems:
 *   For X: ocrCentreX_i = relCentreX_i * cropWidth + cropX
 *   For Y: ocrCentreY_i = relCentreY_i * cropHeight + cropY
 *
 * With 2+ matches, use least-squares to find (cropX, cropWidth) and
 * (cropY, cropHeight).
 *
 * @param matches Vector of matched pairs
 * @param[out] cropRect The solved crop rectangle in pixel coordinates
 * @return true if a valid solution was found
 */
static bool solveCropRectFromMatches(const std::vector<MatchedPair> &matches,
                                     cv::Rect &cropRect) {
  if (matches.size() < 2) {
    std::cerr << "Need at least 2 matched text elements to solve crop rect"
              << std::endl;
    return false;
  }

  // Solve X system: ocrX = relX * cropWidth + cropX
  // Using least squares: minimise sum((ocrX_i - relX_i * w - cx)^2)
  // Normal equations for y = ax + b:
  //   sum(x^2)*a + sum(x)*b = sum(x*y)
  //   sum(x)*a + n*b = sum(y)
  // where x = relCentreX, y = ocrCentreX, a = cropWidth, b = cropX

  int n = static_cast<int>(matches.size());
  double sumRelX = 0, sumRelX2 = 0, sumOcrX = 0, sumRelXOcrX = 0;
  double sumRelY = 0, sumRelY2 = 0, sumOcrY = 0, sumRelYOcrY = 0;

  for (const auto &m : matches) {
    sumRelX += m.relCentreX;
    sumRelX2 += m.relCentreX * m.relCentreX;
    sumOcrX += m.ocrCentreX;
    sumRelXOcrX += m.relCentreX * m.ocrCentreX;

    sumRelY += m.relCentreY;
    sumRelY2 += m.relCentreY * m.relCentreY;
    sumOcrY += m.ocrCentreY;
    sumRelYOcrY += m.relCentreY * m.ocrCentreY;
  }

  // Solve for X: [sumRelX2, sumRelX; sumRelX, n] * [cropWidth; cropX] =
  // [sumRelXOcrX; sumOcrX]
  double detX = sumRelX2 * n - sumRelX * sumRelX;
  if (std::abs(detX) < 1e-10) {
    std::cerr << "X system is singular (all matches at same relativeX?)"
              << std::endl;
    return false;
  }
  double cropWidth = (sumRelXOcrX * n - sumRelX * sumOcrX) / detX;
  double cropX = (sumRelX2 * sumOcrX - sumRelX * sumRelXOcrX) / detX;

  // Solve for Y: same approach
  double detY = sumRelY2 * n - sumRelY * sumRelY;
  if (std::abs(detY) < 1e-10) {
    std::cerr << "Y system is singular (all matches at same relativeY?)"
              << std::endl;
    return false;
  }
  double cropHeight = (sumRelYOcrY * n - sumRelY * sumOcrY) / detY;
  double cropY = (sumRelY2 * sumOcrY - sumRelY * sumRelYOcrY) / detY;

  std::cerr << "Solved crop rect from " << n << " matches:" << std::endl;
  std::cerr << "  cropX=" << cropX << " cropY=" << cropY
            << " cropWidth=" << cropWidth << " cropHeight=" << cropHeight
            << std::endl;

  // Validate the result
  if (cropWidth < 10 || cropHeight < 10) {
    std::cerr << "Solved crop dimensions too small" << std::endl;
    return false;
  }

  // Report per-match residuals
  double totalResidual = 0;
  for (const auto &m : matches) {
    double predX = m.relCentreX * cropWidth + cropX;
    double predY = m.relCentreY * cropHeight + cropY;
    double residual =
        std::sqrt((predX - m.ocrCentreX) * (predX - m.ocrCentreX) +
                  (predY - m.ocrCentreY) * (predY - m.ocrCentreY));
    totalResidual += residual;
    std::cerr << "  Match residual: " << residual << " px" << std::endl;
  }
  std::cerr << "  Average residual: " << totalResidual / n << " px"
            << std::endl;

  cropRect = cv::Rect(static_cast<int>(std::round(cropX)),
                      static_cast<int>(std::round(cropY)),
                      static_cast<int>(std::round(cropWidth)),
                      static_cast<int>(std::round(cropHeight)));
  return true;
}

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
      // Skip text that is mostly underscores (e.g. "_____________________")
      if (!text.text.empty()) {
        size_t underscoreCount =
            std::count(text.text.begin(), text.text.end(), '_');
        if (underscoreCount > text.text.size() / 2) {
          continue;
        }
      }

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

    // If markToFile is provided, auto-crop using OCR and draw bounding boxes
    if (!markToFile.empty()) {
      cv::Mat targetImage = cv::imread(markToFile);
      if (targetImage.empty()) {
        std::cerr << "Warning: Could not load image for marking: " << markToFile
                  << std::endl;
      } else {
        std::cerr << "Loaded image " << markToFile << " (" << targetImage.cols
                  << "x" << targetImage.rows << ")" << std::endl;

        // Step 1: Run OCR on the target image to find all words
        std::cerr << "\n=== OCR-based auto-crop ===" << std::endl;
        std::cerr << "Running OCR on target image..." << std::endl;
        auto ocrWords = ocrDetectWords(targetImage);
        std::cerr << "Detected " << ocrWords.size() << " words via OCR"
                  << std::endl;

        // Log all detected OCR words for debugging
        for (const auto &w : ocrWords) {
          std::cerr << "  OCR word: \"" << w.text << "\" conf=" << w.confidence
                    << " at (" << w.x << "," << w.y << ") " << w.width << "x"
                    << w.height << std::endl;
        }

        // Step 2: Match OCR words against known text from relative map
        // Only use text elements within bounds (0.0-1.0) for matching
        std::vector<MatchedPair> matches;

        for (const auto &elem : result.elements) {
          if (elem.type != RelativeElement::TEXT)
            continue;

          // Use ALL text elements for matching - elements outside bounds
          // (relativeX/Y > 1.0) may still be visible in the image and
          // are valuable for computing the crop rectangle.

          // Skip very short text that could match too many false positives
          std::string normPdfText = normaliseForMatch(elem.text);
          if (normPdfText.length() < 2)
            continue;

          // Find the best matching OCR word
          for (const auto &ocrWord : ocrWords) {
            std::string normOcrText = normaliseForMatch(ocrWord.text);

            // Check if one contains the other (handles partial matches)
            bool match = false;
            if (normPdfText == normOcrText) {
              match = true;
            } else if (normPdfText.length() >= 4 &&
                       normOcrText.find(normPdfText) != std::string::npos) {
              match = true;
            } else if (normOcrText.length() >= 4 &&
                       normPdfText.find(normOcrText) != std::string::npos) {
              match = true;
            }

            if (match) {
              MatchedPair mp;
              mp.relCentreX = elem.relativeX;
              mp.relCentreY = elem.relativeY;
              mp.ocrCentreX = ocrWord.x + ocrWord.width / 2.0;
              mp.ocrCentreY = ocrWord.y + ocrWord.height / 2.0;

              std::cerr << "  Matched: \"" << elem.text << "\" <-> \""
                        << ocrWord.text << "\" rel=(" << mp.relCentreX << ","
                        << mp.relCentreY << ") px=(" << mp.ocrCentreX << ","
                        << mp.ocrCentreY << ")" << std::endl;

              matches.push_back(mp);
              break; // Use first match per element
            }
          }
        }

        std::cerr << "Found " << matches.size() << " text matches" << std::endl;

        // Step 3: Solve for the crop rectangle
        cv::Mat canvas;
        cv::Rect cropRect;
        bool cropped = false;

        if (matches.size() >= 2 &&
            solveCropRectFromMatches(matches, cropRect)) {
          // Clamp crop rect to image bounds
          int clampedX = std::max(0, cropRect.x);
          int clampedY = std::max(0, cropRect.y);
          int clampedW = std::min(cropRect.width, targetImage.cols - clampedX);
          int clampedH = std::min(cropRect.height, targetImage.rows - clampedY);

          if (clampedW > 10 && clampedH > 10) {
            cropRect = cv::Rect(clampedX, clampedY, clampedW, clampedH);
            canvas = targetImage(cropRect).clone();
            cropped = true;
            std::cerr << "Auto-cropped to: (" << cropRect.x << ", "
                      << cropRect.y << ") " << cropRect.width << "x"
                      << cropRect.height << std::endl;

            // Save the cropped image for reference
            std::filesystem::path markPath(markToFile);
            std::string croppedPath = markPath.parent_path().string() + "/" +
                                      markPath.stem().string() + "_cropped" +
                                      markPath.extension().string();
            cv::imwrite(croppedPath, canvas);
            std::cerr << "Cropped image saved: " << croppedPath << std::endl;
          } else {
            std::cerr << "Solved crop rect too small after clamping, "
                         "using full image"
                      << std::endl;
            canvas = targetImage.clone();
          }
        } else if (matches.size() == 1) {
          // Single-match fallback: use the PDF bounds aspect ratio
          // to determine crop dimensions, then position using the
          // single anchor point.
          double expectedAR = result.boundsWidth / result.boundsHeight;
          std::cerr << "Single match fallback using aspect ratio " << expectedAR
                    << std::endl;

          const auto &m = matches[0];

          // From the relationship:
          //   ocrCentreX = relCentreX * cropWidth + cropX
          //   ocrCentreY = relCentreY * cropHeight + cropY
          // With constraint: cropHeight = cropWidth / expectedAR
          // We have 1 match (2 equations) and 3 unknowns, so sweep cropWidth.

          double bestCropWidth = 0, bestCropHeight = 0;
          double bestCropX = 0, bestCropY = 0;
          double bestScore = -1;

          for (double frac = 0.3; frac <= 1.0; frac += 0.01) {
            double tryW = targetImage.cols * frac;
            double tryH = tryW / expectedAR;

            if (tryH > targetImage.rows)
              continue;

            // Derive position from the single match
            double tryX = m.ocrCentreX - m.relCentreX * tryW;
            double tryY = m.ocrCentreY - m.relCentreY * tryH;

            // Check if crop rect fits within image (with 10% tolerance)
            if (tryX < -tryW * 0.1 || tryY < -tryH * 0.1 ||
                tryX + tryW > targetImage.cols * 1.1 ||
                tryY + tryH > targetImage.rows * 1.1) {
              continue;
            }

            // Score: prefer larger rects that stay within bounds
            double clX = std::max(0.0, tryX);
            double clY = std::max(0.0, tryY);
            double clW = std::min(tryW, targetImage.cols - clX);
            double clH = std::min(tryH, targetImage.rows - clY);
            double areaFrac =
                (clW * clH) / (targetImage.cols * targetImage.rows);
            double penalty = 1.0;
            if (tryX < 0)
              penalty *= 0.8;
            if (tryY < 0)
              penalty *= 0.8;
            if (tryX + tryW > targetImage.cols)
              penalty *= 0.8;
            if (tryY + tryH > targetImage.rows)
              penalty *= 0.8;

            double score = areaFrac * penalty;
            if (score > bestScore) {
              bestScore = score;
              bestCropWidth = tryW;
              bestCropHeight = tryH;
              bestCropX = tryX;
              bestCropY = tryY;
            }
          }

          if (bestScore > 0) {
            int cx = std::max(0, static_cast<int>(std::round(bestCropX)));
            int cy = std::max(0, static_cast<int>(std::round(bestCropY)));
            int cw = std::min(static_cast<int>(std::round(bestCropWidth)),
                              targetImage.cols - cx);
            int ch = std::min(static_cast<int>(std::round(bestCropHeight)),
                              targetImage.rows - cy);

            if (cw > 10 && ch > 10) {
              cropRect = cv::Rect(cx, cy, cw, ch);
              canvas = targetImage(cropRect).clone();
              cropped = true;
              std::cerr << "Single-match auto-cropped to: (" << cx << ", " << cy
                        << ") " << cw << "x" << ch << std::endl;

              std::filesystem::path markPath(markToFile);
              std::string croppedPath = markPath.parent_path().string() + "/" +
                                        markPath.stem().string() + "_cropped" +
                                        markPath.extension().string();
              cv::imwrite(croppedPath, canvas);
              std::cerr << "Cropped image saved: " << croppedPath << std::endl;
            }
          }

          if (!cropped) {
            std::cerr << "Single-match fallback failed, using full image"
                      << std::endl;
            canvas = targetImage.clone();
          }
        } else {
          std::cerr << "Could not solve crop rect (no matches), "
                       "using full image"
                    << std::endl;
          canvas = targetImage.clone();
        }

        // Step 4: Draw bounding boxes using relative coordinates
        int canvasWidth = canvas.cols;
        int canvasHeight = canvas.rows;

        cv::Scalar blueColor(255, 0, 0);  // Blue in BGR (for text)
        cv::Scalar greenColor(0, 255, 0); // Green in BGR (for images)
        int drawnCount = 0;

        for (const auto &elem : result.elements) {
          // Only draw elements within or near the bounds (0.0-1.0 range)
          if (elem.relativeX < -0.1 || elem.relativeX > 1.1 ||
              elem.relativeY < -0.1 || elem.relativeY > 1.1) {
            continue;
          }

          // Convert from centre-based relative coords to pixel top-left
          int pixelWidth = static_cast<int>(elem.relativeWidth * canvasWidth);
          int pixelHeight =
              static_cast<int>(elem.relativeHeight * canvasHeight);
          int pixelX = static_cast<int>(
              (elem.relativeX - elem.relativeWidth / 2.0) * canvasWidth);
          int pixelY = static_cast<int>(
              (elem.relativeY - elem.relativeHeight / 2.0) * canvasHeight);

          // Clamp to image bounds
          int drawX1 = std::max(0, std::min(pixelX, canvasWidth - 1));
          int drawY1 = std::max(0, std::min(pixelY, canvasHeight - 1));
          int drawX2 =
              std::max(0, std::min(pixelX + pixelWidth, canvasWidth - 1));
          int drawY2 =
              std::max(0, std::min(pixelY + pixelHeight, canvasHeight - 1));

          if (drawX2 > drawX1 && drawY2 > drawY1) {
            cv::Scalar color =
                (elem.type == RelativeElement::TEXT) ? blueColor : greenColor;
            cv::rectangle(canvas, cv::Point(drawX1, drawY1),
                          cv::Point(drawX2, drawY2), color, 2);
            drawnCount++;

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

        std::cerr << "Drew " << drawnCount << " boxes on "
                  << (cropped ? "cropped" : "full") << " image (" << canvasWidth
                  << "x" << canvasHeight << ")" << std::endl;

        // Save marked image with _relmap suffix
        std::filesystem::path markPath(markToFile);
        std::string outputPath = markPath.parent_path().string() + "/" +
                                 markPath.stem().string() + "_relmap" +
                                 markPath.extension().string();

        if (cv::imwrite(outputPath, canvas)) {
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
