#include "OCRAnalysis.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <unordered_map>
#include <vector>

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
 * @brief Compute Levenshtein edit distance between two strings.
 */
static int levenshtein(const std::string &s, const std::string &t) {
  int m = static_cast<int>(s.size()), n = static_cast<int>(t.size());
  std::vector<int> dp0(n + 1), dp1(n + 1);
  for (int j = 0; j <= n; ++j) dp0[j] = j;
  for (int i = 1; i <= m; ++i) {
    dp1[0] = i;
    for (int j = 1; j <= n; ++j)
      dp1[j] = (s[i-1] == t[j-1])
               ? dp0[j-1]
               : 1 + std::min({dp0[j], dp1[j-1], dp0[j-1]});
    std::swap(dp0, dp1);
  }
  return dp0[n];
}

/**
 * @brief Check if a string is mostly comprised of underscores.
 *        Returns true if more than half the characters are underscores.
 */
static bool isMostlyUnderscores(const std::string &s) {
  if (s.empty())
    return false;
  size_t underscoreCount = std::count(s.begin(), s.end(), '_');
  return underscoreCount > s.size() / 2;
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
  size_t elementIdx;  // absolute index into result.elements
  double relCentreX;  // relative centre X from PDF map
  double relCentreY;  // relative centre Y from PDF map
  double ocrCentreX;  // pixel centre X from OCR detection
  double ocrCentreY;  // pixel centre Y from OCR detection
  int    ocrWidth;    // pixel width of the OCR word bounding box
  int    ocrHeight;   // pixel height of the OCR word bounding box
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

  double detX = sumRelX2 * n - sumRelX * sumRelX;
  if (std::abs(detX) < 1e-10) {
    std::cerr << "X system is singular (all matches at same relativeX?)"
              << std::endl;
    return false;
  }
  double cropWidth = (sumRelXOcrX * n - sumRelX * sumOcrX) / detX;
  double cropX = (sumRelX2 * sumOcrX - sumRelX * sumRelXOcrX) / detX;

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

  if (cropWidth < 10 || cropHeight < 10) {
    std::cerr << "Solved crop dimensions too small" << std::endl;
    return false;
  }

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

// ── Bounds helpers ────────────────────────────────────────────────────────────

struct BoundsResult {
  bool success = false;
  std::string errorMessage;
  double minX = 0, minY = 0, maxX = 0, maxY = 0;
  double width()  const { return maxX - minX; }
  double height() const { return maxY - minY; }
};

static BoundsResult computeBounds(const OCRAnalysis::PDFElements &elements,
                                  OCRAnalysis::RenderBoundsMode boundsMode) {
  BoundsResult r;

  if (boundsMode == OCRAnalysis::RenderBoundsMode::USE_LARGEST_RECTANGLE) {
    double largestArea = 0;
    const OCRAnalysis::PDFRectangle *largestRect = nullptr;
    for (const auto &rect : elements.rectangles) {
      double area = rect.width * rect.height;
      if (area > largestArea) { largestArea = area; largestRect = &rect; }
    }

    if (largestRect == nullptr) {
      if (!elements.images.empty()) {
        double largestImageArea = 0.0;
        const OCRAnalysis::PDFEmbeddedImage *largestImage = nullptr;
        for (const auto &img : elements.images) {
          double area = img.displayWidth * img.displayHeight;
          if (area > largestImageArea) { largestImageArea = area; largestImage = &img; }
        }
        if (largestImage != nullptr) {
          r.minX = largestImage->x;
          r.minY = largestImage->y;
          r.maxX = largestImage->x + largestImage->displayWidth;
          r.maxY = largestImage->y + largestImage->displayHeight;
          r.success = true;
        } else {
          r.errorMessage = "Could not find valid rectangle or image";
        }
      } else {
        r.errorMessage = "No rectangles or images found for USE_LARGEST_RECTANGLE mode";
      }
    } else {
      double rectTopLeftY    = largestRect->y;
      double rectBottomLeftY = rectTopLeftY + largestRect->height;
      r.minX = largestRect->x;
      r.minY = elements.pageHeight - rectBottomLeftY;
      r.maxX = largestRect->x + largestRect->width;
      r.maxY = elements.pageHeight - rectTopLeftY;
      r.success = true;
    }

  } else if (elements.linesBoundingBoxWidth > 0 &&
             elements.linesBoundingBoxHeight > 0) {
    r.minX = elements.linesBoundingBoxX;
    r.minY = elements.linesBoundingBoxY;
    r.maxX = elements.linesBoundingBoxX + elements.linesBoundingBoxWidth;
    r.maxY = elements.linesBoundingBoxY + elements.linesBoundingBoxHeight;
    r.success = true;

  } else {
    r.minX = std::numeric_limits<double>::max();
    r.minY = std::numeric_limits<double>::max();
    r.maxX = std::numeric_limits<double>::lowest();
    r.maxY = std::numeric_limits<double>::lowest();

    for (const auto &text : elements.textLines) {
      if (isMostlyUnderscores(text.text)) continue;
      r.minX = std::min(r.minX, static_cast<double>(text.boundingBox.x));
      r.minY = std::min(r.minY, static_cast<double>(text.boundingBox.y));
      r.maxX = std::max(r.maxX, static_cast<double>(text.boundingBox.x + text.boundingBox.width));
      r.maxY = std::max(r.maxY, static_cast<double>(text.boundingBox.y + text.boundingBox.height));
    }
    for (const auto &img : elements.images) {
      r.minX = std::min(r.minX, img.x);
      r.minY = std::min(r.minY, img.y);
      r.maxX = std::max(r.maxX, img.x + img.displayWidth);
      r.maxY = std::max(r.maxY, img.y + img.displayHeight);
    }

    if (r.minX == std::numeric_limits<double>::max()) {
      r.errorMessage = "No elements found to create relative map";
    } else {
      r.success = true;
    }
  }

  return r;
}

/**
 * @brief Convert raw PDF elements to relative coordinates using the given
 *        bounds, filtering elements outside [0,1], and append to outElements.
 */
static void addPDFElementsToMap(
    const OCRAnalysis::PDFElements &src,
    const BoundsResult &b,
    std::vector<OCRAnalysis::RelativeElement> &outElements)
{
  using RE = OCRAnalysis::RelativeElement;
  double bW = b.width(), bH = b.height();

  // Text elements
  for (const auto &text : src.textLines) {
    if (isMostlyUnderscores(text.text)) continue;

    double tX = (text.preciseWidth  > 0) ? text.preciseX      : text.boundingBox.x;
    double tY = (text.preciseWidth  > 0) ? text.preciseY      : text.boundingBox.y;
    double tW = (text.preciseWidth  > 0) ? text.preciseWidth  : text.boundingBox.width;
    double tH = (text.preciseHeight > 0) ? text.preciseHeight : text.boundingBox.height;

    // tY is screen-top; flip relative to bounds
    double topLeftY = bH - (tY - b.minY + tH);

    RE elem;
    elem.type          = RE::TEXT;
    elem.relativeWidth  = tW / bW;
    elem.relativeHeight = tH / bH;
    elem.relativeX      = (tX - b.minX + tW / 2.0) / bW;
    elem.relativeY      = (topLeftY + tH / 2.0)     / bH;

    double left   = elem.relativeX - elem.relativeWidth  / 2.0;
    double right  = elem.relativeX + elem.relativeWidth  / 2.0;
    double top    = elem.relativeY - elem.relativeHeight / 2.0;
    double bottom = elem.relativeY + elem.relativeHeight / 2.0;
    if (right < 0.0 || left > 1.0 || bottom < 0.0 || top > 1.0) continue;

    elem.text      = text.text;
    elem.fontName  = text.fontName;
    elem.fontSize  = text.fontSize;
    elem.isBold    = text.isBold;
    elem.isItalic  = text.isItalic;
    outElements.push_back(elem);
  }

  // Image elements
  for (const auto &img : src.images) {
    // Use the true AABB extents (max - min) rather than displayWidth/Height.
    // For axis-aligned images these are identical, but for rotated images the
    // CTM column magnitudes (displayWidth/Height) are swapped relative to the
    // axis-aligned extents, causing incorrect centre and size calculations.
    double aabbW = img.aabbMaxX - img.x;
    double aabbH = img.aabbMaxY - img.y;

    double imgTopLeftY = bH - (img.y - b.minY + aabbH);

    RE elem;
    elem.type           = RE::IMAGE;
    elem.relativeWidth  = aabbW / bW;
    elem.relativeHeight = aabbH / bH;
    elem.relativeX      = (img.x - b.minX + aabbW / 2.0) / bW;
    elem.relativeY      = (imgTopLeftY      + aabbH / 2.0) / bH;

    double left   = elem.relativeX - elem.relativeWidth  / 2.0;
    double right  = elem.relativeX + elem.relativeWidth  / 2.0;
    double top    = elem.relativeY - elem.relativeHeight / 2.0;
    double bottom = elem.relativeY + elem.relativeHeight / 2.0;
    if (right < 0.0 || left > 1.0 || bottom < 0.0 || top > 1.0) continue;

    outElements.push_back(elem);
  }
}

// ── OCR anchor helpers ────────────────────────────────────────────────────────

/**
 * @brief Match ALL eligible non-placeholder text elements in [fromIdx, toIdx)
 *        against OCR words. Returns every matched pair with its element index
 *        and OCR bounding box.
 */
static std::vector<MatchedPair>
findAllMatchedPairs(const std::vector<OCRAnalysis::RelativeElement> &elements,
                    size_t fromIdx, size_t toIdx,
                    const std::vector<OcrWord> &ocrWords)
{
  using RE = OCRAnalysis::RelativeElement;
  std::vector<MatchedPair> allMatches;
  std::set<std::string> seenTexts;

  for (size_t i = fromIdx; i < toIdx; ++i) {
    const auto &elem = elements[i];
    if (elem.type != RE::TEXT) continue;
    if (elem.text.find('<') != std::string::npos ||
        elem.text.find('>') != std::string::npos) continue;
    if (elem.relativeX < 0.0 || elem.relativeX > 1.0 ||
        elem.relativeY < 0.0 || elem.relativeY > 1.0) continue;
    std::string normPdf = normaliseForMatch(elem.text);
    if (normPdf.length() < 2) continue;
    if (!seenTexts.insert(normPdf).second) continue;

    // Two-pass: prefer exact matches over substring/fuzzy so that a short
    // element ("Code:") doesn't steal an OCR word ("Ref.Code:") that belongs
    // to a longer, more specific element.
    const OcrWord *bestWord = nullptr;
    for (const auto &word : ocrWords) {
      std::string normOcr = normaliseForMatch(word.text);
      if (normPdf == normOcr) { bestWord = &word; break; } // exact — stop here
    }
    if (!bestWord) {
      for (const auto &word : ocrWords) {
        std::string normOcr = normaliseForMatch(word.text);
        bool isMatch =
            (normPdf.length() >= 4 && normOcr.find(normPdf) != std::string::npos) ||
            (normOcr.length() >= 4 && normPdf.find(normOcr) != std::string::npos);
        if (!isMatch && normPdf.length() >= 6 && normOcr.length() >= 6) {
          int maxDist = std::max(2, static_cast<int>(normPdf.length()) / 5);
          if (levenshtein(normPdf, normOcr) <= maxDist)
            isMatch = true;
        }
        if (isMatch) { bestWord = &word; break; }
      }
    }
    if (bestWord) {
      std::cerr << "  Matched \"" << elem.text << "\" -> OCR \""
                << bestWord->text << "\"  conf=" << bestWord->confidence << std::endl;
      MatchedPair mp;
      mp.elementIdx  = i;
      mp.relCentreX  = elem.relativeX;
      mp.relCentreY  = elem.relativeY;
      mp.ocrCentreX  = bestWord->x + bestWord->width  / 2.0;
      mp.ocrCentreY  = bestWord->y + bestWord->height / 2.0;
      mp.ocrWidth    = bestWord->width;
      mp.ocrHeight   = bestWord->height;
      allMatches.push_back(mp);
    }
  }
  return allMatches;
}

/**
 * @brief From all matched pairs, pick up to maxMatches with maximum vertical
 *        spread (so the Y system in solveCropRectFromMatches is non-singular).
 */
static std::vector<MatchedPair>
findBestMatchedPairs(const std::vector<MatchedPair> &allMatches,
                     int maxMatches = 3)
{
  constexpr double kYSpread = 0.05;

  std::vector<MatchedPair> chosen;
  for (const auto &mp : allMatches) {
    if (static_cast<int>(chosen.size()) >= maxMatches) break;
    bool tooClose = false;
    for (const auto &c : chosen) {
      if (std::abs(mp.relCentreY - c.relCentreY) < kYSpread) {
        tooClose = true;
        break;
      }
    }
    if (!tooClose)
      chosen.push_back(mp);
  }

  // If we couldn't meet the spread requirement, fall back to all matches
  if (chosen.size() < 2 && allMatches.size() >= 2)
    chosen = allMatches;

  return chosen;
}

/**
 * @brief Draw a slice of elements onto canvas using the given transform.
 *        For elements that have a direct OCR match in ocrByIndex, the OCR
 *        bounding box is used directly (exact position and width).
 *        Placeholders have their width expanded by 3.2×.
 *        Text → blue, Image → green.
 */
static int drawElements(
    cv::Mat &canvas,
    const std::vector<OCRAnalysis::RelativeElement> &elements,
    size_t fromIdx, size_t toIdx,
    double scaleX, double scaleY, double offX, double offY,
    const std::vector<MatchedPair> &allMatches = {})
{
  // Build index: element index → matched pair
  std::unordered_map<size_t, const MatchedPair*> ocrByIndex;
  for (const auto &mp : allMatches)
    ocrByIndex[mp.elementIdx] = &mp;

  using RE = OCRAnalysis::RelativeElement;
  int drawn = 0;
  for (size_t i = fromIdx; i < toIdx; ++i) {
    const auto &elem = elements[i];

    int pixW, pixH, pixX, pixY;
    auto it = ocrByIndex.find(i);
    if (it != ocrByIndex.end()) {
      // Use OCR bounding box directly — exact position and width
      const MatchedPair &mp = *it->second;
      pixW = mp.ocrWidth;
      pixH = mp.ocrHeight;
      pixX = static_cast<int>(std::round(mp.ocrCentreX - pixW / 2.0));
      pixY = static_cast<int>(std::round(mp.ocrCentreY - pixH / 2.0));
    } else {
      pixW = std::max(1, static_cast<int>(std::round(elem.relativeWidth  * scaleX)));
      pixH = std::max(1, static_cast<int>(std::round(elem.relativeHeight * scaleY)));
      pixX = static_cast<int>(std::round(elem.relativeX * scaleX + offX - pixW / 2.0));
      pixY = static_cast<int>(std::round(elem.relativeY * scaleY + offY - pixH / 2.0));

      bool isPlaceholder = elem.type == RE::TEXT &&
          (elem.text.find('<') != std::string::npos ||
           elem.text.find('>') != std::string::npos);
      if (isPlaceholder)
        pixW = std::min(static_cast<int>(std::round(pixW * 3.2)),
                        canvas.cols - std::max(0, pixX));
    }

    int x1 = std::max(0, std::min(pixX,        canvas.cols - 1));
    int y1 = std::max(0, std::min(pixY,        canvas.rows - 1));
    int x2 = std::max(0, std::min(pixX + pixW, canvas.cols - 1));
    int y2 = std::max(0, std::min(pixY + pixH, canvas.rows - 1));

    if (x2 > x1 && y2 > y1) {
      cv::Scalar color = (elem.type == RE::TEXT)
                             ? cv::Scalar(255, 0, 0)   // blue  – text
                             : cv::Scalar(0, 255, 0);  // green – image
      cv::rectangle(canvas, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
      ++drawn;
    }
  }
  return drawn;
}

// ── Main function ─────────────────────────────────────────────────────────────

OCRAnalysis::RelativeMapResult
OCRAnalysis::createRelativeMap(const PDFElements &elements,
                               const cv::Mat &image,
                               const std::string &imageFilePath, bool markImage,
                               RenderBoundsMode boundsMode, double dpi,
                               const std::string &l2PdfPath) {
  OCRAnalysis::RelativeMapResult result;

  try {
    // ── L1: compute bounds and relative elements ──────────────────────────────
    // When two PDFs are supplied: first is strictly L1 (largest rectangle).
    // When only one PDF is supplied: auto-detect — try crop marks first,
    // fall back to largest rectangle so it works for either PDF type.
    BoundsResult l1Bounds;
    if (!l2PdfPath.empty()) {
      l1Bounds = computeBounds(elements, boundsMode);
    } else {
      l1Bounds = computeBounds(elements, RenderBoundsMode::USE_CROP_MARKS);
      if (!l1Bounds.success)
        l1Bounds = computeBounds(elements, boundsMode);
    }
    if (!l1Bounds.success) {
      result.errorMessage = l1Bounds.errorMessage;
      return result;
    }

    result.boundsX      = l1Bounds.minX;
    result.boundsY      = l1Bounds.minY;
    result.boundsWidth  = l1Bounds.width();
    result.boundsHeight = l1Bounds.height();

    std::cerr << "L1 bounds: (" << l1Bounds.minX << ", " << l1Bounds.minY
              << ") to (" << l1Bounds.maxX << ", " << l1Bounds.maxY << ")  "
              << l1Bounds.width() << " x " << l1Bounds.height() << " pt"
              << std::endl;

    addPDFElementsToMap(elements, l1Bounds, result.elements);
    const size_t l1Count = result.elements.size();
    std::cerr << "L1 elements added: " << l1Count << std::endl;

    // ── L2: compute its own bounds and relative elements ──────────────────────
    BoundsResult l2Bounds;
    if (!l2PdfPath.empty()) {
      std::cerr << "Extracting L2 elements from: " << l2PdfPath << std::endl;
      OCRAnalysis l2Analyzer;
      auto l2Elements = l2Analyzer.extractPDFElements(l2PdfPath);
      if (!l2Elements.success) {
        std::cerr << "Warning: Could not extract L2 elements: "
                  << l2Elements.errorMessage << std::endl;
      } else {
        // Prefer crop-marks bounds for L2 if available; only fall back to
        // the caller's boundsMode if the PDF has no crop-mark lines.
        l2Bounds = computeBounds(l2Elements, RenderBoundsMode::USE_CROP_MARKS);
        if (!l2Bounds.success)
          l2Bounds = computeBounds(l2Elements, boundsMode);
        if (!l2Bounds.success) {
          std::cerr << "Warning: Could not compute L2 bounds: "
                    << l2Bounds.errorMessage << std::endl;
        } else {
          std::cerr << "L2 bounds: (" << l2Bounds.minX << ", " << l2Bounds.minY
                    << ") to (" << l2Bounds.maxX << ", " << l2Bounds.maxY
                    << ")  " << l2Bounds.width() << " x " << l2Bounds.height()
                    << " pt" << std::endl;
          addPDFElementsToMap(l2Elements, l2Bounds, result.elements);
          std::cerr << "L2 elements added: "
                    << (result.elements.size() - l1Count) << std::endl;
        }
      }
    }

    std::cerr << "Total elements: " << result.elements.size()
              << " (" << l1Count << " L1 + "
              << (result.elements.size() - l1Count) << " L2)" << std::endl;

    // ── marking: solve separate transforms for L1 and L2, draw both ─────────
    if (markImage) {
      if (image.empty()) {
        std::cerr << "Warning: Supplied image is empty, cannot mark" << std::endl;
      } else {
        const cv::Mat &targetImage = image;
        std::cerr << "Image supplied (" << targetImage.cols
                  << "x" << targetImage.rows << ")" << std::endl;

        // Run OCR once for both PDFs
        std::cerr << "Running OCR on target image..." << std::endl;
        auto ocrWords = ocrDetectWords(targetImage);
        std::cerr << "Detected " << ocrWords.size() << " word(s)" << std::endl;
        for (const auto &w : ocrWords)
          std::cerr << "  OCR word: \"" << w.text << "\" at ("
                    << w.x << "," << w.y << ") " << w.width << "x" << w.height
                    << " conf=" << w.confidence << std::endl;

        cv::Mat canvas = targetImage.clone();
        int drawn = 0;

        // ── L1 transform ──────────────────────────────────────────────────────
        {
          std::cerr << "\n=== L1 OCR anchor matching ===" << std::endl;
          auto l1All     = findAllMatchedPairs(result.elements, 0, l1Count, ocrWords);
          auto l1Anchors = findBestMatchedPairs(l1All);
          std::cerr << "Using " << l1Anchors.size() << " anchor pair(s) (of "
                    << l1All.size() << " total matches)" << std::endl;

          cv::Rect cropRect;
          if (l1Anchors.size() >= 2 && solveCropRectFromMatches(l1Anchors, cropRect)) {
            // Y-scale sanity-check: if pixels-per-point differs >15% between X
            // and Y (poor anchor Y-spread), override cropHeight from X-scale.
            if (l1Bounds.width() > 0 && l1Bounds.height() > 0) {
              double xPxPerPt = static_cast<double>(cropRect.width)  / l1Bounds.width();
              double yPxPerPt = static_cast<double>(cropRect.height) / l1Bounds.height();
              if (std::abs(xPxPerPt - yPxPerPt) / xPxPerPt > 0.15) {
                int correctedHeight =
                    static_cast<int>(std::round(xPxPerPt * l1Bounds.height()));
                double sumOffY = 0;
                for (const auto &mp : l1Anchors)
                  sumOffY += mp.ocrCentreY - mp.relCentreY * correctedHeight;
                int correctedY =
                    static_cast<int>(std::round(sumOffY / l1Anchors.size()));
                std::cerr << "  L1 Y-scale corrected: cropHeight "
                          << cropRect.height << " -> " << correctedHeight
                          << ", cropY " << cropRect.y << " -> " << correctedY
                          << " (xPxPerPt=" << xPxPerPt
                          << " yPxPerPt=" << yPxPerPt << ")" << std::endl;
                cropRect.height = correctedHeight;
                cropRect.y = correctedY;
              }
            }
            drawn += drawElements(canvas, result.elements, 0, l1Count,
                                  cropRect.width, cropRect.height,
                                  cropRect.x, cropRect.y, l1All);
            std::cerr << "L1: drew " << drawn << " element(s)" << std::endl;
          } else {
            std::cerr << "L1: could not solve transform ("
                      << l1Anchors.size() << " matches)" << std::endl;
          }
        }

        // ── L2 transform ──────────────────────────────────────────────────────
        const size_t l2Count = result.elements.size() - l1Count;
        if (l2Count > 0) {
          std::cerr << "\n=== L2 OCR anchor matching ===" << std::endl;
          auto l2All     = findAllMatchedPairs(result.elements, l1Count,
                                              result.elements.size(), ocrWords);
          auto l2Anchors = findBestMatchedPairs(l2All);
          std::cerr << "Using " << l2Anchors.size() << " anchor pair(s) (of "
                    << l2All.size() << " total matches)" << std::endl;

          cv::Rect cropRect;
          if (l2Anchors.size() >= 2 && solveCropRectFromMatches(l2Anchors, cropRect)) {
            if (l2Bounds.success && l2Bounds.width() > 0 && l2Bounds.height() > 0) {
              double xPxPerPt = static_cast<double>(cropRect.width)  / l2Bounds.width();
              double yPxPerPt = static_cast<double>(cropRect.height) / l2Bounds.height();
              if (std::abs(xPxPerPt - yPxPerPt) / xPxPerPt > 0.15) {
                int correctedHeight =
                    static_cast<int>(std::round(xPxPerPt * l2Bounds.height()));
                double sumOffY = 0;
                for (const auto &mp : l2Anchors)
                  sumOffY += mp.ocrCentreY - mp.relCentreY * correctedHeight;
                int correctedY =
                    static_cast<int>(std::round(sumOffY / l2Anchors.size()));
                std::cerr << "  L2 Y-scale corrected: cropHeight "
                          << cropRect.height << " -> " << correctedHeight
                          << ", cropY " << cropRect.y << " -> " << correctedY
                          << " (xPxPerPt=" << xPxPerPt
                          << " yPxPerPt=" << yPxPerPt << ")" << std::endl;
                cropRect.height = correctedHeight;
                cropRect.y = correctedY;
              }
            }

            int l2Drawn = drawElements(canvas, result.elements, l1Count,
                                       result.elements.size(),
                                       cropRect.width, cropRect.height,
                                       cropRect.x, cropRect.y, l2All);
            drawn += l2Drawn;
            std::cerr << "L2: drew " << l2Drawn << " element(s)" << std::endl;
          } else {
            std::cerr << "L2: could not solve transform ("
                      << l2Anchors.size() << " matches)" << std::endl;
          }
        }

        std::cerr << "Total drawn: " << drawn << " box(es) on image ("
                  << canvas.cols << "x" << canvas.rows << ")" << std::endl;

        std::filesystem::path markPath(imageFilePath);
        std::string outputPath = markPath.parent_path().string() + "/" +
                                 markPath.stem().string() + "_relmap" +
                                 markPath.extension().string();
        if (cv::imwrite(outputPath, canvas))
          std::cerr << "Marked image saved: " << outputPath << std::endl;
        else
          std::cerr << "ERROR: Failed to save marked image: " << outputPath
                    << std::endl;
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

// ─────────────────────────────────────────────────────────────────────────────
// checkImage helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Replace every occurrence of each placeholder token in @p text with
 *        its corresponding value.
 */
static std::string
applyPlaceholders(const std::string &text,
                  const std::vector<std::pair<std::string, std::string>> &placeholders)
{
  std::string result = text;
  for (const auto &[token, value] : placeholders) {
    size_t pos = 0;
    while ((pos = result.find(token, pos)) != std::string::npos) {
      result.replace(pos, token.size(), value);
      pos += value.size();
    }
  }
  return result;
}

// ─────────────────────────────────────────────────────────────────────────────

bool OCRAnalysis::checkImage(
    const RelativeMapResult &relMap, cv::Mat &image,
    const std::vector<std::pair<std::string, std::string>> &placeholders)
{
  using RE = RelativeElement;

  if (image.empty()) {
    std::cerr << "checkImage: image is empty" << std::endl;
    return false;
  }

  // ── Step 1: OCR the full image once ───────────────────────────────────────
  std::cerr << "checkImage: Running OCR on image ("
            << image.cols << "x" << image.rows << ")" << std::endl;
  auto ocrWords = ocrDetectWords(image);
  std::cerr << "checkImage: Detected " << ocrWords.size() << " word(s)" << std::endl;

  // ── Step 2: Solve crop rect via anchor matching ───────────────────────────
  auto allPairs = findAllMatchedPairs(relMap.elements, 0, relMap.elements.size(), ocrWords);
  auto anchors  = findBestMatchedPairs(allPairs);
  std::cerr << "checkImage: " << anchors.size() << " anchor(s) (of "
            << allPairs.size() << " match(es))" << std::endl;

  cv::Rect cropRect;
  if (anchors.size() < 2 || !solveCropRectFromMatches(anchors, cropRect)) {
    std::cerr << "checkImage: Cannot solve crop rect – need ≥2 non-placeholder "
                 "text anchors in the image" << std::endl;
    return false;
  }

  // Y-scale correction: if X and Y pixels-per-point differ by >15% (poor
  // anchor Y spread), override cropHeight using the X scale and re-solve cropY.
  if (relMap.boundsWidth > 0 && relMap.boundsHeight > 0) {
    double xPxPerPt = static_cast<double>(cropRect.width)  / relMap.boundsWidth;
    double yPxPerPt = static_cast<double>(cropRect.height) / relMap.boundsHeight;
    if (std::abs(xPxPerPt - yPxPerPt) / xPxPerPt > 0.15) {
      int correctedH = static_cast<int>(std::round(xPxPerPt * relMap.boundsHeight));
      double sumOffY = 0;
      for (const auto &mp : anchors)
        sumOffY += mp.ocrCentreY - mp.relCentreY * correctedH;
      int correctedY = static_cast<int>(std::round(sumOffY / anchors.size()));
      std::cerr << "checkImage: Y-scale corrected: height " << cropRect.height
                << " -> " << correctedH << ", y " << cropRect.y
                << " -> " << correctedY << std::endl;
      cropRect.height = correctedH;
      cropRect.y      = correctedY;
    }
  }

  std::cerr << "checkImage: cropRect = (" << cropRect.x << "," << cropRect.y
            << ") " << cropRect.width << "x" << cropRect.height << std::endl;

  // ── Step 3: Check each text element ───────────────────────────────────────
  const cv::Rect imageRect(0, 0, image.cols, image.rows);
  bool allMatch = true;

  for (size_t i = 0; i < relMap.elements.size(); ++i) {
    const auto &elem = relMap.elements[i];
    if (elem.type != RE::TEXT) continue;

    // Substitute placeholders into expected text
    std::string expected     = applyPlaceholders(elem.text, placeholders);
    std::string normExpected = normaliseForMatch(expected);
    if (normExpected.empty()) continue;

    // Map relative centre/size → pixel bounding box
    double centreX = elem.relativeX      * cropRect.width  + cropRect.x;
    double centreY = elem.relativeY      * cropRect.height + cropRect.y;
    double halfW   = elem.relativeWidth  * cropRect.width  / 2.0;
    double halfH   = elem.relativeHeight * cropRect.height / 2.0;

    cv::Rect box(static_cast<int>(std::round(centreX - halfW)),
                 static_cast<int>(std::round(centreY - halfH)),
                 static_cast<int>(std::round(halfW * 2.0)),
                 static_cast<int>(std::round(halfH * 2.0)));
    cv::Rect clipped = box & imageRect;
    if (clipped.area() == 0) {
      std::cerr << "checkImage: [" << i << "] \"" << elem.text
                << "\" box out of image bounds – skipped" << std::endl;
      continue;
    }

    // Collect OCR words whose centre falls within the (clipped) box, sorted
    // left-to-right so multi-word elements are joined in reading order.
    std::vector<const OcrWord *> inBox;
    for (const auto &w : ocrWords) {
      int cx = w.x + w.width  / 2;
      int cy = w.y + w.height / 2;
      if (clipped.contains(cv::Point(cx, cy)))
        inBox.push_back(&w);
    }
    std::sort(inBox.begin(), inBox.end(),
              [](const OcrWord *a, const OcrWord *b) { return a->x < b->x; });

    std::string ocrText;
    for (const auto *w : inBox) {
      if (!ocrText.empty()) ocrText += ' ';
      ocrText += w->text;
    }
    std::string normOcr = normaliseForMatch(ocrText);

    // Match: exact normalised, then containment, then Levenshtein
    bool match = (normExpected == normOcr);
    if (!match)
      match = normOcr.find(normExpected)     != std::string::npos ||
              normExpected.find(normOcr)     != std::string::npos;
    if (!match) {
      int maxDist = std::max(1, static_cast<int>(normExpected.length()) / 5);
      match = levenshtein(normExpected, normOcr) <= maxDist;
    }

    std::cerr << "checkImage: [" << i << "] \"" << elem.text << "\""
              << " expected=\"" << expected << "\""
              << " ocr=\"" << ocrText << "\""
              << " -> " << (match ? "OK" : "FAIL") << std::endl;

    if (match) {
      // Use the union of the OCR word bounding boxes for the green rectangle
      // so it reflects the actual detected text extent, not the map estimate.
      if (!inBox.empty()) {
        int ox1 = inBox.front()->x;
        int oy1 = inBox.front()->y;
        int ox2 = inBox.front()->x + inBox.front()->width;
        int oy2 = inBox.front()->y + inBox.front()->height;
        for (const auto *w : inBox) {
          ox1 = std::min(ox1, w->x);
          oy1 = std::min(oy1, w->y);
          ox2 = std::max(ox2, w->x + w->width);
          oy2 = std::max(oy2, w->y + w->height);
        }
        cv::rectangle(image, cv::Rect(ox1, oy1, ox2 - ox1, oy2 - oy1),
                      cv::Scalar(0, 255, 0), 2);
      }
    } else {
      allMatch = false;
      cv::rectangle(image, clipped, cv::Scalar(0, 0, 255), 2);
    }
  }

  return allMatch;
}

} // namespace ocr
