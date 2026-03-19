#include "OCRAnalysis.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
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
/**
 * @brief Run Tesseract recognition on @p image using an already-initialised
 *        TessBaseAPI, returning all words above 30% confidence.
 *        The caller is responsible for Init() and End(); this function only
 *        calls SetImage / Recognize / Clear.
 */
static std::vector<OcrWord>
ocrDetectWords(const cv::Mat &image, tesseract::TessBaseAPI *ocr,
               tesseract::PageSegMode psm = tesseract::PSM_SINGLE_BLOCK) {
  std::vector<OcrWord> words;

  ocr->SetPageSegMode(psm);
  ocr->SetImage(image.data, image.cols, image.rows, image.channels(),
                static_cast<int>(image.step));
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
    delete ri;
  }

  ocr->Clear();
  return words;
}

/**
 * @brief Convenience overload that creates its own TessBaseAPI.
 *        Used by callers outside checkImage that don't hold a shared instance.
 */
static std::vector<OcrWord> ocrDetectWords(const cv::Mat &image) {
  std::vector<OcrWord> words;

  tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
  if (ocr->Init("C:/tessdata/tessdata", "eng") != 0 &&
      ocr->Init(NULL, "eng") != 0) {
    std::cerr << "Warning: Could not initialize Tesseract" << std::endl;
    delete ocr;
    return words;
  }

  words = ocrDetectWords(image, ocr);
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
    // Use the bounding box of ALL detected rectangles in PDF y-up coordinates.
    // This covers the full content area (e.g. both sticker sections) while
    // excluding elements outside the sticker boundary (e.g. pull-tabs).
    double rectMinX = std::numeric_limits<double>::max();
    double rectMinY = std::numeric_limits<double>::max();
    double rectMaxX = std::numeric_limits<double>::lowest();
    double rectMaxY = std::numeric_limits<double>::lowest();
    for (const auto &rect : elements.rectangles) {
      rectMinX = std::min(rectMinX, rect.x);
      rectMinY = std::min(rectMinY, rect.y);
      rectMaxX = std::max(rectMaxX, rect.x + rect.width);
      rectMaxY = std::max(rectMaxY, rect.y + rect.height);
    }

    if (rectMinX < std::numeric_limits<double>::max()) {
      // Rectangles found: use their bounding box (PDF y-up coords).
      r.minX = rectMinX;
      r.minY = rectMinY;
      r.maxX = rectMaxX;
      r.maxY = rectMaxY;
      r.success = true;
    } else if (!elements.images.empty()) {
      // No rectangles: fall back to bounding box of all embedded images.
      double imgMinX = std::numeric_limits<double>::max();
      double imgMinY = std::numeric_limits<double>::max();
      double imgMaxX = std::numeric_limits<double>::lowest();
      double imgMaxY = std::numeric_limits<double>::lowest();
      for (const auto &img : elements.images) {
        imgMinX = std::min(imgMinX, img.x);
        imgMinY = std::min(imgMinY, img.y);
        imgMaxX = std::max(imgMaxX, img.x + img.displayWidth);
        imgMaxY = std::max(imgMaxY, img.y + img.displayHeight);
      }
      r.minX = imgMinX;
      r.minY = imgMinY;
      r.maxX = imgMaxX;
      r.maxY = imgMaxY;
      r.success = true;
    } else {
      // Last resort: bounding box of all text elements (preciseX/Y, PDF y-up).
      double txtMinX = std::numeric_limits<double>::max();
      double txtMinY = std::numeric_limits<double>::max();
      double txtMaxX = std::numeric_limits<double>::lowest();
      double txtMaxY = std::numeric_limits<double>::lowest();
      for (const auto &text : elements.textLines) {
        if (isMostlyUnderscores(text.text)) continue;
        double tx = (text.preciseWidth > 0) ? text.preciseX : text.boundingBox.x;
        double ty = (text.preciseWidth > 0) ? text.preciseY : text.boundingBox.y;
        double tw = (text.preciseWidth > 0) ? text.preciseWidth : text.boundingBox.width;
        double th = (text.preciseWidth > 0) ? text.preciseHeight : text.boundingBox.height;
        txtMinX = std::min(txtMinX, tx);
        txtMinY = std::min(txtMinY, ty);
        txtMaxX = std::max(txtMaxX, tx + tw);
        txtMaxY = std::max(txtMaxY, ty + th);
      }
      if (txtMinX < std::numeric_limits<double>::max()) {
        r.minX = txtMinX;
        r.minY = txtMinY;
        r.maxX = txtMaxX;
        r.maxY = txtMaxY;
        r.success = true;
      } else {
        r.errorMessage = "No rectangles, images or text found";
      }
    }

  } else if (elements.linesBoundingBoxWidth > 0 &&
             elements.linesBoundingBoxHeight > 0) {
    // Crop marks detected: use their bounding box (PDF y-up coords).
    r.minX = elements.linesBoundingBoxX;
    r.minY = elements.linesBoundingBoxY;
    r.maxX = elements.linesBoundingBoxX + elements.linesBoundingBoxWidth;
    r.maxY = elements.linesBoundingBoxY + elements.linesBoundingBoxHeight;
    r.success = true;

  } else {
    // USE_CROP_MARKS requested but no crop marks found.
    r.errorMessage = "No crop mark lines found";
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

    if (elem.relativeX < 0.0 || elem.relativeX > 1.0 ||
        elem.relativeY < 0.0 || elem.relativeY > 1.0) continue;

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

    if (elem.relativeX < 0.0 || elem.relativeX > 1.0 ||
        elem.relativeY < 0.0 || elem.relativeY > 1.0) continue;

    outElements.push_back(elem);
  }
}

// ── OCR anchor helpers ────────────────────────────────────────────────────────

/**
 * @brief Match ALL eligible non-placeholder text elements in [fromIdx, toIdx)
 *        against OCR words. Returns every matched pair with its element index
 *        and OCR bounding box.
 *
 * Each unique normalised text is used at most once as an anchor (first PDF
 * element wins for duplicate PDF texts; first matching OCR word wins for
 * duplicate OCR detections).  This deduplication is intentional: labels
 * frequently repeat the same field labels ("Ref.Code:", "Med.No.:") in
 * multiple sticker copies, and including both copies with different OCR
 * pixel positions would give the linear solver two contradictory constraints
 * from physically different image regions, destabilising the crop-rect solve.
 *
 * The preferred anchor sources are elements whose text is unique in the PDF
 * (appears only once), because those can be matched to a single OCR word
 * unambiguously.  Duplicate-text elements are still included but only if they
 * are the first element with that text.
 */
static std::vector<MatchedPair>
findAllMatchedPairs(const std::vector<OCRAnalysis::RelativeElement> &elements,
                    size_t fromIdx, size_t toIdx,
                    const std::vector<OcrWord> &ocrWords)
{
  using RE = OCRAnalysis::RelativeElement;

  // Pre-compute: count how many times each normalised text appears in the
  // eligible element range.  Duplicate-text elements are poor anchor
  // candidates because they are ambiguous — the same OCR word could belong
  // to any of the repeated sections (e.g. two identical sticker copies).
  // They are excluded from anchor matching so that only unambiguous,
  // section-unique elements define the crop-rect transform.
  std::unordered_map<std::string, int> normCount;
  for (size_t i = fromIdx; i < toIdx; ++i) {
    const auto &elem = elements[i];
    if (elem.type != RE::TEXT) continue;
    if (elem.text.find('<') != std::string::npos ||
        elem.text.find('>') != std::string::npos) continue;
    std::string norm = normaliseForMatch(elem.text);
    if (norm.length() >= 2) normCount[norm]++;
  }

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
    if (!seenTexts.insert(normPdf).second) continue;   // skip duplicate texts
    if (normCount[normPdf] > 1) continue;  // skip text shared across sections

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
      std::cerr << "  Matched \"" << elem.text << "\" [" << i << "] -> OCR \""
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
                     int maxMatches = 12)
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
                               const std::string &l1PdfPath, double dpi,
                               const std::string &l2PdfPath) {
  OCRAnalysis::RelativeMapResult result;

  try {
    // ── Determine bounds mode from L1 filename ────────────────────────────────
    std::string l1Stem =
        std::filesystem::path(l1PdfPath).filename().string();
    // Case-insensitive prefix check
    auto startsWithCI = [](const std::string &s, const char *prefix) {
      size_t plen = std::strlen(prefix);
      if (s.size() < plen) return false;
      for (size_t i = 0; i < plen; ++i)
        if (std::tolower((unsigned char)s[i]) != prefix[i]) return false;
      return true;
    };
    RenderBoundsMode l1Mode;
    if (startsWithCI(l1Stem, "l1")) {
      l1Mode = RenderBoundsMode::USE_LARGEST_RECTANGLE;
    } else if (startsWithCI(l1Stem, "l2")) {
      l1Mode = RenderBoundsMode::USE_CROP_MARKS;
    } else {
      result.errorMessage =
          "PDF filename must begin with 'L1' or 'L2': " + l1Stem;
      return result;
    }

    // ── L1: compute bounds and relative elements ──────────────────────────────
    BoundsResult l1Bounds = computeBounds(elements, l1Mode);
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

    // ── Crop to backing paper and rotate to match L1 PDF aspect ratio ─────────
    cv::Mat workImg;
    int cwRotations = 0;
    if (!image.empty()) {
      cv::Mat backing = OCRAnalysis::cropToLabel(image, 50, 40, /*tightLabel=*/false);
      if (backing.empty()) {
        std::cerr << "Warning: cropToLabel returned empty; using full image"
                  << std::endl;
        backing = image.clone();
      }
      std::cerr << "Backing crop: " << backing.cols << "x" << backing.rows
                << std::endl;

      if (l1Bounds.width() > 0 && l1Bounds.height() > 0) {
        double pdfAR  = l1Bounds.width() / l1Bounds.height();
        double bestDiff = 1e9;
        cv::Mat current = backing;
        for (int r = 0; r < 4; ++r) {
          double imgAR = static_cast<double>(current.cols) / current.rows;
          double diff  = std::abs(imgAR - pdfAR);
          if (diff < bestDiff) {
            bestDiff   = diff;
            cwRotations = r;
            workImg    = current;
          }
          cv::Mat next;
          cv::rotate(current, next, cv::ROTATE_90_CLOCKWISE);
          current = next;
        }
      } else {
        workImg = backing;
      }
      std::cerr << "Applied " << cwRotations << " CW 90° rotation(s); "
                << "working image: " << workImg.cols << "x" << workImg.rows
                << std::endl;
    }
    result.cwRotations = cwRotations;

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
        // L2 PDFs always use crop marks for bounds.
        l2Bounds = computeBounds(l2Elements, RenderBoundsMode::USE_CROP_MARKS);
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

    // ── OCR + L1 crop rect ────────────────────────────────────────────────────
    // Always run OCR on the reference image so that the crop rect (pixel
    // mapping) can be stored in the result and reused by checkImage without
    // repeating anchor matching on every subsequent call.
    std::vector<OcrWord> ocrWords;
    if (!workImg.empty()) {
      std::cerr << "Running OCR on reference image ("
                << workImg.cols << "x" << workImg.rows << ")..." << std::endl;
      ocrWords = ocrDetectWords(workImg);
      std::cerr << "Detected " << ocrWords.size() << " word(s)" << std::endl;

      std::cerr << "\n=== L1 OCR anchor matching ===" << std::endl;
      auto l1All     = findAllMatchedPairs(result.elements, 0, l1Count, ocrWords);
      auto l1Anchors = findBestMatchedPairs(l1All);
      std::cerr << "Using " << l1Anchors.size() << " anchor pair(s) (of "
                << l1All.size() << " total matches)" << std::endl;

      cv::Rect l1CropRect;
      if (l1Anchors.size() >= 2 && solveCropRectFromMatches(l1Anchors, l1CropRect)) {
        // Y-scale sanity-check.
        if (l1Bounds.width() > 0 && l1Bounds.height() > 0) {
          double xPxPerPt = static_cast<double>(l1CropRect.width)  / l1Bounds.width();
          double yPxPerPt = static_cast<double>(l1CropRect.height) / l1Bounds.height();
          if (std::abs(xPxPerPt - yPxPerPt) / xPxPerPt > 0.15) {
            int correctedHeight =
                static_cast<int>(std::round(xPxPerPt * l1Bounds.height()));
            double sumOffY = 0;
            for (const auto &mp : l1Anchors)
              sumOffY += mp.ocrCentreY - mp.relCentreY * correctedHeight;
            int correctedY =
                static_cast<int>(std::round(sumOffY / l1Anchors.size()));
            std::cerr << "  L1 Y-scale corrected: cropHeight "
                      << l1CropRect.height << " -> " << correctedHeight
                      << ", cropY " << l1CropRect.y << " -> " << correctedY
                      << std::endl;
            l1CropRect.height = correctedHeight;
            l1CropRect.y      = correctedY;
          }
        }
        result.cropX      = l1CropRect.x;
        result.cropY      = l1CropRect.y;
        result.cropWidth  = l1CropRect.width;
        result.cropHeight = l1CropRect.height;
        result.hasCropRect = true;
        std::cerr << "Stored crop rect: (" << l1CropRect.x << ","
                  << l1CropRect.y << ") " << l1CropRect.width
                  << "x" << l1CropRect.height << std::endl;
      } else {
        std::cerr << "Warning: could not solve L1 crop rect ("
                  << l1Anchors.size() << " anchor(s))" << std::endl;
      }

      // ── L2 anchor matching: re-normalise L2 elements into L1 pixel space ───
      // L2 elements were initially normalised against L2 bounds; we now find
      // their actual pixel positions via OCR anchor matching and re-express
      // them in L1 relative coordinates so checkImage can use a single cropRect.
      const size_t l2Count = result.elements.size() - l1Count;
      if (l2Count > 0 && result.hasCropRect) {
        cv::Rect l1CR(result.cropX, result.cropY,
                      result.cropWidth, result.cropHeight);
        std::cerr << "\n=== L2 OCR anchor matching ===" << std::endl;
        auto l2All     = findAllMatchedPairs(result.elements, l1Count,
                                            result.elements.size(), ocrWords);
        auto l2Anchors = findBestMatchedPairs(l2All);
        std::cerr << "Using " << l2Anchors.size() << " anchor pair(s) (of "
                  << l2All.size() << " total matches)" << std::endl;

        cv::Rect l2CR;
        if (l2Anchors.size() >= 2 && solveCropRectFromMatches(l2Anchors, l2CR)) {
          // Y-scale sanity-check (same as for L1).
          if (l2Bounds.success && l2Bounds.width() > 0 && l2Bounds.height() > 0) {
            double xPxPerPt = static_cast<double>(l2CR.width)  / l2Bounds.width();
            double yPxPerPt = static_cast<double>(l2CR.height) / l2Bounds.height();
            if (std::abs(xPxPerPt - yPxPerPt) / xPxPerPt > 0.15) {
              int correctedH = static_cast<int>(std::round(xPxPerPt * l2Bounds.height()));
              double sumOffY = 0;
              for (const auto &mp : l2Anchors)
                sumOffY += mp.ocrCentreY - mp.relCentreY * correctedH;
              l2CR.y      = static_cast<int>(std::round(sumOffY / l2Anchors.size()));
              l2CR.height = correctedH;
            }
          }
          std::cerr << "L2 crop rect: (" << l2CR.x << "," << l2CR.y << ") "
                    << l2CR.width << "x" << l2CR.height << std::endl;

          // Re-express each L2 element in L1 relative coordinates.
          for (size_t i = l1Count; i < result.elements.size(); ++i) {
            auto &elem = result.elements[i];
            double pixCX = elem.relativeX      * l2CR.width  + l2CR.x;
            double pixCY = elem.relativeY      * l2CR.height + l2CR.y;
            double pixW  = elem.relativeWidth  * l2CR.width;
            double pixH  = elem.relativeHeight * l2CR.height;
            elem.relativeX      = (pixCX - l1CR.x) / l1CR.width;
            elem.relativeY      = (pixCY - l1CR.y) / l1CR.height;
            elem.relativeWidth  = pixW / l1CR.width;
            elem.relativeHeight = pixH / l1CR.height;
          }
          std::cerr << "L2 elements re-normalised to L1 coordinate space."
                    << std::endl;
        } else {
          std::cerr << "Warning: L2 anchor matching failed ("
                    << l2Anchors.size() << " anchor(s)); "
                    << "L2 elements may be misaligned in checkImage." << std::endl;
        }
      }
    }

    // ── marking: draw element boxes on a copy of the image ───────────────────
    if (markImage) {
      if (workImg.empty() || !result.hasCropRect) {
        std::cerr << "Warning: cannot mark – image empty or crop rect unavailable"
                  << std::endl;
      } else {
        cv::Rect l1CropRect(result.cropX, result.cropY,
                            result.cropWidth, result.cropHeight);
        cv::Mat canvas = workImg.clone();
        int drawn = 0;

        // L1
        {
          auto l1All = findAllMatchedPairs(result.elements, 0, l1Count, ocrWords);
          drawn += drawElements(canvas, result.elements, 0, l1Count,
                                l1CropRect.width, l1CropRect.height,
                                l1CropRect.x, l1CropRect.y, l1All);
          std::cerr << "L1: drew " << drawn << " element(s)" << std::endl;
        }

        // L2 elements are already re-normalised to L1 coordinate space above;
        // draw them using the same L1 cropRect.
        {
          const size_t l2Cnt = result.elements.size() - l1Count;
          if (l2Cnt > 0) {
            auto l2All = findAllMatchedPairs(result.elements, l1Count,
                                            result.elements.size(), ocrWords);
            int l2Drawn = drawElements(canvas, result.elements, l1Count,
                                       result.elements.size(),
                                       l1CropRect.width, l1CropRect.height,
                                       l1CropRect.x, l1CropRect.y, l2All);
            drawn += l2Drawn;
            std::cerr << "L2: drew " << l2Drawn << " element(s)" << std::endl;
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
    s_lastRelativeMap = result;
    return result;

  } catch (const std::exception &e) {
    result.errorMessage =
        std::string("Error creating relative map: ") + e.what();
    return result;
  }
}

// Static member definition.
OCRAnalysis::RelativeMapResult OCRAnalysis::s_lastRelativeMap;

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

  if (image.empty() || !relMap.hasCropRect)
    return false;

  // ── Crop to backing paper and apply the stored CW rotation ───────────────
  {
    cv::Mat backing = OCRAnalysis::cropToLabel(image, 50, 40, /*tightLabel=*/false);
    if (backing.empty()) {
      std::cerr << "checkImage: cropToLabel returned empty; using full image"
                << std::endl;
      backing = image.clone();
    }
    for (int r = 0; r < relMap.cwRotations; ++r) {
      cv::Mat next;
      cv::rotate(backing, next, cv::ROTATE_90_CLOCKWISE);
      backing = next;
    }
    image = backing;
    std::cerr << "checkImage: backing crop + " << relMap.cwRotations
              << " CW rotation(s); image now "
              << image.cols << "x" << image.rows << std::endl;
  }

  const cv::Rect cropRect(relMap.cropX, relMap.cropY,
                          relMap.cropWidth, relMap.cropHeight);
  const cv::Rect imageRect(0, 0, image.cols, image.rows);
  constexpr double kPadFraction     = 0.07; // fixed-label padding (all sides)
  constexpr double kPhPadFracY      = 0.15; // placeholder vertical padding
  constexpr double kPhPadFracX      = 0.04; // placeholder horizontal padding

  // ── Pass 1: compute ROI and expected text for every text element ──────────
  // No Tesseract yet – defer init until we know there is something to check.
  struct ElemCheck {
    size_t      idx;
    cv::Rect    roi;
    std::string elemText;    // original (for logging)
    std::string normExpected; // normalised expected string
  };
  std::vector<ElemCheck> checks;

  for (size_t i = 0; i < relMap.elements.size(); ++i) {
    const auto &elem = relMap.elements[i];
    if (elem.type != RE::TEXT) continue;

    std::string expected     = applyPlaceholders(elem.text, placeholders);
    std::string normExpected = normaliseForMatch(expected);
    if (normExpected.empty()) continue;

    double centreX = elem.relativeX     * cropRect.width  + cropRect.x;
    double centreY = elem.relativeY     * cropRect.height + cropRect.y;
    double pixW    = elem.relativeWidth  * cropRect.width;
    double pixH    = elem.relativeHeight * cropRect.height;

    cv::Rect roi;
    if (elem.text.find('<') != std::string::npos) {
      // Placeholder: left-anchored at token left edge, widened to fit value.
      // Use asymmetric padding: 12% vertical, 4% horizontal.
      double valueW = std::max(pixW * 1.75,
                               normExpected.length() * pixH * 0.65);
      int padY   = std::max(4, static_cast<int>(std::round(pixH  * kPhPadFracY)));
      int padX   = std::max(4, static_cast<int>(std::round(valueW * kPhPadFracX)));
      int topPx  = static_cast<int>(std::round(centreY - pixH / 2.0)) - padY;
      int height = static_cast<int>(std::round(pixH)) + 2 * padY;
      int tokenL = static_cast<int>(std::round(centreX - pixW / 2.0));
      roi = cv::Rect(tokenL - padX, topPx,
                     static_cast<int>(std::round(valueW)) + 2 * padX, height);
    } else {
      // Fixed label text: symmetric 7% padding on all sides.
      int padY = std::max(4, static_cast<int>(std::round(pixH * kPadFraction)));
      int padX = std::max(4, static_cast<int>(std::round(pixW * kPadFraction)));
      int topPx  = static_cast<int>(std::round(centreY - pixH / 2.0)) - padY;
      int height = static_cast<int>(std::round(pixH)) + 2 * padY;
      roi = cv::Rect(static_cast<int>(std::round(centreX - pixW / 2.0)) - padX,
                     topPx,
                     static_cast<int>(std::round(pixW)) + 2 * padX, height);
    }
    roi &= imageRect;
    if (roi.area() == 0) continue;

    checks.push_back({i, roi, elem.text, normExpected});
  }

  if (checks.empty())
    return true; // nothing to verify – no Tesseract needed

  // ── Pass 2: init Tesseract once, OCR each ROI individually ───────────────
  tesseract::TessBaseAPI ocr;
  if (ocr.Init("C:/tessdata/tessdata", "eng") != 0 &&
      ocr.Init(nullptr, "eng") != 0) {
    std::cerr << "checkImage: cannot initialise Tesseract" << std::endl;
    return false;
  }
  ocr.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);

  bool allMatch = true;
  struct Marking { cv::Rect roi; bool match; };
  std::vector<Marking> markings;
  markings.reserve(checks.size());

  // Helper: run Tesseract on a (possibly non-contiguous) Mat and return text.
  auto ocrMat = [&](const cv::Mat &m) -> std::string {
    ocr.SetImage(m.data, m.cols, m.rows,
                 m.channels(), static_cast<int>(m.step[0]));
    ocr.Recognize(nullptr);
    char *raw = ocr.GetUTF8Text();
    std::string t = raw ? raw : "";
    delete[] raw;
    ocr.Clear();
    return t;
  };

  // Helper: test whether ocrText matches the normalised expected string.
  auto isMatch = [&](const std::string &normExp,
                     const std::string &ocrText) -> bool {
    std::string got = normaliseForMatch(ocrText);
    if (got.empty()) return false;
    if (normExp == got) return true;
    if (got.find(normExp) != std::string::npos) return true;
    if (normExp.find(got) != std::string::npos) return true;
    int maxDist = std::max(1, static_cast<int>(normExp.length()) / 5);
    return levenshtein(normExp, got) <= maxDist;
  };

  for (const auto &chk : checks) {
    // ── Initial OCR on raw ROI subimage ──────────────────────────────────────
    // roi.step[0] is the parent image's row stride so Tesseract can walk rows.
    cv::Mat roi = image(chk.roi);
    std::string ocrTextInitial = ocrMat(roi);
    bool match = isMatch(chk.normExpected, ocrTextInitial);

    // ── Auto-pass for repeated-i sequences in non-substituted text ───────────
    // OCR reliably confuses "ii", "iii" etc. with "(i)", "(1)", etc.
    // If the expected (normalised) text contains "ii" or more and the element
    // is a fixed label (not a placeholder), treat it as a pass automatically.
    if (!match && chk.elemText.find('<') == std::string::npos) {
      const std::string &ne = chk.normExpected;
      if (ne.find("ii") != std::string::npos)
        match = true;
    }

    // ── Cleanup retry: only if initial match failed ───────────────────────────
    cv::Mat cleanedRoi;
    std::string ocrTextCleaned;
    bool usedCleanup = false;
    if (!match) {
      cv::Mat roiCopy = roi.clone(); // cleanupForOCR needs a contiguous image
      cleanedRoi = OCRAnalysis::cleanupForOCR(roiCopy);
      if (!cleanedRoi.empty()) {
        ocrTextCleaned = ocrMat(cleanedRoi);
        if (isMatch(chk.normExpected, ocrTextCleaned)) {
          match = true;
          usedCleanup = true;
        }
      }
    }

    // Winning OCR text for logging: cleanup result if it saved the match.
    const std::string &ocrText = usedCleanup ? ocrTextCleaned : ocrTextInitial;

    std::cerr << "checkImage: [" << chk.idx << "] \"" << chk.elemText << "\""
              << " ocr=\"" << ocrText << "\""
              << (usedCleanup ? " (cleanup)" : "")
              << " -> " << (match ? "OK" : "FAIL") << std::endl;

#ifndef NDEBUG
    if (!match) {
      // Sanitize element text for use as a filename component.
      std::string safe = chk.elemText;
      for (char &c : safe)
        if (c == '<' || c == '>' || c == '/' || c == '\\' || c == ':' ||
            c == '*' || c == '?' || c == '"' || c == '|' || c == ' ')
          c = '_';

      char idxBuf[16];
      std::snprintf(idxBuf, sizeof(idxBuf), "%03zu", chk.idx);

      std::string roiDir = "debug/roi";
      std::filesystem::create_directories(roiDir);
      std::string prefix = roiDir + "/roi_" + idxBuf + "_" + safe;

      // Raw ROI image.
      cv::imwrite(prefix + ".png", roi.clone());

      // Cleaned ROI image (only written when cleanup was attempted).
      if (!cleanedRoi.empty())
        cv::imwrite(prefix + "_cleaned.png", cleanedRoi);

      // Companion text file.
      std::ofstream tf(prefix + ".txt");
      if (tf.is_open()) {
        tf << "element:      " << chk.idx << "\n"
           << "expected:     " << chk.elemText << "\n"
           << "norm_exp:     " << chk.normExpected << "\n"
           << "ocr_initial:  " << ocrTextInitial << "\n"
           << "norm_initial: " << normaliseForMatch(ocrTextInitial) << "\n";
        if (!cleanedRoi.empty()) {
          tf << "ocr_cleaned:  " << ocrTextCleaned << "\n"
             << "norm_cleaned: " << normaliseForMatch(ocrTextCleaned) << "\n"
             << "used_cleanup: " << (usedCleanup ? "yes" : "no (still failed)") << "\n";
        }
        tf << "result:       " << (match ? "PASS" : "FAIL") << "\n";
      }
    }
#endif

    markings.push_back({chk.roi, match});
    if (!match) allMatch = false;
  }

  // ── Pass 3: draw annotations after all matching is done ──────────────────
  // Drawing is deferred so that rectangles from earlier elements do not
  // appear inside the ROI crop of later elements.
  for (const auto &m : markings)
    cv::rectangle(image, m.roi,
                  m.match ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

  ocr.End();
  return allMatch;
}

bool OCRAnalysis::checkImage(
    cv::Mat &image,
    const std::vector<std::pair<std::string, std::string>> &placeholders)
{
  return checkImage(s_lastRelativeMap, image, placeholders);
}

} // namespace ocr
