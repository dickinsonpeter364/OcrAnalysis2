#include "OCRAnalysis.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>

// Cairo for PDF/PNG rendering (if available)
#ifdef HAVE_CAIRO
#include <cairo-pdf.h>
#include <cairo.h>
#endif

// ZXing for barcode / DataMatrix detection (if available)
#ifdef HAVE_ZXING
#include "C:/zxing-cpp/core/src/BarcodeFormat.h"
#include "C:/zxing-cpp/core/src/ImageView.h"
#include "C:/zxing-cpp/core/src/ReadBarcode.h"
#endif

// Define M_PI if not already defined (Windows)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Poppler C++ wrapper
#include <poppler-document.h>
#include <poppler-page-renderer.h>
#include <poppler-page.h>

// Poppler low-level API for embedded image extraction
#include <Error.h>
#include <GfxState.h>
#include <GlobalParams.h>
#include <OutputDev.h>
#include <PDFDoc.h>
#include <Page.h>
#include <Stream.h>
#include <TextOutputDev.h>
#include <goo/GooString.h>
#include <tesseract/resultiterator.h>

// Poppler Splash renderer for full-page rasterization
#include <SplashOutputDev.h>
#include <splash/SplashBitmap.h>

namespace ocr {

OCRAnalysis::OCRAnalysis()
    : m_tesseract(std::make_unique<tesseract::TessBaseAPI>()), m_config(),
      m_initialized(false) {}

OCRAnalysis::OCRAnalysis(const OCRConfig &config)
    : m_tesseract(std::make_unique<tesseract::TessBaseAPI>()), m_config(config),
      m_initialized(false) {}

OCRAnalysis::~OCRAnalysis() {
  if (m_tesseract) {
    m_tesseract->End();
  }
}

OCRAnalysis::OCRAnalysis(OCRAnalysis &&other) noexcept
    : m_tesseract(std::move(other.m_tesseract)),
      m_config(std::move(other.m_config)), m_initialized(other.m_initialized) {
  other.m_initialized = false;
}

OCRAnalysis &OCRAnalysis::operator=(OCRAnalysis &&other) noexcept {
  if (this != &other) {
    if (m_tesseract) {
      m_tesseract->End();
    }
    m_tesseract = std::move(other.m_tesseract);
    m_config = std::move(other.m_config);
    m_initialized = other.m_initialized;
    other.m_initialized = false;
  }
  return *this;
}

bool OCRAnalysis::initialize() {
  if (m_initialized) {
    return true;
  }

  const char *tessDataPath = nullptr;

  // Priority 1: Use config path if provided
  if (!m_config.tessDataPath.empty()) {
    tessDataPath = m_config.tessDataPath.c_str();
  }
  // Priority 2: Check TESSDATA_PREFIX environment variable
  else {
    const char *envPath = std::getenv("TESSDATA_PREFIX");
    if (envPath != nullptr) {
      tessDataPath = envPath;
    } else {
      // Priority 3: Use default path
      tessDataPath = "c:\\tessdata\\tessdata";
      std::cerr << "TESSDATA_PREFIX not set, using default: " << tessDataPath
                << std::endl;
    }
  }

  int result = m_tesseract->Init(tessDataPath, m_config.language.c_str());

  if (result != 0) {
    std::cerr << "Failed to initialize Tesseract with language: "
              << m_config.language << std::endl;
    return false;
  }

  m_tesseract->SetPageSegMode(m_config.pageSegMode);
  m_initialized = true;
  return true;
}

bool OCRAnalysis::isInitialized() const { return m_initialized; }

OCRResult OCRAnalysis::analyzeImage(const std::string &imagePath) {
  OCRResult result;
  result.success = false;

  // Load image using OpenCV
  cv::Mat image = cv::imread(imagePath);
  if (image.empty()) {
    result.errorMessage = "Failed to load image: " + imagePath;
    return result;
  }

  return analyzeImage(image);
}

OCRResult OCRAnalysis::analyzeImage(const cv::Mat &image) {
  OCRResult result;
  result.success = false;

  if (!m_initialized) {
    result.errorMessage =
        "OCR engine not initialized. Call initialize() first.";
    return result;
  }

  if (image.empty()) {
    result.errorMessage = "Input image is empty";
    return result;
  }

  auto startTime = std::chrono::high_resolution_clock::now();

  try {
    // Preprocess image if configured
    cv::Mat processedImage =
        m_config.preprocessImage ? preprocessImage(image) : image;

    // Find the best rotation for the image
    int bestRotation = findBestRotation(processedImage);

    // Apply the best rotation
    cv::Mat orientedImage;
    if (bestRotation != -1) {
      cv::rotate(processedImage, orientedImage, bestRotation);
    } else {
      orientedImage = processedImage.clone();
    }

    // Set the correctly oriented image for Tesseract
    setImage(orientedImage);
    m_tesseract->Recognize(nullptr);

    // Get the recognized text from the correctly oriented image
    char *outText = m_tesseract->GetUTF8Text();
    if (outText) {
      result.fullText = outText;
      delete[] outText;
    }

    // Get detailed text regions (this will also detect orientation internally)
    result.regions = detectTextRegions(processedImage);

    // Filter regions by confidence if configured
    if (m_config.minConfidence > 0) {
      result.regions.erase(
          std::remove_if(result.regions.begin(), result.regions.end(),
                         [this](const TextRegion &region) {
                           return region.confidence < m_config.minConfidence;
                         }),
          result.regions.end());
    }

    result.success = true;
  } catch (const std::exception &e) {
    result.errorMessage = std::string("OCR analysis failed: ") + e.what();
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  result.processingTimeMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();

  return result;
}

OCRResult OCRAnalysis::extractTextFromPDF(const std::string &pdfPath,
                                          PDFExtractionLevel level) {
  OCRResult result;
  result.success = false;

  auto startTime = std::chrono::high_resolution_clock::now();

  try {
    // Use TextOutputDev via displayPage so coordinates are in the same
    // rendered/rotated space as LineExtractorOutputDev.  This ensures
    // text bounding boxes share the same PDF bottom-left origin as the
    // crop-mark and image data later used by renderElementsToPNG.
    GlobalParamsIniter gpi(nullptr);
    auto gooFile = std::make_unique<GooString>(pdfPath);
    std::unique_ptr<PDFDoc> doc(new PDFDoc(std::move(gooFile)));

    if (!doc || !doc->isOk()) {
      result.errorMessage = "Failed to load PDF file: " + pdfPath;
      return result;
    }
    if (doc->getNumPages() < 1) {
      result.errorMessage = "PDF has no pages";
      return result;
    }

    // Get the display height (after rotation) for y-flip.
    Page *pg1 = doc->getPage(1);
    const ::PDFRectangle *mb = pg1->getMediaBox();
    int pageRotate = pg1->getRotate(); // 0, 90, 180, 270
    // After applying rotation, the display width/height may swap.
    double dispW, dispH;
    if (pageRotate == 90 || pageRotate == 270) {
      dispW = mb->y2 - mb->y1; // landscape: display width = portrait height
      dispH = mb->x2 - mb->x1; // landscape: display height = portrait width
    } else {
      dispW = mb->x2 - mb->x1;
      dispH = mb->y2 - mb->y1;
    }
    std::cerr << "DEBUG: extractTextFromPDF pageRotate=" << pageRotate
              << " dispW=" << dispW << " dispH=" << dispH << std::endl;

    // Run TextOutputDev through the same displayPage used by line extraction.
    TextOutputDev textOut(nullptr, true, 0, false, false);
    doc->displayPage(&textOut, 1, 72, 72, 0 /*use PDF rotation*/, false, true,
                     false);
    TextPage *textPage = textOut.takeText();

    std::vector<TextRegion> pageRegions;
    std::string fullText;

    for (const TextFlow *flow = textPage->getFlows(); flow;
         flow = flow->getNext()) {
      for (const TextBlock *blk = flow->getBlocks(); blk;
           blk = blk->getNext()) {
        for (const TextLine *ln = blk->getLines(); ln; ln = ln->getNext()) {
          for (const TextWord *word = ln->getWords(); word;
               word = word->getNext()) {
            if (word->getLength() == 0)
              continue;

            // Build UTF-8 text for this word
            std::string text;
            for (int ci = 0; ci < word->getLength(); ci++) {
              const Unicode *uns = word->getChar(ci);
              if (!uns)
                continue;
              Unicode u = *uns;
              if (u < 0x80) {
                text += static_cast<char>(u);
              } else if (u < 0x800) {
                text += static_cast<char>(0xC0 | (u >> 6));
                text += static_cast<char>(0x80 | (u & 0x3F));
              } else {
                text += static_cast<char>(0xE0 | (u >> 12));
                text += static_cast<char>(0x80 | ((u >> 6) & 0x3F));
                text += static_cast<char>(0x80 | (u & 0x3F));
              }
            }
            if (text.empty())
              continue;

            // getBBox() returns coordinates in screen-Y-down space:
            //   xMin/xMax = left/right (same as PDF x)
            //   yMin = TOP of word  (screen-Y, 0=top of display)
            //   yMax = BOTTOM of word (screen-Y, larger = lower on page)
            // Convert to PDF y-up (bottom-left origin, y-up from bottom of
            // display) using the display height (dispH):
            //   pdf_y_bottom = dispH - yMax   ← bottom edge (y-up)
            //   pdf_y_top    = dispH - yMin   ← top edge (y-up)
            double xMin, yMinScr, xMax, yMaxScr;
            word->getBBox(&xMin, &yMinScr, &xMax, &yMaxScr);
            double pdf_y_bottom = dispH - yMaxScr; // PDF y-up bottom edge
            double pdf_y_top = dispH - yMinScr;    // PDF y-up top edge
            double width = xMax - xMin;
            double height = pdf_y_top - pdf_y_bottom;

            TextRegion region;
            region.text = text;
            // Store bottom-left y-up coordinate (PDF bottom-left origin,
            // same space as images and line-extraction crop marks).
            region.boundingBox = cv::Rect(
                static_cast<int>(xMin), static_cast<int>(pdf_y_bottom),
                static_cast<int>(width), static_cast<int>(height + 0.5));
            region.preciseX = xMin;
            region.preciseY = pdf_y_bottom; // PDF bottom-left y (bottom edge)
            region.preciseWidth = width;
            region.preciseHeight = height;
            region.fontSize = height; // approximation

            // Font info
            const TextFontInfo *fi = word->getFontInfo(0);
            if (fi) {
              const GooString *fn = fi->getFontName();
              if (fn) {
                region.fontName = fn->toStr();
                auto plusPos = region.fontName.find('+');
                if (plusPos != std::string::npos)
                  region.fontName = region.fontName.substr(plusPos + 1);
              }
              region.isBold = fi->isBold();
              region.isItalic = fi->isItalic();
            }

            // Orientation from TextWord rotation (0=0°,1=90°,2=180°,3=270°)
            int rot = word->getRotation();
            if (rot == 1 || rot == 3)
              region.orientation = TextOrientation::Vertical;
            else
              region.orientation = TextOrientation::Horizontal;

            region.confidence = 80.0f;
            region.level = 1; // page 1

            pageRegions.push_back(region);
            fullText += text + " ";
          }
        }
      }
    }
    textPage->decRefCnt();

    // Group words into text lines if requested.
    if (level == PDFExtractionLevel::Word || pageRegions.empty()) {
      result.regions = std::move(pageRegions);
    } else {
      // Simple greedy line grouping by Y proximity (bottom-left coords).
      std::vector<TextRegion> lineRegions;
      std::vector<bool> used(pageRegions.size(), false);
      for (size_t i = 0; i < pageRegions.size(); i++) {
        if (used[i])
          continue;
        TextRegion line = pageRegions[i];
        used[i] = true;
        bool isVert = (line.orientation == TextOrientation::Vertical);
        int tol = isVert ? std::max(5, line.boundingBox.width / 2)
                         : std::max(5, line.boundingBox.height / 2);
        for (size_t j = i + 1; j < pageRegions.size(); j++) {
          if (used[j])
            continue;
          if (pageRegions[j].orientation != line.orientation)
            continue;
          const TextRegion &cand = pageRegions[j];
          bool same;
          if (isVert) {
            same = std::abs(cand.boundingBox.x - line.boundingBox.x) <= tol;
          } else {
            int yc1 = line.boundingBox.y + line.boundingBox.height / 2;
            int yc2 = cand.boundingBox.y + cand.boundingBox.height / 2;
            same = std::abs(yc1 - yc2) <= tol;
          }
          if (same) {
            used[j] = true;
            line.text += " " + cand.text;
            line.boundingBox = line.boundingBox | cand.boundingBox;
            line.preciseX = std::min(line.preciseX, cand.preciseX);
            line.preciseY = std::min(line.preciseY, cand.preciseY);
            double r1 = line.preciseX + line.preciseWidth;
            double r2 = cand.preciseX + cand.preciseWidth;
            double t1 = line.preciseY + line.preciseHeight;
            double t2 = cand.preciseY + cand.preciseHeight;
            line.preciseWidth = std::max(r1, r2) - line.preciseX;
            line.preciseHeight = std::max(t1, t2) - line.preciseY;
          }
        }
        lineRegions.push_back(std::move(line));
      }
      result.regions = std::move(lineRegions);
    }
    result.fullText = fullText;
    result.success = true;

  } catch (const std::exception &e) {
    result.errorMessage =
        std::string("PDF text extraction failed: ") + e.what();
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  result.processingTimeMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();

  return result;
}

OCRAnalysis::PDFGraphicsResult
OCRAnalysis::extractGraphicsFromPDF(const std::string &pdfPath, double dpi) {
  PDFGraphicsResult result;
  result.success = false;

  auto startTime = std::chrono::high_resolution_clock::now();

  try {
    // Load the PDF document
    std::unique_ptr<poppler::document> doc(
        poppler::document::load_from_file(pdfPath));

    if (!doc) {
      result.errorMessage = "Failed to load PDF file: " + pdfPath;
      return result;
    }

    if (doc->is_locked()) {
      result.errorMessage = "PDF file is password protected: " + pdfPath;
      return result;
    }

    int pageCount = doc->pages();
    if (pageCount < 1) {
      result.errorMessage = "PDF has no pages";
      return result;
    }

    // Create page renderer with antialiasing
    poppler::page_renderer renderer;
    renderer.set_render_hint(poppler::page_renderer::antialiasing, true);
    renderer.set_render_hint(poppler::page_renderer::text_antialiasing, true);
    renderer.set_image_format(poppler::image::format_argb32);

    // Render only the first page (index 0)
    int pageIndex = 0;
    std::unique_ptr<poppler::page> page(doc->create_page(pageIndex));

    if (!page) {
      result.errorMessage = "Failed to create first page";
      return result;
    }

    // Render page at specified DPI
    poppler::image popplerImage = renderer.render_page(page.get(), dpi, dpi);

    if (!popplerImage.is_valid()) {
      result.errorMessage = "Failed to render first page";
      return result;
    }

    // Convert Poppler image to OpenCV Mat
    int width = popplerImage.width();
    int height = popplerImage.height();

    // Create OpenCV Mat based on Poppler image format
    cv::Mat mat;

    switch (popplerImage.format()) {
    case poppler::image::format_argb32: {
      // ARGB32 format - 4 bytes per pixel
      mat = cv::Mat(height, width, CV_8UC4,
                    const_cast<char *>(popplerImage.const_data()),
                    popplerImage.bytes_per_row())
                .clone();
      // Convert ARGB to BGRA (OpenCV format)
      cv::cvtColor(mat, mat, cv::COLOR_BGRA2BGR);
      break;
    }
    case poppler::image::format_rgb24: {
      // RGB24 format - 3 bytes per pixel
      mat = cv::Mat(height, width, CV_8UC3,
                    const_cast<char *>(popplerImage.const_data()),
                    popplerImage.bytes_per_row())
                .clone();
      cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
      break;
    }
    case poppler::image::format_bgr24: {
      // BGR24 format - already in OpenCV format
      mat = cv::Mat(height, width, CV_8UC3,
                    const_cast<char *>(popplerImage.const_data()),
                    popplerImage.bytes_per_row())
                .clone();
      break;
    }
    case poppler::image::format_gray8: {
      // Grayscale
      mat = cv::Mat(height, width, CV_8UC1,
                    const_cast<char *>(popplerImage.const_data()),
                    popplerImage.bytes_per_row())
                .clone();
      break;
    }
    default:
      result.errorMessage = "Unsupported image format";
      return result;
    }

    // Create PDFGraphic entry
    PDFGraphic graphic;
    graphic.image = mat;
    graphic.pageNumber = 1;
    graphic.width = width;
    graphic.height = height;
    graphic.dpi = dpi;

    result.pages.push_back(graphic);

    result.success = true;

  } catch (const std::exception &e) {
    result.errorMessage =
        std::string("PDF graphics extraction failed: ") + e.what();
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  result.processingTimeMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();

  return result;
}

// Custom OutputDev to capture embedded images from PDF
namespace {

class ImageExtractorOutputDev : public OutputDev {
public:
  ImageExtractorOutputDev() : doc(nullptr), pageNumber(1), imageIndex(0) {}

  // Get the extracted images
  std::vector<OCRAnalysis::PDFEmbeddedImage> &getImages() { return images; }

  void setPageNumber(int page) {
    pageNumber = page;
    imageIndex = 0;
  }

  void setDoc(PDFDoc *d) { doc = d; }

  // Required OutputDev overrides
  bool upsideDown() override { return false; }
  bool useDrawChar() override { return false; }
  bool interpretType3Chars() override { return false; }
  bool needNonText() override { return true; } // We need images!

  // Handle 1-bit image masks (stencil images).
  // An image mask is a 1-bit-per-pixel image where each pixel is either
  // the current fill colour or transparent. Logos and icons are often
  // stored this way in PDFs.
  void drawImageMask(GfxState *state, Object * /*ref*/, Stream *str, int width,
                     int height, bool invert, bool /*interpolate*/,
                     bool /*inlineImg*/) override {

    std::cerr << "DEBUG: drawImageMask called: " << width << "x" << height
              << " invert=" << invert << std::endl;

    if (width <= 0 || height <= 0) {
      return;
    }

    // Get the current fill colour â€” this is used for opaque mask pixels
    GfxRGB fillRgb;
    state->getFillRGB(&fillRgb);
    unsigned char fgR = colToByte(fillRgb.r);
    unsigned char fgG = colToByte(fillRgb.g);
    unsigned char fgB = colToByte(fillRgb.b);

    // Read the 1-bit mask
    str->reset();
    ImageStream imgStr(str, width, 1, 1);
    imgStr.reset();

    // Composite over white: opaque mask pixels â†’ fill colour, rest â†’ white
    cv::Mat mat(height, width, CV_8UC3);
    for (int row = 0; row < height; row++) {
      unsigned char *line = imgStr.getLine();
      if (!line)
        break;
      unsigned char *imgRow = mat.ptr<unsigned char>(row);
      for (int col = 0; col < width; col++) {
        // Per PDF spec: if invert==false, sample 0 = paint, 1 = transparent
        //               if invert==true,  sample 1 = paint, 0 = transparent
        bool paint = invert ? (line[col] != 0) : (line[col] == 0);
        if (paint) {
          imgRow[col * 3 + 0] = fgB; // BGR for OpenCV
          imgRow[col * 3 + 1] = fgG;
          imgRow[col * 3 + 2] = fgR;
        } else {
          imgRow[col * 3 + 0] = 255;
          imgRow[col * 3 + 1] = 255;
          imgRow[col * 3 + 2] = 255;
        }
      }
    }
    imgStr.close();

    // Position info from CTM
    const auto &ctm = state->getCTM();
    double displayWidth = std::sqrt(ctm[0] * ctm[0] + ctm[1] * ctm[1]);
    double displayHeight = std::sqrt(ctm[2] * ctm[2] + ctm[3] * ctm[3]);

    double x0 = ctm[4], y0 = ctm[5];
    double x1 = ctm[4] + ctm[0], y1 = ctm[5] + ctm[1];
    double x2 = ctm[4] + ctm[2], y2 = ctm[5] + ctm[3];
    double x3 = ctm[4] + ctm[0] + ctm[2], y3 = ctm[5] + ctm[1] + ctm[3];

    double x = std::min({x0, x1, x2, x3});
    double y = std::min({y0, y1, y2, y3});
    double rotationAngle = std::atan2(ctm[1], ctm[0]);

    OCRAnalysis::PDFEmbeddedImage img;
    img.image = mat;
    img.pageNumber = pageNumber;
    img.imageIndex = imageIndex++;
    img.width = width;
    img.height = height;
    img.x = x;
    img.y = y;
    img.displayWidth = displayWidth;
    img.displayHeight = displayHeight;
    img.rotationAngle = rotationAngle;
    img.type = "image_mask";

    std::cerr << "DEBUG: drawImageMask extracted " << width << "x" << height
              << " at (" << x << ", " << y << "), display " << displayWidth
              << "x" << displayHeight << ", fill=(" << (int)fgR << ","
              << (int)fgG << "," << (int)fgB << ")" << std::endl;

    images.push_back(img);
  }

  // This is called for each image in the PDF
  void drawImage(GfxState *state, Object *ref, Stream *str, int width,
                 int height, GfxImageColorMap *colorMap, bool /*interpolate*/,
                 const int * /*maskColors*/, bool inlineImg) override {

    std::cerr << "DEBUG: drawImage called: " << width << "x" << height
              << " nComps=" << (colorMap ? colorMap->getNumPixelComps() : -1)
              << " inline=" << inlineImg << std::endl;

    if (width <= 0 || height <= 0 || !colorMap) {
      return;
    }

    // Get transformation matrix for position info
    const auto &ctm = state->getCTM(); // Returns std::array<double, 6>

    // CTM format: [a, b, c, d, e, f] represents transformation matrix:
    // [a b]   [x]   [e]
    // [c d] * [y] + [f]
    //
    // The CTM maps the unit square (0,0)-(1,1) to the image rectangle
    // For rotated images, we need to calculate the axis-aligned bounding box

    // Calculate display dimensions from CTM
    double displayWidth = std::sqrt(ctm[0] * ctm[0] + ctm[1] * ctm[1]);
    double displayHeight = std::sqrt(ctm[2] * ctm[2] + ctm[3] * ctm[3]);

    // Calculate all four corners of the transformed image
    // (0,0) -> (e, f)
    // (1,0) -> (e+a, f+b)
    // (0,1) -> (e+c, f+d)
    // (1,1) -> (e+a+c, f+b+d)
    double x0 = ctm[4];
    double y0 = ctm[5];
    double x1 = ctm[4] + ctm[0];
    double y1 = ctm[5] + ctm[1];
    double x2 = ctm[4] + ctm[2];
    double y2 = ctm[5] + ctm[3];
    double x3 = ctm[4] + ctm[0] + ctm[2];
    double y3 = ctm[5] + ctm[1] + ctm[3];

    // Find the axis-aligned bounding box (actual rendered position)
    double x = std::min({x0, x1, x2, x3});
    double y = std::min({y0, y1, y2, y3});

    // Determine image type and channels
    int nComps = colorMap->getNumPixelComps();
    int nBits = colorMap->getBits();

    // For non-inline images, fetch a fresh stream from the PDF's XRef table.
    // The stream pointer passed to drawImage has already been consumed by
    // Poppler's rendering pipeline and reads as all-zeros.
    Stream *imgStream = str;
    Object fetchedObj;
    if (!inlineImg && ref && ref->isRef() && doc) {
      fetchedObj = ref->fetch(doc->getXRef());
      if (fetchedObj.isStream()) {
        imgStream = fetchedObj.getStream();
      }
    }

    // Read the image data via ImageStream
    imgStream->reset();
    ImageStream imgStr(imgStream, width, nComps, nBits);
    imgStr.reset();

    // Create OpenCV Mat - always use 3-channel output for simplicity
    cv::Mat mat(height, width, CV_8UC3);

    GfxRGB rgb;

    for (int row = 0; row < height; row++) {
      unsigned char *line = imgStr.getLine();
      if (!line)
        break;

      unsigned char *imgRow = mat.ptr<unsigned char>(row);

      for (int col = 0; col < width; col++) {
        // Convert pixel to RGB using color map
        colorMap->getRGB(&line[col * nComps], &rgb);
        imgRow[col * 3 + 0] = colToByte(rgb.b); // OpenCV uses BGR
        imgRow[col * 3 + 1] = colToByte(rgb.g);
        imgRow[col * 3 + 2] = colToByte(rgb.r);
      }
    }

    imgStr.close();

    // Calculate rotation angle from CTM
    // The CTM is [a, b, c, d, e, f] where:
    // - (a, b) is the transformed (1, 0) vector
    // - (c, d) is the transformed (0, 1) vector
    // Rotation angle = atan2(b, a)
    double rotationAngle = std::atan2(ctm[1], ctm[0]);

    // Create the embedded image entry
    OCRAnalysis::PDFEmbeddedImage img;
    img.image = mat;
    img.pageNumber = pageNumber;
    img.imageIndex = imageIndex++;
    img.width = width;
    img.height = height;
    img.x = x;
    img.y = y;
    img.displayWidth = displayWidth;
    img.displayHeight = displayHeight;
    img.rotationAngle = rotationAngle;
    img.type = "raw";

    images.push_back(img);
  }

  // Handle images with soft masks (SMask / alpha channel).
  // In some PDFs, text like "<MED>" is stored as a solid-fill image
  // whose letter shapes are cut out via the soft mask.
  void drawSoftMaskedImage(GfxState *state, Object *ref, Stream *str, int width,
                           int height, GfxImageColorMap *colorMap,
                           bool /*interpolate*/, Stream *maskStr, int maskWidth,
                           int maskHeight, GfxImageColorMap *maskColorMap,
                           bool /*maskInterpolate*/) override {

    std::cerr << "DEBUG: drawSoftMaskedImage called: base " << width << "x"
              << height << ", mask " << maskWidth << "x" << maskHeight
              << std::endl;

    if (maskWidth <= 0 || maskHeight <= 0 || !maskColorMap) {
      // Fall back to regular drawImage if no usable mask
      drawImage(state, ref, str, width, height, colorMap, false, nullptr,
                false);
      return;
    }

    // --- Read the foreground colour from the base image ---------
    // The base image is typically a tiny solid-fill rectangle.
    int fgNComps = colorMap ? colorMap->getNumPixelComps() : 0;
    int fgBits = colorMap ? colorMap->getBits() : 8;
    GfxRGB fgRgb;
    fgRgb.r = 0;
    fgRgb.g = 0;
    fgRgb.b = 0; // default black

    if (colorMap && width > 0 && height > 0) {
      Stream *fgStream = str;
      Object fetchedFg;
      if (ref && ref->isRef() && doc) {
        fetchedFg = ref->fetch(doc->getXRef());
        if (fetchedFg.isStream())
          fgStream = fetchedFg.getStream();
      }
      fgStream->reset();
      ImageStream fgImgStr(fgStream, width, fgNComps, fgBits);
      fgImgStr.reset();
      unsigned char *fgLine = fgImgStr.getLine();
      if (fgLine) {
        colorMap->getRGB(fgLine, &fgRgb);
      }
      fgImgStr.close();
    }

    unsigned char fgB = colToByte(fgRgb.r);
    unsigned char fgG = colToByte(fgRgb.g);
    unsigned char fgR = colToByte(fgRgb.b); // swap for BGR

    // --- Read the soft mask (alpha channel) --------------------
    // Re-fetch maskStr from XRef if possible to avoid consumed stream.
    // maskStr is separate from str; Poppler passes it directly from
    // the SMask object.  We still try a fresh fetch via ref lookup.
    maskStr->reset();
    int maskNComps = maskColorMap->getNumPixelComps();
    int maskBits = maskColorMap->getBits();
    ImageStream maskImgStr(maskStr, maskWidth, maskNComps, maskBits);
    maskImgStr.reset();

    // Build a BGR image by compositing foreground colour + mask over white
    cv::Mat mat(maskHeight, maskWidth, CV_8UC3);

    for (int row = 0; row < maskHeight; row++) {
      unsigned char *mLine = maskImgStr.getLine();
      if (!mLine)
        break;

      unsigned char *imgRow = mat.ptr<unsigned char>(row);
      for (int col = 0; col < maskWidth; col++) {
        // Get mask alpha (0 = transparent, 255 = fully opaque)
        GfxGray gray;
        maskColorMap->getGray(&mLine[col * maskNComps], &gray);
        unsigned char alpha = colToByte(gray);

        // Composite over white: out = fg * alpha + white * (1 - alpha)
        unsigned char invAlpha = 255 - alpha;
        imgRow[col * 3 + 0] =
            (unsigned char)((fgR * alpha + 255 * invAlpha) / 255);
        imgRow[col * 3 + 1] =
            (unsigned char)((fgG * alpha + 255 * invAlpha) / 255);
        imgRow[col * 3 + 2] =
            (unsigned char)((fgB * alpha + 255 * invAlpha) / 255);
      }
    }
    maskImgStr.close();

    // --- Position info from CTM --------------------------------
    const auto &ctm = state->getCTM();
    double displayWidth = std::sqrt(ctm[0] * ctm[0] + ctm[1] * ctm[1]);
    double displayHeight = std::sqrt(ctm[2] * ctm[2] + ctm[3] * ctm[3]);

    double x0 = ctm[4], y0 = ctm[5];
    double x1 = ctm[4] + ctm[0], y1 = ctm[5] + ctm[1];
    double x2 = ctm[4] + ctm[2], y2 = ctm[5] + ctm[3];
    double x3 = ctm[4] + ctm[0] + ctm[2], y3 = ctm[5] + ctm[1] + ctm[3];

    double x = std::min({x0, x1, x2, x3});
    double y = std::min({y0, y1, y2, y3});
    double rotationAngle = std::atan2(ctm[1], ctm[0]);

    OCRAnalysis::PDFEmbeddedImage img;
    img.image = mat;
    img.pageNumber = pageNumber;
    img.imageIndex = imageIndex++;
    img.width = maskWidth;
    img.height = maskHeight;
    img.x = x;
    img.y = y;
    img.displayWidth = displayWidth;
    img.displayHeight = displayHeight;
    img.rotationAngle = rotationAngle;
    img.type = "soft_masked";

    images.push_back(img);
  }

  // Handle images with a 1-bit (hard) mask.
  // Some logos and graphics use a 1-bit transparency mask instead of
  // a soft (8-bit) mask.  Without this override the image is silently
  // dropped by Poppler.
  void drawMaskedImage(GfxState *state, Object *ref, Stream *str, int width,
                       int height, GfxImageColorMap *colorMap,
                       bool /*interpolate*/, Stream *maskStr, int maskWidth,
                       int maskHeight, bool maskInvert,
                       bool /*maskInterpolate*/) override {

    std::cerr << "DEBUG: drawMaskedImage called: image " << width << "x"
              << height << ", mask " << maskWidth << "x" << maskHeight
              << ", invert=" << maskInvert << std::endl;

    if (width <= 0 || height <= 0 || !colorMap) {
      return;
    }

    // --- Read the base (colour) image ---------------------------
    int nComps = colorMap->getNumPixelComps();
    int nBits = colorMap->getBits();

    Stream *imgStream = str;
    Object fetchedObj;
    if (ref && ref->isRef() && doc) {
      fetchedObj = ref->fetch(doc->getXRef());
      if (fetchedObj.isStream()) {
        imgStream = fetchedObj.getStream();
      }
    }
    imgStream->reset();
    ImageStream imgStr(imgStream, width, nComps, nBits);
    imgStr.reset();

    // Read colour pixels into a BGR Mat
    cv::Mat colourMat(height, width, CV_8UC3);
    GfxRGB rgb;
    for (int row = 0; row < height; row++) {
      unsigned char *line = imgStr.getLine();
      if (!line)
        break;
      unsigned char *imgRow = colourMat.ptr<unsigned char>(row);
      for (int col = 0; col < width; col++) {
        colorMap->getRGB(&line[col * nComps], &rgb);
        imgRow[col * 3 + 0] = colToByte(rgb.b); // BGR
        imgRow[col * 3 + 1] = colToByte(rgb.g);
        imgRow[col * 3 + 2] = colToByte(rgb.r);
      }
    }
    imgStr.close();

    // --- Read the 1-bit mask ------------------------------------
    maskStr->reset();
    ImageStream maskImgStr(maskStr, maskWidth, 1, 1);
    maskImgStr.reset();

    cv::Mat maskMat(maskHeight, maskWidth, CV_8UC1);
    for (int row = 0; row < maskHeight; row++) {
      unsigned char *mLine = maskImgStr.getLine();
      if (!mLine)
        break;
      unsigned char *mRow = maskMat.ptr<unsigned char>(row);
      for (int col = 0; col < maskWidth; col++) {
        // mask bit: 1 = opaque (or transparent if inverted)
        unsigned char bit = mLine[col] ? 255 : 0;
        if (maskInvert)
          bit = 255 - bit;
        mRow[col] = bit;
      }
    }
    maskImgStr.close();

    // --- Composite over white -----------------------------------
    // Resize mask to match colour image if needed
    if (maskMat.rows != colourMat.rows || maskMat.cols != colourMat.cols) {
      cv::resize(maskMat, maskMat, colourMat.size(), 0, 0, cv::INTER_NEAREST);
    }

    cv::Mat result(colourMat.rows, colourMat.cols, CV_8UC3);
    for (int row = 0; row < colourMat.rows; row++) {
      const unsigned char *cRow = colourMat.ptr<unsigned char>(row);
      const unsigned char *mRow = maskMat.ptr<unsigned char>(row);
      unsigned char *oRow = result.ptr<unsigned char>(row);
      for (int col = 0; col < colourMat.cols; col++) {
        unsigned char alpha = mRow[col];
        unsigned char invAlpha = 255 - alpha;
        oRow[col * 3 + 0] =
            (unsigned char)((cRow[col * 3 + 0] * alpha + 255 * invAlpha) / 255);
        oRow[col * 3 + 1] =
            (unsigned char)((cRow[col * 3 + 1] * alpha + 255 * invAlpha) / 255);
        oRow[col * 3 + 2] =
            (unsigned char)((cRow[col * 3 + 2] * alpha + 255 * invAlpha) / 255);
      }
    }

    // --- Position info from CTM --------------------------------
    const auto &ctm = state->getCTM();
    double displayWidth = std::sqrt(ctm[0] * ctm[0] + ctm[1] * ctm[1]);
    double displayHeight = std::sqrt(ctm[2] * ctm[2] + ctm[3] * ctm[3]);

    double x0 = ctm[4], y0 = ctm[5];
    double x1 = ctm[4] + ctm[0], y1 = ctm[5] + ctm[1];
    double x2 = ctm[4] + ctm[2], y2 = ctm[5] + ctm[3];
    double x3 = ctm[4] + ctm[0] + ctm[2], y3 = ctm[5] + ctm[1] + ctm[3];

    double x = std::min({x0, x1, x2, x3});
    double y = std::min({y0, y1, y2, y3});
    double rotationAngle = std::atan2(ctm[1], ctm[0]);

    OCRAnalysis::PDFEmbeddedImage img;
    img.image = result;
    img.pageNumber = pageNumber;
    img.imageIndex = imageIndex++;
    img.width = colourMat.cols;
    img.height = colourMat.rows;
    img.x = x;
    img.y = y;
    img.displayWidth = displayWidth;
    img.displayHeight = displayHeight;
    img.rotationAngle = rotationAngle;
    img.type = "masked";

    std::cerr << "DEBUG: drawMaskedImage extracted " << img.width << "x"
              << img.height << " at (" << x << ", " << y << "), display "
              << displayWidth << "x" << displayHeight << std::endl;

    images.push_back(img);
  }

private:
  std::vector<OCRAnalysis::PDFEmbeddedImage> images;
  PDFDoc *doc;
  int pageNumber;
  int imageIndex;
};

} // anonymous namespace

OCRAnalysis::PDFEmbeddedImagesResult
OCRAnalysis::extractEmbeddedImagesFromPDF(const std::string &pdfPath) {
  PDFEmbeddedImagesResult result;
  result.success = false;

  auto startTime = std::chrono::high_resolution_clock::now();

  try {
    // Initialize Poppler's global parameters (required for low-level API)
    // GlobalParamsIniter is a RAII class that manages the lifecycle
    GlobalParamsIniter globalParamsInit(nullptr);

    // Load PDF using low-level Poppler API
    auto fileName = std::make_unique<GooString>(pdfPath);
    std::unique_ptr<PDFDoc> doc(new PDFDoc(std::move(fileName)));

    if (!doc->isOk()) {
      result.errorMessage = "Failed to load PDF file: " + pdfPath;
      return result;
    }

    int pageCount = doc->getNumPages();
    if (pageCount < 1) {
      result.errorMessage = "PDF has no pages";
      return result;
    }

    // Create our custom output device to capture images
    std::cerr << "DEBUG: Creating ImageExtractorOutputDev..." << std::endl;
    ImageExtractorOutputDev outputDev;
    outputDev.setDoc(doc.get());

    // Process only the first page (page 1)
    int pageIndex = 1;
    outputDev.setPageNumber(pageIndex);

    // Display (render) the page to our output device
    // This triggers drawImage callbacks for each image
    std::cerr << "DEBUG: Calling displayPage for embedded images..."
              << std::endl;
    doc->displayPage(&outputDev, pageIndex, 72.0,
                     72.0,   // DPI (not used since we capture raw)
                     0,      // rotation
                     true,   // useMediaBox
                     false,  // crop
                     false); // printing
    std::cerr << "DEBUG: displayPage completed for embedded images"
              << std::endl;

    // Get the extracted images
    result.images = std::move(outputDev.getImages());
    result.success = true;

  } catch (const std::exception &e) {
    result.errorMessage =
        std::string("PDF embedded image extraction failed: ") + e.what();
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  result.processingTimeMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();

  return result;
}

// Custom OutputDev to extract rectangles from PDF paths
namespace {

class RectangleExtractorOutputDev : public OutputDev {
public:
  RectangleExtractorOutputDev(double minSz) : pageNumber(1), minSize(minSz) {}

  std::vector<OCRAnalysis::PDFRectangle> &getRectangles() { return rectangles; }

  void setPageNumber(int page) { pageNumber = page; }

  // Required OutputDev overrides
  bool upsideDown() override { return false; }
  bool useDrawChar() override { return false; }
  bool interpretType3Chars() override { return false; }
  bool needNonText() override { return true; }

  // Called when a path is stroked (drawn with lines)
  void stroke(GfxState *state) override {
    extractRectangleFromPath(state, false, true);
  }

  // Called when a path is filled
  void fill(GfxState *state) override {
    extractRectangleFromPath(state, true, false);
  }

private:
  void extractRectangleFromPath(GfxState *state, bool isFilled,
                                bool isStroked) {
    const GfxPath *path = state->getPath();
    if (!path)
      return;

    // We're looking for closed rectangles: 4 corners + closed
    for (int i = 0; i < path->getNumSubpaths(); i++) {
      const GfxSubpath *subpath = path->getSubpath(i);

      // A rectangle has exactly 4 or 5 points (4 corners, possibly with
      // closing point)
      int numPoints = subpath->getNumPoints();
      if (numPoints < 4 || numPoints > 5)
        continue;

      // Must be closed to be a rectangle
      if (!subpath->isClosed() && numPoints != 5)
        continue;

      // Check that all points are line segments (not curves)
      bool hasCurves = false;
      for (int j = 0; j < numPoints; j++) {
        if (subpath->getCurve(j)) {
          hasCurves = true;
          break;
        }
      }
      if (hasCurves)
        continue;

      // Get the 4 corner points
      double x[4], y[4];
      for (int j = 0; j < 4; j++) {
        x[j] = subpath->getX(j);
        y[j] = subpath->getY(j);
      }

      // Transform points through CTM to get actual page coordinates
      const auto &ctm = state->getCTM();
      for (int j = 0; j < 4; j++) {
        double tx = ctm[0] * x[j] + ctm[2] * y[j] + ctm[4];
        double ty = ctm[1] * x[j] + ctm[3] * y[j] + ctm[5];
        x[j] = tx;
        y[j] = ty;
      }

      // Check if it's a rectangle: opposite sides parallel and equal
      // A rectangle has points where x or y values repeat
      if (!isRectangle(x, y))
        continue;

      // Calculate bounding box
      double minX = std::min({x[0], x[1], x[2], x[3]});
      double maxX = std::max({x[0], x[1], x[2], x[3]});
      double minY = std::min({y[0], y[1], y[2], y[3]});
      double maxY = std::max({y[0], y[1], y[2], y[3]});

      double width = maxX - minX;
      double height = maxY - minY;

      // Skip if too small
      if (width < minSize || height < minSize)
        continue;

      OCRAnalysis::PDFRectangle rect;
      rect.pageNumber = pageNumber;
      rect.x = minX;
      rect.y = minY;
      rect.width = width;
      rect.height = height;
      rect.lineWidth = state->getLineWidth();
      rect.filled = isFilled;
      rect.stroked = isStroked;

      rectangles.push_back(rect);
    }
  }

  bool isRectangle(double x[4], double y[4]) {
    // A rectangle should have 4 corners where:
    // - At least 2 unique X values
    // - At least 2 unique Y values
    // - Each corner connects at right angles

    const double tolerance = 0.5; // Half a point tolerance

    // Check if we have 2 distinct X values and 2 distinct Y values
    std::vector<double> xVals, yVals;
    for (int i = 0; i < 4; i++) {
      bool foundX = false, foundY = false;
      for (double xv : xVals) {
        if (std::abs(x[i] - xv) < tolerance) {
          foundX = true;
          break;
        }
      }
      for (double yv : yVals) {
        if (std::abs(y[i] - yv) < tolerance) {
          foundY = true;
          break;
        }
      }
      if (!foundX)
        xVals.push_back(x[i]);
      if (!foundY)
        yVals.push_back(y[i]);
    }

    // A proper rectangle has exactly 2 unique X values and 2 unique Y
    // values
    return xVals.size() == 2 && yVals.size() == 2;
  }

  std::vector<OCRAnalysis::PDFRectangle> rectangles;
  int pageNumber;
  double minSize;
};

} // anonymous namespace

OCRAnalysis::PDFRectanglesResult
OCRAnalysis::extractRectanglesFromPDF(const std::string &pdfPath,
                                      double minSize) {
  PDFRectanglesResult result;
  result.success = false;

  auto startTime = std::chrono::high_resolution_clock::now();

  try {
    // Initialize Poppler's global parameters
    GlobalParamsIniter globalParamsInit(nullptr);

    // Load PDF
    auto fileName = std::make_unique<GooString>(pdfPath);
    std::unique_ptr<PDFDoc> doc(new PDFDoc(std::move(fileName)));

    if (!doc->isOk()) {
      result.errorMessage = "Failed to load PDF file: " + pdfPath;
      return result;
    }

    int pageCount = doc->getNumPages();
    if (pageCount < 1) {
      result.errorMessage = "PDF has no pages";
      return result;
    }

    // Create our custom output device
    RectangleExtractorOutputDev outputDev(minSize);

    // Process only the first page (page 1)
    int pageIndex = 1;
    outputDev.setPageNumber(pageIndex);

    doc->displayPage(&outputDev, pageIndex, 72.0, 72.0, // DPI
                     0,                                 // rotation
                     true,                              // useMediaBox
                     false,                             // crop
                     false);                            // printing

    result.rectangles = std::move(outputDev.getRectangles());
    result.success = true;

  } catch (const std::exception &e) {
    result.errorMessage =
        std::string("PDF rectangle extraction failed: ") + e.what();
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  result.processingTimeMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();

  return result;
}

// Custom OutputDev to extract lines from PDF paths
namespace {

class LineExtractorOutputDev : public OutputDev {
public:
  LineExtractorOutputDev(double minLen) : pageNumber(1), minLength(minLen) {}

  std::vector<OCRAnalysis::PDFLine> &getLines() { return lines; }

  void setPageNumber(int page) { pageNumber = page; }

  // Required OutputDev overrides
  bool upsideDown() override { return false; }
  bool useDrawChar() override { return false; }
  bool interpretType3Chars() override { return false; }
  bool needNonText() override { return true; }

  // Called when a path is stroked (drawn with lines)
  void stroke(GfxState *state) override { extractLinesFromPath(state); }

private:
  void extractLinesFromPath(GfxState *state) {
    const GfxPath *path = state->getPath();
    if (!path)
      return;

    const auto &ctm = state->getCTM();
    double lineWidth = state->getLineWidth();

    // Process each subpath
    for (int i = 0; i < path->getNumSubpaths(); i++) {
      const GfxSubpath *subpath = path->getSubpath(i);
      int numPoints = subpath->getNumPoints();

      if (numPoints < 2)
        continue;

      // Extract line segments between consecutive points
      for (int j = 0; j < numPoints - 1; j++) {
        // Skip curve control points
        if (subpath->getCurve(j) || subpath->getCurve(j + 1))
          continue;

        // Get endpoints
        double x1 = subpath->getX(j);
        double y1 = subpath->getY(j);
        double x2 = subpath->getX(j + 1);
        double y2 = subpath->getY(j + 1);

        // Transform through CTM
        double tx1 = ctm[0] * x1 + ctm[2] * y1 + ctm[4];
        double ty1 = ctm[1] * x1 + ctm[3] * y1 + ctm[5];
        double tx2 = ctm[0] * x2 + ctm[2] * y2 + ctm[4];
        double ty2 = ctm[1] * x2 + ctm[3] * y2 + ctm[5];

        // Calculate length
        double dx = tx2 - tx1;
        double dy = ty2 - ty1;
        double length = std::sqrt(dx * dx + dy * dy);

        // Skip if too short
        if (length < minLength)
          continue;

        // Determine orientation
        const double angleTolerance = 5.0; // degrees
        double angle = std::atan2(std::abs(dy), std::abs(dx)) * 180.0 / M_PI;
        bool isHorizontal = angle < angleTolerance;
        bool isVertical = angle > (90.0 - angleTolerance);

        OCRAnalysis::PDFLine line;
        line.pageNumber = pageNumber;
        line.x1 = tx1;
        line.y1 = ty1;
        line.x2 = tx2;
        line.y2 = ty2;
        line.lineWidth = lineWidth;
        line.length = length;
        line.isHorizontal = isHorizontal;
        line.isVertical = isVertical;

        lines.push_back(line);
      }

      // If closed, also add the closing segment
      if (subpath->isClosed() && numPoints >= 2) {
        double x1 = subpath->getX(numPoints - 1);
        double y1 = subpath->getY(numPoints - 1);
        double x2 = subpath->getX(0);
        double y2 = subpath->getY(0);

        // Transform through CTM
        double tx1 = ctm[0] * x1 + ctm[2] * y1 + ctm[4];
        double ty1 = ctm[1] * x1 + ctm[3] * y1 + ctm[5];
        double tx2 = ctm[0] * x2 + ctm[2] * y2 + ctm[4];
        double ty2 = ctm[1] * x2 + ctm[3] * y2 + ctm[5];

        double dx = tx2 - tx1;
        double dy = ty2 - ty1;
        double length = std::sqrt(dx * dx + dy * dy);

        if (length >= minLength) {
          double angle = std::atan2(std::abs(dy), std::abs(dx)) * 180.0 / M_PI;

          OCRAnalysis::PDFLine line;
          line.pageNumber = pageNumber;
          line.x1 = tx1;
          line.y1 = ty1;
          line.x2 = tx2;
          line.y2 = ty2;
          line.lineWidth = lineWidth;
          line.length = length;
          line.isHorizontal = angle < 5.0;
          line.isVertical = angle > 85.0;

          lines.push_back(line);
        }
      }
    }
  }

  std::vector<OCRAnalysis::PDFLine> lines;
  int pageNumber;
  double minLength;
};

} // anonymous namespace

OCRAnalysis::PDFLinesResult
OCRAnalysis::extractLinesFromPDF(const std::string &pdfPath, double minLength) {
  PDFLinesResult result;
  result.success = false;

  auto startTime = std::chrono::high_resolution_clock::now();

  try {
    // Initialize Poppler's global parameters
    GlobalParamsIniter globalParamsInit(nullptr);

    // Load PDF
    auto fileName = std::make_unique<GooString>(pdfPath);
    std::unique_ptr<PDFDoc> doc(new PDFDoc(std::move(fileName)));

    if (!doc->isOk()) {
      result.errorMessage = "Failed to load PDF file: " + pdfPath;
      return result;
    }

    int pageCount = doc->getNumPages();
    if (pageCount < 1) {
      result.errorMessage = "PDF has no pages";
      return result;
    }

    // Create our custom output device
    LineExtractorOutputDev outputDev(minLength);

    // Process only the first page (page 1)
    int pageIndex = 1;
    outputDev.setPageNumber(pageIndex);

    doc->displayPage(&outputDev, pageIndex, 72.0, 72.0, // DPI
                     0,                                 // rotation
                     true,                              // useMediaBox
                     false,                             // crop
                     false);                            // printing

    result.lines = std::move(outputDev.getLines());

    // Calculate the largest box that is INSIDE the found lines
    // (i.e., the interior space bounded by the lines)
    if (!result.lines.empty()) {
      // Separate horizontal and vertical lines
      std::vector<double> horizontalYs;
      std::vector<double> verticalXs;

      for (const auto &line : result.lines) {
        if (line.isHorizontal) {
          horizontalYs.push_back((line.y1 + line.y2) / 2.0);
        } else if (line.isVertical) {
          verticalXs.push_back((line.x1 + line.x2) / 2.0);
        }
      }

      // Sort to find innermost boundaries
      if (!horizontalYs.empty() && !verticalXs.empty()) {
        std::sort(horizontalYs.begin(), horizontalYs.end());
        std::sort(verticalXs.begin(), verticalXs.end());

        // The interior box is bounded by the lines
        // Assume lines form a rectangle - interior is between innermost lines
        double leftX = verticalXs.front();
        double rightX = verticalXs.back();
        double topY = horizontalYs.front();
        double bottomY = horizontalYs.back();

        // If we have multiple lines, find the interior space
        // (between the innermost pair of lines on each side)
        if (verticalXs.size() >= 2) {
          // Interior left edge is after the leftmost vertical line
          // Interior right edge is before the rightmost vertical line
          leftX = verticalXs[0];
          rightX = verticalXs[verticalXs.size() - 1];
        }

        if (horizontalYs.size() >= 2) {
          // Interior top edge is after the topmost horizontal line
          // Interior bottom edge is before the bottommost horizontal line
          topY = horizontalYs[0];
          bottomY = horizontalYs[horizontalYs.size() - 1];
        }

        // Set interior box in result
        result.boundingBoxX = leftX;
        result.boundingBoxY = topY;
        result.boundingBoxWidth = rightX - leftX;
        result.boundingBoxHeight = bottomY - topY;
      }
    }

    result.success = true;

  } catch (const std::exception &e) {
    result.errorMessage =
        std::string("PDF line extraction failed: ") + e.what();
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  result.processingTimeMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();

  return result;
}

OCRAnalysis::PDFElements
OCRAnalysis::extractPDFElements(const std::string &pdfPath, double minRectSize,
                                double minLineLength,
                                const std::string &imageOutputDir) {
  // NOTE: This function processes ONLY the first page of the PDF.
  // All sub-functions (text, images, rectangles, lines) are hardcoded to
  // page 1.  Multi-page PDFs are accepted but only page 1 is ever read.

  PDFElements result;
  result.success = false;

  // Verify the PDF is loadable and has at least one page before doing any work.
  {
    std::unique_ptr<poppler::document> checkDoc(
        poppler::document::load_from_file(pdfPath));
    if (!checkDoc) {
      result.errorMessage = "Failed to open PDF: " + pdfPath;
      return result;
    }
    int totalPages = checkDoc->pages();
    if (totalPages < 1) {
      result.errorMessage = "PDF has no pages: " + pdfPath;
      return result;
    }
    if (totalPages > 1) {
      std::cerr << "DEBUG: PDF has " << totalPages
                << " pages â€” only page 1 will be processed." << std::endl;
    }
  }

  auto startTime = std::chrono::high_resolution_clock::now();

  try {
    // Extract text as individual words (preserves exact positioning) from first
    // page
    std::cerr << "DEBUG: Extracting text from first page..." << std::endl;
    try {
      OCRResult textResult =
          extractTextFromPDF(pdfPath, PDFExtractionLevel::Word);
      std::cerr << "DEBUG: Text extraction completed, success="
                << textResult.success << std::endl;
      if (textResult.success) {
        result.fullText = textResult.fullText;
        result.textLines = std::move(textResult.regions);
        result.textLineCount = static_cast<int>(result.textLines.size());
      } else {
        std::cerr << "DEBUG: Text extraction failed: "
                  << textResult.errorMessage << std::endl;
      }
    } catch (const std::exception &e) {
      std::cerr << "DEBUG: Text extraction threw exception: " << e.what()
                << std::endl;
    }

    // Enrich font names using Poppler's internal TextOutputDev, which provides
    // the actual embedded font name and accurate isBold/isItalic flags from
    // font descriptor bits.  The C++ text_list() API returns "*ignored*" for
    // most PDFs, so this is the only reliable source of font identity.
    if (!result.textLines.empty()) {
      try {
        GlobalParamsIniter gpi(nullptr);
        auto gooFile = std::make_unique<GooString>(pdfPath);
        std::unique_ptr<PDFDoc> fontDoc(new PDFDoc(std::move(gooFile)));
        if (fontDoc->isOk() && fontDoc->getNumPages() >= 1) {
          // Get page height so we can convert TextWord PDF y-up coords to
          // the screen y-down coords used by preciseX/Y.
          Page *pg1 = fontDoc->getPage(1);
          const ::PDFRectangle *mb = pg1->getMediaBox();
          // Get display height (after rotation) for y-flip.
          // For Rotate=90/270, page displays with width/height swapped.
          int pageRotate = pg1->getRotate();
          double dispH_font;
          if (pageRotate == 90 || pageRotate == 270)
            dispH_font = mb->x2 - mb->x1; // landscape: dispH = portrait width
          else
            dispH_font = mb->y2 - mb->y1;

          TextOutputDev textOut(nullptr, true, 0, false, false);
          fontDoc->displayPage(&textOut, 1, 72, 72, 0, false, true, false);
          TextPage *textPage = textOut.takeText();

          // Collect per-word font entries.
          // TextWord getBBox() returns PDF y-up coords (yMin=bottom, yMax=top).
          // text regions are stored in PDF bottom-left coords (y = bottom
          // edge).
          struct WordFontEntry {
            std::string fontName;
            bool isBold = false, isItalic = false;
            double xMin = 0, yBottom = 0, xMax = 0, yTop = 0;
          };
          std::vector<WordFontEntry> wordFonts;

          for (const TextFlow *flow = textPage->getFlows(); flow;
               flow = flow->getNext()) {
            for (const TextBlock *blk = flow->getBlocks(); blk;
                 blk = blk->getNext()) {
              for (const TextLine *ln = blk->getLines(); ln;
                   ln = ln->getNext()) {
                for (const TextWord *word = ln->getWords(); word;
                     word = word->getNext()) {
                  if (word->getLength() == 0)
                    continue;
                  const TextFontInfo *fi = word->getFontInfo(0);
                  if (!fi)
                    continue;
                  WordFontEntry e;
                  const GooString *fn = fi->getFontName();
                  if (fn) {
                    e.fontName = fn->toStr();
                    auto plusPos = e.fontName.find('+');
                    if (plusPos != std::string::npos)
                      e.fontName = e.fontName.substr(plusPos + 1);
                  }
                  e.isBold = fi->isBold();
                  e.isItalic = fi->isItalic();
                  // getBBox returns screen-Y-down coords; convert to PDF y-up
                  double gxMin, gyMinScr, gxMax, gyMaxScr;
                  word->getBBox(&gxMin, &gyMinScr, &gxMax, &gyMaxScr);
                  e.xMin = gxMin;
                  e.xMax = gxMax;
                  e.yBottom = dispH_font - gyMaxScr; // PDF y-up bottom
                  e.yTop = dispH_font - gyMinScr;    // PDF y-up top
                  wordFonts.push_back(e);
                }
              }
            }
          }
          textPage->decRefCnt();

          // Match each text region to the word with the greatest bbox overlap.
          // tr.preciseX/preciseY are now in PDF bottom-left coords (y =
          // bottom).
          for (auto &tr : result.textLines) {
            double trL = tr.preciseX;
            double trB = tr.preciseY; // bottom edge (y-up)
            double trR = trL + tr.preciseWidth;
            double trT = trB + tr.preciseHeight; // top edge (y-up)
            double bestOverlap = 0.0;
            const WordFontEntry *best = nullptr;
            for (const auto &wf : wordFonts) {
              double ol = std::max(trL, wf.xMin);
              double ob = std::max(trB, wf.yBottom);
              double orx = std::min(trR, wf.xMax);
              double ot = std::min(trT, wf.yTop);
              if (orx > ol && ot > ob) {
                double overlap = (orx - ol) * (ot - ob);
                if (overlap > bestOverlap) {
                  bestOverlap = overlap;
                  best = &wf;
                }
              }
            }
            if (best && !best->fontName.empty()) {
              tr.fontName = best->fontName;
              tr.isBold = best->isBold;
              tr.isItalic = best->isItalic;
            }
          }
        }
      } catch (const std::exception &e) {
        std::cerr << "DEBUG: Font enrichment failed: " << e.what() << std::endl;
      }
    }

    // Move symbol-font text lines (Wingdings family) to ignoredTextLines so
    // they are excluded from rendering but remain available for diagnostics.
    {
      auto isWingdings = [](const std::string &name) {
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        return lower.find("wingdings") != std::string::npos;
      };
      std::vector<TextRegion> kept, ignored;
      for (auto &tr : result.textLines) {
        if (isWingdings(tr.fontName))
          ignored.push_back(std::move(tr));
        else
          kept.push_back(std::move(tr));
      }
      result.ignoredTextLines = std::move(ignored);
      result.textLines = std::move(kept);
      result.textLineCount = static_cast<int>(result.textLines.size());
    }

    // Extract embedded images from first page
    std::cerr << "DEBUG: Extracting embedded images from first page..."
              << std::endl;
    try {
      PDFEmbeddedImagesResult imageResult =
          extractEmbeddedImagesFromPDF(pdfPath);
      std::cerr << "DEBUG: Image extraction completed" << std::endl;
      if (imageResult.success) {
        result.images = std::move(imageResult.images);
        result.imageCount = static_cast<int>(result.images.size());
      }
    } catch (const std::exception &e) {
      std::cerr << "DEBUG: Image extraction threw exception: " << e.what()
                << std::endl;
    }

    // Scan for DataMatrix barcodes
#ifdef HAVE_ZXING
    std::cerr << "DEBUG: Scanning for DataMatrix barcodes..." << std::endl;
    try {
      ZXing::ReaderOptions opts;
      opts.setFormats(ZXing::BarcodeFormat::DataMatrix);
      opts.setTryHarder(true);
      opts.setTryRotate(true);

      // Strategy 1: Scan each embedded image
      for (size_t imgIdx = 0; imgIdx < result.images.size(); imgIdx++) {
        const auto &pdfImage = result.images[imgIdx];
        if (pdfImage.image.empty())
          continue;

        cv::Mat scanImg;
        if (pdfImage.image.channels() == 4) {
          cv::cvtColor(pdfImage.image, scanImg, cv::COLOR_BGRA2BGR);
        } else {
          scanImg = pdfImage.image;
        }

        auto fmt = ZXing::ImageFormat::None;
        switch (scanImg.channels()) {
        case 1:
          fmt = ZXing::ImageFormat::Lum;
          break;
        case 3:
          fmt = ZXing::ImageFormat::BGR;
          break;
        default:
          continue;
        }

        ZXing::ImageView iv(scanImg.data, scanImg.cols, scanImg.rows, fmt);
        auto barcodes = ZXing::ReadBarcodes(iv, opts);

        double scaleX = pdfImage.displayWidth / pdfImage.image.cols;
        double scaleY = pdfImage.displayHeight / pdfImage.image.rows;

        for (const auto &bc : barcodes) {
          auto pos = bc.position();
          int pxMinX =
              std::max(0, std::min({pos[0].x, pos[1].x, pos[2].x, pos[3].x}));
          int pxMinY =
              std::max(0, std::min({pos[0].y, pos[1].y, pos[2].y, pos[3].y}));
          int pxMaxX = std::min(
              scanImg.cols, std::max({pos[0].x, pos[1].x, pos[2].x, pos[3].x}));
          int pxMaxY = std::min(
              scanImg.rows, std::max({pos[0].y, pos[1].y, pos[2].y, pos[3].y}));

          PDFDataMatrix dm;
          dm.text = bc.text();
          dm.x = pdfImage.x + pxMinX * scaleX;
          dm.y = pdfImage.y + pxMinY * scaleY;
          dm.width = (pxMaxX - pxMinX) * scaleX;
          dm.height = (pxMaxY - pxMinY) * scaleY;
          dm.sourceImageIndex = static_cast<int>(imgIdx);

          int cropW = pxMaxX - pxMinX;
          int cropH = pxMaxY - pxMinY;
          if (cropW > 0 && cropH > 0) {
            dm.image =
                pdfImage.image(cv::Rect(pxMinX, pxMinY, cropW, cropH)).clone();
          }

          std::cerr << "DEBUG: DataMatrix in image " << imgIdx << ": \""
                    << dm.text.substr(0, 30) << "\" at PDF (" << dm.x << ", "
                    << dm.y << ") size " << dm.width << "x" << dm.height
                    << std::endl;
          result.dataMatrices.push_back(std::move(dm));
        }
      }

      // Strategy 2: Rasterise the full page and scan for vector-drawn
      // DataMatrix codes that won't appear as embedded images
      std::cerr << "DEBUG: Rasterising page for vector DataMatrix detection..."
                << std::endl;
      try {
        std::unique_ptr<poppler::document> doc(
            poppler::document::load_from_file(pdfPath));
        if (doc && doc->pages() > 0) {
          std::unique_ptr<poppler::page> page(doc->create_page(0));
          if (page) {
            poppler::page_renderer renderer;
            renderer.set_render_hint(poppler::page_renderer::antialiasing,
                                     true);
            renderer.set_render_hint(poppler::page_renderer::text_antialiasing,
                                     true);
            renderer.set_image_format(poppler::image::format_argb32);

            // Render at 600 DPI for reliable barcode detection
            const double scanDpi = 600.0;
            poppler::image popplerImg =
                renderer.render_page(page.get(), scanDpi, scanDpi);

            if (popplerImg.is_valid()) {
              int w = popplerImg.width();
              int h = popplerImg.height();
              cv::Mat pageMat(h, w, CV_8UC4,
                              const_cast<char *>(popplerImg.const_data()),
                              popplerImg.bytes_per_row());
              cv::Mat pageBGR;
              cv::cvtColor(pageMat, pageBGR, cv::COLOR_BGRA2BGR);

              ZXing::ImageView pageIV(pageBGR.data, pageBGR.cols, pageBGR.rows,
                                      ZXing::ImageFormat::BGR);
              auto pageBarcodes = ZXing::ReadBarcodes(pageIV, opts);

              // Scale from raster pixels to PDF points
              poppler::rectf pageRect = page->page_rect();
              double pixToPtX = pageRect.width() / w;
              double pixToPtY = pageRect.height() / h;

              for (const auto &bc : pageBarcodes) {
                auto pos = bc.position();
                int pxMinX = std::max(
                    0, std::min({pos[0].x, pos[1].x, pos[2].x, pos[3].x}));
                int pxMinY = std::max(
                    0, std::min({pos[0].y, pos[1].y, pos[2].y, pos[3].y}));
                int pxMaxX = std::min(
                    w, std::max({pos[0].x, pos[1].x, pos[2].x, pos[3].x}));
                int pxMaxY = std::min(
                    h, std::max({pos[0].y, pos[1].y, pos[2].y, pos[3].y}));

                double pdfX = pageRect.x() + pxMinX * pixToPtX;
                double pdfY = pageRect.y() + pxMinY * pixToPtY;
                double pdfW = (pxMaxX - pxMinX) * pixToPtX;
                double pdfH = (pxMaxY - pxMinY) * pixToPtY;

                // Check if this barcode was already found in an embedded image
                // (dedup by position overlap)
                bool isDuplicate = false;
                for (const auto &existing : result.dataMatrices) {
                  double overlapX = std::max(
                      0.0, std::min(existing.x + existing.width, pdfX + pdfW) -
                               std::max(existing.x, pdfX));
                  double overlapY = std::max(
                      0.0, std::min(existing.y + existing.height, pdfY + pdfH) -
                               std::max(existing.y, pdfY));
                  double overlapArea = overlapX * overlapY;
                  double existingArea = existing.width * existing.height;
                  if (existingArea > 0 && overlapArea / existingArea > 0.3) {
                    isDuplicate = true;
                    break;
                  }
                }

                if (!isDuplicate) {
                  PDFDataMatrix dm;
                  dm.text = bc.text();
                  dm.x = pdfX;
                  dm.y = pdfY;
                  dm.width = pdfW;
                  dm.height = pdfH;
                  dm.sourceImageIndex = -1; // from rasterised page

                  int cropW = pxMaxX - pxMinX;
                  int cropH = pxMaxY - pxMinY;
                  if (cropW > 0 && cropH > 0) {
                    dm.image =
                        pageBGR(cv::Rect(pxMinX, pxMinY, cropW, cropH)).clone();
                  }

                  std::cerr << "DEBUG: DataMatrix in rasterised page: \""
                            << dm.text.substr(0, 30) << "\" at PDF (" << dm.x
                            << ", " << dm.y << ") size " << dm.width << "x"
                            << dm.height << std::endl;
                  result.dataMatrices.push_back(std::move(dm));
                }
              }
            }
          }
        }
      } catch (const std::exception &e) {
        std::cerr << "DEBUG: Page rasterisation for DataMatrix failed: "
                  << e.what() << std::endl;
      }

      result.dataMatrixCount = static_cast<int>(result.dataMatrices.size());
      std::cerr << "DEBUG: Total DataMatrix barcodes found: "
                << result.dataMatrixCount << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "DEBUG: DataMatrix scanning threw exception: " << e.what()
                << std::endl;
    }
#endif // HAVE_ZXING

    // Save images and DataMatrix barcodes to PNG files if output dir specified
    if (!imageOutputDir.empty()) {
      try {
        std::filesystem::path outDir(imageOutputDir);
        std::filesystem::create_directories(outDir);

        // Get PDF stem for naming files
        std::filesystem::path pdfFilePath(pdfPath);
        std::string pdfStem = pdfFilePath.stem().string();

        // Save embedded images
        for (size_t i = 0; i < result.images.size(); i++) {
          const auto &img = result.images[i];
          if (!img.image.empty()) {
            std::string filename = (outDir / (pdfStem + "_image_" +
                                              std::to_string(i + 1) + ".png"))
                                       .string();
            if (cv::imwrite(filename, img.image)) {
              std::cerr << "DEBUG: Saved image " << (i + 1) << " ("
                        << img.image.cols << "x" << img.image.rows
                        << ") to: " << filename << std::endl;
            } else {
              std::cerr << "DEBUG: Failed to save image " << (i + 1)
                        << " to: " << filename << std::endl;
            }
          }
        }

        // Save DataMatrix barcode images
        for (size_t i = 0; i < result.dataMatrices.size(); i++) {
          const auto &dm = result.dataMatrices[i];
          if (!dm.image.empty()) {
            std::string filename = (outDir / (pdfStem + "_datamatrix_" +
                                              std::to_string(i + 1) + ".png"))
                                       .string();
            if (cv::imwrite(filename, dm.image)) {
              std::cerr << "DEBUG: Saved DataMatrix " << (i + 1) << " (\""
                        << dm.text.substr(0, 30) << "\") to: " << filename
                        << std::endl;
            } else {
              std::cerr << "DEBUG: Failed to save DataMatrix " << (i + 1)
                        << " to: " << filename << std::endl;
            }
          }
        }
      } catch (const std::exception &e) {
        std::cerr << "DEBUG: Failed to save images: " << e.what() << std::endl;
      }
    }

    // Extract rectangles from first page
    std::cerr << "DEBUG: Extracting rectangles from first page..." << std::endl;
    try {
      PDFRectanglesResult rectResult =
          extractRectanglesFromPDF(pdfPath, minRectSize);
      std::cerr << "DEBUG: Rectangle extraction completed" << std::endl;
      if (rectResult.success) {
        result.rectangles = std::move(rectResult.rectangles);
        result.rectangleCount = static_cast<int>(result.rectangles.size());
      }
    } catch (const std::exception &e) {
      std::cerr << "DEBUG: Rectangle extraction threw exception: " << e.what()
                << std::endl;
    }

    // Extract drawn lines (vector graphics) from first page
    std::cerr << "DEBUG: Extracting lines from first page..." << std::endl;
    try {
      PDFLinesResult lineResult = extractLinesFromPDF(pdfPath, minLineLength);
      std::cerr << "DEBUG: Line extraction completed" << std::endl;
      if (lineResult.success) {
        // First, detect rectangles formed by 4 lines
        // Group horizontal and vertical lines
        std::vector<PDFLine> horizontalLines, verticalLines;
        const double parallelTolerance = 2.0;

        for (const auto &line : lineResult.lines) {
          if (line.isHorizontal) {
            horizontalLines.push_back(line);
          } else if (line.isVertical) {
            verticalLines.push_back(line);
          }
        }

        std::cerr << "DEBUG: Found " << horizontalLines.size()
                  << " horizontal lines, " << verticalLines.size()
                  << " vertical lines" << std::endl;

        // Simple case: if we have exactly 2 horizontal and 2 vertical lines,
        // they likely form a rectangle
        if (horizontalLines.size() == 2 && verticalLines.size() == 2) {
          const auto &h1 = horizontalLines[0];
          const auto &h2 = horizontalLines[1];
          const auto &v1 = verticalLines[0];
          const auto &v2 = verticalLines[1];

          if (h1.pageNumber == h2.pageNumber &&
              h1.pageNumber == v1.pageNumber &&
              h1.pageNumber == v2.pageNumber) {
            double y1 = (h1.y1 + h1.y2) / 2.0;
            double y2 = (h2.y1 + h2.y2) / 2.0;
            double x1 = (v1.x1 + v1.x2) / 2.0;
            double x2 = (v2.x1 + v2.x2) / 2.0;

            double minX = std::min(x1, x2);
            double maxX = std::max(x1, x2);
            double minY = std::min(y1, y2);
            double maxY = std::max(y1, y2);

            PDFRectangle rect;
            rect.pageNumber = h1.pageNumber;
            rect.x = minX;
            rect.y = minY;
            rect.width = maxX - minX;
            rect.height = maxY - minY;
            rect.filled = false;
            rect.stroked = true;
            rect.lineWidth = 1.0;

            result.rectangles.push_back(rect);
            std::cerr << "DEBUG: Detected rectangle from 4 lines at (" << rect.x
                      << ", " << rect.y << ") size: " << rect.width << "x"
                      << rect.height << std::endl;
          }
        }

        // Try to find rectangles formed by 2 horizontal + 2 vertical lines
        for (size_t h1 = 0; h1 < horizontalLines.size(); h1++) {
          for (size_t h2 = h1 + 1; h2 < horizontalLines.size(); h2++) {
            const auto &hLine1 = horizontalLines[h1];
            const auto &hLine2 = horizontalLines[h2];

            // Check if they're on the same page and parallel
            if (hLine1.pageNumber != hLine2.pageNumber)
              continue;

            double y1 = (hLine1.y1 + hLine1.y2) / 2.0;
            double y2 = (hLine2.y1 + hLine2.y2) / 2.0;
            if (std::abs(y1 - y2) < 10.0)
              continue; // Too close, not a rectangle

            // Find vertical lines that connect these horizontals
            for (size_t v1 = 0; v1 < verticalLines.size(); v1++) {
              for (size_t v2 = v1 + 1; v2 < verticalLines.size(); v2++) {
                const auto &vLine1 = verticalLines[v1];
                const auto &vLine2 = verticalLines[v2];

                if (vLine1.pageNumber != hLine1.pageNumber)
                  continue;

                double x1 = (vLine1.x1 + vLine1.x2) / 2.0;
                double x2 = (vLine2.x1 + vLine2.x2) / 2.0;
                if (std::abs(x1 - x2) < 10.0)
                  continue; // Too close

                // Check if these 4 lines form a rectangle
                double minX = std::min(x1, x2);
                double maxX = std::max(x1, x2);
                double minY = std::min(y1, y2);
                double maxY = std::max(y1, y2);

                // Check if horizontal lines span the width
                double h1MinX = std::min(hLine1.x1, hLine1.x2);
                double h1MaxX = std::max(hLine1.x1, hLine1.x2);
                double h2MinX = std::min(hLine2.x1, hLine2.x2);
                double h2MaxX = std::max(hLine2.x1, hLine2.x2);

                // Check if vertical lines span the height
                double v1MinY = std::min(vLine1.y1, vLine1.y2);
                double v1MaxY = std::max(vLine1.y1, vLine1.y2);
                double v2MinY = std::min(vLine2.y1, vLine2.y2);
                double v2MaxY = std::max(vLine2.y1, vLine2.y2);

                const double spanTolerance = 10.0;
                bool h1Spans = (h1MinX <= minX + spanTolerance &&
                                h1MaxX >= maxX - spanTolerance);
                bool h2Spans = (h2MinX <= minX + spanTolerance &&
                                h2MaxX >= maxX - spanTolerance);
                bool v1Spans = (v1MinY <= minY + spanTolerance &&
                                v1MaxY >= maxY - spanTolerance);
                bool v2Spans = (v2MinY <= minY + spanTolerance &&
                                v2MaxY >= maxY - spanTolerance);

                if (h1Spans && h2Spans && v1Spans && v2Spans) {
                  // Found a rectangle!
                  PDFRectangle rect;
                  rect.pageNumber = hLine1.pageNumber;
                  rect.x = minX;
                  rect.y = minY;
                  rect.width = maxX - minX;
                  rect.height = maxY - minY;
                  rect.filled = false;
                  rect.stroked = true;
                  rect.lineWidth = 1.0;

                  // Check if we already have this rectangle (avoid duplicates)
                  bool isDuplicate = false;
                  for (const auto &existing : result.rectangles) {
                    if (existing.pageNumber == rect.pageNumber &&
                        std::abs(existing.x - rect.x) < 5.0 &&
                        std::abs(existing.y - rect.y) < 5.0 &&
                        std::abs(existing.width - rect.width) < 5.0 &&
                        std::abs(existing.height - rect.height) < 5.0) {
                      isDuplicate = true;
                      break;
                    }
                  }

                  if (!isDuplicate) {
                    result.rectangles.push_back(rect);
                    std::cerr << "DEBUG: Detected rectangle from lines at ("
                              << rect.x << ", " << rect.y
                              << ") size: " << rect.width << "x" << rect.height
                              << std::endl;
                  }
                }
              }
            }
          }
        }

        // Filter out lines that are part of rectangles
        std::vector<PDFLine> filteredLines;
        const double tolerance =
            2.0; // Tolerance for matching line to rectangle edge

        for (const auto &line : lineResult.lines) {
          bool isPartOfRectangle = false;

          // Check if this line is part of any rectangle
          for (const auto &rect : result.rectangles) {
            // Only check rectangles on the same page
            if (rect.pageNumber != line.pageNumber) {
              continue;
            }

            // Skip small rectangles (potential crop marks)
            const double cropMarkMaxSize = 30.0;
            if (rect.width <= cropMarkMaxSize &&
                rect.height <= cropMarkMaxSize) {
              continue;
            }

            // Calculate rectangle edges
            double rectLeft = rect.x;
            double rectRight = rect.x + rect.width;
            double rectTop = rect.y;
            double rectBottom = rect.y + rect.height;

            // Check if line is horizontal and matches top or bottom edge
            if (line.isHorizontal) {
              double lineY = (line.y1 + line.y2) / 2.0;
              double lineMinX = std::min(line.x1, line.x2);
              double lineMaxX = std::max(line.x1, line.x2);

              // Check if line aligns with top or bottom edge
              if ((std::abs(lineY - rectTop) < tolerance ||
                   std::abs(lineY - rectBottom) < tolerance) &&
                  lineMinX >= rectLeft - tolerance &&
                  lineMaxX <= rectRight + tolerance) {
                isPartOfRectangle = true;
                break;
              }
            }

            // Check if line is vertical and matches left or right edge
            if (line.isVertical) {
              double lineX = (line.x1 + line.x2) / 2.0;
              double lineMinY = std::min(line.y1, line.y2);
              double lineMaxY = std::max(line.y1, line.y2);

              // Check if line aligns with left or right edge
              if ((std::abs(lineX - rectLeft) < tolerance ||
                   std::abs(lineX - rectRight) < tolerance) &&
                  lineMinY >= rectTop - tolerance &&
                  lineMaxY <= rectBottom + tolerance) {
                isPartOfRectangle = true;
                break;
              }
            }
          }

          // Only keep lines that are not part of rectangles
          if (!isPartOfRectangle) {
            filteredLines.push_back(line);
          }
        }

        std::cerr << "DEBUG: Filtered "
                  << (lineResult.lines.size() - filteredLines.size())
                  << " lines that are part of rectangles" << std::endl;

        result.graphicLines = std::move(filteredLines);
        result.graphicLineCount = static_cast<int>(result.graphicLines.size());

        // Detect crop marks from perpendicular line intersections
        // (only if we don't already have a linesBoundingBox from TrimBox)
        if (result.linesBoundingBoxWidth > 0 &&
            result.linesBoundingBoxHeight > 0) {
          std::cerr << "DEBUG: Skipping crop mark detection â€” using TrimBox"
                    << std::endl;
        } else {

          std::vector<std::pair<double, double>> cropMarkCorners;

          // Find short horizontal and vertical lines
          std::vector<PDFLine> horizontalCropLines;
          std::vector<PDFLine> verticalCropLines;

          const double cropMarkMaxLength = 30.0;
          const double cropMarkMinLength = 10.0;

          for (const auto &line : lineResult.lines) {
            double length = std::sqrt(std::pow(line.x2 - line.x1, 2) +
                                      std::pow(line.y2 - line.y1, 2));

            if (length >= cropMarkMinLength && length <= cropMarkMaxLength) {
              if (line.isHorizontal) {
                horizontalCropLines.push_back(line);
              } else if (line.isVertical) {
                verticalCropLines.push_back(line);
              }
            }
          }

          // Filter out bleed mark lines: bleed marks are rows of small
          // coloured boxes that all sit on the same horizontal line.
          // Strategy:
          //   1. Group horizontal lines by Y coordinate. Any Y with >4
          //      horizontal lines is a bleed-mark row.
          //   2. Determine the Y-band(s) of those rows.
          //   3. Remove ALL horizontal lines in those Y groups.
          //   4. Remove ALL vertical lines whose endpoints fall within
          //      a bleed-mark Y-band â€” these are the box edges.
          {
            const double gridTolerance = 3.0;

            // Step 1: Group horizontal lines by Y coordinate
            std::map<double, std::vector<int>> hLinesByY;
            for (size_t i = 0; i < horizontalCropLines.size(); i++) {
              double y =
                  (horizontalCropLines[i].y1 + horizontalCropLines[i].y2) / 2.0;
              bool found = false;
              for (auto &[groupY, indices] : hLinesByY) {
                if (std::abs(y - groupY) < gridTolerance) {
                  indices.push_back(static_cast<int>(i));
                  found = true;
                  break;
                }
              }
              if (!found) {
                hLinesByY[y].push_back(static_cast<int>(i));
              }
            }

            // Step 2: Identify bleed-mark Y-bands (groups with >4 lines)
            const int maxGroupSize = 4;
            std::set<int> hLinesToRemove;
            // Collect the Y-band ranges from bleed-mark rows
            struct YBand {
              double minY, maxY;
              double minX = 0, maxX = 0; // interior X range of bleed marks
            };
            std::vector<YBand> bleedBands;

            for (const auto &[y, indices] : hLinesByY) {
              if (static_cast<int>(indices.size()) > maxGroupSize) {
                // This Y coordinate has many horizontal lines â€” it's a
                // bleed-mark row.  However, crop mark lines may also
                // exist at this same Y (at the extreme left/right edge
                // of the content area).  To distinguish:
                //  - Find the overall X extent of ALL lines in the group
                //  - Lines that start/end near the overall left or right
                //    edge are crop marks; lines in the interior are bleed
                //    mark box edges.

                // Find overall X extent
                double groupMinX = std::numeric_limits<double>::max();
                double groupMaxX = std::numeric_limits<double>::lowest();
                for (int idx : indices) {
                  const auto &line = horizontalCropLines[idx];
                  groupMinX = std::min(groupMinX, std::min(line.x1, line.x2));
                  groupMaxX = std::max(groupMaxX, std::max(line.x1, line.x2));
                }

                // The crop marks will be at the extreme edges.
                // Define interior zone: lines that don't touch the
                // leftmost or rightmost cropMarkMaxLength are bleed marks.
                double leftEdge = groupMinX + cropMarkMaxLength;
                double rightEdge = groupMaxX - cropMarkMaxLength;

                for (int idx : indices) {
                  const auto &line = horizontalCropLines[idx];
                  double lineMinX = std::min(line.x1, line.x2);
                  double lineMaxX = std::max(line.x1, line.x2);

                  // A line is a crop mark if it touches the left or
                  // right edge of the group
                  bool isAtLeftEdge = lineMinX < leftEdge;
                  bool isAtRightEdge = lineMaxX > rightEdge;

                  if (!isAtLeftEdge && !isAtRightEdge) {
                    // Interior line â€” bleed mark, remove it
                    hLinesToRemove.insert(idx);
                  }
                }

                // Calculate the Y-band for vertical line filtering
                double bandMinY = std::numeric_limits<double>::max();
                double bandMaxY = std::numeric_limits<double>::lowest();
                for (int idx : indices) {
                  const auto &line = horizontalCropLines[idx];
                  bandMinY = std::min(bandMinY, std::min(line.y1, line.y2));
                  bandMaxY = std::max(bandMaxY, std::max(line.y1, line.y2));
                }
                // Expand band slightly to catch box edges
                bleedBands.push_back(
                    {bandMinY - gridTolerance, bandMaxY + gridTolerance});

                // Also compute the bleed mark interior X range so we
                // only remove vertical lines in the interior too
                bleedBands.back().minX = leftEdge;
                bleedBands.back().maxX = rightEdge;

                std::cerr << "DEBUG: Bleed mark Y-band: " << bandMinY << " to "
                          << bandMaxY << ", X interior: " << leftEdge << " to "
                          << rightEdge << " (" << indices.size()
                          << " horizontal lines, " << hLinesToRemove.size()
                          << " marked)" << std::endl;
              }
            }

            // Merge nearby Y-bands (top and bottom edges of the same
            // row of boxes will produce separate bands with a gap
            // equal to the box height; merge if gap < cropMarkMaxLength)
            if (bleedBands.size() > 1) {
              std::sort(bleedBands.begin(), bleedBands.end(),
                        [](const YBand &a, const YBand &b) {
                          return a.minY < b.minY;
                        });
              std::vector<YBand> merged;
              merged.push_back(bleedBands[0]);
              for (size_t i = 1; i < bleedBands.size(); i++) {
                // Merge if gap between bands is less than box height
                if (bleedBands[i].minY <=
                    merged.back().maxY + cropMarkMaxLength) {
                  merged.back().maxY =
                      std::max(merged.back().maxY, bleedBands[i].maxY);
                  // Widen X range to union of both bands
                  merged.back().minX =
                      std::min(merged.back().minX, bleedBands[i].minX);
                  merged.back().maxX =
                      std::max(merged.back().maxX, bleedBands[i].maxX);
                } else {
                  merged.push_back(bleedBands[i]);
                }
              }
              bleedBands = std::move(merged);
            }

            for (const auto &band : bleedBands) {
              std::cerr << "DEBUG: Merged bleed Y-band: Y=" << band.minY
                        << " to " << band.maxY << ", X interior=" << band.minX
                        << " to " << band.maxX << std::endl;
            }

            // Step 3: Remove vertical lines within bleed-mark Y-bands
            std::set<int> vLinesToRemove;
            for (size_t i = 0; i < verticalCropLines.size(); i++) {
              double vMinY =
                  std::min(verticalCropLines[i].y1, verticalCropLines[i].y2);
              double vMaxY =
                  std::max(verticalCropLines[i].y1, verticalCropLines[i].y2);
              double vX =
                  (verticalCropLines[i].x1 + verticalCropLines[i].x2) / 2.0;
              for (const auto &band : bleedBands) {
                // Vertical line is part of bleed mark if:
                //  - Its Y range is within the bleed band
                //  - Its X position is in the bleed mark interior
                //    (not at the extreme edges where crop marks are)
                if (vMinY >= band.minY && vMaxY <= band.maxY &&
                    vX >= band.minX && vX <= band.maxX) {
                  vLinesToRemove.insert(static_cast<int>(i));
                  break;
                }
              }
            }

            // Step 4: Apply filtering
            if (!hLinesToRemove.empty() || !vLinesToRemove.empty()) {
              std::vector<PDFLine> filteredH, filteredV;
              for (size_t i = 0; i < horizontalCropLines.size(); i++) {
                if (hLinesToRemove.find(static_cast<int>(i)) ==
                    hLinesToRemove.end()) {
                  filteredH.push_back(horizontalCropLines[i]);
                }
              }
              for (size_t i = 0; i < verticalCropLines.size(); i++) {
                if (vLinesToRemove.find(static_cast<int>(i)) ==
                    vLinesToRemove.end()) {
                  filteredV.push_back(verticalCropLines[i]);
                }
              }

              std::cerr << "DEBUG: Filtered " << hLinesToRemove.size()
                        << " horizontal and " << vLinesToRemove.size()
                        << " vertical bleed mark lines" << std::endl;

              horizontalCropLines = std::move(filteredH);
              verticalCropLines = std::move(filteredV);
            }
          }

          std::cerr << "DEBUG: Found " << horizontalCropLines.size()
                    << " horizontal crop mark lines, "
                    << verticalCropLines.size() << " vertical" << std::endl;

          // Find intersection points
          const double intersectionTolerance = 5.0;

          for (const auto &hLine : horizontalCropLines) {
            double hY = (hLine.y1 + hLine.y2) / 2.0;

            for (const auto &vLine : verticalCropLines) {
              double vX = (vLine.x1 + vLine.x2) / 2.0;

              double hMinX = std::min(hLine.x1, hLine.x2);
              double hMaxX = std::max(hLine.x1, hLine.x2);
              double vMinY = std::min(vLine.y1, vLine.y2);
              double vMaxY = std::max(vLine.y1, vLine.y2);

              bool hLineNearVLine = (vX >= hMinX - intersectionTolerance &&
                                     vX <= hMaxX + intersectionTolerance);
              bool vLineNearHLine = (hY >= vMinY - intersectionTolerance &&
                                     hY <= vMaxY + intersectionTolerance);

              if (hLineNearVLine && vLineNearHLine) {
                cropMarkCorners.push_back({vX, hY});
              }
            }
          }

          std::cerr << "DEBUG: Found " << cropMarkCorners.size()
                    << " crop mark intersection points" << std::endl;

          // Calculate interior box from crop mark corners
          if (cropMarkCorners.size() >= 4) {
            // Cluster nearby corners together (many duplicates from multiple
            // line widths)
            const double clusterTolerance = 5.0;
            std::vector<std::pair<double, double>> uniqueCorners;

            for (const auto &corner : cropMarkCorners) {
              bool foundCluster = false;
              for (auto &unique : uniqueCorners) {
                double dx = corner.first - unique.first;
                double dy = corner.second - unique.second;
                double dist = std::sqrt(dx * dx + dy * dy);

                if (dist < clusterTolerance) {
                  unique.first = (unique.first + corner.first) / 2.0;
                  unique.second = (unique.second + corner.second) / 2.0;
                  foundCluster = true;
                  break;
                }
              }

              if (!foundCluster) {
                uniqueCorners.push_back(corner);
              }
            }

            std::cerr << "DEBUG: Clustered " << cropMarkCorners.size()
                      << " corners to " << uniqueCorners.size()
                      << " unique corners" << std::endl;

            // Find the 4 crop mark corners by grouping by X and Y coordinates
            // The 4 actual crop marks will share 2 X values and 2 Y values
            std::map<double, int> xCounts;
            std::map<double, int> yCounts;

            const double coordTolerance = 10.0;

            // Count corners at each X and Y coordinate
            for (const auto &corner : uniqueCorners) {
              // Find or increment X count
              bool foundX = false;
              for (auto &[x, count] : xCounts) {
                if (std::abs(corner.first - x) < coordTolerance) {
                  count++;
                  foundX = true;
                  break;
                }
              }
              if (!foundX) {
                xCounts[corner.first] = 1;
              }

              // Find or increment Y count
              bool foundY = false;
              for (auto &[y, count] : yCounts) {
                if (std::abs(corner.second - y) < coordTolerance) {
                  count++;
                  foundY = true;
                  break;
                }
              }
              if (!foundY) {
                yCounts[corner.second] = 1;
              }
            }

            // Find the 2 X values with most corners
            std::vector<std::pair<double, int>> xList(xCounts.begin(),
                                                      xCounts.end());
            std::sort(xList.begin(), xList.end(),
                      [](const auto &a, const auto &b) {
                        return a.second > b.second;
                      });

            // Find the 2 Y values with most corners
            std::vector<std::pair<double, int>> yList(yCounts.begin(),
                                                      yCounts.end());
            std::sort(yList.begin(), yList.end(),
                      [](const auto &a, const auto &b) {
                        return a.second > b.second;
                      });

            std::cerr << "DEBUG: Found " << xCounts.size()
                      << " unique X coordinates, " << yCounts.size()
                      << " unique Y coordinates" << std::endl;

            if (xList.size() >= 2 && yList.size() >= 2) {
              double leftX = std::min(xList[0].first, xList[1].first);
              double rightX = std::max(xList[0].first, xList[1].first);
              double bottomY = std::min(yList[0].first, yList[1].first);
              double topY = std::max(yList[0].first, yList[1].first);

              double minX = leftX;
              double maxX = rightX;
              double minY = bottomY;
              double maxY = topY;

              std::cerr << "DEBUG: Most common coords - X: " << xList[0].first
                        << " (n=" << xList[0].second << "), " << xList[1].first
                        << " (n=" << xList[1].second
                        << "); Y: " << yList[0].first
                        << " (n=" << yList[0].second << "), " << yList[1].first
                        << " (n=" << yList[1].second << ")" << std::endl;

              result.linesBoundingBoxX = minX;
              result.linesBoundingBoxY = minY;
              result.linesBoundingBoxWidth = maxX - minX;
              result.linesBoundingBoxHeight = maxY - minY;
              result.hasCropMarks = true;

              std::cerr << "DEBUG: Crop box from crop marks: (" << minX << ", "
                        << minY << ") to (" << maxX << ", " << maxY << ")"
                        << std::endl;
              std::cerr << "DEBUG: Bounding rectangle size (crop marks): "
                        << result.linesBoundingBoxWidth << " x "
                        << result.linesBoundingBoxHeight << " pt" << std::endl;
            } else {
              // Fallback to original bounding box
              result.linesBoundingBoxX = lineResult.boundingBoxX;
              result.linesBoundingBoxY = lineResult.boundingBoxY;
              result.linesBoundingBoxWidth = lineResult.boundingBoxWidth;
              result.linesBoundingBoxHeight = lineResult.boundingBoxHeight;

              std::cerr << "DEBUG: Not enough crop marks found ("
                        << cropMarkCorners.size()
                        << "), using line bounding box" << std::endl;
              std::cerr
                  << "DEBUG: Bounding rectangle size (largest rect/lines): "
                  << result.linesBoundingBoxWidth << " x "
                  << result.linesBoundingBoxHeight << " pt" << std::endl;
            }
          }
        } // end else (no TrimBox â€” use crop mark detection)
      }
    } catch (const std::exception &e) {
      std::cerr << "DEBUG: Line extraction threw exception: " << e.what()
                << std::endl;
    }

    // Get page count by loading the PDF once more (or we could track it
    // from earlier)
    std::cerr << "DEBUG: Getting page count..." << std::endl;
    try {
      std::unique_ptr<poppler::document> doc(
          poppler::document::load_from_file(pdfPath));
      std::cerr << "DEBUG: PDF loaded for page count" << std::endl;
      if (doc) {
        result.pageCount = doc->pages();
        std::cerr << "DEBUG: Page count retrieved: " << result.pageCount
                  << std::endl;

        // Get page dimensions from first page
        std::unique_ptr<poppler::page> page(doc->create_page(0));
        if (page) {
          poppler::rectf pageRect = page->page_rect();
          int pageRotation = page->orientation(); // 0=portrait,1=landscape,etc.

          // page_rect() returns unrotated MediaBox dimensions.
          // For Rotate=90/270 the display axes are swapped, so width and
          // height need to be exchanged to reflect the actual display extent
          // (the same coordinate space used by LineExtractorOutputDev /
          // crop-mark detection).
          bool swapped = (pageRotation == poppler::page::landscape ||
                          pageRotation == poppler::page::seascape);

          result.pageX = pageRect.x();
          result.pageY = pageRect.y();
          result.pageWidth = swapped ? pageRect.height() : pageRect.width();
          result.pageHeight = swapped ? pageRect.width() : pageRect.height();

          std::cerr << "DEBUG: Page crop box: origin(" << pageRect.x() << ", "
                    << pageRect.y() << ") size(" << result.pageWidth << " x "
                    << result.pageHeight << ") points"
                    << " rotation=" << pageRotation << std::endl;

          // Check for TrimBox â€” if the PDF has one that's smaller than the
          // MediaBox/CropBox, it defines the intended content area precisely
          // and we can skip unreliable geometric crop-mark detection.
          poppler::rectf trimRect = page->page_rect(poppler::trim_box);
          poppler::rectf mediaRect = page->page_rect(poppler::media_box);

          // TrimBox is meaningful if it's strictly smaller than MediaBox
          // (with a tolerance of 1pt to avoid floating-point noise)
          const double trimTolerance = 1.0;
          bool hasTrimBox =
              (trimRect.width() > 0 && trimRect.height() > 0 &&
               (trimRect.width() < mediaRect.width() - trimTolerance ||
                trimRect.height() < mediaRect.height() - trimTolerance));

          if (hasTrimBox) {
            std::cerr << "DEBUG: TrimBox found: origin(" << trimRect.x() << ", "
                      << trimRect.y() << ") size(" << trimRect.width() << " x "
                      << trimRect.height() << ") points" << std::endl;
            // TrimBox is logged but NOT used as content area â€”
            // crop mark detection is more reliable for these PDFs.
          } else {
            std::cerr << "DEBUG: No meaningful TrimBox found, will use crop "
                         "mark detection"
                      << std::endl;
          }
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "DEBUG: Exception getting page count: " << e.what()
                << std::endl;
      result.pageCount = 1; // Default to 1 if we can't get the count
    }

    // Note: pageCount reflects total pages in PDF, but only first page was
    // processed

    // Detect vector graphic regions (logos, illustrations, etc.) that are not
    // captured by extractEmbeddedImagesFromPDF, which only finds raster images.
    // We rasterize the page at low DPI, mask out areas covered by known
    // text/rectangle/line elements, and treat remaining non-white blobs of
    // significant size as vector-drawn image regions.
    if (result.pageWidth > 0 && result.pageHeight > 0) {
      std::cerr << "DEBUG: Scanning for vector graphic regions..." << std::endl;
      try {
        std::unique_ptr<poppler::document> vgDoc(
            poppler::document::load_from_file(pdfPath));
        if (vgDoc && vgDoc->pages() > 0) {
          std::unique_ptr<poppler::page> vgPage(vgDoc->create_page(0));
          if (vgPage) {
            const double vgDpi = 96.0;
            const double vgScale = vgDpi / 72.0;

            poppler::page_renderer vgRenderer;
            vgRenderer.set_render_hint(poppler::page_renderer::antialiasing,
                                       true);
            vgRenderer.set_image_format(poppler::image::format_argb32);
            poppler::image vgPopplerImg =
                vgRenderer.render_page(vgPage.get(), vgDpi, vgDpi);

            if (vgPopplerImg.is_valid()) {
              int vgW = vgPopplerImg.width();
              int vgH = vgPopplerImg.height();

              cv::Mat vgMat(vgH, vgW, CV_8UC4,
                            const_cast<char *>(vgPopplerImg.const_data()),
                            vgPopplerImg.bytes_per_row());
              cv::Mat vgBGR;
              cv::cvtColor(vgMat, vgBGR, cv::COLOR_BGRA2BGR);

              // Threshold to find non-white pixels
              cv::Mat vgGray, vgNonWhite;
              cv::cvtColor(vgBGR, vgGray, cv::COLOR_BGR2GRAY);
              cv::threshold(vgGray, vgNonWhite, 240, 255,
                            cv::THRESH_BINARY_INV);

              // Build a coverage mask for all known elements so we can
              // subtract them and find uncovered graphical content.
              cv::Mat vgCoverage(vgH, vgW, CV_8UC1, cv::Scalar(0));
              const int vgPad = 3; // extra pixel padding around each element

              // Helper: mark a rectangle in top-left pixel coordinates
              auto markRectTL = [&](double tlX, double tlY, double w,
                                    double h) {
                int px = static_cast<int>(tlX * vgScale);
                int py = static_cast<int>(tlY * vgScale);
                int pw = static_cast<int>(w * vgScale) + 1;
                int ph = static_cast<int>(h * vgScale) + 1;
                px = std::max(0, px - vgPad);
                py = std::max(0, py - vgPad);
                pw = std::min(vgW - px, pw + 2 * vgPad);
                ph = std::min(vgH - py, ph + 2 * vgPad);
                if (pw > 0 && ph > 0)
                  cv::rectangle(vgCoverage, cv::Rect(px, py, pw, ph), 255,
                                cv::FILLED);
              };

              // Text regions are already in top-left coords
              for (const auto &t : result.textLines)
                markRectTL(t.boundingBox.x, t.boundingBox.y,
                           t.boundingBox.width, t.boundingBox.height);

              // Rectangles and lines use bottom-left PDF coords â€” convert
              for (const auto &r : result.rectangles)
                markRectTL(r.x, result.pageHeight - r.y - r.height, r.width,
                           r.height);

              for (const auto &ln : result.graphicLines) {
                int lx1 = static_cast<int>(ln.x1 * vgScale);
                int ly1 =
                    static_cast<int>((result.pageHeight - ln.y1) * vgScale);
                int lx2 = static_cast<int>(ln.x2 * vgScale);
                int ly2 =
                    static_cast<int>((result.pageHeight - ln.y2) * vgScale);
                int lw = std::max(vgPad * 2,
                                  static_cast<int>(ln.lineWidth * vgScale + 2));
                cv::line(vgCoverage, cv::Point(lx1, ly1), cv::Point(lx2, ly2),
                         255, lw);
              }

              // Existing embedded images (bottom-left â†’ top-left)
              for (const auto &im : result.images)
                markRectTL(im.x, result.pageHeight - im.y - im.displayHeight,
                           im.displayWidth, im.displayHeight);

              // Find uncovered non-white content
              cv::Mat vgUnmasked;
              cv::bitwise_and(vgNonWhite, ~vgCoverage, vgUnmasked);

              // Close small holes to merge nearby pixels into blobs
              auto morphK =
                  cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
              cv::morphologyEx(vgUnmasked, vgUnmasked, cv::MORPH_CLOSE, morphK);

              // Content area limits in pixel space (top-left origin).
              // linesBoundingBox stores bottom-left PDF coords.
              int cntLeft = 0, cntTop = 0, cntRight = vgW, cntBottom = vgH;
              if (result.linesBoundingBoxWidth > 0 &&
                  result.linesBoundingBoxHeight > 0) {
                cntLeft = static_cast<int>(result.linesBoundingBoxX * vgScale);
                cntRight = static_cast<int>(
                    (result.linesBoundingBoxX + result.linesBoundingBoxWidth) *
                    vgScale);
                cntTop = static_cast<int>((result.pageHeight -
                                           result.linesBoundingBoxY -
                                           result.linesBoundingBoxHeight) *
                                          vgScale);
                cntBottom = static_cast<int>(
                    (result.pageHeight - result.linesBoundingBoxY) * vgScale);
              }

              // Find connected components in the uncovered mask
              cv::Mat vgLabels, vgStats, vgCentroids;
              int numCC = cv::connectedComponentsWithStats(
                  vgUnmasked, vgLabels, vgStats, vgCentroids);

              // Minimum size threshold: 40 PDF points in each dimension
              const double minPtSize = 40.0;
              const int minPxSize = static_cast<int>(minPtSize * vgScale);

              std::filesystem::path pdfFP(pdfPath);
              std::string pdfStemVG = pdfFP.stem().string();
              int vecIdx = static_cast<int>(result.images.size());

              for (int ci = 1; ci < numCC; ci++) {
                int bx = vgStats.at<int>(ci, cv::CC_STAT_LEFT);
                int by = vgStats.at<int>(ci, cv::CC_STAT_TOP);
                int bw = vgStats.at<int>(ci, cv::CC_STAT_WIDTH);
                int bh = vgStats.at<int>(ci, cv::CC_STAT_HEIGHT);

                if (bw < minPxSize || bh < minPxSize)
                  continue;

                // Require the region to overlap with the content area
                if (bx + bw <= cntLeft || bx >= cntRight || by + bh <= cntTop ||
                    by >= cntBottom)
                  continue;

                // Clip the bounding box to the content area
                int cx = std::max(bx, cntLeft);
                int cy = std::max(by, cntTop);
                int cx2 = std::min(bx + bw, cntRight);
                int cy2 = std::min(by + bh, cntBottom);
                int cw = cx2 - cx;
                int ch = cy2 - cy;
                if (cw <= 0 || ch <= 0)
                  continue;

                // Convert pixel region to PDF bottom-left coordinates
                double pdfX = cx / vgScale;
                double pdfY = result.pageHeight - cy2 / vgScale;
                double pdfW = cw / vgScale;
                double pdfH = ch / vgScale;

                PDFEmbeddedImage vgEmbImg;
                int safeCX = std::max(0, cx);
                int safeCY = std::max(0, cy);
                int safeCW = std::min(vgW - safeCX, cw);
                int safeCH = std::min(vgH - safeCY, ch);
                if (safeCW > 0 && safeCH > 0)
                  vgEmbImg.image =
                      vgBGR(cv::Rect(safeCX, safeCY, safeCW, safeCH)).clone();
                vgEmbImg.pageNumber = 1;
                vgEmbImg.imageIndex = vecIdx;
                vgEmbImg.width = safeCW;
                vgEmbImg.height = safeCH;
                vgEmbImg.x = pdfX;
                vgEmbImg.y = pdfY;
                vgEmbImg.displayWidth = pdfW;
                vgEmbImg.displayHeight = pdfH;
                vgEmbImg.rotationAngle = 0.0;
                vgEmbImg.type = "vector_graphic";

                std::cerr << "DEBUG: Vector graphic at PDF (" << pdfX << ", "
                          << pdfY << ") size " << pdfW << "x" << pdfH
                          << " pts, " << safeCW << "x" << safeCH << " px"
                          << std::endl;

                // Save to output directory if one was specified
                if (!imageOutputDir.empty() && !vgEmbImg.image.empty()) {
                  std::filesystem::path outDir(imageOutputDir);
                  std::string fn =
                      (outDir / (pdfStemVG + "_vecgfx_" +
                                 std::to_string(vecIdx + 1) + ".png"))
                          .string();
                  if (cv::imwrite(fn, vgEmbImg.image))
                    std::cerr << "DEBUG: Saved vector graphic " << (vecIdx + 1)
                              << " to: " << fn << std::endl;
                }

                result.images.push_back(std::move(vgEmbImg));
                vecIdx++;
              }

              result.imageCount = static_cast<int>(result.images.size());
              std::cerr << "DEBUG: Total images after vector graphic scan: "
                        << result.imageCount << std::endl;
            }
          }
        }
      } catch (const std::exception &e) {
        std::cerr << "DEBUG: Vector graphic detection error: " << e.what()
                  << std::endl;
      }
    }

    // --- OCR on embedded images for "L1" PDFs ----------------------------
    // When the PDF file name begins with "L1" (case-insensitive), run
    // Tesseract OCR on every embedded image.  Words whose per-word Tesseract
    // confidence exceeds 85 % are added to result.textLines as TextRegion
    // entries (top-left PDF-point coordinates, matching native Poppler text).
    {
      std::filesystem::path ocrPdfPath(pdfPath);
      std::string ocrStem = ocrPdfPath.stem().string();
      bool doImageOCR =
          ocrStem.size() >= 2 &&
          std::toupper(static_cast<unsigned char>(ocrStem[0])) == 'L' &&
          ocrStem[1] == '1';

      if (doImageOCR && !result.images.empty()) {
        std::cerr << "DEBUG: Running OCR on " << result.images.size()
                  << " image(s) (L1 PDF rule)" << std::endl;

        // Initialise a single Tesseract instance for all images.
        tesseract::TessBaseAPI *tess = new tesseract::TessBaseAPI();
        bool tessOk = (tess->Init("C:/tessdata/tessdata", "eng") == 0 ||
                       tess->Init(NULL, "eng") == 0);
        if (!tessOk) {
          std::cerr << "DEBUG: Could not initialise Tesseract for image OCR"
                    << std::endl;
        } else {
          tess->SetPageSegMode(tesseract::PSM_AUTO);

          const float kConfThreshold = 85.0f;

          for (const auto &img : result.images) {
            if (img.image.empty())
              continue;

            // Convert to grayscale for Tesseract.
            cv::Mat gray;
            if (img.image.channels() == 1)
              gray = img.image;
            else
              cv::cvtColor(img.image, gray, cv::COLOR_BGR2GRAY);

            // Scale factors: image pixels â†’ PDF points.
            double scaleX =
                (img.width > 0) ? img.displayWidth / img.width : 1.0;
            double scaleY =
                (img.height > 0) ? img.displayHeight / img.height : 1.0;

            // Top-left corner of the image in PDF top-leftâ€“origin coords.
            double imgTopLeftPtX = img.x;
            double imgTopLeftPtY =
                result.pageHeight - img.y - img.displayHeight;

            tess->SetImage(gray.data, gray.cols, gray.rows, 1,
                           static_cast<int>(gray.step));
            tess->Recognize(0);

            // Iterate over words.
            tesseract::ResultIterator *ri = tess->GetIterator();
            if (ri == nullptr)
              continue;

            // Collect high-confidence words.
            struct OcrWordPt {
              std::string text;
              float conf;
              // Top-left origin, PDF points:
              double x, y, w, h;
            };
            std::vector<OcrWordPt> goodWords;

            do {
              const char *wordRaw = ri->GetUTF8Text(tesseract::RIL_WORD);
              float conf = ri->Confidence(tesseract::RIL_WORD);
              // Trim and skip whitespace-only results
              std::string wordStr = wordRaw ? wordRaw : "";
              delete[] wordRaw;
              auto wsStart = wordStr.find_first_not_of(" \t\r\n\f\v");
              if (wsStart != std::string::npos)
                wordStr = wordStr.substr(
                    wsStart,
                    wordStr.find_last_not_of(" \t\r\n\f\v") - wsStart + 1);
              else
                wordStr.clear();
              if (!wordStr.empty() && conf >= kConfThreshold) {
                int wx1, wy1, wx2, wy2;
                ri->BoundingBox(tesseract::RIL_WORD, &wx1, &wy1, &wx2, &wy2);
                OcrWordPt wp;
                wp.text = wordStr;
                wp.conf = conf;
                wp.x = imgTopLeftPtX + wx1 * scaleX;
                wp.y = imgTopLeftPtY + wy1 * scaleY;
                wp.w = (wx2 - wx1) * scaleX;
                wp.h = (wy2 - wy1) * scaleY;
                goodWords.push_back(wp);
                std::cerr << "DEBUG: OCR word \"" << wp.text
                          << "\" conf=" << conf << " at PDF (" << wp.x << ","
                          << wp.y << ") " << wp.w << "x" << wp.h << " pt"
                          << std::endl;
              }
            } while (ri->Next(tesseract::RIL_WORD));
            delete ri;

            if (goodWords.empty())
              continue;

            // Group words into text lines: words within half a line-height
            // of each other (sorted top-to-bottom) belong to the same line.
            std::sort(goodWords.begin(), goodWords.end(),
                      [](const OcrWordPt &a, const OcrWordPt &b) {
                        return a.y < b.y;
                      });

            std::vector<std::vector<size_t>> lines; // groups of word indices
            for (size_t wi = 0; wi < goodWords.size(); wi++) {
              bool placed = false;
              for (auto &line : lines) {
                const OcrWordPt &ref = goodWords[line[0]];
                double tol = ref.h * 0.6;
                if (std::abs(goodWords[wi].y - ref.y) <= tol) {
                  line.push_back(wi);
                  placed = true;
                  break;
                }
              }
              if (!placed)
                lines.push_back({wi});
            }

            // Build one TextRegion per line.
            for (const auto &line : lines) {
              if (line.empty())
                continue;

              // Sort words left-to-right within the line.
              std::vector<size_t> sorted = line;
              std::sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) {
                return goodWords[a].x < goodWords[b].x;
              });

              // Compute bounding box and joined text.
              double lx = goodWords[sorted[0]].x;
              double ly = goodWords[sorted[0]].y;
              double lx2 = lx, ly2 = ly;
              float totalConf = 0.0f;
              std::string lineText;
              for (size_t idx : sorted) {
                const auto &w = goodWords[idx];
                lx = std::min(lx, w.x);
                ly = std::min(ly, w.y);
                lx2 = std::max(lx2, w.x + w.w);
                ly2 = std::max(ly2, w.y + w.h);
                totalConf += w.conf;
                if (!lineText.empty())
                  lineText += ' ';
                lineText += w.text;
              }
              float avgConf = totalConf / static_cast<float>(sorted.size());

              TextRegion tr;
              // boundingBox is integer, top-left PDF-point coords.
              tr.boundingBox =
                  cv::Rect(static_cast<int>(lx), static_cast<int>(ly),
                           static_cast<int>(std::ceil(lx2 - lx)),
                           static_cast<int>(std::ceil(ly2 - ly)));
              tr.preciseX = lx;
              tr.preciseY = ly;
              tr.preciseWidth = lx2 - lx;
              tr.preciseHeight = ly2 - ly;
              tr.text = lineText;
              tr.confidence = avgConf;
              tr.level = 2; // line level
              tr.orientation = TextOrientation::Horizontal;

              std::cerr << "DEBUG: OCR line \"" << lineText
                        << "\" conf=" << avgConf << " bbox=("
                        << tr.boundingBox.x << "," << tr.boundingBox.y << ","
                        << tr.boundingBox.width << "x" << tr.boundingBox.height
                        << ") pt" << std::endl;

              result.textLines.push_back(std::move(tr));
              result.textLineCount = static_cast<int>(result.textLines.size());
            }
          } // for each image

          tess->End();
        }
        delete tess;
      }
    }
    // --- end OCR on images -----------------------------------------------

    result.success = true;
    std::cerr << "DEBUG: All extractions completed successfully" << std::endl;

  } catch (const std::exception &e) {
    result.errorMessage =
        std::string("PDF element extraction failed: ") + e.what();
    std::cerr << "DEBUG: Top-level exception: " << e.what() << std::endl;
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  result.processingTimeMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();

  return result;
}

int OCRAnalysis::writeAllImages(const std::string &pdfPath,
                                const std::string &outputDir) {
  try {
    // Create output directory if it doesn't exist
    std::filesystem::path outDir(outputDir);
    std::filesystem::create_directories(outDir);

    // Get PDF stem for naming files
    std::filesystem::path pdfFilePath(pdfPath);
    std::string pdfStem = pdfFilePath.stem().string();

    // Load PDF using low-level Poppler API
    GlobalParamsIniter globalParamsInit(nullptr);
    auto fileName = std::make_unique<GooString>(pdfPath);
    std::unique_ptr<PDFDoc> doc(new PDFDoc(std::move(fileName)));

    if (!doc->isOk()) {
      std::cerr << "ERROR: Failed to load PDF file: " << pdfPath << std::endl;
      return -1;
    }

    int pageCount = doc->getNumPages();
    if (pageCount < 1) {
      std::cerr << "ERROR: PDF has no pages" << std::endl;
      return -1;
    }

    std::cerr << "DEBUG: PDF has " << pageCount << " page(s)" << std::endl;

    int totalSaved = 0;

    for (int pageNum = 1; pageNum <= pageCount; pageNum++) {
      // Extract images from this page
      ImageExtractorOutputDev outputDev;
      outputDev.setDoc(doc.get());
      outputDev.setPageNumber(pageNum);

      doc->displayPage(&outputDev, pageNum, 72.0, 72.0, 0, true, false, false);

      auto images = std::move(outputDev.getImages());
      std::cerr << "DEBUG: Page " << pageNum << ": found " << images.size()
                << " embedded image(s)" << std::endl;

      for (size_t i = 0; i < images.size(); i++) {
        const auto &img = images[i];
        if (img.image.empty()) {
          std::cerr << "DEBUG: Page " << pageNum << " image " << (i + 1)
                    << ": empty, skipping" << std::endl;
          continue;
        }

        std::string filename =
            (outDir / (pdfStem + "_page" + std::to_string(pageNum) + "_image" +
                       std::to_string(i + 1) + ".png"))
                .string();

        if (cv::imwrite(filename, img.image)) {
          std::cout << "Saved: " << filename << " (" << img.image.cols << "x"
                    << img.image.rows << ", " << img.image.channels()
                    << " channels)" << std::endl;
          totalSaved++;
        } else {
          std::cerr << "ERROR: Failed to save: " << filename << std::endl;
        }
      }
    }

    std::cout << "Total images saved: " << totalSaved << std::endl;
    return totalSaved;

  } catch (const std::exception &e) {
    std::cerr << "ERROR: writeAllImages failed: " << e.what() << std::endl;
    return -1;
  }
}

std::string OCRAnalysis::getTextFromRegion(const cv::Mat &image,
                                           const cv::Rect &roi) {
  if (!m_initialized || image.empty()) {
    return "";
  }

  // Validate and clamp ROI to image bounds
  cv::Rect validRoi = roi & cv::Rect(0, 0, image.cols, image.rows);
  if (validRoi.empty()) {
    return "";
  }

  cv::Mat regionImage = image(validRoi);
  cv::Mat processedRegion =
      m_config.preprocessImage ? preprocessImage(regionImage) : regionImage;

  setImage(processedRegion);

  char *outText = m_tesseract->GetUTF8Text();
  std::string result;
  if (outText) {
    result = outText;
    delete[] outText;
  }

  return result;
}

std::vector<TextRegion> OCRAnalysis::detectTextRegions(const cv::Mat &image) {
  std::vector<TextRegion> regions;

  if (!m_initialized || image.empty()) {
    return regions;
  }

  // Find the best rotation for the image
  int bestRotation = findBestRotation(image);

  // Apply the best rotation
  cv::Mat workingImage;
  if (bestRotation != -1) {
    cv::rotate(image, workingImage, bestRotation);
  } else {
    workingImage = image.clone();
  }

  setImage(workingImage);

  // Must call Recognize before GetIterator
  m_tesseract->Recognize(nullptr);

  // Structure to hold region info including Tesseract orientation for later
  // processing
  struct RegionInfo {
    TextRegion region;
    tesseract::Orientation tessOrientation;
  };
  std::vector<RegionInfo> regionInfos;

  // Use Tesseract's component analysis
  tesseract::ResultIterator *ri = m_tesseract->GetIterator();
  tesseract::PageIteratorLevel level = tesseract::RIL_WORD;

  // First pass: collect all regions with their orientations
  // Do NOT call setImage or Recognize during this loop as it invalidates
  // the iterator
  if (ri != nullptr) {
    do {
      const char *word = ri->GetUTF8Text(level);
      float conf = ri->Confidence(level);

      if (word != nullptr && *word != '\0') {
        RegionInfo info;
        info.region.text = word;
        info.region.confidence = conf;
        info.region.level = static_cast<int>(level);

        int x1, y1, x2, y2;
        ri->BoundingBox(level, &x1, &y1, &x2, &y2);
        info.region.boundingBox = cv::Rect(x1, y1, x2 - x1, y2 - y1);

        // Determine text orientation using Tesseract's orientation
        // detection
        tesseract::WritingDirection writingDirection;
        tesseract::TextlineOrder textlineOrder;
        float deskewAngle;
        ri->Orientation(&info.tessOrientation, &writingDirection,
                        &textlineOrder, &deskewAngle);

        // Map Tesseract orientation to our TextOrientation enum
        // PAGE_UP = normal, PAGE_DOWN = upside-down (180Â°)
        // PAGE_LEFT/RIGHT = vertical (90Â° rotated)
        switch (info.tessOrientation) {
        case tesseract::ORIENTATION_PAGE_UP:
          info.region.orientation = TextOrientation::Horizontal;
          break;
        case tesseract::ORIENTATION_PAGE_DOWN:
          // Upside-down text - mark as horizontal but will need 180Â°
          // rotation
          info.region.orientation = TextOrientation::Horizontal;
          break;
        case tesseract::ORIENTATION_PAGE_LEFT:
        case tesseract::ORIENTATION_PAGE_RIGHT:
          info.region.orientation = TextOrientation::Vertical;
          break;
        default:
          info.region.orientation = TextOrientation::Unknown;
          break;
        }

        regionInfos.push_back(info);
      }

      delete[] word;
    } while (ri->Next(level));

    delete ri;
  }

  // Second pass: re-OCR rotated regions after correcting their orientation
  // Now it's safe to call setImage and Recognize since the iterator is gone
  for (auto &info : regionInfos) {
    // Only process regions that need rotation (not PAGE_UP which is normal)
    bool needsRotation =
        (info.tessOrientation != tesseract::ORIENTATION_PAGE_UP);

    if (needsRotation) {
      // Add padding around the bounding box for better OCR
      int padding = 10;
      cv::Rect paddedBox = info.region.boundingBox;
      paddedBox.x = std::max(0, paddedBox.x - padding);
      paddedBox.y = std::max(0, paddedBox.y - padding);
      paddedBox.width = std::min(workingImage.cols - paddedBox.x,
                                 paddedBox.width + 2 * padding);
      paddedBox.height = std::min(workingImage.rows - paddedBox.y,
                                  paddedBox.height + 2 * padding);

      // Validate the padded bounding box
      cv::Rect validBox =
          paddedBox & cv::Rect(0, 0, workingImage.cols, workingImage.rows);
      if (!validBox.empty() && validBox.width > 0 && validBox.height > 0) {
        // Extract the region from the working (orientation-corrected) image
        cv::Mat regionImage = workingImage(validBox).clone();

        // Rotate the region based on its detected orientation
        cv::Mat rotatedRegion;
        switch (info.tessOrientation) {
        case tesseract::ORIENTATION_PAGE_DOWN:
          // Upside-down text, rotate 180 degrees
          cv::rotate(regionImage, rotatedRegion, cv::ROTATE_180);
          break;
        case tesseract::ORIENTATION_PAGE_RIGHT:
          // Text reads top-to-bottom, rotate clockwise to make horizontal
          cv::rotate(regionImage, rotatedRegion, cv::ROTATE_90_CLOCKWISE);
          break;
        case tesseract::ORIENTATION_PAGE_LEFT:
          // Text reads bottom-to-top, rotate counter-clockwise
          cv::rotate(regionImage, rotatedRegion,
                     cv::ROTATE_90_COUNTERCLOCKWISE);
          break;
        default:
          rotatedRegion = regionImage;
          break;
        }

        // Add a white border for additional context
        cv::Mat borderedRegion;
        cv::copyMakeBorder(rotatedRegion, borderedRegion, 10, 10, 10, 10,
                           cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

        // Use PSM_SINGLE_WORD for better single word recognition
        tesseract::PageSegMode originalMode = m_tesseract->GetPageSegMode();
        m_tesseract->SetPageSegMode(tesseract::PSM_SINGLE_WORD);

        // Perform OCR on the rotated region
        setImage(borderedRegion);
        m_tesseract->Recognize(nullptr);

        char *rotatedText = m_tesseract->GetUTF8Text();
        if (rotatedText != nullptr && *rotatedText != '\0') {
          // Update the region text with the improved recognition
          std::string newText = rotatedText;
          // Trim whitespace and newlines
          size_t start = newText.find_first_not_of(" \t\n\r");
          size_t end = newText.find_last_not_of(" \t\n\r");
          if (start != std::string::npos && end != std::string::npos) {
            info.region.text = newText.substr(start, end - start + 1);
          }

          // Update confidence with the new recognition result
          info.region.confidence =
              static_cast<float>(m_tesseract->MeanTextConf());
        }
        if (rotatedText) {
          delete[] rotatedText;
        }

        // Restore original page segmentation mode
        m_tesseract->SetPageSegMode(originalMode);
      }
    }

    regions.push_back(info.region);
  }

  return regions;
}

std::vector<TextRegion> OCRAnalysis::identifyTextRegions(const cv::Mat &image) {
  std::vector<TextRegion> allRegions;

  if (!m_initialized || image.empty()) {
    return allRegions;
  }

  // Try all 4 orientations to find text in any direction
  std::vector<std::pair<int, TextOrientation>> rotations = {
      {-1, TextOrientation::Horizontal},                    // 0Â° - normal
      {cv::ROTATE_90_CLOCKWISE, TextOrientation::Vertical}, // 90Â° CW
      {cv::ROTATE_180, TextOrientation::Horizontal}, // 180Â° - upside down
      {cv::ROTATE_90_COUNTERCLOCKWISE, TextOrientation::Vertical} // 270Â°
  };

  for (const auto &rotation : rotations) {
    cv::Mat rotatedImage;
    if (rotation.first == -1) {
      rotatedImage = image.clone();
    } else {
      cv::rotate(image, rotatedImage, rotation.first);
    }

    setImage(rotatedImage);
    m_tesseract->Recognize(nullptr);

    tesseract::ResultIterator *ri = m_tesseract->GetIterator();
    if (ri == nullptr) {
      continue;
    }

    // Use LINE level for text line detection
    tesseract::PageIteratorLevel level = tesseract::RIL_TEXTLINE;

    do {
      const char *text = ri->GetUTF8Text(level);
      float conf = ri->Confidence(level);

      if (text != nullptr && *text != '\0' && conf > 10.0f) {
        TextRegion region;
        region.text = text;
        region.confidence = conf;
        region.level = static_cast<int>(level);

        int x1, y1, x2, y2;
        ri->BoundingBox(level, &x1, &y1, &x2, &y2);

        // Transform bounding box back to original image coordinates
        cv::Rect box(x1, y1, x2 - x1, y2 - y1);

        // Reverse the rotation to get coordinates in original image space
        switch (rotation.first) {
        case cv::ROTATE_90_CLOCKWISE:
          // (x, y) in rotated -> (y, width - x - w) in original
          region.boundingBox = cv::Rect(box.y, image.cols - box.x - box.width,
                                        box.height, box.width);
          region.orientation = TextOrientation::Vertical;
          break;
        case cv::ROTATE_180:
          // (x, y) in rotated -> (width - x - w, height - y - h) in
          // original
          region.boundingBox =
              cv::Rect(image.cols - box.x - box.width,
                       image.rows - box.y - box.height, box.width, box.height);
          region.orientation = TextOrientation::Horizontal;
          break;
        case cv::ROTATE_90_COUNTERCLOCKWISE:
          // (x, y) in rotated -> (height - y - h, x) in original
          region.boundingBox = cv::Rect(image.rows - box.y - box.height, box.x,
                                        box.height, box.width);
          region.orientation = TextOrientation::Vertical;
          break;
        default:
          // No rotation
          region.boundingBox = box;
          region.orientation = TextOrientation::Horizontal;
          break;
        }

        // Clamp to image bounds
        region.boundingBox &= cv::Rect(0, 0, image.cols, image.rows);

        if (!region.boundingBox.empty()) {
          allRegions.push_back(region);
        }
      }

      delete[] text;
    } while (ri->Next(level));

    delete ri;
  }

  // Remove duplicate/overlapping regions (keep highest confidence)
  std::vector<TextRegion> filteredRegions;
  for (const auto &region : allRegions) {
    bool isDuplicate = false;
    for (auto &existing : filteredRegions) {
      // Check for significant overlap (IoU > 0.5)
      cv::Rect intersection = region.boundingBox & existing.boundingBox;
      if (!intersection.empty()) {
        double intersectionArea = intersection.area();
        double unionArea = region.boundingBox.area() +
                           existing.boundingBox.area() - intersectionArea;
        double iou = intersectionArea / unionArea;

        if (iou > 0.5) {
          isDuplicate = true;
          // Keep the one with higher confidence
          if (region.confidence > existing.confidence) {
            existing = region;
          }
          break;
        }
      }
    }
    if (!isDuplicate) {
      filteredRegions.push_back(region);
    }
  }

  return filteredRegions;
}

cv::Mat OCRAnalysis::maskNonTextRegions(const cv::Mat &image) {
  if (image.empty()) {
    return image.clone();
  }

  cv::Mat result = image.clone();
  cv::Mat gray;

  // Convert to grayscale for analysis
  if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  } else {
    gray = image.clone();
  }

  // Apply edge detection to find contours
  cv::Mat edges;
  cv::Canny(gray, edges, 50, 150);

  // Dilate to connect nearby edges
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::dilate(edges, edges, kernel, cv::Point(-1, -1), 2);

  // Find contours
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  // Minimum size thresholds (in pixels) - smaller regions are likely noise
  const int minWidth = 50;
  const int minHeight = 50;
  const int minArea = 2500;

  // Maximum size thresholds - don't mask very large regions (likely text
  // blocks)
  const int maxWidthRatio = 3;  // Max width as fraction of image width
  const int maxHeightRatio = 3; // Max height as fraction of image height

  std::vector<cv::Rect> graphicRegions;

  for (size_t i = 0; i < contours.size(); i++) {
    cv::Rect boundingRect = cv::boundingRect(contours[i]);

    // Skip regions that are too small
    if (boundingRect.width < minWidth || boundingRect.height < minHeight ||
        boundingRect.area() < minArea) {
      continue;
    }

    // Skip regions that are too large (likely encompassing text)
    if (boundingRect.width > image.cols / maxWidthRatio ||
        boundingRect.height > image.rows / maxHeightRatio) {
      continue;
    }

    // Analyze if this region looks like a graphic
    if (isLikelyGraphic(image, contours[i], boundingRect)) {
      // Add some padding around the graphic
      int padding = 5;
      cv::Rect paddedRect(std::max(0, boundingRect.x - padding),
                          std::max(0, boundingRect.y - padding),
                          std::min(boundingRect.width + 2 * padding,
                                   image.cols - boundingRect.x + padding),
                          std::min(boundingRect.height + 2 * padding,
                                   image.rows - boundingRect.y + padding));

      graphicRegions.push_back(paddedRect);
    }
  }

  // Merge overlapping regions
  std::vector<cv::Rect> mergedRegions;
  std::vector<bool> merged(graphicRegions.size(), false);

  for (size_t i = 0; i < graphicRegions.size(); i++) {
    if (merged[i])
      continue;

    cv::Rect current = graphicRegions[i];
    bool didMerge;

    do {
      didMerge = false;
      for (size_t j = i + 1; j < graphicRegions.size(); j++) {
        if (merged[j])
          continue;

        cv::Rect intersection = current & graphicRegions[j];
        if (!intersection.empty()) {
          // Merge by taking the bounding rect of both
          current = current | graphicRegions[j];
          merged[j] = true;
          didMerge = true;
        }
      }
    } while (didMerge);

    mergedRegions.push_back(current);
  }

  // Fill the graphic regions with white
  for (const auto &rect : mergedRegions) {
    cv::Rect validRect = rect & cv::Rect(0, 0, result.cols, result.rows);
    if (!validRect.empty()) {
      if (result.channels() == 1) {
        result(validRect).setTo(cv::Scalar(255));
      } else if (result.channels() == 3) {
        result(validRect).setTo(cv::Scalar(255, 255, 255));
      } else if (result.channels() == 4) {
        result(validRect).setTo(cv::Scalar(255, 255, 255, 255));
      }
    }
  }

  return result;
}

bool OCRAnalysis::isLikelyGraphic(const cv::Mat &image,
                                  const std::vector<cv::Point> &contour,
                                  const cv::Rect &boundingRect) {
  // Extract the region of interest
  cv::Rect validRect = boundingRect & cv::Rect(0, 0, image.cols, image.rows);
  if (validRect.empty() || validRect.width < 10 || validRect.height < 10) {
    return false;
  }

  cv::Mat roi = image(validRect);

  // 1. Aspect ratio analysis - logos often have more square-ish aspect
  // ratios
  //    while text is usually very wide or very tall
  double aspectRatio =
      static_cast<double>(boundingRect.width) / boundingRect.height;
  bool isSquarish = (aspectRatio >= 0.5 && aspectRatio <= 2.0);

  // 2. Contour solidity - ratio of contour area to convex hull area
  //    Logos often have lower solidity due to complex shapes
  double contourArea = cv::contourArea(contour);
  std::vector<cv::Point> hull;
  cv::convexHull(contour, hull);
  double hullArea = cv::contourArea(hull);
  double solidity = (hullArea > 0) ? (contourArea / hullArea) : 0;
  bool hasComplexShape = (solidity < 0.7);

  // 3. Edge density analysis - graphics typically have higher edge density
  cv::Mat grayRoi;
  if (roi.channels() == 3) {
    cv::cvtColor(roi, grayRoi, cv::COLOR_BGR2GRAY);
  } else if (roi.channels() == 4) {
    cv::cvtColor(roi, grayRoi, cv::COLOR_BGRA2GRAY);
  } else {
    grayRoi = roi;
  }

  cv::Mat edges;
  cv::Canny(grayRoi, edges, 50, 150);
  double edgeDensity = static_cast<double>(cv::countNonZero(edges)) /
                       (validRect.width * validRect.height);
  bool hasHighEdgeDensity = (edgeDensity > 0.15);

  // 4. Color complexity - logos often have multiple distinct colors
  bool hasColorComplexity = false;
  if (roi.channels() >= 3) {
    cv::Mat hsvRoi;
    if (roi.channels() == 4) {
      cv::Mat tempRoi;
      cv::cvtColor(roi, tempRoi, cv::COLOR_BGRA2BGR);
      cv::cvtColor(tempRoi, hsvRoi, cv::COLOR_BGR2HSV);
    } else {
      cv::cvtColor(roi, hsvRoi, cv::COLOR_BGR2HSV);
    }

    // Calculate histogram of hue channel
    cv::Mat hueChannel;
    cv::extractChannel(hsvRoi, hueChannel, 0);

    // Count distinct hue ranges (simplified color counting)
    std::vector<int> hueHist(18, 0); // 18 bins for hue (0-180)
    for (int y = 0; y < hueChannel.rows; y++) {
      for (int x = 0; x < hueChannel.cols; x++) {
        int bin = hueChannel.at<uchar>(y, x) / 10;
        if (bin < 18)
          hueHist[bin]++;
      }
    }

    int significantBins = 0;
    int minPixelsPerBin =
        (hueChannel.rows * hueChannel.cols) / 50; // 2% threshold
    for (int count : hueHist) {
      if (count > minPixelsPerBin)
        significantBins++;
    }

    hasColorComplexity = (significantBins >= 3);
  }

  // 5. Check for filled regions - graphics often have solid filled areas
  cv::Mat binaryRoi;
  cv::threshold(grayRoi, binaryRoi, 0, 255,
                cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  double fillRatio = static_cast<double>(cv::countNonZero(binaryRoi)) /
                     (validRect.width * validRect.height);
  bool hasSignificantFill = (fillRatio > 0.2 && fillRatio < 0.8);

  // Scoring system: classify as graphic if multiple criteria are met
  int graphicScore = 0;
  if (isSquarish)
    graphicScore += 2;
  if (hasComplexShape)
    graphicScore += 2;
  if (hasHighEdgeDensity)
    graphicScore += 1;
  if (hasColorComplexity)
    graphicScore += 3;
  if (hasSignificantFill && isSquarish)
    graphicScore += 2;

  // Region is likely a graphic if score is high enough
  return graphicScore >= 5;
}

bool OCRAnalysis::setLanguage(const std::string &language) {
  m_config.language = language;

  if (m_initialized) {
    m_tesseract->End();
    m_initialized = false;
    return initialize();
  }

  return true;
}

void OCRAnalysis::setPageSegMode(tesseract::PageSegMode mode) {
  m_config.pageSegMode = mode;
  if (m_initialized) {
    m_tesseract->SetPageSegMode(mode);
  }
}

const OCRConfig &OCRAnalysis::getConfig() const { return m_config; }

void OCRAnalysis::setConfig(const OCRConfig &config) {
  m_config = config;
  if (m_initialized) {
    m_tesseract->End();
    m_initialized = false;
  }
}

std::string OCRAnalysis::getTesseractVersion() {
  return tesseract::TessBaseAPI::Version();
}

std::vector<std::string> OCRAnalysis::getAvailableLanguages() const {
  std::vector<std::string> languages;

  if (m_initialized) {
    m_tesseract->GetAvailableLanguagesAsVector(&languages);
  }

  return languages;
}

cv::Mat OCRAnalysis::preprocessImage(const cv::Mat &image) {
  cv::Mat processed;

  // Convert to grayscale if color
  if (image.channels() == 3) {
    cv::cvtColor(image, processed, cv::COLOR_BGR2GRAY);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, processed, cv::COLOR_BGRA2GRAY);
  } else {
    processed = image.clone();
  }

  // Apply Gaussian blur to reduce noise
  cv::GaussianBlur(processed, processed, cv::Size(3, 3), 0);

  // Apply adaptive thresholding for better text recognition
  cv::adaptiveThreshold(processed, processed, 255,
                        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11,
                        2);

  return processed;
}

// static
cv::Mat OCRAnalysis::cleanupForOCR(const cv::Mat &input,
                                    CleanupDiagnostics *diag) {
  if (input.empty())
    return {};

  // ── Step 1: separate colour from transparency ─────────────────────────────
  // For RGBA images the root cause of "bleed-through" is that compositing a
  // semi-transparent dark edge onto white produces a mid-grey pixel.  If we
  // then threshold on grey level those mid-grey pixels incorrectly appear as
  // ink.  The fix is to treat the alpha channel as a hard gate: only pixels
  // with alpha > 200 (≈ 78 % opaque) are even allowed to be classified as
  // ink.  Everything else is forced to white before any thresholding occurs.
  //
  // For the colour channels we convert to greyscale WITHOUT compositing so
  // the actual ink darkness is preserved at its true value, not diluted by
  // the blend with white.

  cv::Mat gray;
  cv::Mat opaqueMask; // non-empty only for RGBA input

  if (input.channels() == 4) {
    std::vector<cv::Mat> ch;
    cv::split(input, ch); // B, G, R, A  (all CV_8U)

    // Opaque mask: alpha > 200 → 255 (include), ≤ 200 → 0 (force white).
    cv::threshold(ch[3], opaqueMask, 200, 255, cv::THRESH_BINARY);

    // Greyscale from the colour channels only (no alpha blending).
    cv::Mat bgr;
    cv::merge(std::vector<cv::Mat>{ch[0], ch[1], ch[2]}, bgr);
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

  } else if (input.channels() == 3) {
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = input.clone();
  }

  // ── Step 2: gentle noise reduction ───────────────────────────────────────
  cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);

  // ── Step 3: find the optimal threshold from the histogram ─────────────────
  // We compute the histogram only over the opaque pixels (via opaqueMask for
  // RGBA, or the whole image for other formats).  This means the semi-
  // transparent edges that caused the 57 % bleed-through are excluded from
  // the analysis entirely.  The remaining opaque pixels form a naturally
  // bimodal distribution – a peak near 255 for the label paper / background
  // and a dark cluster for the real ink strokes.  We locate the valley
  // between those two peaks and use it as a global binary threshold.
  //
  // Algorithm:
  //   a) Build the 256-bin histogram (masked to opaque pixels if RGBA).
  //   b) Smooth with a Gaussian to suppress per-bin noise.
  //   c) Find the background peak – the tallest bin in [128, 255].
  //   d) Find the ink peak        – the tallest bin in [0, bgPeak-10].
  //   e) Walk from inkPeak to bgPeak; record the bin with the lowest count
  //      (the valley), which is the natural ink / background boundary.
  //   f) Apply a global binary threshold at that valley value.
  //      Pixels at or above the valley → white; below → black ink.

  cv::Mat hist;
  const int   histSize  = 256;
  const float range[]   = {0.0f, 256.0f};
  const float *histRange = range;
  cv::calcHist(&gray, 1, nullptr,
               opaqueMask.empty() ? cv::Mat() : opaqueMask,
               hist, 1, &histSize, &histRange, true, false);

  // (b) Smooth the 256×1 histogram column with a vertical Gaussian.
  cv::GaussianBlur(hist, hist, cv::Size(1, 11), 0);

  // (c) Find the first local maximum walking from the dark end (bin 0).
  //     This is the dark-ink cluster.  We detect it by watching for the
  //     histogram to rise then fall.
  int   firstPeakBin   = -1;
  float firstPeakCount = 0.0f;
  bool  wasAscending   = false;
  for (int b = 1; b < 240; b++) {
    float cur  = hist.at<float>(b);
    float prev = hist.at<float>(b - 1);
    if (cur > prev) {
      wasAscending = true;
      if (cur > firstPeakCount) { firstPeakCount = cur; firstPeakBin = b; }
    } else if (wasAscending && cur < prev) {
      break; // just passed the crest of the first peak
    }
  }
  if (firstPeakBin < 0) firstPeakBin = 0; // no peak: nothing is ink

  // (d) Walk from the ink peak to find the first valley.
  //     Track the running minimum; stop the moment the histogram rises —
  //     that upward step marks the bottom of the valley between the dark
  //     ink cluster and the next lighter cluster.
  float valleyCount = firstPeakCount;
  int   valleyBin   = firstPeakBin;
  for (int b = firstPeakBin + 1; b < 256; b++) {
    float v = hist.at<float>(b);
    if (v < valleyCount) { valleyCount = v; valleyBin = b; }
    else { break; } // histogram turned upward — valley is behind us
  }

  // (e) Find the first local peak AFTER the valley.
  //     Walk forward from valleyBin until the histogram rises then falls;
  //     the crest of that first rise is the next cluster peak.  Using this
  //     first local maximum rather than the global maximum keeps the threshold
  //     close to the ink/background boundary and avoids being pulled all the
  //     way to a very bright background peak at the far end of the histogram.
  int   nextPeakBin   = valleyBin;
  float nextPeakCount = valleyCount;
  bool  nextAscend    = false;
  for (int b = valleyBin + 1; b < 256; b++) {
    float cur  = hist.at<float>(b);
    float prev = hist.at<float>(b - 1);
    if (cur > prev) {
      nextAscend = true;
      if (cur > nextPeakCount) { nextPeakCount = cur; nextPeakBin = b; }
    } else if (nextAscend && cur < prev) {
      break; // just past the crest of the next peak
    }
  }

  // (f) Find the first valley AFTER the next peak — same logic as (d).
  float nextValleyCount = nextPeakCount;
  int   nextValleyBin   = nextPeakBin;
  for (int b = nextPeakBin + 1; b < 256; b++) {
    float v = hist.at<float>(b);
    if (v < nextValleyCount) { nextValleyCount = v; nextValleyBin = b; }
    else { break; }
  }

  // (g) Use the valley after the next peak as the threshold.
  //     This sits at the natural low point between the two lighter clusters
  //     immediately beyond the ink region.
  int threshBin = nextValleyBin;

  // (g) Global binary threshold: pixels at or below threshBin → black ink.
  cv::Mat binary;
  cv::threshold(gray, binary, threshBin, 255, cv::THRESH_BINARY);

  // ── Step 4: force semi-transparent pixels to white ────────────────────────
  // Any pixel that failed the alpha gate in Step 1 is set to white (255)
  // regardless of what the grey-level threshold decided.
  if (!opaqueMask.empty())
    binary.setTo(255, ~opaqueMask);

  // ── Step 5: optional diagnostics ──────────────────────────────────────────
  if (diag) {
    diag->firstPeakBin  = firstPeakBin;
    diag->valleyBin     = valleyBin;
    diag->nextPeakBin   = nextPeakBin;
    diag->nextValleyBin = nextValleyBin;
    diag->threshBin     = threshBin;

    // Render a 512 × 360 histogram image (2 px per bin, sqrt-scaled bars).
    const int barH     = 300;
    const int marginT  = 40;  // room for labels above bars
    const int marginB  = 20;
    const int W        = 512;
    const int H        = marginT + barH + marginB;

    cv::Mat histImg(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

    // Find max sqrt value for scaling.
    float sqrtMax = 0.0f;
    for (int b = 0; b < 256; b++)
      sqrtMax = std::max(sqrtMax, std::sqrt(hist.at<float>(b)));
    if (sqrtMax < 1.0f) sqrtMax = 1.0f;

    // Draw grey bars (2 px wide per bin).
    for (int b = 0; b < 256; b++) {
      int h  = static_cast<int>(barH * std::sqrt(hist.at<float>(b)) / sqrtMax);
      int x0 = b * 2;
      int y0 = marginT + barH - h;
      cv::rectangle(histImg,
                    cv::Point(x0, y0),
                    cv::Point(x0 + 1, marginT + barH - 1),
                    cv::Scalar(160, 160, 160), cv::FILLED);
    }

    // Draw a thin baseline.
    cv::line(histImg,
             cv::Point(0,   marginT + barH),
             cv::Point(W-1, marginT + barH),
             cv::Scalar(0, 0, 0), 1);

    // Helper: draw a vertical coloured marker + label above bars.
    auto drawMarker = [&](int bin, cv::Scalar colour, const std::string &label) {
      int x = bin * 2 + 1;
      cv::line(histImg,
               cv::Point(x, marginT),
               cv::Point(x, marginT + barH),
               colour, 1);
      // Place label above the bar area; clamp to image edges.
      int tx = std::max(2, std::min(x - 18, W - 60));
      cv::putText(histImg, label,
                  cv::Point(tx, marginT - 4),
                  cv::FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv::LINE_AA);
    };

    // Markers: red=ink peak, green=valley1, orange=next peak,
    //          purple=valley2 (threshold), blue=threshold line.
    drawMarker(firstPeakBin,  cv::Scalar(200,   0,   0),
               "pk=" + std::to_string(firstPeakBin));
    drawMarker(valleyBin,     cv::Scalar(  0, 160,   0),
               "vl=" + std::to_string(valleyBin));
    drawMarker(nextPeakBin,   cv::Scalar(  0, 140, 255),
               "np=" + std::to_string(nextPeakBin));
    drawMarker(nextValleyBin, cv::Scalar(160,   0, 160),
               "v2=" + std::to_string(nextValleyBin));
    drawMarker(threshBin,    cv::Scalar(  0,   0, 220),
               "th=" + std::to_string(threshBin));

    diag->histImage = histImg;
  }

  return binary;
}

void OCRAnalysis::setImage(const cv::Mat &image) {
  cv::Mat rgbImage;

  // Convert to RGB if necessary (Tesseract expects RGB)
  if (image.channels() == 1) {
    cv::cvtColor(image, rgbImage, cv::COLOR_GRAY2RGB);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, rgbImage, cv::COLOR_BGRA2RGB);
  } else {
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
  }

  m_tesseract->SetImage(rgbImage.data, rgbImage.cols, rgbImage.rows, 3,
                        static_cast<int>(rgbImage.step));
}

int OCRAnalysis::findBestRotation(const cv::Mat &image) {
  if (!m_initialized || image.empty()) {
    return -1;
  }

  struct RotationResult {
    int rotationCode;
    double avgConfidence;
    int wordCount;
  };

  std::vector<RotationResult> rotationResults;

  // Define rotations to try: no rotation, 90Â° CW, 180Â°, 90Â° CCW
  std::vector<int> rotations = {-1, cv::ROTATE_90_CLOCKWISE, cv::ROTATE_180,
                                cv::ROTATE_90_COUNTERCLOCKWISE};

  tesseract::PageSegMode originalMode = m_tesseract->GetPageSegMode();

  for (int rotationCode : rotations) {
    cv::Mat testImage;
    if (rotationCode == -1) {
      testImage = image.clone();
    } else {
      cv::rotate(image, testImage, rotationCode);
    }

    setImage(testImage);
    m_tesseract->Recognize(nullptr);

    // Calculate average confidence for this rotation
    double totalConfidence = 0.0;
    int wordCount = 0;

    tesseract::ResultIterator *ri = m_tesseract->GetIterator();
    if (ri != nullptr) {
      tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
      do {
        const char *word = ri->GetUTF8Text(level);
        if (word != nullptr && *word != '\0') {
          float conf = ri->Confidence(level);
          totalConfidence += conf;
          wordCount++;
        }
        delete[] word;
      } while (ri->Next(level));
      delete ri;
    }

    double avgConf = (wordCount > 0) ? (totalConfidence / wordCount) : 0.0;
    rotationResults.push_back({rotationCode, avgConf, wordCount});
  }

  // Find the rotation with highest average confidence
  int bestRotation = -1;
  double bestConfidence = -1.0;
  for (const auto &result : rotationResults) {
    if (result.wordCount >= 1 && result.avgConfidence > bestConfidence) {
      bestConfidence = result.avgConfidence;
      bestRotation = result.rotationCode;
    }
  }

  // Restore original page segmentation mode
  m_tesseract->SetPageSegMode(originalMode);

  return bestRotation;
}

OCRAnalysis::PNGRenderResult OCRAnalysis::renderElementsToPNG(
    const PDFElements &elements, const std::string &pdfPath, double dpi,
    const std::string &outputDir, RenderBoundsMode boundsMode,
    const std::string &markToFile) {

  PNGRenderResult result;

  try {
    // Find the actual bounding box of all elements to determine the offset
    // This allows us to normalize coordinates so rendering starts at (0,0)
    // Only include elements that are within the crop box
    double minX, minY, maxX, maxY;

    if (boundsMode == RenderBoundsMode::USE_LARGEST_RECTANGLE) {
      // Use the largest rectangle to determine bounds
      // Find the largest rectangle by area
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
        std::cerr << "DEBUG: No rectangles found, checking for images..."
                  << std::endl;

        if (!elements.images.empty()) {
          // Find the largest image by area
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

            std::cerr << "DEBUG: Using largest image bounds: (" << minX << ", "
                      << minY << ") to (" << maxX << ", " << maxY << ")"
                      << " (area: " << largestImageArea << ")" << std::endl;
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
        // Use only the rectangle bounds - do not expand
        // Rectangle coords are in top-left origin (from OutputDev with
        // upsideDown()=false). Text coords are in bottom-left origin (flipped
        // in extractTextFromPDF via: y = pageHeight - bbox.y - bbox.height).
        // Convert rectangle to bottom-left to match text coordinate system.
        double rectTopLeftY = largestRect->y;
        double rectBottomLeftY = rectTopLeftY + largestRect->height;

        minX = largestRect->x;
        minY = elements.pageHeight - rectBottomLeftY; // Convert to bottom-left
        maxX = largestRect->x + largestRect->width;
        maxY = elements.pageHeight - rectTopLeftY; // Convert to bottom-left

        std::cerr << "DEBUG: Found " << elements.rectangles.size()
                  << " rectangle(s)" << std::endl;
        for (size_t i = 0; i < elements.rectangles.size(); i++) {
          const auto &rect = elements.rectangles[i];
          double area = rect.width * rect.height;
          std::cerr << "  Rectangle " << i << ": (" << rect.x << ", " << rect.y
                    << ") size: " << rect.width << "x" << rect.height
                    << " area: " << area << std::endl;
        }

        std::cerr << "DEBUG: Rectangle top-left coords: (" << largestRect->x
                  << ", " << largestRect->y << ") to ("
                  << largestRect->x + largestRect->width << ", "
                  << largestRect->y + largestRect->height << ")" << std::endl;
        std::cerr << "DEBUG: Converted to bottom-left coords (no expansion): ("
                  << minX << ", " << minY << ") to (" << maxX << ", " << maxY
                  << ")"
                  << " (area: " << largestArea << ")" << std::endl;
      }

    } else if (elements.linesBoundingBoxWidth > 0 &&
               elements.linesBoundingBoxHeight > 0) {
      // Use the bounding box from the lines/crop marks
      minX = elements.linesBoundingBoxX;
      minY = elements.linesBoundingBoxY;
      maxX = elements.linesBoundingBoxX + elements.linesBoundingBoxWidth;
      maxY = elements.linesBoundingBoxY + elements.linesBoundingBoxHeight;

      std::cerr
          << "DEBUG: Using linesBoundingBox (crop marks) as content area: ("
          << minX << ", " << minY << ") to (" << maxX << ", " << maxY << ")"
          << std::endl;
    } else {
      // Fall back to calculating bounding box from all elements
      minX = std::numeric_limits<double>::max();
      minY = std::numeric_limits<double>::max();
      maxX = std::numeric_limits<double>::lowest();
      maxY = std::numeric_limits<double>::lowest();

      for (const auto &text : elements.textLines) {
        // Convert text.boundingBox from screen coords (y-down) to PDF coords
        // (y-up) before accumulating bounds.
        double tx = text.boundingBox.x;
        double tw = text.boundingBox.width;
        double tbottom =
            elements.pageHeight - text.boundingBox.y - text.boundingBox.height;
        double ttop = elements.pageHeight - text.boundingBox.y;
        // Skip elements outside page bounds
        if (tx < elements.pageX || tbottom < elements.pageY ||
            tx + tw > elements.pageX + elements.pageWidth ||
            ttop > elements.pageY + elements.pageHeight) {
          continue;
        }
        minX = std::min(minX, tx);
        minY = std::min(minY, tbottom);
        maxX = std::max(maxX, tx + tw);
        maxY = std::max(maxY, ttop);
      }

      for (const auto &img : elements.images) {
        // Skip elements outside crop box
        if (img.x < elements.pageX || img.y < elements.pageY ||
            img.x + img.displayWidth > elements.pageX + elements.pageWidth ||
            img.y + img.displayHeight > elements.pageY + elements.pageHeight) {
          continue;
        }
        minX = std::min(minX, img.x);
        minY = std::min(minY, img.y);
        maxX = std::max(maxX, img.x + img.displayWidth);
        maxY = std::max(maxY, img.y + img.displayHeight);
      }

      for (const auto &rect : elements.rectangles) {
        // Skip elements outside crop box
        if (rect.x < elements.pageX || rect.y < elements.pageY ||
            rect.x + rect.width > elements.pageX + elements.pageWidth ||
            rect.y + rect.height > elements.pageY + elements.pageHeight) {
          continue;
        }
        minX = std::min(minX, rect.x);
        minY = std::min(minY, rect.y);
        maxX = std::max(maxX, rect.x + rect.width);
        maxY = std::max(maxY, rect.y + rect.height);
      }

      for (const auto &line : elements.graphicLines) {
        double lineMinX = std::min(line.x1, line.x2);
        double lineMaxX = std::max(line.x1, line.x2);
        double lineMinY = std::min(line.y1, line.y2);
        double lineMaxY = std::max(line.y1, line.y2);
        // Skip elements outside crop box
        if (lineMinX < 0 || lineMinY < 0 || lineMaxX > elements.pageWidth ||
            lineMaxY > elements.pageHeight) {
          continue;
        }
        minX = std::min(minX, lineMinX);
        minY = std::min(minY, lineMinY);
        maxX = std::max(maxX, lineMaxX);
        maxY = std::max(maxY, lineMaxY);
      }

      std::cerr << "DEBUG: Calculated bounding box from elements: (" << minX
                << ", " << minY << ") to (" << maxX << ", " << maxY << ")"
                << std::endl;
    }

    // If no elements found, use page dimensions with origin at (0,0)
    if (minX == std::numeric_limits<double>::max()) {
      minX = 0;
      minY = 0;
      maxX = elements.pageWidth;
      maxY = elements.pageHeight;
    }

    // Ensure we don't exceed page dimensions (crop to page bounds)
    minX = std::max(elements.pageX, minX);
    minY = std::max(elements.pageY, minY);
    maxX = std::min(elements.pageX + elements.pageWidth, maxX);
    maxY = std::min(elements.pageY + elements.pageHeight, maxY);

    std::cerr << "DEBUG: After clamping - minX=" << minX << ", maxX=" << maxX
              << ", minY=" << minY << ", maxY=" << maxY << std::endl;
    std::cerr << "DEBUG: pageWidth=" << elements.pageWidth
              << ", pageHeight=" << elements.pageHeight << std::endl;

    if (maxX <= minX || maxY <= minY) {
      std::cerr << "DEBUG: Bounding box invalid after clamping, falling back "
                   "to full page dimensions"
                << std::endl;
      minX = elements.pageX;
      minY = elements.pageY;
      maxX = elements.pageX + elements.pageWidth;
      maxY = elements.pageY + elements.pageHeight;
    }

    // Calculate dimensions in points and pixels
    // Save the crop box dimensions (for image size)
    double cropBoxMinX = minX;
    double cropBoxMinY = minY;
    double cropBoxMaxX = maxX;
    double cropBoxMaxY = maxY;

    // Find the actual bounds of elements that will be rendered
    // (within the crop box, but elements may not start at crop box edge)
    double renderMinX = std::numeric_limits<double>::max();
    double renderMinY = std::numeric_limits<double>::max();
    double renderMaxX = std::numeric_limits<double>::lowest();
    double renderMaxY = std::numeric_limits<double>::lowest();

    // Check text elements within crop box
    const double textBaselineOffset = 10.0; // Must match rendering offset
    for (const auto &text : elements.textLines) {
      double textLeft = text.boundingBox.x;
      double textTop = text.boundingBox.y;
      double textRight = text.boundingBox.x + text.boundingBox.width;
      double textBottom = text.boundingBox.y + text.boundingBox.height;

      // Include if element overlaps with crop box (not strict containment)
      // Also account for baseline offset extending below bounding box
      if (textRight > cropBoxMinX && textLeft < cropBoxMaxX &&
          textBottom > cropBoxMinY &&
          textTop - textBaselineOffset < cropBoxMaxY) {
        renderMinX = std::min(renderMinX, textLeft);
        renderMinY = std::min(renderMinY, textTop - textBaselineOffset);
        renderMaxX = std::max(renderMaxX, textRight);
        renderMaxY = std::max(renderMaxY, textBottom);
      }
    }

    // Check image elements within crop box
    for (const auto &img : elements.images) {
      double imgLeft = img.x;
      double imgTop = img.y;
      double imgRight = img.x + img.displayWidth;
      double imgBottom = img.y + img.displayHeight;

      // Include if element overlaps with crop box
      if (imgRight > cropBoxMinX && imgLeft < cropBoxMaxX &&
          imgBottom > cropBoxMinY && imgTop < cropBoxMaxY) {
        renderMinX = std::min(renderMinX, imgLeft);
        renderMinY = std::min(renderMinY, imgTop);
        renderMaxX = std::max(renderMaxX, imgRight);
        renderMaxY = std::max(renderMaxY, imgBottom);
      }
    }

    // Use crop box from crop marks for positioning
    // Crop box rendered from (0,0) with elements positioned relative to it
    minX = cropBoxMinX;
    minY = cropBoxMinY;
    maxX = cropBoxMaxX;
    maxY = cropBoxMaxY;

    std::cerr << "DEBUG: Final content area: (" << minX << ", " << minY
              << ") to (" << maxX << ", " << maxY << ")"
              << (elements.hasCropMarks ? " [from crop marks]"
                                        : " [from largest rect/elements]")
              << std::endl;

    // Perform OCR on embedded images ONLY if crop marks were NOT detected.
    // When crop marks are present, images are treated as purely visual
    // elements to render rather than sources of additional text.
    std::vector<TextRegion> ocrTextLines;
    if (elements.hasCropMarks) {
      if (!elements.images.empty()) {
        std::cerr << "DEBUG: Crop marks detected â€” skipping OCR on "
                  << elements.images.size()
                  << " embedded image(s); will render as-is" << std::endl;
      }
    } else if (!elements.images.empty()) {
      std::cerr << "DEBUG: Performing OCR on " << elements.images.size()
                << " embedded image(s)..." << std::endl;

      // Initialize OCR engine if not already initialized
      if (!m_initialized) {
        std::cerr << "DEBUG: Initializing OCR engine..." << std::endl;
        if (!initialize()) {
          std::cerr << "DEBUG: Failed to initialize OCR engine, skipping OCR"
                    << std::endl;
        }
      }

      // Only proceed with OCR if engine is initialized
      if (m_initialized) {
        for (const auto &pdfImage : elements.images) {
          if (pdfImage.image.empty()) {
            continue;
          }

          // Perform OCR on the image
          std::cerr << "DEBUG: Performing OCR on image " << pdfImage.image.cols
                    << "x" << pdfImage.image.rows << " pixels..." << std::endl;
          OCRResult ocrResult = analyzeImage(pdfImage.image);

          if (!ocrResult.success) {
            std::cerr << "DEBUG: OCR failed: " << ocrResult.errorMessage
                      << std::endl;
          } else if (ocrResult.regions.empty()) {
            std::cerr << "DEBUG: OCR succeeded but found no text regions"
                      << std::endl;
          } else {
            std::cerr << "DEBUG: OCR found " << ocrResult.regions.size()
                      << " text regions in image at (" << pdfImage.x << ", "
                      << pdfImage.y << ")" << std::endl;

            // Convert OCR results to PDF coordinates
            // OCR coordinates are in pixels relative to the image
            // We need to convert them to PDF points relative to the page

            // Calculate scale factor from image pixels to PDF points
            double scaleX = pdfImage.displayWidth / pdfImage.image.cols;
            double scaleY = pdfImage.displayHeight / pdfImage.image.rows;

            for (const auto &region : ocrResult.regions) {
              TextRegion textRegion;
              textRegion.text = region.text;
              textRegion.confidence = region.confidence;
              textRegion.orientation = region.orientation;

              // Convert pixel coordinates to PDF points
              // Note: OCR uses top-left origin, PDF images use bottom-left
              // origin
              textRegion.boundingBox.x =
                  pdfImage.x + (region.boundingBox.x * scaleX);
              textRegion.boundingBox.y =
                  pdfImage.y + (region.boundingBox.y * scaleY);
              textRegion.boundingBox.width = region.boundingBox.width * scaleX;
              textRegion.boundingBox.height =
                  region.boundingBox.height * scaleY;

              ocrTextLines.push_back(textRegion);
            }
          }
        }

        if (!ocrTextLines.empty()) {
          std::cerr << "DEBUG: Total OCR text regions extracted: "
                    << ocrTextLines.size() << std::endl;
        }
      } // end if (m_initialized)
    } // end if (!elements.images.empty())

    // -----------------------------------------------------------
    // Use DataMatrix barcodes already detected in extractPDFElements
    // -----------------------------------------------------------
    // dataMatrixZones tracks bounding boxes in PDF coordinates so that
    // overlapping OCR text can be excluded later.
    struct DMZone {
      double left, top, right, bottom; // PDF coordinates
    };
    std::vector<DMZone> dataMatrixZones;

    for (const auto &dm : elements.dataMatrices) {
      dataMatrixZones.push_back(
          {dm.x, dm.y, dm.x + dm.width, dm.y + dm.height});

      // Store a DATAMATRIX rendered element; its relative
      // coordinates are computed later once imageWidth/imageHeight
      // are determined.  For now keep the raw PDF coordinates in a
      // temporary struct so we can convert after the image size is set.
      RenderedElement elem;
      elem.type = RenderedElement::DATAMATRIX;
      elem.barcodeText = dm.text;
      elem.image = dm.image;
      // Temporarily store PDF coords â€” will be converted to relative
      // after imageWidth/imageHeight are known.
      elem.relativeX = dm.x + dm.width / 2.0;  // centre X in PDF pts
      elem.relativeY = dm.y + dm.height / 2.0; // centre Y in PDF pts
      elem.relativeWidth = dm.width;           // width in PDF pts
      elem.relativeHeight = dm.height;         // height in PDF pts
      result.elements.push_back(elem);

      std::cerr << "DEBUG: Using pre-detected DataMatrix \""
                << dm.text.substr(0, 30) << "\" at PDF (" << dm.x << ", "
                << dm.y << ") size " << dm.width << "x" << dm.height
                << std::endl;
    }

    // Remove OCR text lines that overlap with DataMatrix zones
    if (!dataMatrixZones.empty() && !ocrTextLines.empty()) {
      auto before = ocrTextLines.size();
      ocrTextLines.erase(
          std::remove_if(ocrTextLines.begin(), ocrTextLines.end(),
                         [&dataMatrixZones](const TextRegion &tr) {
                           double cx =
                               tr.boundingBox.x + tr.boundingBox.width / 2.0;
                           double cy =
                               tr.boundingBox.y + tr.boundingBox.height / 2.0;
                           for (const auto &zone : dataMatrixZones) {
                             if (cx >= zone.left && cx <= zone.right &&
                                 cy >= zone.top && cy <= zone.bottom) {
                               return true; // overlaps â€” remove
                             }
                           }
                           return false;
                         }),
          ocrTextLines.end());
      auto removed = before - ocrTextLines.size();
      if (removed > 0) {
        std::cerr << "DEBUG: Removed " << removed
                  << " OCR text line(s) overlapping DataMatrix zone(s)"
                  << std::endl;
      }
    }

    // Use crop box dimensions for image size
    const double margin = 0.0;
    double pageWidthPt = maxX - minX;
    double pageHeightPt = maxY - minY;

    // Convert to pixels based on DPI (72 points = 1 inch)
    double scale = dpi / 72.0;
    int imageWidth = static_cast<int>(pageWidthPt * scale);
    int imageHeight = static_cast<int>(pageHeightPt * scale);

    result.imageWidth = imageWidth;
    result.imageHeight = imageHeight;

    // Now convert any DATAMATRIX elements from temporary PDF coords
    // to proper relative centre-point coordinates.
    for (auto &elem : result.elements) {
      if (elem.type != RenderedElement::DATAMATRIX)
        continue;
      // elem.relativeX/Y currently hold PDF centre coords;
      // elem.relativeWidth/Height hold PDF extent.
      double pdfCX = elem.relativeX;
      double pdfCY = elem.relativeY;
      double pdfW = elem.relativeWidth;
      double pdfH = elem.relativeHeight;

      // Convert to rendering coords â€” these are derived from image
      // pixel space (top-left origin) so NO Y-flip is needed.
      double renderCX = pdfCX - minX;
      double renderCY = pdfCY - minY; // already top-left origin
      double pxCX = renderCX * scale;
      double pxCY = renderCY * scale;
      double pxW = pdfW * scale;
      double pxH = pdfH * scale;

      elem.relativeX = pxCX / imageWidth;
      elem.relativeY = pxCY / imageHeight;
      elem.relativeWidth = pxW / imageWidth;
      elem.relativeHeight = pxH / imageHeight;
    }

    // Create output filename
    std::filesystem::path pdfFilePath(pdfPath);
    std::string baseName = pdfFilePath.stem().string();

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(outputDir);

    std::string outputPath = outputDir + "/" + baseName + "_rendered.png";
    result.outputPath = outputPath;

    std::cerr << "Rendering to PNG: " << outputPath << std::endl;
    std::cerr << "  DPI: " << dpi << ", Scale: " << scale << std::endl;
    std::cerr << "  Page dimensions: " << pageWidthPt << " x " << pageHeightPt
              << " pt" << std::endl;
    std::cerr << "  Image size: " << imageWidth << "x" << imageHeight
              << " pixels" << std::endl;

#ifdef HAVE_CAIRO

    // ---- Full-page rasterization path for crop-mark PDFs ----------
    // When crop marks are detected, use Poppler's SplashOutputDev to
    // rasterize the entire page at the target DPI, then crop to the
    // crop box.  This captures EVERYTHING â€” vector graphics, images,
    // text â€” without relying on element-by-element reconstruction.
    if (elements.hasCropMarks) {
      std::cerr << "DEBUG: Using SplashOutputDev full-page rasterization "
                   "(crop marks mode)"
                << std::endl;

      try {
        GlobalParamsIniter globalParamsInit(nullptr);

        auto fileName = std::make_unique<GooString>(pdfPath);
        std::unique_ptr<PDFDoc> doc(new PDFDoc(std::move(fileName)));

        if (!doc->isOk()) {
          result.errorMessage = "Failed to load PDF for rasterization";
          return result;
        }

        // Create a SplashOutputDev that renders to RGB
        SplashColor paperColor;
        paperColor[0] = 255;
        paperColor[1] = 255;
        paperColor[2] = 255;
        SplashOutputDev splashOut(splashModeRGB8, 4, false, paperColor);
        splashOut.startDoc(doc.get());

        // Render page 1 at the requested DPI
        doc->displayPage(&splashOut, 1, dpi, dpi, 0, true, false, false);

        SplashBitmap *bitmap = splashOut.getBitmap();
        if (!bitmap) {
          result.errorMessage = "SplashOutputDev returned null bitmap";
          return result;
        }

        int bmpW = bitmap->getWidth();
        int bmpH = bitmap->getHeight();
        int bmpRowSize = bitmap->getRowSize();

        std::cerr << "DEBUG: Full page raster: " << bmpW << "x" << bmpH
                  << " pixels (row size " << bmpRowSize << ")" << std::endl;

        // Convert SplashBitmap (RGB) â†’ OpenCV Mat (BGR)
        cv::Mat fullPage(bmpH, bmpW, CV_8UC3);
        unsigned char *splashData = bitmap->getDataPtr();
        for (int row = 0; row < bmpH; row++) {
          const unsigned char *src = splashData + row * bmpRowSize;
          unsigned char *dst = fullPage.ptr<unsigned char>(row);
          for (int col = 0; col < bmpW; col++) {
            dst[col * 3 + 0] = src[col * 3 + 2]; // B
            dst[col * 3 + 1] = src[col * 3 + 1]; // G
            dst[col * 3 + 2] = src[col * 3 + 0]; // R
          }
        }

        // Crop to the crop box.
        // minX/minY/maxX/maxY are in PDF points (origin top-left in
        // Poppler's coordinate system when upsideDown() is true, which
        // SplashOutputDev defaults to).
        // Poppler renders with (0,0) at the top-left of the MediaBox.
        // PDF coordinates have (0,0) at bottom-left, but the text/image
        // extraction already converts to top-left.  However, the crop
        // box values we have (minX, minY) are in the original PDF
        // coordinate system (bottom-left origin), so we need to convert Y.
        double fullPageHeightPt = elements.pageY + elements.pageHeight;
        int cropX = static_cast<int>((minX - elements.pageX) * scale);
        int cropY = static_cast<int>((minY - elements.pageY) * scale);
        int cropW = static_cast<int>((maxX - minX) * scale);
        int cropH = static_cast<int>((maxY - minY) * scale);

        // Clamp to bitmap bounds
        cropX = std::max(0, std::min(cropX, bmpW - 1));
        cropY = std::max(0, std::min(cropY, bmpH - 1));
        cropW = std::min(cropW, bmpW - cropX);
        cropH = std::min(cropH, bmpH - cropY);

        std::cerr << "DEBUG: Crop region: x=" << cropX << ", y=" << cropY
                  << ", w=" << cropW << ", h=" << cropH << std::endl;

        cv::Mat cropped =
            fullPage(cv::Rect(cropX, cropY, cropW, cropH)).clone();

        // Write PNG
        cv::imwrite(outputPath, cropped);

        std::cerr << "PNG rendered successfully (rasterised): " << outputPath
                  << std::endl;
        std::cerr << "  Final image: " << cropped.cols << "x" << cropped.rows
                  << " pixels" << std::endl;

        // Update result dimensions
        result.imageWidth = cropped.cols;
        result.imageHeight = cropped.rows;

        // Add IMAGE elements for every entry in elements.images (which
        // includes both embedded raster images and vector graphics detected
        // by extractPDFElements).  Save a high-DPI crop from the rasterised
        // page so callers can verify the returned images are at render DPI.
        int savedImageCount = 0;
        for (const auto &img : elements.images) {
          // Convert PDF coords to top-left crop-relative pixels.
          double imgPxX = (img.x - minX) * scale;
          double imgPxY =
              (pageHeightPt - (img.y - minY + img.displayHeight)) * scale;
          double imgPxW = img.displayWidth * scale;
          double imgPxH = img.displayHeight * scale;

          // Skip elements that fall entirely outside the rendered area
          if (imgPxX + imgPxW <= 0 || imgPxX >= result.imageWidth ||
              imgPxY + imgPxH <= 0 || imgPxY >= result.imageHeight)
            continue;

          // Crop the rasterised image at render DPI for this element
          int cx = std::max(0, static_cast<int>(imgPxX));
          int cy = std::max(0, static_cast<int>(imgPxY));
          int cw = std::min(static_cast<int>(std::ceil(imgPxW)),
                            result.imageWidth - cx);
          int ch = std::min(static_cast<int>(std::ceil(imgPxH)),
                            result.imageHeight - cy);

          cv::Mat imgCrop;
          if (cw > 0 && ch > 0) {
            imgCrop = cropped(cv::Rect(cx, cy, cw, ch)).clone();
            if (!outputDir.empty()) {
              std::string imgSavePath =
                  outputDir + "/" + baseName + "_rendered_image_" +
                  std::to_string(++savedImageCount) + ".png";
              cv::imwrite(imgSavePath, imgCrop);
              std::cerr << "DEBUG: Saved rendered image crop: " << imgSavePath
                        << std::endl;
            }
          } else {
            imgCrop = img.image.clone();
          }

          RenderedElement imgElem;
          imgElem.type = RenderedElement::IMAGE;
          imgElem.relativeX = (imgPxX + imgPxW / 2.0) / result.imageWidth;
          imgElem.relativeY = (imgPxY + imgPxH / 2.0) / result.imageHeight;
          imgElem.relativeWidth = imgPxW / result.imageWidth;
          imgElem.relativeHeight = imgPxH / result.imageHeight;
          imgElem.image = imgCrop;
          imgElem.rotationAngle = img.rotationAngle;
          result.elements.push_back(imgElem);
        }

        // Add TEXT elements from elements.textLines.
        // text.boundingBox stores PDF bottom-left coordinates
        // (y = bottom edge of text, same space as images and crop-mark bounds).
        // Use the same formula as for image elements to convert to pixels.
        for (const auto &text : elements.textLines) {
          double textLeft = text.boundingBox.x;
          double textRight = textLeft + text.boundingBox.width;
          double textBottom = text.boundingBox.y; // PDF y-up bottom
          double textTop = textBottom + text.boundingBox.height; // PDF y-up top

          // Skip text outside the crop area (PDF bottom-left coords)
          if (textLeft < minX || textRight > maxX || textBottom < minY ||
              textTop > maxY)
            continue;

          // Convert PDF bottom-left to top-left crop-relative pixels.
          // Same formula as images:
          //   pxY_top = (pageHeightPt - (textBottom - minY + height)) * scale
          double pxX = (textLeft - minX) * scale;
          double pxY =
              (pageHeightPt - (textBottom - minY + text.boundingBox.height)) *
              scale;
          double pxW = text.boundingBox.width * scale;
          double pxH = text.boundingBox.height * scale;

          RenderedElement textElem;
          textElem.type = RenderedElement::TEXT;
          textElem.relativeX = (pxX + pxW / 2.0) / result.imageWidth;
          textElem.relativeY = (pxY + pxH / 2.0) / result.imageHeight;
          textElem.relativeWidth = pxW / result.imageWidth;
          textElem.relativeHeight = pxH / result.imageHeight;
          textElem.text = text.text;
          textElem.fontName = text.fontName.empty() ? "Sans" : text.fontName;
          textElem.fontSize = (text.fontSize > 0) ? text.fontSize * 0.75 : 10.0;
          textElem.isBold = text.isBold;
          textElem.isItalic = text.isItalic;
          result.elements.push_back(textElem);
        }

        // Draw bounding boxes for all elements and save alongside the plain
        // render.
        {
          cv::Mat annotated = drawElementBoxes(cropped, result.elements);
          std::string annotPath = outputDir + "/" + baseName + "_annotated.png";
          cv::imwrite(annotPath, annotated);
          std::cerr << "DEBUG: Saved annotated image: " << annotPath
                    << std::endl;
        }
        result.success = true;
        return result;

      } catch (const std::exception &e) {
        std::cerr << "WARNING: SplashOutputDev rasterisation failed ("
                  << e.what() << "), falling back to Cairo" << std::endl;
        // Fall through to Cairo element-by-element rendering
      }
    }
    // ---- End SplashOutputDev path ---------------------------------

    // Create Cairo image surface
    cairo_surface_t *surface = cairo_image_surface_create(
        CAIRO_FORMAT_ARGB32, imageWidth, imageHeight);

    if (cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS) {
      result.errorMessage = "Failed to create Cairo image surface";
      cairo_surface_destroy(surface);
      return result;
    }

    cairo_t *cr = cairo_create(surface);

    // Scale to match DPI
    cairo_scale(cr, scale, scale);

    // Fill with white background
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
    cairo_paint(cr);

    // Draw rectangles (PDF bottom-left origin -> convert to top-left)
    // DISABLED: Rectangles are structural/decorative elements (borders, crop
    // marks) They should not be rendered in content extraction - only text
    // and images matter
    /*
      cairo_set_source_rgb(cr, 0.8, 0.8, 0.8);
      cairo_set_line_width(cr, 1.0 / scale);
      for (const auto &rect : elements.rectangles) {
        // Only render rectangles that are mostly inside the content area
        // This prevents crop mark rectangles from being rendered
        double rectLeft = std::max(rect.x, minX);
        double rectTop = std::max(rect.y, minY);
        double rectRight = std::min(rect.x + rect.width, maxX);
        double rectBottom = std::min(rect.y + rect.height, maxY);

        // Skip if rectangle is completely outside content area
        if (rectLeft >= rectRight || rectTop >= rectBottom) {
          continue;
        }

        // Calculate what percentage of the rectangle is inside the content
      area double clippedArea = (rectRight - rectLeft) * (rectBottom -
      rectTop); double totalArea = rect.width * rect.height; double
      insideRatio = clippedArea / totalArea;

        // Only render if more than 50% of the rectangle is inside
        if (insideRatio <= 0.5) {
          continue;
        }

        // Use clipped dimensions for rendering
        double clippedX = rectLeft;
        double clippedY = rectTop;
        double clippedWidth = rectRight - rectLeft;
        double clippedHeight = rectBottom - rectTop;

        double x = clippedX - minX + margin;
        // Convert from PDF bottom-left to Cairo top-left
        double y = pageHeightPt - (clippedY - minY + clippedHeight) - margin;
        cairo_rectangle(cr, x, y, clippedWidth, clippedHeight);
        cairo_stroke(cr);

        // Add to result -- centre-point relative coordinates
        double pxX = x * scale;
        double pxY = y * scale;
        double pxW = clippedWidth * scale;
        double pxH = clippedHeight * scale;

        RenderedElement elem;
        elem.type = RenderedElement::RECTANGLE;
        elem.relativeX = (pxX + pxW / 2.0) / imageWidth;
        elem.relativeY = (pxY + pxH / 2.0) / imageHeight;
        elem.relativeWidth = pxW / imageWidth;
        elem.relativeHeight = pxH / imageHeight;
        result.elements.push_back(elem);
      }
      */

    // Draw lines (PDF bottom-left origin -> convert to top-left)
    // Only render lines within the content area
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    cairo_set_line_width(cr, 0.5 / scale);
    for (const auto &line : elements.graphicLines) {
      double lx1 = line.x1;
      double ly1 = line.y1;
      double lx2 = line.x2;
      double ly2 = line.y2;

      double lineMinX = std::min(lx1, lx2);
      double lineMaxX = std::max(lx1, lx2);
      double lineMinY = std::min(ly1, ly2);
      double lineMaxY = std::max(ly1, ly2);

      // Skip if line is outside the content area (strict filtering)
      if (lineMinX < minX || lineMaxX > maxX || lineMinY < minY ||
          lineMaxY > maxY) {
        continue;
      }

      double x1 = lx1 - minX + margin;
      double x2 = lx2 - minX + margin;
      // Convert from PDF bottom-left to Cairo top-left
      double y1 = pageHeightPt - (ly1 - minY) - margin;
      double y2 = pageHeightPt - (ly2 - minY) - margin;
      cairo_move_to(cr, x1, y1);
      cairo_line_to(cr, x2, y2);
      cairo_stroke(cr);

      // Add to result -- start/end relative coordinates
      double pxX1 = x1 * scale;
      double pxY1 = y1 * scale;
      double pxX2 = x2 * scale;
      double pxY2 = y2 * scale;

      RenderedElement elem;
      elem.type = RenderedElement::LINE;
      elem.relativeX = pxX1 / imageWidth;
      elem.relativeY = pxY1 / imageHeight;
      elem.relativeX2 = pxX2 / imageWidth;
      elem.relativeY2 = pxY2 / imageHeight;
      elem.relativeWidth = std::abs(pxX2 - pxX1) / imageWidth;
      elem.relativeHeight = std::abs(pxY2 - pxY1) / imageHeight;
      result.elements.push_back(elem);
    }

    // Draw images (PDF bottom-left origin -> convert to top-left)
    // Only render images when crop marks define the content area;
    // in that case images are purely visual and not OCR'd.
    if (elements.hasCropMarks) {
      for (const auto &img : elements.images) {
        // Clip image to content bounding box
        double imgLeft = std::max(img.x, minX);
        double imgTop = std::max(img.y, minY);
        double imgRight = std::min(img.x + img.displayWidth, maxX);
        double imgBottom = std::min(img.y + img.displayHeight, maxY);

        std::cerr << "DEBUG: Image at (" << img.x << ", " << img.y
                  << "), size: " << img.displayWidth << " x "
                  << img.displayHeight << ", clipped bounds: (" << imgLeft
                  << ", " << imgTop << ") to (" << imgRight << ", " << imgBottom
                  << ")" << std::endl;

        // Skip if image is completely outside content area
        if (imgLeft >= imgRight || imgTop >= imgBottom) {
          std::cerr << "DEBUG: Skipping image - outside content area"
                    << std::endl;
          continue;
        }

        double x = img.x - minX + margin;
        // Convert from PDF bottom-left to Cairo top-left
        double y = pageHeightPt - (img.y - minY + img.displayHeight) - margin;

        if (!img.image.empty()) {
          // Convert cv::Mat to Cairo surface and draw
          cairo_save(cr);

          // For rotated images, we need special handling
          bool is90DegRotation =
              std::abs(std::abs(img.rotationAngle) - M_PI / 2.0) < 0.1;

          if (is90DegRotation) {
            cairo_translate(cr, x, y);
            cairo_translate(cr, img.displayWidth / 2.0,
                            img.displayHeight / 2.0);
            cairo_rotate(cr, -img.rotationAngle);

            double scaleX =
                img.displayHeight / static_cast<double>(img.image.cols);
            double scaleY =
                img.displayWidth / static_cast<double>(img.image.rows);
            cairo_scale(cr, scaleX, scaleY);
            cairo_translate(cr, -img.image.cols / 2.0, -img.image.rows / 2.0);
          } else {
            cairo_translate(cr, x, y);
            double scaleX =
                img.displayWidth / static_cast<double>(img.image.cols);
            double scaleY =
                img.displayHeight / static_cast<double>(img.image.rows);
            cairo_scale(cr, scaleX, scaleY);
          }

          // Convert BGR to RGB
          cv::Mat rgbImage;
          if (img.image.channels() == 1) {
            cv::cvtColor(img.image, rgbImage, cv::COLOR_GRAY2RGB);
          } else if (img.image.channels() == 3) {
            cv::cvtColor(img.image, rgbImage, cv::COLOR_BGR2RGB);
          } else if (img.image.channels() == 4) {
            cv::cvtColor(img.image, rgbImage, cv::COLOR_BGRA2RGB);
          } else {
            cairo_restore(cr);
            continue;
          }

          // Create Cairo surface from RGB image
          cairo_surface_t *imgSurface = cairo_image_surface_create(
              CAIRO_FORMAT_RGB24, rgbImage.cols, rgbImage.rows);
          unsigned char *data = cairo_image_surface_get_data(imgSurface);
          int stride = cairo_image_surface_get_stride(imgSurface);

          // Copy pixel data (Cairo uses BGRA on little-endian)
          for (int row = 0; row < rgbImage.rows; row++) {
            for (int col = 0; col < rgbImage.cols; col++) {
              cv::Vec3b pixel = rgbImage.at<cv::Vec3b>(row, col);
              int offset = row * stride + col * 4;
              data[offset + 0] = pixel[2]; // B
              data[offset + 1] = pixel[1]; // G
              data[offset + 2] = pixel[0]; // R
              data[offset + 3] = 255;      // A
            }
          }

          cairo_surface_mark_dirty(imgSurface);
          cairo_set_source_surface(cr, imgSurface, 0, 0);
          cairo_paint(cr);
          cairo_surface_destroy(imgSurface);
          cairo_restore(cr);
        }

        // Add to result -- centre-point relative coordinates
        double pxX = x * scale;
        double pxY = y * scale;
        double pxW = img.displayWidth * scale;
        double pxH = img.displayHeight * scale;

        RenderedElement elem;
        elem.type = RenderedElement::IMAGE;
        elem.relativeX = (pxX + pxW / 2.0) / imageWidth;
        elem.relativeY = (pxY + pxH / 2.0) / imageHeight;
        elem.relativeWidth = pxW / imageWidth;
        elem.relativeHeight = pxH / imageHeight;
        elem.image = img.image.clone();
        elem.rotationAngle = img.rotationAngle;
        result.elements.push_back(elem);
      }
    }

    // Draw text (PDF bottom-left origin -> convert to top-left)
    // Only render text within the crop box
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);

    for (const auto &text : elements.textLines) {
      // text.boundingBox.y is screen y of the TOP of the text box (y-down).
      // Convert to PDF bottom-left coords (y-up) for filtering and positioning.
      double pdfTextLeft = text.boundingBox.x;
      double pdfTextRight = text.boundingBox.x + text.boundingBox.width;
      double pdfTextBottom =
          elements.pageHeight - text.boundingBox.y - text.boundingBox.height;
      double pdfTextTop = elements.pageHeight - text.boundingBox.y;

      // Skip text outside the content area (strict containment)
      if (pdfTextLeft < minX || pdfTextRight > maxX || pdfTextBottom < minY ||
          pdfTextTop > maxY) {
        std::cerr << "DEBUG: Filtering out text \"" << text.text.substr(0, 20)
                  << "\" pdfBox=(" << pdfTextLeft << "," << pdfTextBottom
                  << ") to (" << pdfTextRight << "," << pdfTextTop
                  << "), bounds: (" << minX << "," << minY << ") to (" << maxX
                  << "," << maxY << ")" << std::endl;
        continue;
      }

      double x = pdfTextLeft - minX + margin;
      // Cairo y of text baseline = distance from content top to text bottom
      double y = pageHeightPt - (pdfTextBottom - minY) + margin;

      std::cerr << "DEBUG: Rendering text at (" << x << ", " << y
                << "), original PDF pos: (" << text.boundingBox.x << ", "
                << text.boundingBox.y
                << "), bbox width: " << text.boundingBox.width << ", text: \""
                << text.text.substr(0, 20) << "\", font: " << text.fontName
                << " " << text.fontSize << "pt" << (text.isBold ? " bold" : "")
                << (text.isItalic ? " italic" : "") << std::endl;

      // Set font based on text region properties
      std::string fontFamily = text.fontName.empty() ? "Sans" : text.fontName;
      // Scale font size down - bounding box height includes
      // ascenders/descenders so it's larger than the actual font size. Use
      // 0.75 as a scaling factor.
      double fontSize = (text.fontSize > 0) ? (text.fontSize * 0.75) : 10.0;

      // Determine Cairo font slant
      cairo_font_slant_t slant =
          text.isItalic ? CAIRO_FONT_SLANT_ITALIC : CAIRO_FONT_SLANT_NORMAL;

      // Determine Cairo font weight
      cairo_font_weight_t weight =
          text.isBold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL;

      cairo_select_font_face(cr, fontFamily.c_str(), slant, weight);
      cairo_set_font_size(cr, fontSize);

      cairo_move_to(cr, x, y);
      cairo_show_text(cr, text.text.c_str());

      // Add to result -- centre-point relative coordinates.
      // pxY is the pixel y of the baseline (bottom of text box); centre is
      // pxY - pxH/2 (moving upward by half the height).
      double pxX = x * scale;
      double pxY = y * scale;
      double pxW = text.boundingBox.width * scale;
      double pxH = text.boundingBox.height * scale;

      RenderedElement elem;
      elem.type = RenderedElement::TEXT;
      elem.relativeX = (pxX + pxW / 2.0) / imageWidth;
      elem.relativeY = (pxY - pxH / 2.0) / imageHeight;
      elem.relativeWidth = pxW / imageWidth;
      elem.relativeHeight = pxH / imageHeight;
      elem.text = text.text;
      elem.fontName = fontFamily;
      elem.fontSize = fontSize;
      elem.isBold = text.isBold;
      elem.isItalic = text.isItalic;
      result.elements.push_back(elem);
    }

    // Draw OCR text from images (if any)
    for (const auto &text : ocrTextLines) {
      // Skip if text is outside the content area
      double textLeft = text.boundingBox.x;
      double textTop = text.boundingBox.y;
      double textRight = text.boundingBox.x + text.boundingBox.width;
      double textBottom = text.boundingBox.y + text.boundingBox.height;

      if (textLeft < minX || textRight > maxX || textTop < minY ||
          textBottom > maxY) {
        continue;
      }

      double x = text.boundingBox.x - minX + margin;
      // OCR coordinates are already in top-left origin (from image pixel
      // space), so NO Y-flip is needed â€” unlike PDF text which uses
      // bottom-left origin.
      double y = text.boundingBox.y - minY + margin;

      std::cerr << "DEBUG: Rendering OCR text at (" << x << ", " << y
                << "), text: \"" << text.text.substr(0, 20) << "\""
                << std::endl;

      // Use default font for OCR text
      std::string fontFamily = "Sans";
      double fontSize = 10.0;

      cairo_select_font_face(cr, fontFamily.c_str(), CAIRO_FONT_SLANT_NORMAL,
                             CAIRO_FONT_WEIGHT_NORMAL);
      cairo_set_font_size(cr, fontSize);

      cairo_move_to(cr, x, y);
      cairo_show_text(cr, text.text.c_str());

      // Add to result -- centre-point relative coordinates
      double pxX = x * scale;
      double pxY = y * scale;
      double pxW = text.boundingBox.width * scale;
      double pxH = text.boundingBox.height * scale;

      RenderedElement elem;
      elem.type = RenderedElement::TEXT;
      elem.relativeX = (pxX + pxW / 2.0) / imageWidth;
      elem.relativeY = (pxY + pxH / 2.0) / imageHeight;
      elem.relativeWidth = pxW / imageWidth;
      elem.relativeHeight = pxH / imageHeight;
      elem.text = text.text;
      elem.fontName = fontFamily;
      elem.fontSize = fontSize;
      elem.isBold = false;
      elem.isItalic = false;
      elem.ocrConfidence = text.confidence;
      result.elements.push_back(elem);
    }

    // Write to PNG
    cairo_surface_write_to_png(surface, outputPath.c_str());

    cairo_destroy(cr);
    cairo_surface_destroy(surface);

    std::cerr << "PNG rendered successfully: " << outputPath << std::endl;
    std::cerr << "  Total elements: " << result.elements.size() << std::endl;

    // Marking is now handled by the separate alignAndMarkElements() function
    // which provides OCR-aligned bounding boxes

    result.success = true;
    return result;

#else
    result.errorMessage =
        "Cairo not available - PNG rendering requires Cairo library";
    return result;
#endif
  } catch (const std::exception &e) {
    result.errorMessage = std::string("Error rendering PNG: ") + e.what();
    return result;
  }
}

void OCRAnalysis::sortByPosition(PNGRenderResult &result) {
  // Sort elements by position: top to bottom, left to right
  // Elements on the same horizontal line (similar Y) are sorted by X
  std::sort(result.elements.begin(), result.elements.end(),
            [](const RenderedElement &a, const RenderedElement &b) {
              // Define tolerance for "same line" â€“ fraction of image height
              // (roughly equivalent to ~5px on a 1000px image)
              const double Y_TOLERANCE = 0.005;

              // If Y positions are similar (within tolerance), sort by X
              if (std::abs(a.relativeY - b.relativeY) <= Y_TOLERANCE) {
                return a.relativeX < b.relativeX;
              }

              // Otherwise, sort by Y (top to bottom)
              return a.relativeY < b.relativeY;
            });
}

cv::Mat
OCRAnalysis::drawElementBoxes(const cv::Mat &image,
                              const std::vector<RenderedElement> &elements) {
  cv::Mat annotated = image.clone();
  const int W = annotated.cols;
  const int H = annotated.rows;

  for (const auto &elem : elements) {
    int pw = static_cast<int>(elem.relativeWidth * W);
    int ph = static_cast<int>(elem.relativeHeight * H);
    int rx = static_cast<int>(elem.relativeX * W) - pw / 2;
    int ry = static_cast<int>(elem.relativeY * H) - ph / 2;
    rx = std::max(0, rx);
    ry = std::max(0, ry);
    pw = std::min(pw, W - rx);
    ph = std::min(ph, H - ry);
    if (pw <= 0 || ph <= 0)
      continue;

    cv::Scalar color;
    if (elem.type == RenderedElement::TEXT)
      color = cv::Scalar(0, 200, 0); // green  â€” text
    else if (elem.type == RenderedElement::IMAGE)
      color = cv::Scalar(200, 0, 0); // blue   â€” image
    else
      color = cv::Scalar(0, 0, 200); // red    â€” datamatrix

    cv::rectangle(annotated, cv::Rect(rx, ry, pw, ph), color, 2);
  }
  return annotated;
}

bool OCRAnalysis::alignAndMarkElements(const std::string &renderedImagePath,
                                       const std::string &originalImagePath,
                                       const PNGRenderResult &renderResult,
                                       const std::string &outputPath) {
  try {
    // Load the original image for OCR analysis and marking
    cv::Mat originalImage = cv::imread(originalImagePath, cv::IMREAD_COLOR);
    if (originalImage.empty()) {
      std::cerr << "ERROR: Could not load original image: " << originalImagePath
                << std::endl;
      return false;
    }

    // Find the first text element in the render result
    const RenderedElement *firstTextElement = nullptr;
    for (const auto &elem : renderResult.elements) {
      if (elem.type == RenderedElement::TEXT && !elem.text.empty()) {
        firstTextElement = &elem;
        break;
      }
    }

    if (!firstTextElement) {
      std::cerr << "ERROR: No text elements found in render result"
                << std::endl;
      return false;
    }

    std::cerr << "First text element: \"" << firstTextElement->text
              << "\" at rel(" << firstTextElement->relativeX << ", "
              << firstTextElement->relativeY << ")" << std::endl;

    // Calculate scale factors if images have different dimensions
    double scaleX =
        static_cast<double>(originalImage.cols) / renderResult.imageWidth;
    double scaleY =
        static_cast<double>(originalImage.rows) / renderResult.imageHeight;

    std::cerr << "Image dimensions - Rendered: " << renderResult.imageWidth
              << "x" << renderResult.imageHeight
              << ", Original: " << originalImage.cols << "x"
              << originalImage.rows << std::endl;
    std::cerr << "Scale factors: X=" << scaleX << ", Y=" << scaleY << std::endl;

    // Use Tesseract OCR on the original image in WORD mode
    tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
    // Try to initialize with explicit tessdata path first, then fall back to
    // default
    if (ocr->Init("C:/tessdata/tessdata", "eng") != 0 &&
        ocr->Init(NULL, "eng") != 0) {
      std::cerr << "ERROR: Could not initialize Tesseract" << std::endl;
      std::cerr << "Tried paths: C:/tessdata/tessdata and default" << std::endl;
      delete ocr;
      return false;
    }

    // Set image to the ORIGINAL image (not rendered)
    ocr->SetImage(originalImage.data, originalImage.cols, originalImage.rows, 3,
                  originalImage.step);
    ocr->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK); // WORD mode for detecting
                                                      // individual words

    // Get word-level bounding boxes from OCR and store them
    ocr->Recognize(0);
    tesseract::ResultIterator *ri = ocr->GetIterator();

    // Store all OCR word boxes with their text
    struct OcrBox {
      int x, y, width, height;
      std::string text;
    };
    std::vector<OcrBox> ocrBoxes;

    if (ri != nullptr) {
      do {
        const char *word = ri->GetUTF8Text(tesseract::RIL_WORD);
        if (word != nullptr) {
          int x1, y1, x2, y2;
          ri->BoundingBox(tesseract::RIL_WORD, &x1, &y1, &x2, &y2);

          // Clean up the text for matching
          std::string wordStr(word);
          std::string cleanText = wordStr;
          cleanText.erase(
              std::remove_if(cleanText.begin(), cleanText.end(),
                             [](unsigned char c) { return std::isspace(c); }),
              cleanText.end());
          std::transform(cleanText.begin(), cleanText.end(), cleanText.begin(),
                         [](unsigned char c) { return std::tolower(c); });

          ocrBoxes.push_back({x1, y1, x2 - x1, y2 - y1, cleanText});
          delete[] word;
        }
      } while (ri->Next(tesseract::RIL_WORD));
      delete ri;
    }

    ocr->End();
    delete ocr;

    if (ocrBoxes.empty()) {
      std::cerr << "WARNING: Could not find any OCR words in the original image"
                << std::endl;
      std::cerr << "Creating marked image without alignment adjustment"
                << std::endl;
    } else {
      std::cerr << "Found " << ocrBoxes.size()
                << " OCR word boxes for alignment" << std::endl;
    }

    // Store per-element alignment data
    struct ElementAlignment {
      int offsetX, offsetY, width, height;
      bool found;
    };
    std::map<size_t, ElementAlignment> elementAlignments; // Index -> alignment

    if (!ocrBoxes.empty()) {
      std::cerr << "Performing per-element OCR alignment for "
                << renderResult.elements.size() << " elements" << std::endl;

      // For each text element, search for it individually
      for (size_t elemIdx = 0; elemIdx < renderResult.elements.size();
           elemIdx++) {
        const auto &elem = renderResult.elements[elemIdx];

        if (elem.type != RenderedElement::TEXT || elem.text.empty()) {
          continue;
        }

        // Skip elements that are mostly underscores (difficult for OCR)
        size_t underscoreCount =
            std::count(elem.text.begin(), elem.text.end(), '_');
        if (underscoreCount > elem.text.length() / 2) {
          std::cerr << "Skipping element " << elemIdx
                    << " (mostly underscores): \"" << elem.text.substr(0, 20)
                    << "\"" << std::endl;
          continue;
        }

        // Convert relative coordinates to pixel coordinates in rendered image
        // For TEXT, relativeX/Y is the centre, so derive top-left
        int elemPixelX =
            static_cast<int>((elem.relativeX - elem.relativeWidth / 2.0) *
                             renderResult.imageWidth);
        int elemPixelY =
            static_cast<int>((elem.relativeY - elem.relativeHeight / 2.0) *
                             renderResult.imageHeight);
        int elemPixelW =
            static_cast<int>(elem.relativeWidth * renderResult.imageWidth);
        int elemPixelH =
            static_cast<int>(elem.relativeHeight * renderResult.imageHeight);

        // Scale element coordinates to original image space
        int scaledElemX = static_cast<int>(elemPixelX * scaleX);
        int scaledElemY = static_cast<int>(elemPixelY * scaleY);
        int scaledElemWidth = static_cast<int>(elemPixelW * scaleX);
        int scaledElemHeight = static_cast<int>(elemPixelH * scaleY);

        // Define search region around the expected position (in original
        // image space)
        const int SEARCH_RADIUS = 150; // pixels to search in each direction
        int roiX = std::max(0, scaledElemX - SEARCH_RADIUS);
        int roiY = std::max(0, scaledElemY - SEARCH_RADIUS);
        int roiWidth = std::min(SEARCH_RADIUS * 2 + scaledElemWidth,
                                originalImage.cols - roiX);
        int roiHeight = std::min(SEARCH_RADIUS * 2 + scaledElemHeight,
                                 originalImage.rows - roiY);

        // Skip if ROI is invalid
        if (roiWidth <= 0 || roiHeight <= 0)
          continue;

        cv::Rect roiRect(roiX, roiY, roiWidth, roiHeight);
        cv::Mat roi = originalImage(roiRect);

        // Run OCR on this ROI
        tesseract::TessBaseAPI *roiOcr = new tesseract::TessBaseAPI();
        if (roiOcr->Init("C:/tessdata/tessdata", "eng") != 0 &&
            roiOcr->Init(NULL, "eng") != 0) {
          delete roiOcr;
          continue;
        }

        roiOcr->SetImage(roi.data, roi.cols, roi.rows, 3, roi.step);
        roiOcr->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
        roiOcr->Recognize(0);

        // Clean element text for matching
        std::string elemText = elem.text;
        elemText.erase(
            std::remove_if(elemText.begin(), elemText.end(),
                           [](unsigned char c) { return std::isspace(c); }),
            elemText.end());
        std::transform(elemText.begin(), elemText.end(), elemText.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        // Find matching text in ROI
        tesseract::ResultIterator *roiRi = roiOcr->GetIterator();
        int bestOffsetX = 0;
        int bestOffsetY = 0;
        int bestOcrWidth = 0;
        int bestOcrHeight = 0;
        double bestDistance = std::numeric_limits<double>::max();
        bool foundMatch = false;

        if (roiRi != nullptr) {
          do {
            const char *word = roiRi->GetUTF8Text(tesseract::RIL_WORD);
            if (word != nullptr) {
              std::string wordStr(word);
              std::string cleanWord = wordStr;
              cleanWord.erase(std::remove_if(cleanWord.begin(), cleanWord.end(),
                                             [](unsigned char c) {
                                               return std::isspace(c);
                                             }),
                              cleanWord.end());
              std::transform(cleanWord.begin(), cleanWord.end(),
                             cleanWord.begin(),
                             [](unsigned char c) { return std::tolower(c); });

              // Calculate similarity between cleanWord and elemText
              auto calculateSimilarity = [](const std::string &s1,
                                            const std::string &s2) -> double {
                if (s1.empty() || s2.empty())
                  return 0.0;
                size_t matches = 0;
                size_t maxLen = std::max(s1.length(), s2.length());
                size_t minLen = std::min(s1.length(), s2.length());
                for (size_t i = 0; i < minLen; i++) {
                  if (s1[i] == s2[i])
                    matches++;
                }
                return static_cast<double>(matches) / maxLen;
              };

              double similarity = calculateSimilarity(cleanWord, elemText);

              // Check if text matches (exact, substring, or fuzzy)
              bool textMatches =
                  (cleanWord == elemText) ||
                  (cleanWord.find(elemText) != std::string::npos) ||
                  (elemText.find(cleanWord) != std::string::npos) ||
                  (similarity >= 0.7); // 70% similarity threshold

              if (textMatches) {
                int x1, y1, x2, y2;
                roiRi->BoundingBox(tesseract::RIL_WORD, &x1, &y1, &x2, &y2);

                // Convert ROI coordinates to original image coordinates
                int absX = roiX + x1;
                int absY = roiY + y1;

                // Calculate offset in original image space
                double dx = absX - scaledElemX;
                double dy = absY - scaledElemY;
                double distance = std::sqrt(dx * dx + dy * dy);

                if (distance < bestDistance) {
                  bestDistance = distance;
                  bestOffsetX = absX - scaledElemX;
                  bestOffsetY = absY - scaledElemY;
                  bestOcrWidth = x2 - x1;
                  bestOcrHeight = y2 - y1;
                  foundMatch = true;
                }
              }

              delete[] word;
            }
          } while (roiRi->Next(tesseract::RIL_WORD));
          delete roiRi;
        }

        roiOcr->End();
        delete roiOcr;

        if (foundMatch) {
          elementAlignments[elemIdx] = {bestOffsetX, bestOffsetY, bestOcrWidth,
                                        bestOcrHeight, true};
          std::cerr << "Element " << elemIdx << " \"" << elem.text.substr(0, 20)
                    << "\": offset (" << bestOffsetX << ", " << bestOffsetY
                    << "), OCR size: " << bestOcrWidth << "x" << bestOcrHeight
                    << ", distance: " << bestDistance << std::endl;
        } else {
          elementAlignments[elemIdx] = {0, 0, -1, -1, false};
        }
      }

      // Fill in missing alignments by interpolating from nearby elements
      std::cerr << "Filling in missing alignments..." << std::endl;
      for (size_t elemIdx = 0; elemIdx < renderResult.elements.size();
           elemIdx++) {
        const auto &elem = renderResult.elements[elemIdx];
        if (elem.type != RenderedElement::TEXT || elem.text.empty()) {
          continue;
        }

        // Skip elements that are mostly underscores
        size_t underscoreCount =
            std::count(elem.text.begin(), elem.text.end(), '_');
        if (underscoreCount > elem.text.length() / 2) {
          continue;
        }

        auto it = elementAlignments.find(elemIdx);
        if (it != elementAlignments.end() && it->second.found) {
          continue; // Already has alignment
        }

        // Find nearest aligned element
        int nearestIdx = -1;
        double nearestDist = std::numeric_limits<double>::max();

        for (size_t otherIdx = 0; otherIdx < renderResult.elements.size();
             otherIdx++) {
          if (otherIdx == elemIdx)
            continue;

          auto otherIt = elementAlignments.find(otherIdx);
          if (otherIt == elementAlignments.end() || !otherIt->second.found) {
            continue;
          }

          const auto &otherElem = renderResult.elements[otherIdx];
          double dx = elem.relativeX - otherElem.relativeX;
          double dy = elem.relativeY - otherElem.relativeY;
          double dist = std::sqrt(dx * dx + dy * dy);

          if (dist < nearestDist) {
            nearestDist = dist;
            nearestIdx = otherIdx;
          }
        }

        if (nearestIdx >= 0) {
          auto nearestAlignment = elementAlignments[nearestIdx];
          // Copy both X and Y offsets from nearest neighbor
          elementAlignments[elemIdx] = {nearestAlignment.offsetX,
                                        nearestAlignment.offsetY, -1, -1, true};
          std::cerr << "Element " << elemIdx << " \"" << elem.text.substr(0, 20)
                    << "\": using offset from element " << nearestIdx << " ("
                    << nearestAlignment.offsetX << ", "
                    << nearestAlignment.offsetY << ")" << std::endl;
        }
      }
    }

    // Detect and resolve overlaps
    std::cerr << "Checking for overlapping boxes..." << std::endl;

    // Build list of boxes with their element info
    struct BoxInfo {
      size_t elemIdx;
      int x, y, width, height;
      std::string text;
      double fontSize; // PDF font size in points
    };
    std::vector<BoxInfo> boxes;

    for (size_t elemIdx = 0; elemIdx < renderResult.elements.size();
         elemIdx++) {
      const auto &elem = renderResult.elements[elemIdx];

      if (elem.type != RenderedElement::TEXT || elem.text.empty()) {
        continue;
      }

      // Skip underscore elements
      size_t underscoreCount =
          std::count(elem.text.begin(), elem.text.end(), '_');
      if (underscoreCount > elem.text.length() / 2) {
        continue;
      }

      // Convert relative coordinates to pixel coordinates in rendered image
      int elemPixelX =
          static_cast<int>((elem.relativeX - elem.relativeWidth / 2.0) *
                           renderResult.imageWidth);
      int elemPixelY =
          static_cast<int>((elem.relativeY - elem.relativeHeight / 2.0) *
                           renderResult.imageHeight);
      int elemPixelW =
          static_cast<int>(elem.relativeWidth * renderResult.imageWidth);
      int elemPixelH =
          static_cast<int>(elem.relativeHeight * renderResult.imageHeight);

      int scaledElemX = static_cast<int>(elemPixelX * scaleX);
      int scaledElemY = static_cast<int>(elemPixelY * scaleY);
      int scaledElemWidth = static_cast<int>(elemPixelW * scaleX);
      int scaledElemHeight = static_cast<int>(elemPixelH * scaleY);

      int adjustedX = scaledElemX;
      int adjustedY = scaledElemY;

      auto it = elementAlignments.find(elemIdx);
      if (it != elementAlignments.end() && it->second.found) {
        adjustedX = scaledElemX + it->second.offsetX;
        adjustedY = scaledElemY + it->second.offsetY;
      }

      boxes.push_back({elemIdx, adjustedX, adjustedY, scaledElemWidth,
                       scaledElemHeight, elem.text, elem.fontSize});
    }

    // Group boxes by fontSize and calculate uniform heights
    std::map<double, int> fontSizeToUniformHeight;
    std::cerr << "Calculating uniform heights for each fontSize group..."
              << std::endl;

    // First pass: find maximum height needed for each fontSize
    for (const auto &box : boxes) {
      double fontSize = box.fontSize;
      if (fontSizeToUniformHeight.find(fontSize) ==
          fontSizeToUniformHeight.end()) {
        fontSizeToUniformHeight[fontSize] = 0;
      }
      // Track the maximum height for this fontSize
      fontSizeToUniformHeight[fontSize] =
          std::max(fontSizeToUniformHeight[fontSize], box.height);
    }

    // Add padding to each fontSize group's height
    const int VERTICAL_PADDING = 4; // pixels of padding above and below
    for (auto &pair : fontSizeToUniformHeight) {
      pair.second += VERTICAL_PADDING * 2;
      std::cerr << "  fontSize " << pair.first
                << "pt: uniform height = " << pair.second
                << " pixels (includes " << (VERTICAL_PADDING * 2)
                << "px padding)" << std::endl;
    }

    // Second pass: apply uniform heights and center vertically
    for (auto &box : boxes) {
      int oldHeight = box.height;
      int uniformHeight = fontSizeToUniformHeight[box.fontSize];
      int heightDiff = uniformHeight - oldHeight;

      // Center the box vertically by adjusting Y position
      box.y -= heightDiff / 2;
      box.height = uniformHeight;

      std::cerr << "Element " << box.elemIdx << " \"" << box.text.substr(0, 20)
                << "\": fontSize=" << box.fontSize << "pt, adjusted height "
                << oldHeight << " â†’ " << uniformHeight << ", y offset "
                << (heightDiff / 2) << std::endl;
    }

    // Check for vertical overlaps and adjust Y positions (preserve uniform
    // heights)
    std::cerr << "Resolving overlaps while preserving uniform heights..."
              << std::endl;
    for (size_t i = 0; i < boxes.size(); i++) {
      for (size_t j = i + 1; j < boxes.size(); j++) {
        auto &box1 = boxes[i];
        auto &box2 = boxes[j];

        // Check if boxes overlap horizontally (same column)
        int hOverlap = std::min(box1.x + box1.width, box2.x + box2.width) -
                       std::max(box1.x, box2.x);
        if (hOverlap <= 0)
          continue; // No horizontal overlap

        // Check vertical overlap
        int box1Bottom = box1.y + box1.height;
        int box2Bottom = box2.y + box2.height;
        int vOverlap =
            std::min(box1Bottom, box2Bottom) - std::max(box1.y, box2.y);

        if (vOverlap > 0) {
          std::cerr << "Overlap detected between element " << box1.elemIdx
                    << " and " << box2.elemIdx << " (" << vOverlap << " pixels)"
                    << std::endl;

          // Adjust Y position of lower box to avoid overlap (preserve
          // height!)
          if (box1.y < box2.y) {
            int newBox2Y = box1Bottom;
            box2.y = newBox2Y;
            std::cerr << "  Moved element " << box2.elemIdx
                      << " down to Y=" << newBox2Y << " (height preserved at "
                      << box2.height << ")" << std::endl;
          } else {
            int newBox1Y = box2Bottom;
            box1.y = newBox1Y;
            std::cerr << "  Moved element " << box1.elemIdx
                      << " down to Y=" << newBox1Y << " (height preserved at "
                      << box1.height << ")" << std::endl;
          }
        }
      }
    }

    // Expand boxes horizontally to maximize width without overlapping
    std::cerr << "Expanding boxes horizontally..." << std::endl;
    for (auto &box : boxes) {
      // Find the maximum width we can expand to
      int maxExpandLeft = box.x; // Can expand to left edge of image
      int maxExpandRight =
          originalImage.cols - (box.x + box.width); // Can expand to right edge

      // Check for horizontal neighbors that would limit expansion
      for (const auto &other : boxes) {
        if (other.elemIdx == box.elemIdx)
          continue;

        // Check if boxes are vertically aligned (could interfere
        // horizontally)
        int vOverlap = std::min(box.y + box.height, other.y + other.height) -
                       std::max(box.y, other.y);
        if (vOverlap <= 0)
          continue; // No vertical overlap, won't interfere

        // Check if other box is to the left
        if (other.x + other.width <= box.x) {
          int gap = box.x - (other.x + other.width);
          maxExpandLeft = std::min(maxExpandLeft, gap);
        }
        // Check if other box is to the right
        else if (other.x >= box.x + box.width) {
          int gap = other.x - (box.x + box.width);
          maxExpandRight = std::min(maxExpandRight, gap);
        }
      }

      // Apply expansion (leave 2px gap on each side for safety)
      // Also limit expansion to reasonable amount (50% of original width on
      // each side)
      const int HORIZONTAL_GAP = 2;
      int maxReasonableExpand =
          box.width / 2; // Don't expand more than 50% of original width
      int expandLeft = std::max(
          0, std::min(maxExpandLeft - HORIZONTAL_GAP, maxReasonableExpand));
      int expandRight = std::max(
          0, std::min(maxExpandRight - HORIZONTAL_GAP, maxReasonableExpand));

      if (expandLeft > 0 || expandRight > 0) {
        int oldX = box.x;
        int oldWidth = box.width;
        box.x -= expandLeft;
        box.width += expandLeft + expandRight;
        std::cerr << "Element " << box.elemIdx << ": expanded width "
                  << oldWidth << " â†’ " << box.width << " (left+" << expandLeft
                  << ", right+" << expandRight << ")" << std::endl;
      }
    }

    // Verify each box still contains correct text using OCR
    std::cerr << "Verifying boxes with OCR..." << std::endl;
    tesseract::TessBaseAPI *verifyOcr = new tesseract::TessBaseAPI();
    if (verifyOcr->Init("C:/tessdata/tessdata", "eng") != 0 &&
        verifyOcr->Init(NULL, "eng") != 0) {
      std::cerr << "WARNING: Could not initialize OCR for verification"
                << std::endl;
    } else {
      for (auto &box : boxes) {
        // Extract ROI
        cv::Rect roi(box.x, box.y, box.width, box.height);
        roi = roi & cv::Rect(0, 0, originalImage.cols,
                             originalImage.rows); // Clamp to image

        if (roi.width <= 0 || roi.height <= 0)
          continue;

        cv::Mat roiImage = originalImage(roi);

        // Run OCR
        verifyOcr->SetImage(roiImage.data, roiImage.cols, roiImage.rows, 3,
                            roiImage.step);
        char *ocrText = verifyOcr->GetUTF8Text();

        if (ocrText) {
          std::string detectedText(ocrText);
          // Clean up
          detectedText.erase(
              std::remove_if(detectedText.begin(), detectedText.end(),
                             [](unsigned char c) { return std::isspace(c); }),
              detectedText.end());
          std::transform(detectedText.begin(), detectedText.end(),
                         detectedText.begin(),
                         [](unsigned char c) { return std::tolower(c); });

          std::string expectedText = box.text;
          expectedText.erase(
              std::remove_if(expectedText.begin(), expectedText.end(),
                             [](unsigned char c) { return std::isspace(c); }),
              expectedText.end());
          std::transform(expectedText.begin(), expectedText.end(),
                         expectedText.begin(),
                         [](unsigned char c) { return std::tolower(c); });

          // Calculate character-level similarity for fuzzy matching
          auto calculateSimilarity = [](const std::string &s1,
                                        const std::string &s2) -> double {
            if (s1.empty() || s2.empty())
              return 0.0;
            size_t matches = 0;
            size_t maxLen = std::max(s1.length(), s2.length());
            size_t minLen = std::min(s1.length(), s2.length());

            // Count matching characters at same positions
            for (size_t i = 0; i < minLen; i++) {
              if (s1[i] == s2[i])
                matches++;
              // Handle common OCR confusions
              else if ((s1[i] == 'i' && s2[i] == '1') ||
                       (s1[i] == '1' && s2[i] == 'i'))
                matches++;
              else if ((s1[i] == 'o' && s2[i] == '0') ||
                       (s1[i] == '0' && s2[i] == 'o'))
                matches++;
              else if ((s1[i] == 'l' && s2[i] == '1') ||
                       (s1[i] == '1' && s2[i] == 'l'))
                matches++;
            }
            return static_cast<double>(matches) / maxLen;
          };

          double similarity = calculateSimilarity(detectedText, expectedText);

          bool matches =
              (detectedText.find(expectedText) != std::string::npos) ||
              (expectedText.find(detectedText) != std::string::npos) ||
              (similarity >= 0.7); // 70% similarity threshold

          std::cerr << "Element " << box.elemIdx << " \"" << box.text
                    << "\": OCR=\"" << detectedText
                    << "\" similarity=" << std::fixed << std::setprecision(2)
                    << similarity << " " << (matches ? "âœ“" : "âœ—")
                    << std::endl;

          delete[] ocrText;

          // If box is too tall compared to font size, try to shrink it
          // Calculate expected height from font size
          double fontSizePixels = box.fontSize * 1.333 * scaleY;
          int expectedHeight = static_cast<int>(fontSizePixels * 1.2);

          if (box.height > expectedHeight * 1.5 && expectedText.length() <= 5) {
            std::cerr << "  Box too tall (" << box.height << " vs expected "
                      << expectedHeight << " from " << box.fontSize
                      << "pt font), attempting to shrink..." << std::endl;

            // Try shrinking from bottom
            int originalHeight = box.height;
            int minHeight = expectedHeight; // At least the expected font height

            for (int testHeight = minHeight; testHeight < originalHeight;
                 testHeight += 5) {
              cv::Rect testRoi(box.x, box.y, box.width, testHeight);
              testRoi = testRoi &
                        cv::Rect(0, 0, originalImage.cols, originalImage.rows);

              if (testRoi.width <= 0 || testRoi.height <= 0)
                continue;

              cv::Mat testImage = originalImage(testRoi);
              verifyOcr->SetImage(testImage.data, testImage.cols,
                                  testImage.rows, 3, testImage.step);
              char *testText = verifyOcr->GetUTF8Text();

              if (testText) {
                std::string testDetected(testText);
                testDetected.erase(std::remove_if(testDetected.begin(),
                                                  testDetected.end(),
                                                  [](unsigned char c) {
                                                    return std::isspace(c);
                                                  }),
                                   testDetected.end());
                std::transform(testDetected.begin(), testDetected.end(),
                               testDetected.begin(),
                               [](unsigned char c) { return std::tolower(c); });

                double testSim =
                    calculateSimilarity(testDetected, expectedText);
                bool testMatches =
                    (testDetected.find(expectedText) != std::string::npos) ||
                    (expectedText.find(testDetected) != std::string::npos) ||
                    (testSim >= 0.7);

                if (testMatches && testSim >= similarity) {
                  box.height = testHeight;
                  similarity = testSim;
                  std::cerr << "  Shrunk to height " << testHeight
                            << ", similarity=" << testSim << std::endl;
                  break;
                }

                delete[] testText;
              }
            }
          }
        }
      }
      verifyOcr->End();
    }
    delete verifyOcr;

    // Create a copy of the original image for marking
    cv::Mat markedImage = originalImage.clone();

    // Draw adjusted bounding boxes in blue
    cv::Scalar blueColor(255, 0, 0); // Blue in BGR
    int drawnCount = 0;

    for (const auto &box : boxes) {
      // Clamp coordinates to original image bounds
      int drawX1 = std::max(0, std::min(box.x, originalImage.cols));
      int drawY1 = std::max(0, std::min(box.y, originalImage.rows));
      int drawX2 = std::max(0, std::min(box.x + box.width, originalImage.cols));
      int drawY2 =
          std::max(0, std::min(box.y + box.height, originalImage.rows));

      // Only draw if we have a valid rectangle
      if (drawX2 > drawX1 && drawY2 > drawY1) {
        cv::rectangle(markedImage, cv::Point(drawX1, drawY1),
                      cv::Point(drawX2, drawY2), blueColor, 2);
        drawnCount++;
      }
    }

    std::cerr << "Drew " << drawnCount << " non-overlapping boxes" << std::endl;

    // Save the marked image
    if (cv::imwrite(outputPath, markedImage)) {
      std::cerr << "Aligned marked image saved: " << outputPath << std::endl;
      return true;
    } else {
      std::cerr << "ERROR: Failed to save aligned marked image: " << outputPath
                << std::endl;
      return false;
    }
  } catch (const std::exception &e) {
    std::cerr << "ERROR in alignAndMarkElements: " << e.what() << std::endl;
    return false;
  }
}

OCRAnalysis::PDFElements
OCRAnalysis::stripBleedMarks(const std::string &pdfPath) {
  PDFElements result;
  result.success = false;

  auto startTime = std::chrono::high_resolution_clock::now();

  try {
    // Extract all PDF elements first
    PDFElements allElements = extractPDFElements(pdfPath);

    if (!allElements.success) {
      result.errorMessage =
          "Failed to extract PDF elements: " + allElements.errorMessage;
      return result;
    }

    // STEP 1: Remove bleed marks (rectangles on same horizontal line +
    // associated lines)
    std::set<int> rectanglesToRemove;
    std::set<int> linesToRemove;

    const double yTolerance = 2.0; // tolerance for same horizontal line
    const double connectionTolerance = 2.0; // tolerance for line/box connection

    // Group rectangles by Y position (horizontal alignment)
    std::vector<std::vector<int>> rectangleGroups;
    std::vector<bool> processed(allElements.rectangles.size(), false);

    for (size_t i = 0; i < allElements.rectangles.size(); i++) {
      if (processed[i])
        continue;

      const auto &rect1 = allElements.rectangles[i];
      std::vector<int> group;
      group.push_back(i);
      processed[i] = true;

      // Find all rectangles at approximately the same Y position
      for (size_t j = i + 1; j < allElements.rectangles.size(); j++) {
        if (processed[j])
          continue;

        const auto &rect2 = allElements.rectangles[j];
        bool sameY = std::abs(rect1.y - rect2.y) < yTolerance;

        if (sameY) {
          group.push_back(j);
          processed[j] = true;
        }
      }

      // Only keep groups with 2 or more rectangles (bleed marks)
      if (group.size() >= 2) {
        // Check if this group is near the page edges (likely crop marks, not
        // bleed marks) Get the Y position of this group
        double groupY = allElements.rectangles[group[0]].y;
        double edgeMargin = 50.0; // points from edge

        bool nearTopEdge = groupY < edgeMargin;
        bool nearBottomEdge = groupY > (allElements.pageHeight - edgeMargin);

        // Skip groups near edges - they're likely crop marks, not bleed marks
        if (!nearTopEdge && !nearBottomEdge) {
          rectangleGroups.push_back(group);
        }
      }
    }

    // Mark rectangles and associated lines for removal
    for (const auto &group : rectangleGroups) {
      for (int rectIdx : group) {
        rectanglesToRemove.insert(rectIdx);
      }

      // Find Y and X range of this group
      double minY = std::numeric_limits<double>::max();
      double maxY = std::numeric_limits<double>::lowest();
      double minX = std::numeric_limits<double>::max();
      double maxX = std::numeric_limits<double>::lowest();

      for (int rectIdx : group) {
        const auto &rect = allElements.rectangles[rectIdx];
        minY = std::min(minY, rect.y);
        maxY = std::max(maxY, rect.y + rect.height);
        minX = std::min(minX, rect.x);
        maxX = std::max(maxX, rect.x + rect.width);
      }

      // Find lines that are part of these boxes or between them
      // BUT exclude lines near page corners (those are likely crop marks)
      const double cornerMargin = 100.0; // distance from corner

      for (size_t lineIdx = 0; lineIdx < allElements.graphicLines.size();
           lineIdx++) {
        const auto &line = allElements.graphicLines[lineIdx];

        // Check if line is near any corner
        double lineMinX = std::min(line.x1, line.x2);
        double lineMaxX = std::max(line.x1, line.x2);
        double lineMinY = std::min(line.y1, line.y2);
        double lineMaxY = std::max(line.y1, line.y2);

        bool nearTopLeftCorner =
            (lineMinX < cornerMargin && lineMinY < cornerMargin);
        bool nearTopRightCorner =
            (lineMaxX > allElements.pageWidth - cornerMargin &&
             lineMinY < cornerMargin);
        bool nearBottomLeftCorner =
            (lineMinX < cornerMargin &&
             lineMaxY > allElements.pageHeight - cornerMargin);
        bool nearBottomRightCorner =
            (lineMaxX > allElements.pageWidth - cornerMargin &&
             lineMaxY > allElements.pageHeight - cornerMargin);

        bool nearAnyCorner = nearTopLeftCorner || nearTopRightCorner ||
                             nearBottomLeftCorner || nearBottomRightCorner;

        if (nearAnyCorner)
          continue; // Don't remove lines near corners - they're likely crop
                    // marks

        double lineMinYPos = std::min(line.y1, line.y2);
        double lineMaxYPos = std::max(line.y1, line.y2);

        bool lineInYRange = !(lineMaxYPos < minY - connectionTolerance ||
                              lineMinYPos > maxY + connectionTolerance);

        if (!lineInYRange)
          continue;

        bool lineInXRange = !(lineMaxX < minX - connectionTolerance ||
                              lineMinX > maxX + connectionTolerance);

        if (lineInXRange) {
          linesToRemove.insert(lineIdx);
        }
      }
    }

    // Filter out bleed mark rectangles and lines
    std::vector<PDFRectangle> filteredRectangles;
    std::vector<PDFLine> filteredLines;

    for (size_t i = 0; i < allElements.rectangles.size(); i++) {
      if (rectanglesToRemove.find(i) == rectanglesToRemove.end()) {
        filteredRectangles.push_back(allElements.rectangles[i]);
      }
    }

    for (size_t i = 0; i < allElements.graphicLines.size(); i++) {
      if (linesToRemove.find(i) == linesToRemove.end()) {
        filteredLines.push_back(allElements.graphicLines[i]);
      }
    }

    std::cerr << "DEBUG: Removed " << rectanglesToRemove.size()
              << " bleed mark rectangles and " << linesToRemove.size()
              << " associated lines" << std::endl;
    std::cerr << "DEBUG: Remaining: " << filteredRectangles.size()
              << " rectangles, " << filteredLines.size() << " lines"
              << std::endl;

    // Show first few remaining lines for debugging
    std::cerr << "DEBUG: First 10 remaining lines:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), filteredLines.size()); i++) {
      const auto &line = filteredLines[i];
      std::cerr << "  Line " << i << ": (" << line.x1 << "," << line.y1
                << ") to (" << line.x2 << "," << line.y2 << ")" << std::endl;
    }

    // STEP 2: Detect crop marks from remaining lines
    struct CropMark {
      int line1Idx;
      int line2Idx;
      double cropX;
      double cropY;
    };

    std::vector<CropMark> cropMarks;
    const double perpendicularTolerance = 5.0; // degrees
    const double proximityTolerance = 50.0;    // max box size for L-shape
    const double cornerMargin =
        100.0; // distance from page corner to look for crop marks

    // Define the 4 corner regions where crop marks should be
    struct CornerRegion {
      double minX, maxX, minY, maxY;
      std::string name;
    };

    std::vector<CornerRegion> corners = {
        {0, cornerMargin, 0, cornerMargin, "top-left"},
        {allElements.pageWidth - cornerMargin, allElements.pageWidth, 0,
         cornerMargin, "top-right"},
        {0, cornerMargin, allElements.pageHeight - cornerMargin,
         allElements.pageHeight, "bottom-left"},
        {allElements.pageWidth - cornerMargin, allElements.pageWidth,
         allElements.pageHeight - cornerMargin, allElements.pageHeight,
         "bottom-right"}};

    // Find perpendicular line pairs in filtered lines, but only in corner
    // regions
    for (size_t i = 0; i < filteredLines.size(); i++) {
      const auto &line1 = filteredLines[i];

      for (size_t j = i + 1; j < filteredLines.size(); j++) {
        const auto &line2 = filteredLines[j];

        // Calculate angles
        double angle1 =
            std::atan2(line1.y2 - line1.y1, line1.x2 - line1.x1) * 180.0 / M_PI;
        double angle2 =
            std::atan2(line2.y2 - line2.y1, line2.x2 - line2.x1) * 180.0 / M_PI;

        angle1 = std::fmod(std::abs(angle1), 180.0);
        angle2 = std::fmod(std::abs(angle2), 180.0);

        // Check perpendicularity
        double angleDiff = std::abs(angle1 - angle2);
        if (angleDiff > 90.0)
          angleDiff = 180.0 - angleDiff;

        bool isPerpendicular =
            std::abs(angleDiff - 90.0) < perpendicularTolerance;

        if (!isPerpendicular)
          continue;

        // Check proximity (can be enclosed in small box)
        double minX = std::min({line1.x1, line1.x2, line2.x1, line2.x2});
        double maxX = std::max({line1.x1, line1.x2, line2.x1, line2.x2});
        double minY = std::min({line1.y1, line1.y2, line2.y1, line2.y2});
        double maxY = std::max({line1.y1, line1.y2, line2.y1, line2.y2});

        double boxSize = std::max(maxX - minX, maxY - minY);

        if (boxSize > proximityTolerance)
          continue;

        // Crop marks can be anywhere on the page, not just in corners
        // (commenting out corner restriction)
        /*
        // Check if this L-shape is in one of the corner regions
        bool inCorner = false;
        for (const auto &corner : corners) {
          bool xInRange = minX >= corner.minX && maxX <= corner.maxX;
          bool yInRange = minY >= corner.minY && maxY <= corner.maxY;
          if (xInRange && yInRange) {
            inCorner = true;
            break;
          }
        }

        if (!inCorner)
          continue; // Skip L-shapes not in corners
        */

        // Calculate intersection point when extended
        double x1 = line1.x1, y1 = line1.y1, x2 = line1.x2, y2 = line1.y2;
        double x3 = line2.x1, y3 = line2.y1, x4 = line2.x2, y4 = line2.y2;

        double denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

        if (std::abs(denom) < 1e-10)
          continue;

        double intersectX = ((x1 * y2 - y1 * x2) * (x3 - x4) -
                             (x1 - x2) * (x3 * y4 - y3 * x4)) /
                            denom;
        double intersectY = ((x1 * y2 - y1 * x2) * (y3 - y4) -
                             (y1 - y2) * (x3 * y4 - y3 * x4)) /
                            denom;

        // For crop marks, use the closest endpoints to avoid extending too
        // far Determine which line is horizontal and which is vertical
        bool line1IsHorizontal = std::abs(y2 - y1) < std::abs(x2 - x1);

        double cropX, cropY;
        if (line1IsHorizontal) {
          // Line1 is horizontal, Line2 is vertical
          cropX = (x3 + x4) / 2.0; // X from vertical line (line2)
          double horizY = (y1 + y2) / 2.0;

          // Check if vertical line crosses or touches the horizontal line
          double vertMinY = std::min(y3, y4);
          double vertMaxY = std::max(y3, y4);

          if (horizY >= vertMinY && horizY <= vertMaxY) {
            // Vertical line crosses horizontal - use closest endpoint of
            // vertical
            cropY = (std::abs(y3 - horizY) < std::abs(y4 - horizY)) ? y3 : y4;
          } else if (vertMinY > horizY) {
            // Vertical line is entirely above horizontal (top crop mark) -
            // use horizontal Y
            cropY = horizY;
          } else {
            // Vertical line is entirely below horizontal (bottom crop mark) -
            // use top endpoint of vertical
            cropY = vertMaxY;
          }
        } else {
          // Line1 is vertical, Line2 is horizontal
          cropX = (x1 + x2) / 2.0; // X from vertical line (line1)
          double horizY = (y3 + y4) / 2.0;

          // Check if vertical line crosses or touches the horizontal line
          double vertMinY = std::min(y1, y2);
          double vertMaxY = std::max(y1, y2);

          if (horizY >= vertMinY && horizY <= vertMaxY) {
            // Vertical line crosses horizontal - use closest endpoint of
            // vertical
            cropY = (std::abs(y1 - horizY) < std::abs(y2 - horizY)) ? y1 : y2;
          } else if (vertMinY > horizY) {
            // Vertical line is entirely above horizontal (top crop mark) -
            // use horizontal Y
            cropY = horizY;
          } else {
            // Vertical line is entirely below horizontal (bottom crop mark) -
            // use top endpoint of vertical
            cropY = vertMaxY;
          }
        }

        CropMark mark;
        mark.line1Idx = i;
        mark.line2Idx = j;
        mark.cropX = cropX;
        mark.cropY = cropY;
        cropMarks.push_back(mark);
      }
    }

    if (cropMarks.size() < 4) {
      result.errorMessage = "Could not find 4 crop marks. Found: " +
                            std::to_string(cropMarks.size());
      return result;
    }

    std::cerr << "DEBUG: Found " << cropMarks.size()
              << " crop marks:" << std::endl;
    for (size_t i = 0; i < cropMarks.size(); i++) {
      std::cerr << "  Crop mark " << i << ": (" << cropMarks[i].cropX << ", "
                << cropMarks[i].cropY << ")" << std::endl;
    }

    // Select the 4 corner-most crop marks
    if (cropMarks.size() > 4) {
      double minX = std::numeric_limits<double>::max();
      double maxX = std::numeric_limits<double>::lowest();
      double minY = std::numeric_limits<double>::max();
      double maxY = std::numeric_limits<double>::lowest();

      for (const auto &mark : cropMarks) {
        minX = std::min(minX, mark.cropX);
        maxX = std::max(maxX, mark.cropX);
        minY = std::min(minY, mark.cropY);
        maxY = std::max(maxY, mark.cropY);
      }

      CropMark *topLeft = nullptr, *topRight = nullptr, *bottomLeft = nullptr,
               *bottomRight = nullptr;

      for (auto &mark : cropMarks) {
        bool isLeft = mark.cropX < (minX + maxX) / 2.0;
        bool isTop = mark.cropY < (minY + maxY) / 2.0;

        if (isLeft && isTop) {
          if (!topLeft ||
              mark.cropX + mark.cropY < topLeft->cropX + topLeft->cropY)
            topLeft = &mark;
        } else if (!isLeft && isTop) {
          if (!topRight ||
              mark.cropX - mark.cropY > topRight->cropX - topRight->cropY)
            topRight = &mark;
        } else if (isLeft && !isTop) {
          if (!bottomLeft ||
              mark.cropY - mark.cropX > bottomLeft->cropY - bottomLeft->cropX)
            bottomLeft = &mark;
        } else {
          if (!bottomRight ||
              mark.cropX + mark.cropY > bottomRight->cropX + bottomRight->cropY)
            bottomRight = &mark;
        }
      }

      cropMarks.clear();
      if (topLeft)
        cropMarks.push_back(*topLeft);
      if (topRight)
        cropMarks.push_back(*topRight);
      if (bottomLeft)
        cropMarks.push_back(*bottomLeft);
      if (bottomRight)
        cropMarks.push_back(*bottomRight);
    }

    if (cropMarks.size() != 4) {
      result.errorMessage = "Could not identify exactly 4 crop marks";
      return result;
    }

    // Calculate crop box from the 4 crop points
    double cropMinX = std::numeric_limits<double>::max();
    double cropMaxX = std::numeric_limits<double>::lowest();
    double cropMinY = std::numeric_limits<double>::max();
    double cropMaxY = std::numeric_limits<double>::lowest();

    std::set<int> cropMarkLineIndices;

    for (const auto &mark : cropMarks) {
      cropMinX = std::min(cropMinX, mark.cropX);
      cropMaxX = std::max(cropMaxX, mark.cropX);
      cropMinY = std::min(cropMinY, mark.cropY);
      cropMaxY = std::max(cropMaxY, mark.cropY);

      cropMarkLineIndices.insert(mark.line1Idx);
      cropMarkLineIndices.insert(mark.line2Idx);
    }

    // Validate crop box dimensions
    double cropWidth = cropMaxX - cropMinX;
    double cropHeight = cropMaxY - cropMinY;

    if (cropWidth < 100.0 || cropHeight < 100.0) {
      result.errorMessage = "Detected crop box is too small (" +
                            std::to_string(cropWidth) + " x " +
                            std::to_string(cropHeight) + " points). " +
                            "Crop marks may not be correctly detected.";
      return result;
    }

    // Remove crop mark lines from filtered lines
    std::vector<PDFLine> finalLines;
    for (size_t i = 0; i < filteredLines.size(); i++) {
      if (cropMarkLineIndices.find(i) == cropMarkLineIndices.end()) {
        finalLines.push_back(filteredLines[i]);
      }
    }

    // Build final result
    result = allElements;
    result.rectangles = filteredRectangles;
    result.graphicLines = finalLines;
    result.rectangleCount = filteredRectangles.size();
    result.graphicLineCount = finalLines.size();

    // Set crop box
    result.pageX = cropMinX;
    result.pageY = cropMinY;
    result.pageWidth = cropWidth;
    result.pageHeight = cropHeight;

    // Also update linesBoundingBox to match crop box (for
    // renderElementsToPNG)
    result.linesBoundingBoxX = cropMinX;
    result.linesBoundingBoxY = cropMinY;
    result.linesBoundingBoxWidth = cropWidth;
    result.linesBoundingBoxHeight = cropHeight;

    result.success = true;

  } catch (const std::exception &e) {
    result.errorMessage =
        std::string("Crop mark detection failed: ") + e.what();
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  result.processingTimeMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();

  return result;
}

} // namespace ocr
