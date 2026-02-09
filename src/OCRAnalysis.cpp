#include "OCRAnalysis.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

// Cairo for PDF/PNG rendering (if available)
#ifdef HAVE_CAIRO
#include <cairo-pdf.h>
#include <cairo.h>
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
#include <goo/GooString.h>

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

  const char *tessDataPath =
      m_config.tessDataPath.empty() ? nullptr : m_config.tessDataPath.c_str();

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
    // Load the PDF document using Poppler
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

    // Extract text from the first page with position information
    std::string fullText;
    int pageCount = doc->pages();
    std::cerr << "DEBUG: PDF has " << pageCount
              << " pages, processing first page only" << std::endl;

    // Only process the first page (pageIndex = 0)
    for (int pageIndex = 0; pageIndex < 1 && pageIndex < pageCount;
         pageIndex++) {
      std::cerr << "DEBUG: Processing page " << (pageIndex + 1) << std::endl;

      try {
        std::unique_ptr<poppler::page> page(doc->create_page(pageIndex));

        if (!page) {
          std::cerr << "DEBUG: Failed to create page " << (pageIndex + 1)
                    << ", skipping" << std::endl;
          continue;
        }

        // Get page dimensions for coordinate context
        poppler::rectf pageRect = page->page_rect();

        // Get text list with font information for detailed extraction
        std::cerr << "DEBUG: Getting text list for page " << (pageIndex + 1)
                  << std::endl;
        std::vector<poppler::text_box> textBoxes;
        try {
          // Try with font information first
          std::cerr << "DEBUG: Attempting text extraction with font info..."
                    << std::endl;
          textBoxes = page->text_list(); // Use simpler version without font
                                         // info to avoid crashes
          std::cerr << "DEBUG: Found " << textBoxes.size()
                    << " text boxes on page " << (pageIndex + 1) << std::endl;
        } catch (const std::exception &e) {
          std::cerr << "DEBUG: Exception getting text list for page "
                    << (pageIndex + 1) << ": " << e.what() << std::endl;
          continue;
        } catch (...) {
          std::cerr << "DEBUG: Unknown exception getting text list for page "
                    << (pageIndex + 1) << std::endl;
          continue;
        }

        // Temporary storage for word-level regions on this page
        std::vector<TextRegion> pageRegions;
        std::string pageText;

        for (auto &textBox : textBoxes) {
          try {
            // Get the text content
            poppler::byte_array textBytes = textBox.text().to_utf8();
            std::string text(textBytes.begin(), textBytes.end());

            if (text.empty()) {
              continue;
            }

            // Get bounding box (in PDF coordinates - origin at bottom-left)
            poppler::rectf bbox = textBox.bbox();

            // Convert PDF coordinates to image-style coordinates (origin at
            // top-left) PDF y increases upward, image y increases downward
            double x = bbox.x();
            double y = pageRect.height() - bbox.y() - bbox.height();
            double width = bbox.width();
            double height = bbox.height();

            // Create TextRegion with position information
            TextRegion region;
            region.text = text;
            region.boundingBox =
                cv::Rect(static_cast<int>(x), static_cast<int>(y),
                         static_cast<int>(width), static_cast<int>(height));

            // Get rotation (in degrees)
            int rotation = textBox.rotation();

            // Determine orientation using rotation and aspect ratio
            double aspectRatio = (width > 0) ? (height / width) : 1.0;

            bool isLikelyVerticalByShape = false;
            if (text.length() > 1) {
              isLikelyVerticalByShape = (aspectRatio > 1.5);
            } else {
              isLikelyVerticalByShape = (aspectRatio > 3.0);
            }

            // Combine rotation and shape to determine orientation
            if (rotation == 90 || rotation == 270 || rotation == -90 ||
                rotation == -270) {
              region.orientation = TextOrientation::Vertical;
            } else if (isLikelyVerticalByShape) {
              region.orientation = TextOrientation::Vertical;
            } else if (rotation == 0 || rotation == 180 || rotation == -180) {
              region.orientation = TextOrientation::Horizontal;
            } else {
              region.orientation = isLikelyVerticalByShape
                                       ? TextOrientation::Vertical
                                       : TextOrientation::Horizontal;
            }

            // Set default confidence (font size not available without font
            // info)
            region.confidence = 80.0f;

            // Store page number in level field (1-indexed)
            region.level = pageIndex + 1;

            pageRegions.push_back(region);

            // Build full text
            pageText += text;
            if (textBox.has_space_after()) {
              pageText += " ";
            }
          } catch (const std::exception &e) {
            std::cerr << "DEBUG: Exception processing text box on page "
                      << (pageIndex + 1) << ": " << e.what() << std::endl;
            continue;
          } catch (...) {
            std::cerr << "DEBUG: Unknown exception processing text box on page "
                      << (pageIndex + 1) << std::endl;
            continue;
          }
        }

        // Post-processing: Reclassify horizontal words that are spatially
        // aligned with vertical text. This catches short words like "No."
        // that don't meet the aspect ratio threshold but are clearly part of
        // a vertical text column.
        for (size_t i = 0; i < pageRegions.size(); i++) {
          if (pageRegions[i].orientation != TextOrientation::Horizontal) {
            continue; // Already vertical or unknown
          }

          const cv::Rect &hBox = pageRegions[i].boundingBox;

          // First, check if this word is horizontally adjacent to other
          // horizontal text. If so, it's likely part of a horizontal line and
          // shouldn't be reclassified.
          bool hasHorizontalNeighbor = false;
          for (size_t k = 0; k < pageRegions.size(); k++) {
            if (i == k ||
                pageRegions[k].orientation != TextOrientation::Horizontal) {
              continue;
            }

            const cv::Rect &otherBox = pageRegions[k].boundingBox;

            // Check if Y positions are similar (same horizontal line)
            int yCenter1 = hBox.y + hBox.height / 2;
            int yCenter2 = otherBox.y + otherBox.height / 2;
            int yDiff = std::abs(yCenter1 - yCenter2);
            int yTolerance =
                std::max(5, std::max(hBox.height, otherBox.height) / 2);

            if (yDiff <= yTolerance) {
              // Check if horizontally adjacent (close in X)
              int gap;
              if (hBox.x > otherBox.x + otherBox.width) {
                gap = hBox.x - (otherBox.x + otherBox.width);
              } else if (otherBox.x > hBox.x + hBox.width) {
                gap = otherBox.x - (hBox.x + hBox.width);
              } else {
                gap = 0; // overlapping
              }

              // Consider adjacent if gap is less than average word width
              int avgWidth = (hBox.width + otherBox.width) / 2;
              if (gap < avgWidth * 2) {
                hasHorizontalNeighbor = true;
                break;
              }
            }
          }

          // If this word has horizontal neighbors, don't reclassify it
          if (hasHorizontalNeighbor) {
            continue;
          }

          // Check if this horizontal word aligns with any vertical word
          for (size_t j = 0; j < pageRegions.size(); j++) {
            if (i == j ||
                pageRegions[j].orientation != TextOrientation::Vertical) {
              continue;
            }

            const cv::Rect &vBox = pageRegions[j].boundingBox;

            // Check X alignment: centers should be close
            int hCenterX = hBox.x + hBox.width / 2;
            int vCenterX = vBox.x + vBox.width / 2;
            int xDiff = std::abs(hCenterX - vCenterX);

            // Check vertical proximity: should be adjacent or overlapping
            int verticalGap;
            if (hBox.y > vBox.y + vBox.height) {
              // horizontal box is below vertical box
              verticalGap = hBox.y - (vBox.y + vBox.height);
            } else if (vBox.y > hBox.y + hBox.height) {
              // vertical box is below horizontal box
              verticalGap = vBox.y - (hBox.y + hBox.height);
            } else {
              // overlapping
              verticalGap = 0;
            }

            // If X positions are very close and vertically adjacent,
            // reclassify
            int xTolerance = std::max(5, std::max(hBox.width, vBox.width) / 2);
            int verticalTolerance = std::max(10, vBox.height);

            if (xDiff <= xTolerance && verticalGap <= verticalTolerance) {
              pageRegions[i].orientation = TextOrientation::Vertical;
              break; // Found a match, no need to check more
            }
          }
        }

        // If line-level extraction is requested, group words into lines
        if (level == PDFExtractionLevel::Line && !pageRegions.empty()) {
          std::vector<TextRegion> lineRegions;
          std::vector<bool> used(pageRegions.size(), false);

          for (size_t i = 0; i < pageRegions.size(); i++) {
            if (used[i])
              continue;

            TextRegion lineRegion = pageRegions[i];
            used[i] = true;

            // Determine grouping tolerance based on orientation
            // For horizontal text, group by similar Y position
            // For vertical text, group by similar X position
            bool isVertical =
                (lineRegion.orientation == TextOrientation::Vertical);

            // Tolerance for considering words on the same line
            // Use half the height/width as tolerance
            int tolerance =
                isVertical ? std::max(5, lineRegion.boundingBox.width / 2)
                           : std::max(5, lineRegion.boundingBox.height / 2);

            // Find all words that belong to this line
            for (size_t j = i + 1; j < pageRegions.size(); j++) {
              if (used[j])
                continue;

              const TextRegion &candidate = pageRegions[j];

              // Must have same orientation
              if (candidate.orientation != lineRegion.orientation)
                continue;

              bool onSameLine = false;
              if (isVertical) {
                // For vertical text, check if X positions are close
                int xDiff = std::abs(candidate.boundingBox.x -
                                     lineRegion.boundingBox.x);
                // Also check they're vertically adjacent
                int verticalGap =
                    std::min(std::abs(candidate.boundingBox.y -
                                      (lineRegion.boundingBox.y +
                                       lineRegion.boundingBox.height)),
                             std::abs(lineRegion.boundingBox.y -
                                      (candidate.boundingBox.y +
                                       candidate.boundingBox.height)));
                onSameLine = (xDiff <= tolerance &&
                              verticalGap < lineRegion.boundingBox.height * 2);
              } else {
                // For horizontal text, check if Y positions are close
                int yCenter1 = lineRegion.boundingBox.y +
                               lineRegion.boundingBox.height / 2;
                int yCenter2 =
                    candidate.boundingBox.y + candidate.boundingBox.height / 2;
                int yDiff = std::abs(yCenter1 - yCenter2);
                // Also check they're horizontally close (not too far apart)
                int horizontalGap =
                    std::min(std::abs(candidate.boundingBox.x -
                                      (lineRegion.boundingBox.x +
                                       lineRegion.boundingBox.width)),
                             std::abs(lineRegion.boundingBox.x -
                                      (candidate.boundingBox.x +
                                       candidate.boundingBox.width)));
                onSameLine = (yDiff <= tolerance &&
                              horizontalGap < lineRegion.boundingBox.width * 3);
              }

              if (onSameLine) {
                // Merge this word into the line
                used[j] = true;

                // Expand bounding box to include this word
                int newX =
                    std::min(lineRegion.boundingBox.x, candidate.boundingBox.x);
                int newY =
                    std::min(lineRegion.boundingBox.y, candidate.boundingBox.y);
                int newRight = std::max(
                    lineRegion.boundingBox.x + lineRegion.boundingBox.width,
                    candidate.boundingBox.x + candidate.boundingBox.width);
                int newBottom = std::max(
                    lineRegion.boundingBox.y + lineRegion.boundingBox.height,
                    candidate.boundingBox.y + candidate.boundingBox.height);

                lineRegion.boundingBox =
                    cv::Rect(newX, newY, newRight - newX, newBottom - newY);

                // Append text with space
                lineRegion.text += " " + candidate.text;

                // Keep higher confidence
                lineRegion.confidence =
                    std::max(lineRegion.confidence, candidate.confidence);
              }
            }

            lineRegions.push_back(lineRegion);
          }

          // Add line regions to result
          for (auto &region : lineRegions) {
            result.regions.push_back(region);
          }
        } else {
          // Word-level extraction - add all word regions
          for (auto &region : pageRegions) {
            result.regions.push_back(region);
          }
        }

        // Add page text to full document text
        if (!pageText.empty()) {
          fullText += pageText;
        }
      } catch (const std::exception &e) {
        std::cerr << "DEBUG: Exception processing page " << (pageIndex + 1)
                  << ": " << e.what() << std::endl;
        // Continue to next page
      } catch (...) {
        std::cerr << "DEBUG: Unknown exception processing page "
                  << (pageIndex + 1) << std::endl;
        // Continue to next page
      }
    }

    result.fullText = fullText;
    result.success = true;
  } catch (const std::exception &e) {
    result.errorMessage = std::string("PDF extraction failed: ") + e.what();
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
  ImageExtractorOutputDev() : pageNumber(1), imageIndex(0) {}

  // Get the extracted images
  std::vector<OCRAnalysis::PDFEmbeddedImage> &getImages() { return images; }

  void setPageNumber(int page) {
    pageNumber = page;
    imageIndex = 0;
  }

  // Required OutputDev overrides
  bool upsideDown() override { return false; }
  bool useDrawChar() override { return false; }
  bool interpretType3Chars() override { return false; }
  bool needNonText() override { return true; } // We need images!

  // This is called for each image in the PDF
  void drawImage(GfxState *state, Object * /*ref*/, Stream *str, int width,
                 int height, GfxImageColorMap *colorMap, bool /*interpolate*/,
                 const int * /*maskColors*/, bool /*inlineImg*/) override {

    if (width <= 0 || height <= 0 || !colorMap) {
      return;
    }

    // Get transformation matrix for position info
    const auto &ctm = state->getCTM(); // Returns std::array<double, 6>
    double x = ctm[4];                 // Translation X
    double y = ctm[5];                 // Translation Y

    // Calculate display dimensions from CTM
    // CTM contains the transformation matrix in points
    // The magnitude of the transformation vectors gives us the display size
    double displayWidth = std::sqrt(ctm[0] * ctm[0] + ctm[1] * ctm[1]);
    double displayHeight = std::sqrt(ctm[2] * ctm[2] + ctm[3] * ctm[3]);

    // Determine image type and channels
    int nComps = colorMap->getNumPixelComps();
    int nBits = colorMap->getBits();

    // Create an ImageStream to read the image data properly
    ImageStream imgStr(str, width, nComps, nBits);
    imgStr.reset(); // Initialize the stream

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
    img.type = "raw";

    images.push_back(img);
  }

private:
  std::vector<OCRAnalysis::PDFEmbeddedImage> images;
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
                                double minLineLength) {
  PDFElements result;
  result.success = false;

  auto startTime = std::chrono::high_resolution_clock::now();

  try {
    // Extract text as lines (grouped words with orientation) from first page
    std::cerr << "DEBUG: Extracting text from first page..." << std::endl;
    try {
      OCRResult textResult =
          extractTextFromPDF(pdfPath, PDFExtractionLevel::Line);
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

        // Copy interior bounding box dimensions
        result.linesBoundingBoxX = lineResult.boundingBoxX;
        result.linesBoundingBoxY = lineResult.boundingBoxY;
        result.linesBoundingBoxWidth = lineResult.boundingBoxWidth;
        result.linesBoundingBoxHeight = lineResult.boundingBoxHeight;
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
          result.pageX = pageRect.x();
          result.pageWidth = pageRect.width();
          result.pageY = pageRect.y();
          result.pageHeight = pageRect.height();
          std::cerr << "DEBUG: Page crop box: origin(" << pageRect.x() << ", "
                    << pageRect.y() << ") size(" << result.pageWidth << " x "
                    << result.pageHeight << ") points" << std::endl;
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "DEBUG: Exception getting page count: " << e.what()
                << std::endl;
      result.pageCount = 1; // Default to 1 if we can't get the count
    }

    // Note: pageCount reflects total pages in PDF, but only first page was
    // processed

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
        // PAGE_UP = normal, PAGE_DOWN = upside-down (180°)
        // PAGE_LEFT/RIGHT = vertical (90° rotated)
        switch (info.tessOrientation) {
        case tesseract::ORIENTATION_PAGE_UP:
          info.region.orientation = TextOrientation::Horizontal;
          break;
        case tesseract::ORIENTATION_PAGE_DOWN:
          // Upside-down text - mark as horizontal but will need 180°
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
      {-1, TextOrientation::Horizontal},                    // 0° - normal
      {cv::ROTATE_90_CLOCKWISE, TextOrientation::Vertical}, // 90° CW
      {cv::ROTATE_180, TextOrientation::Horizontal}, // 180° - upside down
      {cv::ROTATE_90_COUNTERCLOCKWISE, TextOrientation::Vertical} // 270°
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

  // Define rotations to try: no rotation, 90° CW, 180°, 90° CCW
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

OCRAnalysis::PNGRenderResult
OCRAnalysis::renderElementsToPNG(const PDFElements &elements,
                                 const std::string &pdfPath, double dpi,
                                 const std::string &outputDir) {

  PNGRenderResult result;

  try {
    // Find the actual bounding box of all elements to determine the offset
    // This allows us to normalize coordinates so rendering starts at (0,0)
    // Only include elements that are within the crop box
    double minX, minY, maxX, maxY;

    // Use the interior bounding box (from crop marks) if available
    if (elements.linesBoundingBoxWidth > 0 &&
        elements.linesBoundingBoxHeight > 0) {
      // Use the interior box calculated from crop marks/lines
      minX = elements.linesBoundingBoxX;
      minY = elements.linesBoundingBoxY;
      maxX = elements.linesBoundingBoxX + elements.linesBoundingBoxWidth;
      maxY = elements.linesBoundingBoxY + elements.linesBoundingBoxHeight;

      std::cerr << "DEBUG: Using linesBoundingBox as content area: (" << minX
                << ", " << minY << ") to (" << maxX << ", " << maxY << ")"
                << std::endl;
    } else {
      // Fall back to calculating bounding box from all elements
      minX = std::numeric_limits<double>::max();
      minY = std::numeric_limits<double>::max();
      maxX = std::numeric_limits<double>::lowest();
      maxY = std::numeric_limits<double>::lowest();

      // Check all elements to find actual bounds (only those within crop box)
      for (const auto &text : elements.textLines) {
        // Skip elements outside crop box
        if (text.boundingBox.x < elements.pageX ||
            text.boundingBox.y < elements.pageY ||
            text.boundingBox.x + text.boundingBox.width >
                elements.pageX + elements.pageWidth ||
            text.boundingBox.y + text.boundingBox.height >
                elements.pageY + elements.pageHeight) {
          continue;
        }
        minX = std::min(minX, static_cast<double>(text.boundingBox.x));
        minY = std::min(minY, static_cast<double>(text.boundingBox.y));
        maxX = std::max(maxX, static_cast<double>(text.boundingBox.x +
                                                  text.boundingBox.width));
        maxY = std::max(maxY, static_cast<double>(text.boundingBox.y +
                                                  text.boundingBox.height));
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
    minX = std::max(0.0, minX);
    minY = std::max(0.0, minY);
    maxX = std::min(elements.pageWidth, maxX);
    maxY = std::min(elements.pageHeight, maxY);

    if (maxX <= minX || maxY <= minY) {
      result.errorMessage = "Invalid bounding box dimensions";
      return result;
    }

    // Calculate dimensions in points and pixels
    const double margin = 10.0;
    double pageWidthPt = maxX - minX + 2 * margin;
    double pageHeightPt = maxY - minY + 2 * margin;

    // Convert to pixels based on DPI (72 points = 1 inch)
    double scale = dpi / 72.0;
    int imageWidth = static_cast<int>(pageWidthPt * scale);
    int imageHeight = static_cast<int>(pageHeightPt * scale);

    result.imageWidth = imageWidth;
    result.imageHeight = imageHeight;

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
    // Only render rectangles within the crop box
    cairo_set_source_rgb(cr, 0.8, 0.8, 0.8);
    cairo_set_line_width(cr, 1.0 / scale);
    for (const auto &rect : elements.rectangles) {
      // Clip rectangle to content bounding box
      double rectLeft = std::max(rect.x, minX);
      double rectTop = std::max(rect.y, minY);
      double rectRight = std::min(rect.x + rect.width, maxX);
      double rectBottom = std::min(rect.y + rect.height, maxY);

      // Skip if rectangle is completely outside content area
      if (rectLeft >= rectRight || rectTop >= rectBottom) {
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

      // Add to result
      RenderedElement elem;
      elem.type = RenderedElement::RECTANGLE;
      elem.pixelX = static_cast<int>(x * scale);
      elem.pixelY = static_cast<int>(y * scale);
      elem.pixelWidth = static_cast<int>(clippedWidth * scale);
      elem.pixelHeight = static_cast<int>(clippedHeight * scale);
      result.elements.push_back(elem);
    }

    // Draw lines (PDF bottom-left origin -> convert to top-left)
    // Only render lines within the crop box
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    cairo_set_line_width(cr, 0.5 / scale);
    for (const auto &line : elements.graphicLines) {
      // Clip line to content bounding box
      double lx1 = line.x1;
      double ly1 = line.y1;
      double lx2 = line.x2;
      double ly2 = line.y2;

      // Simple clipping: clip each endpoint to the bounding box
      lx1 = std::max(minX, std::min(maxX, lx1));
      ly1 = std::max(minY, std::min(maxY, ly1));
      lx2 = std::max(minX, std::min(maxX, lx2));
      ly2 = std::max(minY, std::min(maxY, ly2));

      // Skip if line is completely outside or collapsed to a point
      if ((lx1 == lx2 && ly1 == ly2) || (line.x1 < minX && line.x2 < minX) ||
          (line.x1 > maxX && line.x2 > maxX) ||
          (line.y1 < minY && line.y2 < minY) ||
          (line.y1 > maxY && line.y2 > maxY)) {
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

      // Add to result
      RenderedElement elem;
      elem.type = RenderedElement::LINE;
      elem.pixelX = static_cast<int>(std::min(x1, x2) * scale);
      elem.pixelY = static_cast<int>(std::min(y1, y2) * scale);
      elem.pixelWidth = static_cast<int>(std::abs(x2 - x1) * scale);
      elem.pixelHeight = static_cast<int>(std::abs(y2 - y1) * scale);
      result.elements.push_back(elem);
    }

    // Draw images (PDF bottom-left origin -> convert to top-left)
    // Only render images within the crop box
    for (const auto &img : elements.images) {
      // Clip image to content bounding box
      double imgLeft = std::max(img.x, minX);
      double imgTop = std::max(img.y, minY);
      double imgRight = std::min(img.x + img.displayWidth, maxX);
      double imgBottom = std::min(img.y + img.displayHeight, maxY);

      // Skip if image is completely outside content area
      if (imgLeft >= imgRight || imgTop >= imgBottom) {
        continue;
      }

      // Calculate clipped dimensions
      double clippedWidth = imgRight - imgLeft;
      double clippedHeight = imgBottom - imgTop;
      double offsetX = imgLeft - img.x; // How much we clipped from left
      double offsetY = imgTop - img.y;  // How much we clipped from top

      double x = img.x - minX + margin;
      // Convert from PDF bottom-left to Cairo top-left
      double y = pageHeightPt - (img.y - minY + img.displayHeight) - margin;

      if (!img.image.empty()) {
        // Convert cv::Mat to Cairo surface and draw
        cairo_save(cr);
        cairo_translate(cr, x, y);

        // Scale image to match display dimensions in points
        // displayWidth/Height are in points, image.cols/rows are in pixels
        double scaleX = img.displayWidth / static_cast<double>(img.image.cols);
        double scaleY = img.displayHeight / static_cast<double>(img.image.rows);
        cairo_scale(cr, scaleX, scaleY);

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

        // Copy pixel data (Cairo uses BGRA on little-endian, but we have RGB)
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

        cairo_set_source_surface(cr, imgSurface, 0, 0);
        cairo_paint(cr);
        cairo_surface_destroy(imgSurface);
        cairo_restore(cr);
      }

      // Add to result
      RenderedElement elem;
      elem.type = RenderedElement::IMAGE;
      elem.pixelX = static_cast<int>(x * scale);
      elem.pixelY = static_cast<int>(y * scale);
      elem.pixelWidth = static_cast<int>(img.displayWidth * scale);
      elem.pixelHeight = static_cast<int>(img.displayHeight * scale);
      elem.image = img.image.clone();
      result.elements.push_back(elem);
    }

    // Draw text (PDF bottom-left origin -> convert to top-left)
    // Only render text within the crop box
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL,
                           CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, 10.0);

    for (const auto &text : elements.textLines) {
      // Clip text to content bounding box
      double textLeft = std::max(static_cast<double>(text.boundingBox.x), minX);
      double textTop = std::max(static_cast<double>(text.boundingBox.y), minY);
      double textRight = std::min(
          static_cast<double>(text.boundingBox.x + text.boundingBox.width),
          maxX);
      double textBottom = std::min(
          static_cast<double>(text.boundingBox.y + text.boundingBox.height),
          maxY);

      // Skip if text is completely outside content area
      if (textLeft >= textRight || textTop >= textBottom) {
        continue;
      }

      double x = text.boundingBox.x - minX + margin;
      // Convert from PDF bottom-left to Cairo top-left
      double y = pageHeightPt - (text.boundingBox.y - minY) - margin +
                 10; // Baseline offset

      cairo_move_to(cr, x, y);
      cairo_show_text(cr, text.text.c_str());

      // Add to result
      RenderedElement elem;
      elem.type = RenderedElement::TEXT;
      elem.pixelX = static_cast<int>(x * scale);
      elem.pixelY =
          static_cast<int>((y - 10) * scale); // Subtract baseline offset
      elem.pixelWidth = static_cast<int>(text.boundingBox.width * scale);
      elem.pixelHeight = static_cast<int>(text.boundingBox.height * scale);
      elem.text = text.text;
      result.elements.push_back(elem);
    }

    // Write to PNG
    cairo_surface_write_to_png(surface, outputPath.c_str());

    cairo_destroy(cr);
    cairo_surface_destroy(surface);

    std::cerr << "PNG rendered successfully: " << outputPath << std::endl;
    std::cerr << "  Total elements: " << result.elements.size() << std::endl;

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

} // namespace ocr
