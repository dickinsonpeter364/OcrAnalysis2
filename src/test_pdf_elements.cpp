#include "OCRAnalysis.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>

// Helper function to sanitize text for display
std::string sanitizeText(const std::string &text) {
  std::string result;
  result.reserve(text.size());

  for (size_t i = 0; i < text.size(); ++i) {
    unsigned char c = static_cast<unsigned char>(text[i]);

    // ASCII printable characters and common whitespace
    if ((c >= 32 && c <= 126) || c == '\n' || c == '\r' || c == '\t') {
      result += c;
    }
    // UTF-8 multi-byte sequence detection
    else if (c >= 0x80) {
      // This is a UTF-8 multi-byte character
      // For Windows console compatibility, replace with '?'
      // Skip the rest of the multi-byte sequence
      if ((c & 0xE0) == 0xC0) {
        // 2-byte sequence
        i += 1;
      } else if ((c & 0xF0) == 0xE0) {
        // 3-byte sequence (like the bullet points)
        i += 2;
      } else if ((c & 0xF8) == 0xF0) {
        // 4-byte sequence
        i += 3;
      }
      result += '?'; // Replace with question mark for visibility
    }
    // Other control characters
    else {
      result += ' ';
    }
  }
  return result;
}

int main(int argc, char *argv[]) {
  std::cout << "=== PDF Elements Extraction Test ===" << std::endl << std::endl;

  // Check for PDF file argument
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <pdf_file> [options]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --no-crop-filter        Extract all elements regardless of "
                 "crop marks"
              << std::endl;
    std::cerr << "  --render-output <file>  Render extracted elements to "
                 "visualization file"
              << std::endl;
    return 1;
  }

  std::string pdfPath = argv[1];
  bool applyCropFilter = true;  // Default: apply crop box filtering
  std::string renderOutputPath; // Empty = no rendering

  // Parse command-line options
  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--no-crop-filter") {
      applyCropFilter = false;
      std::cout << "Crop box filtering DISABLED - will extract all elements"
                << std::endl;
    } else if (arg == "--render-output" && i + 1 < argc) {
      renderOutputPath = argv[++i];
      std::cout << "Will render elements to: " << renderOutputPath << std::endl;
    }
  }

  std::cout << "Loading PDF: " << pdfPath << std::endl << std::endl;

  // Create OCR analyzer
  ocr::OCRAnalysis analyzer;

  // Extract all elements
  std::cout << "Extracting all elements from PDF..." << std::endl;
  // Determine the paired LAF PDF (L1 <-> L2 in the same directory) so
  // both cropped content PDFs can share the same page height.
  std::string pairPdfPath;
  {
    std::filesystem::path p(pdfPath);
    std::string stem = p.filename().string();
    bool isL1 = stem.size() >= 2 &&
                std::tolower((unsigned char)stem[0]) == 'l' && stem[1] == '1';
    bool isL2 = stem.size() >= 2 &&
                std::tolower((unsigned char)stem[0]) == 'l' && stem[1] == '2';
    if (isL1 || isL2) {
      std::string other = stem;
      other[1] = isL1 ? '2' : '1';
      std::filesystem::path candidate = p.parent_path() / other;
      if (std::filesystem::exists(candidate))
        pairPdfPath = candidate.string();
    }
  }

  ocr::OCRAnalysis::PDFElements elements =
      analyzer.extractPDFElements(pdfPath, 5.0, 5.0, "", true, pairPdfPath);

  if (!elements.success) {
    std::cerr << "Failed to extract elements: " << elements.errorMessage
              << std::endl;
    return 1;
  }

  // Rotate the input image 3.png 90° clockwise and drop it into the same
  // graphics folder as the cropped PDF, then append its dimensions
  // (assuming 300 DPI to match the PDF→PNG render) to dimensions.txt.
  {
    constexpr double kRenderDpi = 300.0;
    std::filesystem::path pdfDir =
        std::filesystem::path(pdfPath).parent_path();
    std::filesystem::path srcImg = pdfDir / "3.png";
    std::filesystem::path graphicsDir = pdfDir / "graphics";
    std::filesystem::path dstImgEarly = graphicsDir / "3_rotated.png";
    if (std::filesystem::exists(srcImg) &&
        std::filesystem::exists(graphicsDir) &&
        !std::filesystem::exists(dstImgEarly)) {
      cv::Mat in = cv::imread(srcImg.string(), cv::IMREAD_UNCHANGED);
      if (!in.empty()) {
        cv::Mat rotated;
        cv::rotate(in, rotated, cv::ROTATE_90_CLOCKWISE);
        std::filesystem::path dstImg = graphicsDir / "3_rotated.png";
        if (cv::imwrite(dstImg.string(), rotated)) {
          double widthIn  = rotated.cols / kRenderDpi;
          double heightIn = rotated.rows / kRenderDpi;
          double widthMm  = widthIn  * 25.4;
          double heightMm = heightIn * 25.4;
          std::ofstream dimsOut(graphicsDir / "dimensions.txt", std::ios::app);
          if (dimsOut) {
            dimsOut << dstImg.filename().string()
                    << "  " << std::fixed << std::setprecision(2)
                    << widthMm << " x " << heightMm << " mm  ("
                    << std::setprecision(3)
                    << widthIn << " x " << heightIn << " in)  "
                    << rotated.cols << " x " << rotated.rows
                    << " px @ " << std::setprecision(0) << kRenderDpi
                    << " dpi\n";
          }
          std::cout << "Rotated 3.png -> " << dstImg.string() << std::endl;
        }
      }
    }
  }

  // Locate the LAF1 region in the camera image using the existing
  // createRelativeMap pipeline: OCR-based anchor matching against
  // non-placeholder PDF text (findAllMatchedPairs already skips every
  // element whose text contains '<' or '>').  Detection is performed
  // on a 72-DPI downsample *of the 300-DPI backing crop* — that way
  // work72 and work300 descend from the same cropToLabel output and
  // their pixel ratios are exactly 300/72, so scaling the solved
  // crop rect back to 300 DPI doesn't introduce aspect errors.
  {
    constexpr double kRenderDpi = 300.0;
    constexpr double kDetectDpi = 72.0;
    std::filesystem::path pdfDir =
        std::filesystem::path(pdfPath).parent_path();
    std::filesystem::path graphicsDir = pdfDir / "graphics";
    std::filesystem::path templatePath = graphicsDir / "L1Packing_content.png";
    std::filesystem::path scenePath    = graphicsDir / "3_rotated.png";
    std::filesystem::path outPath      = graphicsDir / "extracted image.png";

    // Only run on the L1 pass so we have the L1 PDFElements in hand and
    // skip on L2 so we don't clobber a good result.
    std::string stem = std::filesystem::path(pdfPath).filename().string();
    bool isL1 = stem.size() >= 2 &&
                std::tolower((unsigned char)stem[0]) == 'l' &&
                stem[1] == '1';
    if (isL1 &&
        std::filesystem::exists(templatePath) &&
        std::filesystem::exists(scenePath) &&
        !std::filesystem::exists(outPath)) {
      cv::Mat tmpl   = cv::imread(templatePath.string(), cv::IMREAD_GRAYSCALE);
      cv::Mat scene  = cv::imread(scenePath.string(),    cv::IMREAD_GRAYSCALE);
      if (tmpl.empty() || scene.empty()) {
        std::cerr << "Extract skipped — failed to load template or scene"
                  << std::endl;
      } else {
        // Identify the actual content rectangle (largest rectangle
        // that is NOT the outer frame) so we can later trim the
        // extracted image to only the content-bearing area.  This
        // avoids pulling in adjacent LAF2 data from the empty white
        // portion of LAF1's bounds.
        double contentRectX = 0, contentRectW = 0;
        bool haveContentRect = false;
        if (!elements.rectangles.empty()) {
          double overallMinX =  std::numeric_limits<double>::infinity();
          double overallMaxX = -std::numeric_limits<double>::infinity();
          for (const auto &r : elements.rectangles) {
            overallMinX = std::min(overallMinX, r.x);
            overallMaxX = std::max(overallMaxX, r.x + r.width);
          }
          double overallWidth = overallMaxX - overallMinX;
          const double eps = 0.5;
          int bestIdx = -1;
          double bestArea = 0.0;
          for (size_t i = 0; i < elements.rectangles.size(); ++i) {
            const auto &r = elements.rectangles[i];
            if (r.width >= overallWidth - eps) continue;
            double a = r.width * r.height;
            if (a > bestArea) { bestArea = a; bestIdx = int(i); }
          }
          if (bestIdx >= 0) {
            contentRectX = elements.rectangles[bestIdx].x;
            contentRectW = elements.rectangles[bestIdx].width;
            haveContentRect = true;
            std::cout << "Content rect: x=" << contentRectX
                      << " w=" << contentRectW << " pt (bounds "
                      << overallMinX << ".." << overallMaxX << ")"
                      << std::endl;
          }
        }

        // Stage 1: build work300 ourselves so both detection and
        // extraction share the same cropToLabel result.
        cv::Mat backing300 =
            ocr::OCRAnalysis::cropToLabel(scene, 50, 40, false);
        if (backing300.empty()) backing300 = scene.clone();

        // Downsample backing300 to 72 DPI by the exact ratio.
        double detectScale = kDetectDpi / kRenderDpi;
        cv::Mat work72;
        cv::resize(backing300, work72, cv::Size(), detectScale, detectScale,
                   cv::INTER_AREA);
        std::cout << "Detect: backing300 " << backing300.cols << "x"
                  << backing300.rows << " -> work72 " << work72.cols << "x"
                  << work72.rows << std::endl;

        // Run the existing OCR-anchor-based localisation pipeline on
        // the 72-DPI downsample.  createRelativeMap will call
        // cropToLabel internally, but since work72 is already tightly
        // backing-cropped the second call is a near-no-op, so the
        // inner work image stays proportional to backing300.
        ocr::OCRAnalysis::RelativeMapResult rel =
            analyzer.createRelativeMap(
                elements, work72, /*imageFilePath=*/"",
                /*markImage=*/false, /*l1PdfPath=*/pdfPath,
                /*dpi=*/kDetectDpi);

        if (!rel.success || !rel.hasCropRect) {
          std::cerr << "Extract skipped — createRelativeMap failed: "
                    << rel.errorMessage << std::endl;
        } else {
          std::cout << "Detect: cropRect72 (" << rel.cropX << ", "
                    << rel.cropY << ") " << rel.cropWidth << "x"
                    << rel.cropHeight << "  cwRotations="
                    << rel.cwRotations << std::endl;

          // Replicate cropToLabel + rotation on backing300 to match
          // the inner work coordinate system.
          cv::Mat innerBacking300 =
              ocr::OCRAnalysis::cropToLabel(backing300, 50, 40, false);
          if (innerBacking300.empty()) innerBacking300 = backing300.clone();
          cv::Mat work300 = innerBacking300;
          for (int r = 0; r < rel.cwRotations; ++r) {
            cv::Mat tmp;
            cv::rotate(work300, tmp, cv::ROTATE_90_CLOCKWISE);
            work300 = tmp;
          }

          // Inner 72-DPI work image dimensions (what createRelativeMap
          // actually OCR'd against).
          cv::Mat innerWork72 =
              ocr::OCRAnalysis::cropToLabel(work72, 50, 40, false);
          if (innerWork72.empty()) innerWork72 = work72.clone();
          for (int r = 0; r < rel.cwRotations; ++r) {
            cv::Mat tmp;
            cv::rotate(innerWork72, tmp, cv::ROTATE_90_CLOCKWISE);
            innerWork72 = tmp;
          }

          double sx = double(work300.cols) / innerWork72.cols;
          double sy = double(work300.rows) / innerWork72.rows;
          std::cout << "Detect: innerWork72 " << innerWork72.cols << "x"
                    << innerWork72.rows << "  work300 " << work300.cols
                    << "x" << work300.rows << "  sx=" << std::fixed
                    << std::setprecision(4) << sx << " sy=" << sy
                    << " (expected " << kRenderDpi/kDetectDpi << ")"
                    << std::endl;

          cv::Rect rect300(
              int(std::round(rel.cropX      * sx)),
              int(std::round(rel.cropY      * sy)),
              int(std::round(rel.cropWidth  * sx)),
              int(std::round(rel.cropHeight * sy)));
          rect300.x = std::max(0, rect300.x);
          rect300.y = std::max(0, rect300.y);
          rect300.width  = std::min(work300.cols - rect300.x, rect300.width);
          rect300.height = std::min(work300.rows - rect300.y, rect300.height);

          if (rect300.width <= 0 || rect300.height <= 0) {
            std::cerr << "Extract skipped — scaled crop rect empty"
                      << std::endl;
          } else {
            cv::Mat rawCrop = work300(rect300).clone();
            // Resize the extracted region to the physical size of the
            // full LAF1 bounds at 300 DPI so the output shares the
            // same coordinate space as L1Packing_content.png and any
            // element found in LAF1 can be located by pixel position.
            int fullW = int(std::round(rel.boundsWidth  / 72.0 * kRenderDpi));
            int fullH = int(std::round(rel.boundsHeight / 72.0 * kRenderDpi));
            cv::Mat crop;
            cv::resize(rawCrop, crop, cv::Size(fullW, fullH), 0, 0,
                       cv::INTER_AREA);
            std::cout << "Detect: rawCrop " << rawCrop.cols << "x"
                      << rawCrop.rows << " -> final "
                      << crop.cols << "x" << crop.rows << std::endl;
            if (cv::imwrite(outPath.string(), crop)) {
              double widthIn  = crop.cols / kRenderDpi;
              double heightIn = crop.rows / kRenderDpi;
              double widthMm  = widthIn  * 25.4;
              double heightMm = heightIn * 25.4;
              std::ofstream dimsOut(graphicsDir / "dimensions.txt",
                                    std::ios::app);
              if (dimsOut) {
                dimsOut << outPath.filename().string()
                        << "  " << std::fixed << std::setprecision(2)
                        << widthMm << " x " << heightMm << " mm  ("
                        << std::setprecision(3)
                        << widthIn << " x " << heightIn << " in)  "
                        << crop.cols << " x " << crop.rows
                        << " px @ " << std::setprecision(0) << kRenderDpi
                        << " dpi\n";
              }
              std::cout << "Extracted LAF1 region -> "
                        << outPath.string() << std::endl;
            }
          }
        }
      }
    }
  }

  // Note: Rendering to PDF is not currently supported
  // The renderElementsToPDF function has been removed
  if (!renderOutputPath.empty()) {
    std::cerr << "Warning: PDF rendering is not currently supported"
              << std::endl;
  }

  // Summary
  std::cout << std::endl;
  std::cout << "+======================================================+"
            << std::endl;
  std::cout << "|              PDF EXTRACTION SUMMARY                  |"
            << std::endl;
  std::cout << "+======================================================+"
            << std::endl;
  std::cout << "|  Processing time: " << std::setw(10) << std::fixed
            << std::setprecision(2) << elements.processingTimeMs
            << " ms                     |" << std::endl;
  std::cout << "|  Pages:           " << std::setw(10) << elements.pageCount
            << "                       |" << std::endl;
  std::cout << "+======================================================+"
            << std::endl;
  std::cout << "|  Text lines:      " << std::setw(10) << elements.textLineCount
            << "                       |" << std::endl;
  std::cout << "|  Embedded images: " << std::setw(10) << elements.imageCount
            << "                       |" << std::endl;
  std::cout << "|  Rectangles:      " << std::setw(10)
            << elements.rectangleCount << "                       |"
            << std::endl;
  std::cout << "|  Graphic lines:   " << std::setw(10)
            << elements.graphicLineCount << "                       |"
            << std::endl;
  std::cout << "+======================================================+"
            << std::endl;

  // Crop mark detection info
  std::cout << std::endl << "=== CROP MARK DETECTION ===" << std::endl;
  std::cout << std::string(60, '-') << std::endl;
  // Note: Detailed crop mark info is no longer available in PDFElements
  // The linesBoundingBox fields contain the calculated interior box
  if (elements.linesBoundingBoxWidth > 0 &&
      elements.linesBoundingBoxHeight > 0) {
    std::cout << "Interior bounding box detected from lines:" << std::endl;
    std::cout << "  Position: (" << std::fixed << std::setprecision(1)
              << elements.linesBoundingBoxX << ", "
              << elements.linesBoundingBoxY << ")" << std::endl;
    std::cout << "  Dimensions: " << elements.linesBoundingBoxWidth << " x "
              << elements.linesBoundingBoxHeight << " points" << std::endl;
  } else {
    std::cout << "No interior bounding box detected" << std::endl;
  }
  std::cout << std::string(60, '-') << std::endl;

  // Text preview (first 500 chars)
  std::cout << std::endl << "=== TEXT PREVIEW ===" << std::endl;
  std::cout << std::string(60, '-') << std::endl;
  if (!elements.fullText.empty()) {
    std::string preview = sanitizeText(elements.fullText.substr(
        0, std::min(size_t(500), elements.fullText.size())));
    if (elements.fullText.size() > 500) {
      preview += "...\n[truncated]";
    }
    std::cout << preview << std::endl;
  } else {
    std::cout << "(No text extracted)" << std::endl;
  }
  std::cout << std::string(60, '-') << std::endl;

  // Text lines with orientation
  if (!elements.textLines.empty()) {
    std::cout << std::endl << "=== TEXT LINES (first 15) ===" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(12) << "Position" << std::setw(8)
              << "Orient"
              << "Text" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    int shown = 0;
    for (const auto &line : elements.textLines) {
      if (shown >= 15)
        break;

      std::ostringstream posStr;
      posStr << "(" << line.boundingBox.x << "," << line.boundingBox.y << ")";

      std::string orient =
          (line.orientation == ocr::TextOrientation::Horizontal) ? "[H]"
                                                                 : "[V]";
      std::string text = sanitizeText(line.text);
      if (text.size() > 45) {
        text = text.substr(0, 45) + "...";
      }

      std::cout << std::left << std::setw(12) << posStr.str() << std::setw(8)
                << orient << text << std::endl;
      shown++;
    }
    if (elements.textLineCount > 15) {
      std::cout << "  ... and " << (elements.textLineCount - 15) << " more"
                << std::endl;
    }
    std::cout << std::string(80, '-') << std::endl;
  }

  // Embedded images
  if (!elements.images.empty()) {
    std::cout << std::endl << "=== EMBEDDED IMAGES ===" << std::endl;

    // Create images folder if it doesn't exist
    std::filesystem::path imagesDir = "images";
    try {
      if (!std::filesystem::exists(imagesDir)) {
        std::filesystem::create_directory(imagesDir);
        std::cout << "Created images folder: "
                  << std::filesystem::absolute(imagesDir) << std::endl;
      }
    } catch (const std::exception &e) {
      std::cerr << "Warning: Failed to create images folder: " << e.what()
                << std::endl;
    }

    int imageIndex = 0;
    for (const auto &img : elements.images) {
      std::cout << "  Page " << img.pageNumber << ": " << img.width << "x"
                << img.height << " at (" << static_cast<int>(img.x) << ", "
                << static_cast<int>(img.y) << ")";

      // Save image if it has valid data
      if (!img.image.empty()) {
        std::ostringstream filename;
        filename << "images/image_page" << img.pageNumber << "_" << imageIndex
                 << ".png";

        try {
          cv::imwrite(filename.str(), img.image);
          std::cout << " -> Saved to " << filename.str();
        } catch (const std::exception &e) {
          std::cout << " -> Failed to save: " << e.what();
        }
      }
      std::cout << std::endl;
      imageIndex++;
    }
  }

  // Rectangles
  if (!elements.rectangles.empty()) {
    std::cout << std::endl << "=== RECTANGLES (first 10) ===" << std::endl;
    int shown = 0;
    for (const auto &rect : elements.rectangles) {
      if (shown >= 10)
        break;
      std::cout << "  Page " << rect.pageNumber << ": " << std::fixed
                << std::setprecision(1) << rect.width << "x" << rect.height
                << " at (" << rect.x << ", " << rect.y << ")"
                << (rect.filled ? " [filled]" : "")
                << (rect.stroked ? " [stroked]" : "") << std::endl;
      shown++;
    }
    if (elements.rectangleCount > 10) {
      std::cout << "  ... and " << (elements.rectangleCount - 10) << " more"
                << std::endl;
    }
  }

  // Graphic lines details
  if (!elements.graphicLines.empty()) {
    std::cout << std::endl << "=== GRAPHIC LINES (first 15) ===" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    std::cout << std::left << std::setw(6) << "Page" << std::setw(25)
              << "From (x1,y1)" << std::setw(25) << "To (x2,y2)"
              << std::setw(12) << "Length" << std::setw(10) << "Width"
              << "Orient" << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    int shown = 0;
    for (const auto &line : elements.graphicLines) {
      if (shown >= 15)
        break;

      std::ostringstream fromStr, toStr;
      fromStr << std::fixed << std::setprecision(1) << "(" << line.x1 << ", "
              << line.y1 << ")";
      toStr << std::fixed << std::setprecision(1) << "(" << line.x2 << ", "
            << line.y2 << ")";

      std::string orient =
          line.isHorizontal ? "[H]" : (line.isVertical ? "[V]" : "[D]");

      std::cout << std::left << std::setw(6) << line.pageNumber << std::setw(25)
                << fromStr.str() << std::setw(25) << toStr.str()
                << std::setw(12) << std::fixed << std::setprecision(1)
                << line.length << std::setw(10) << std::fixed
                << std::setprecision(2) << line.lineWidth << orient
                << std::endl;
      shown++;
    }

    if (elements.graphicLineCount > 15) {
      std::cout << "  ... and " << (elements.graphicLineCount - 15) << " more"
                << std::endl;
    }
    std::cout << std::string(100, '-') << std::endl;

    // Also show summary counts
    int hCount = 0, vCount = 0, dCount = 0;
    for (const auto &line : elements.graphicLines) {
      if (line.isHorizontal)
        hCount++;
      else if (line.isVertical)
        vCount++;
      else
        dCount++;
    }
    std::cout << "Summary: " << hCount << " horizontal, " << vCount
              << " vertical, " << dCount << " diagonal" << std::endl;

    // Display interior bounding box (largest box inside the found lines)
    std::cout << std::endl << "=== INTERIOR BOUNDING BOX ===" << std::endl;
    std::cout << "  (Largest box inside the found lines)" << std::endl;
    std::cout << "  Position: (" << std::fixed << std::setprecision(1)
              << elements.linesBoundingBoxX << ", "
              << elements.linesBoundingBoxY << ")" << std::endl;
    std::cout << "  Dimensions: " << elements.linesBoundingBoxWidth << " x "
              << elements.linesBoundingBoxHeight << " points" << std::endl;
  }

  std::cout << std::endl << "Extraction completed successfully!" << std::endl;

  return 0;
}
