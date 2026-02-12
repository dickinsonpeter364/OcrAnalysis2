#include "OCRAnalysis.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
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
  ocr::OCRAnalysis::PDFElements elements = analyzer.extractPDFElements(pdfPath);

  if (!elements.success) {
    std::cerr << "Failed to extract elements: " << elements.errorMessage
              << std::endl;
    return 1;
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
