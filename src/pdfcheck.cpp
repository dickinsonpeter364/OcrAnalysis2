#include "OCRAnalysis.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

static bool isL1(const std::string &stem) {
  return stem.size() >= 2 &&
         (stem[0] == 'L' || stem[0] == 'l') && stem[1] == '1';
}

static bool isL2(const std::string &stem) {
  return stem.size() >= 2 &&
         (stem[0] == 'L' || stem[0] == 'l') && stem[1] == '2';
}

static bool regionInBounds(const ocr::TextRegion &r, double minX, double minY,
                           double maxX, double maxY) {
  double cx = r.preciseX + r.preciseWidth / 2.0;
  double cy = r.preciseY + r.preciseHeight / 2.0;
  return cx >= minX && cx <= maxX && cy >= minY && cy <= maxY;
}

static bool dataMatrixInBounds(const ocr::OCRAnalysis::PDFDataMatrix &dm,
                               double minX, double minY, double maxX,
                               double maxY) {
  double cx = dm.x + dm.width / 2.0;
  double cy = dm.y + dm.height / 2.0;
  return cx >= minX && cx <= maxX && cy >= minY && cy <= maxY;
}

static void ensureEmptyDir(const fs::path &dir) {
  if (fs::exists(dir)) {
    for (auto &entry : fs::directory_iterator(dir))
      fs::remove_all(entry.path());
  } else {
    fs::create_directories(dir);
  }
}

// Draw annotation boxes on a rendered image.
// Red 5px boxes for hidden elements, green 5px boxes for DataMatrix.
static void annotateImage(
    cv::Mat &img, double dpi, double roiMinX, double roiMinY, double roiMaxY,
    const std::vector<ocr::TextRegion> &hiddenInROI,
    const std::vector<ocr::OCRAnalysis::PDFDataMatrix> &dmInROI) {

  double scale = dpi / 72.0;
  double roiH = roiMaxY - roiMinY;

  for (const auto &h : hiddenInROI) {
    double pxX = (h.preciseX - roiMinX) * scale;
    double pxY =
        (roiH - (h.preciseY - roiMinY + h.preciseHeight)) * scale;
    double pxW = h.preciseWidth * scale;
    double pxH = h.preciseHeight * scale;
    int x1 = std::max(0, static_cast<int>(pxX));
    int y1 = std::max(0, static_cast<int>(pxY));
    int x2 = std::min(img.cols - 1, static_cast<int>(pxX + pxW));
    int y2 = std::min(img.rows - 1, static_cast<int>(pxY + pxH));
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                  cv::Scalar(0, 0, 255), 5);
  }

  for (const auto &dm : dmInROI) {
    double pxX = (dm.x - roiMinX) * scale;
    double pxY = (roiH - (dm.y - roiMinY + dm.height)) * scale;
    double pxW = dm.width * scale;
    double pxH = dm.height * scale;
    int x1 = std::max(0, static_cast<int>(pxX));
    int y1 = std::max(0, static_cast<int>(pxY));
    int x2 = std::min(img.cols - 1, static_cast<int>(pxX + pxW));
    int y2 = std::min(img.rows - 1, static_cast<int>(pxY + pxH));
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                  cv::Scalar(0, 200, 0), 5);
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: pdfcheck <folder>" << std::endl;
    std::cerr << "  Processes all PDF files in <folder>." << std::endl;
    std::cerr << "  Creates Processed/ and Anomalies/ subfolders." << std::endl;
    return 1;
  }

  fs::path inputDir(argv[1]);
  if (!fs::is_directory(inputDir)) {
    std::cerr << "Error: \"" << inputDir.string() << "\" is not a directory."
              << std::endl;
    return 1;
  }

  fs::path processedDir = inputDir / "Processed";
  fs::path anomaliesDir = inputDir / "Anomalies";

  std::cout << "Input folder : " << inputDir.string() << std::endl;
  std::cout << "Processed    : " << processedDir.string() << std::endl;
  std::cout << "Anomalies    : " << anomaliesDir.string() << std::endl;

  ensureEmptyDir(processedDir);
  ensureEmptyDir(anomaliesDir);

  // Collect PDF files
  std::vector<fs::path> pdfFiles;
  for (auto &entry : fs::directory_iterator(inputDir)) {
    if (!entry.is_regular_file())
      continue;
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".pdf")
      pdfFiles.push_back(entry.path());
  }

  if (pdfFiles.empty()) {
    std::cout << "No PDF files found in " << inputDir.string() << std::endl;
    return 0;
  }

  std::sort(pdfFiles.begin(), pdfFiles.end());
  std::cout << "Found " << pdfFiles.size() << " PDF file(s)." << std::endl
            << std::endl;

  ocr::OCRAnalysis analyzer;
  const double dpi = 300.0;

  for (const auto &pdfPath : pdfFiles) {
    std::string stem = pdfPath.stem().string();
    std::string pdfStr = pdfPath.string();
    std::cout << "--- Processing: " << stem << " ---" << std::endl;

    // 1. Extract elements
    ocr::OCRAnalysis::PDFElements elements =
        analyzer.extractPDFElements(pdfStr);

    if (!elements.success) {
      std::cerr << "  FAILED to extract: " << elements.errorMessage
                << std::endl;
      std::ofstream note(anomaliesDir / (stem + "_note.txt"));
      note << "Failed to extract PDF elements: " << elements.errorMessage
           << std::endl;
      continue;
    }

    // Report DataMatrix barcodes
    if (!elements.dataMatrices.empty()) {
      std::cout << "  DataMatrix barcodes: " << elements.dataMatrices.size()
                << std::endl;
      for (const auto &dm : elements.dataMatrices)
        std::cout << "    \"" << dm.text.substr(0, 40) << "\"" << std::endl;
    }

    // 2. Determine bounds mode and ROI
    bool l1 = isL1(stem);
    bool l2 = isL2(stem);
    ocr::OCRAnalysis::RenderBoundsMode boundsMode =
        l2 ? ocr::OCRAnalysis::RenderBoundsMode::USE_CROP_MARKS
           : ocr::OCRAnalysis::RenderBoundsMode::USE_LARGEST_RECTANGLE;

    double roiMinX = 0, roiMinY = 0, roiMaxX = 0, roiMaxY = 0;

    if (boundsMode ==
        ocr::OCRAnalysis::RenderBoundsMode::USE_LARGEST_RECTANGLE) {
      double largestArea = 0;
      const ocr::OCRAnalysis::PDFRectangle *largestRect = nullptr;
      for (const auto &rect : elements.rectangles) {
        double area = rect.width * rect.height;
        if (area > largestArea) {
          largestArea = area;
          largestRect = &rect;
        }
      }

      if (largestRect) {
        double rectTopLeftY = largestRect->y;
        double rectBottomLeftY = rectTopLeftY + largestRect->height;
        roiMinX = largestRect->x;
        roiMinY = elements.pageHeight - rectBottomLeftY;
        roiMaxX = largestRect->x + largestRect->width;
        roiMaxY = elements.pageHeight - rectTopLeftY;
        std::cout << "  ROI: largest rectangle (" << largestRect->width
                  << " x " << largestRect->height << " pt)" << std::endl;
      } else if (l1 && !elements.images.empty()) {
        const ocr::OCRAnalysis::PDFEmbeddedImage *largestImg = nullptr;
        double largestImgArea = 0;
        for (const auto &img : elements.images) {
          double area = img.displayWidth * img.displayHeight;
          if (area > largestImgArea) {
            largestImgArea = area;
            largestImg = &img;
          }
        }

        if (largestImg &&
            largestImgArea > elements.pageWidth * elements.pageHeight * 0.1) {
          std::cout << "  ANOMALY: L1 has no rectangle but has image "
                       "resembling one ("
                    << largestImg->displayWidth << " x "
                    << largestImg->displayHeight << " pt)" << std::endl;

          ocr::OCRAnalysis::PNGRenderResult renderResult =
              analyzer.renderElementsToPNG(elements, pdfStr, dpi,
                                           anomaliesDir.string(), boundsMode);
          std::ofstream note(anomaliesDir / (stem + "_note.txt"));
          note << "L1 file has no rectangle but contains an image "
                  "resembling a rectangle."
               << std::endl;
          note << "Image: " << largestImg->displayWidth << " x "
               << largestImg->displayHeight << " pt at ("
               << largestImg->x << ", " << largestImg->y << ")" << std::endl;
          if (renderResult.success)
            note << "Rendered to: " << renderResult.outputPath << std::endl;
          continue;
        }

        std::cout << "  ANOMALY: L1 has no rectangle and no suitable image"
                  << std::endl;
        std::ofstream note(anomaliesDir / (stem + "_note.txt"));
        note << "L1 file has no rectangle of interest and no image "
                "resembling a rectangle."
             << std::endl;
        continue;
      } else {
        std::cout << "  ANOMALY: No rectangle found" << std::endl;
        std::ofstream note(anomaliesDir / (stem + "_note.txt"));
        note << "No rectangle of interest found in PDF." << std::endl;
        continue;
      }
    } else {
      // Crop marks mode
      if (elements.linesBoundingBoxWidth > 0 &&
          elements.linesBoundingBoxHeight > 0) {
        roiMinX = elements.linesBoundingBoxX;
        roiMinY = elements.linesBoundingBoxY;
        roiMaxX = roiMinX + elements.linesBoundingBoxWidth;
        roiMaxY = roiMinY + elements.linesBoundingBoxHeight;
        std::cout << "  ROI: crop marks (" << elements.linesBoundingBoxWidth
                  << " x " << elements.linesBoundingBoxHeight << " pt)"
                  << std::endl;
      } else {
        std::cout << "  WARNING: No crop marks found, using full page"
                  << std::endl;
        roiMinX = elements.pageX;
        roiMinY = elements.pageY;
        roiMaxX = elements.pageX + elements.pageWidth;
        roiMaxY = elements.pageY + elements.pageHeight;
      }
    }

    // 3. Collect hidden text within ROI
    std::vector<ocr::TextRegion> hiddenInROI;
    for (const auto &hidden : elements.hiddenTextLines) {
      if (regionInBounds(hidden, roiMinX, roiMinY, roiMaxX, roiMaxY))
        hiddenInROI.push_back(hidden);
    }

    // 4. Collect DataMatrix barcodes within ROI (these are valid)
    std::vector<ocr::OCRAnalysis::PDFDataMatrix> dmInROI;
    for (const auto &dm : elements.dataMatrices) {
      if (dataMatrixInBounds(dm, roiMinX, roiMinY, roiMaxX, roiMaxY))
        dmInROI.push_back(dm);
    }

    // 5. Route: hidden elements in ROI → Anomalies, otherwise → Processed
    bool isAnomaly = !hiddenInROI.empty();
    fs::path destDir = isAnomaly ? anomaliesDir : processedDir;

    ocr::OCRAnalysis::PNGRenderResult renderResult =
        analyzer.renderElementsToPNG(elements, pdfStr, dpi,
                                     destDir.string(), boundsMode);

    if (!renderResult.success) {
      std::cerr << "  FAILED to render: " << renderResult.errorMessage
                << std::endl;
      std::ofstream note(anomaliesDir / (stem + "_note.txt"));
      note << "Failed to render PDF: " << renderResult.errorMessage
           << std::endl;
      continue;
    }

    // 6. Annotate: red boxes for hidden elements, green for DataMatrix
    if (!hiddenInROI.empty() || !dmInROI.empty()) {
      cv::Mat img = cv::imread(renderResult.outputPath, cv::IMREAD_COLOR);
      if (!img.empty()) {
        annotateImage(img, dpi, roiMinX, roiMinY, roiMaxY, hiddenInROI,
                      dmInROI);
        // Overwrite the rendered image with the annotated version
        cv::imwrite(renderResult.outputPath, img);
      }
    }

    if (isAnomaly) {
      std::cout << "  ANOMALY: " << hiddenInROI.size()
                << " hidden text element(s) in ROI → Anomalies" << std::endl;
      for (const auto &h : hiddenInROI)
        std::cout << "    \"" << h.text << "\"" << std::endl;

      std::ofstream note(anomaliesDir / (stem + "_note.txt"));
      note << "Hidden text elements found within the rectangle of interest:"
           << std::endl;
      for (const auto &h : hiddenInROI)
        note << "  \"" << h.text << "\" at (" << h.preciseX << ", "
             << h.preciseY << ") size " << h.preciseWidth << " x "
             << h.preciseHeight << " pt" << std::endl;
    } else {
      std::cout << "  OK → Processed";
      if (!dmInROI.empty())
        std::cout << " [" << dmInROI.size() << " DataMatrix marked green]";
      std::cout << std::endl;
    }

    std::cout << std::endl;
  }

  // Summary
  int processedCount = 0, anomalyCount = 0;
  for (auto &entry : fs::directory_iterator(processedDir)) {
    if (entry.path().extension() == ".png")
      processedCount++;
  }
  if (fs::exists(anomaliesDir)) {
    for (auto &entry : fs::directory_iterator(anomaliesDir)) {
      if (entry.path().extension() == ".txt")
        anomalyCount++;
    }
  }

  std::cout << "=== Summary ===" << std::endl;
  std::cout << "  Processed: " << processedCount << " PDF(s) rendered OK"
            << std::endl;
  std::cout << "  Anomalies: " << anomalyCount << " PDF(s) flagged"
            << std::endl;

  return 0;
}
