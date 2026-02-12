// Simplified stripBleedMarks - uses existing linesBoundingBox from
// extractPDFElements

OCRAnalysis::PDFElements
OCRAnalysis::stripBleedMarks(const std::string &pdfPath) {
  PDFElements result;
  result.success = false;

  auto startTime = std::chrono::high_resolution_clock::now();

  try {
    // Extract all PDF elements - this already detects crop marks and calculates
    // linesBoundingBox
    PDFElements allElements = extractPDFElements(pdfPath);

    if (!allElements.success) {
      result.errorMessage =
          "Failed to extract PDF elements: " + allElements.errorMessage;
      return result;
    }

    // Use the linesBoundingBox as the crop box (this is already calculated from
    // crop marks)
    result = allElements;

    // Set the page dimensions to the lines bounding box (crop box)
    result.pageX = allElements.linesBoundingBoxX;
    result.pageY = allElements.linesBoundingBoxY;
    result.pageWidth = allElements.linesBoundingBoxWidth;
    result.pageHeight = allElements.linesBoundingBoxHeight;

    // Validate that the crop box has reasonable dimensions
    if (result.pageWidth < 100.0 || result.pageHeight < 100.0) {
      result.errorMessage = "Detected crop box is too small (" +
                            std::to_string(result.pageWidth) + " x " +
                            std::to_string(result.pageHeight) + " points). " +
                            "Crop marks may not be correctly detected.";
      return result;
    }

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
