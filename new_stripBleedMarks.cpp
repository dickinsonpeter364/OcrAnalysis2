// Temporary file with new stripBleedMarks implementation
// This will be copied into OCRAnalysis.cpp

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

    // Structure to hold a crop mark (two perpendicular lines)
    struct CropMark {
      int line1Idx; // Index of first line
      int line2Idx; // Index of second line
      double cropX; // Intersection point X
      double cropY; // Intersection point Y
    };

    std::vector<CropMark> cropMarks;
    const double perpendicularTolerance = 5.0; // degrees tolerance
    const double proximityTolerance = 20.0;    // max box size for crop mark

    // Find all pairs of perpendicular lines that could be crop marks
    for (size_t i = 0; i < allElements.graphicLines.size(); i++) {
      const auto &line1 = allElements.graphicLines[i];

      for (size_t j = i + 1; j < allElements.graphicLines.size(); j++) {
        const auto &line2 = allElements.graphicLines[j];

        // Calculate angles of both lines
        double angle1 =
            std::atan2(line1.y2 - line1.y1, line1.x2 - line1.x1) * 180.0 / M_PI;
        double angle2 =
            std::atan2(line2.y2 - line2.y1, line2.x2 - line2.x1) * 180.0 / M_PI;

        // Normalize angles to [0, 180)
        angle1 = std::fmod(std::abs(angle1), 180.0);
        angle2 = std::fmod(std::abs(angle2), 180.0);

        // Check if lines are perpendicular (90 degrees apart)
        double angleDiff = std::abs(angle1 - angle2);
        if (angleDiff > 90.0)
          angleDiff = 180.0 - angleDiff;

        bool isPerpendicular =
            std::abs(angleDiff - 90.0) < perpendicularTolerance;

        if (!isPerpendicular)
          continue;

        // Check if lines are close (can be enclosed in a small box)
        double minX = std::min({line1.x1, line1.x2, line2.x1, line2.x2});
        double maxX = std::max({line1.x1, line1.x2, line2.x1, line2.x2});
        double minY = std::min({line1.y1, line1.y2, line2.y1, line2.y2});
        double maxY = std::max({line1.y1, line1.y2, line2.y1, line2.y2});

        double boxSize = std::max(maxX - minX, maxY - minY);

        if (boxSize > proximityTolerance)
          continue;

        // Calculate intersection point when lines are extended
        double x1 = line1.x1, y1 = line1.y1, x2 = line1.x2, y2 = line1.y2;
        double x3 = line2.x1, y3 = line2.y1, x4 = line2.x2, y4 = line2.y2;

        double denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

        if (std::abs(denom) < 1e-10)
          continue; // Lines are parallel

        double intersectX = ((x1 * y2 - y1 * x2) * (x3 - x4) -
                             (x1 - x2) * (x3 * y4 - y3 * x4)) /
                            denom;
        double intersectY = ((x1 * y2 - y1 * x2) * (y3 - y4) -
                             (y1 - y2) * (x3 * y4 - y3 * x4)) /
                            denom;

        // Store this crop mark
        CropMark mark;
        mark.line1Idx = i;
        mark.line2Idx = j;
        mark.cropX = intersectX;
        mark.cropY = intersectY;
        cropMarks.push_back(mark);
      }
    }

    // We should have exactly 4 crop marks
    if (cropMarks.size() < 4) {
      result.errorMessage = "Could not find 4 crop marks. Found: " +
                            std::to_string(cropMarks.size());
      return result;
    }

    // If we have more than 4, select the 4 corner-most ones
    if (cropMarks.size() > 4) {
      // Find extremes
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

      // Select the 4 corners
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

    // Calculate the crop box from the 4 crop points
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

      // Mark these lines for removal
      cropMarkLineIndices.insert(mark.line1Idx);
      cropMarkLineIndices.insert(mark.line2Idx);
    }

    // Build the filtered result - copy all elements except crop mark lines
    result = allElements;
    result.graphicLines.clear();

    for (size_t i = 0; i < allElements.graphicLines.size(); i++) {
      if (cropMarkLineIndices.find(i) == cropMarkLineIndices.end()) {
        result.graphicLines.push_back(allElements.graphicLines[i]);
      }
    }

    // Update the crop box in the result
    result.pageX = cropMinX;
    result.pageY = cropMinY;
    result.pageWidth = cropMaxX - cropMinX;
    result.pageHeight = cropMaxY - cropMinY;

    // Update counts
    result.graphicLineCount = result.graphicLines.size();
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
