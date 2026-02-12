// Complete stripBleedMarks implementation
// Step 1: Remove bleed marks (horizontal rectangles + lines)
// Step 2: Detect crop marks (L-shaped perpendicular line pairs)
// Step 3: Calculate crop box from 4 crop points

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
        rectangleGroups.push_back(group);
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
      for (size_t lineIdx = 0; lineIdx < allElements.graphicLines.size();
           lineIdx++) {
        const auto &line = allElements.graphicLines[lineIdx];

        double lineMinY = std::min(line.y1, line.y2);
        double lineMaxY = std::max(line.y1, line.y2);

        bool lineInYRange = !(lineMaxY < minY - connectionTolerance ||
                              lineMinY > maxY + connectionTolerance);

        if (!lineInYRange)
          continue;

        double lineMinX = std::min(line.x1, line.x2);
        double lineMaxX = std::max(line.x1, line.x2);

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

    // Find perpendicular line pairs in filtered lines
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

        CropMark mark;
        mark.line1Idx = i;
        mark.line2Idx = j;
        mark.cropX = intersectX;
        mark.cropY = intersectY;
        cropMarks.push_back(mark);
      }
    }

    if (cropMarks.size() < 4) {
      result.errorMessage = "Could not find 4 crop marks. Found: " +
                            std::to_string(cropMarks.size());
      return result;
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
