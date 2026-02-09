
OCRAnalysis::PNGRenderResult
OCRAnalysis::renderElementsToPNG(const PDFElements &elements,
                                 const std::string &pdfPath, double dpi,
                                 const std::string &outputDir) {

  PNGRenderResult result;

  try {
    // Find bounding box of all elements
    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double maxY = std::numeric_limits<double>::lowest();

    // Check text lines (already in top-left coordinates)
    for (const auto &text : elements.textLines) {
      minX = std::min(minX, static_cast<double>(text.boundingBox.x));
      minY = std::min(minY, static_cast<double>(text.boundingBox.y));
      maxX = std::max(maxX, static_cast<double>(text.boundingBox.x +
                                                text.boundingBox.width));
      maxY = std::max(maxY, static_cast<double>(text.boundingBox.y +
                                                text.boundingBox.height));
    }

    // Check images (PDF bottom-left coordinates)
    for (const auto &img : elements.images) {
      minX = std::min(minX, img.x);
      minY = std::min(minY, img.y);
      maxX = std::max(maxX, img.x + img.displayWidth);
      maxY = std::max(maxY, img.y + img.displayHeight);
    }

    // Check rectangles (PDF bottom-left coordinates)
    for (const auto &rect : elements.rectangles) {
      minX = std::min(minX, rect.x);
      minY = std::min(minY, rect.y);
      maxX = std::max(maxX, rect.x + rect.width);
      maxY = std::max(maxY, rect.y + rect.height);
    }

    // Check lines (PDF bottom-left coordinates)
    for (const auto &line : elements.graphicLines) {
      minX = std::min(minX, std::min(line.x1, line.x2));
      minY = std::min(minY, std::min(line.y1, line.y2));
      maxX = std::max(maxX, std::max(line.x1, line.x2));
      maxY = std::max(maxY, std::max(line.y1, line.y2));
    }

    if (minX == std::numeric_limits<double>::max()) {
      result.errorMessage = "No elements to render";
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
    cairo_set_source_rgb(cr, 0.8, 0.8, 0.8);
    cairo_set_line_width(cr, 1.0 / scale);
    for (const auto &rect : elements.rectangles) {
      double x = rect.x - minX + margin;
      // Convert from PDF bottom-left to Cairo top-left
      double y = pageHeightPt - (rect.y - minY + rect.height) - margin;
      cairo_rectangle(cr, x, y, rect.width, rect.height);
      cairo_stroke(cr);

      // Add to result
      RenderedElement elem;
      elem.type = RenderedElement::RECTANGLE;
      elem.pixelX = static_cast<int>(x * scale);
      elem.pixelY = static_cast<int>(y * scale);
      elem.pixelWidth = static_cast<int>(rect.width * scale);
      elem.pixelHeight = static_cast<int>(rect.height * scale);
      result.elements.push_back(elem);
    }

    // Draw lines (PDF bottom-left origin -> convert to top-left)
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    cairo_set_line_width(cr, 0.5 / scale);
    for (const auto &line : elements.graphicLines) {
      double x1 = line.x1 - minX + margin;
      double x2 = line.x2 - minX + margin;
      // Convert from PDF bottom-left to Cairo top-left
      double y1 = pageHeightPt - (line.y1 - minY) - margin;
      double y2 = pageHeightPt - (line.y2 - minY) - margin;
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
    for (const auto &img : elements.images) {
      double x = img.x - minX + margin;
      // Convert from PDF bottom-left to Cairo top-left
      double y = pageHeightPt - (img.y - minY + img.displayHeight) - margin;

      if (!img.image.empty()) {
        // Convert cv::Mat to Cairo surface and draw
        cairo_save(cr);
        cairo_translate(cr, x, y);

        // Scale to display dimensions
        cairo_scale(cr, img.displayWidth / static_cast<double>(img.image.cols),
                    img.displayHeight / static_cast<double>(img.image.rows));

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

    // Draw text (already in top-left coordinates, no conversion needed)
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL,
                           CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, 10.0);

    for (const auto &text : elements.textLines) {
      double x = text.boundingBox.x - minX + margin;
      double y = text.boundingBox.y - minY + margin + 10; // Baseline offset

      cairo_move_to(cr, x, y);
      cairo_show_text(cr, text.text.c_str());

      // Add to result
      RenderedElement elem;
      elem.type = RenderedElement::TEXT;
      elem.pixelX = static_cast<int>(x * scale);
      elem.pixelY = static_cast<int>((y - 10) * scale);
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
