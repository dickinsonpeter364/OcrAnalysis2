#ifndef OCR_ANALYSIS_HPP
#define OCR_ANALYSIS_HPP

#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>

#include <memory>
#include <string>
#include <vector>

namespace ocr {

/**
 * @brief Text orientation detected in a region
 */
enum class TextOrientation {
  Horizontal, ///< Text is oriented horizontally (left-to-right or
              ///< right-to-left)
  Vertical,   ///< Text is oriented vertically (top-to-bottom or bottom-to-top)
  Unknown     ///< Orientation could not be determined
};

/**
 * @brief Represents a detected text region with its bounding box and confidence
 */
struct TextRegion {
  cv::Rect boundingBox; ///< Bounding rectangle of the text region
  std::string text;     ///< Recognized text content
  float confidence;     ///< Confidence score (0-100)
  int level;            ///< Hierarchy level (word, line, paragraph, block)
  TextOrientation orientation; ///< Detected text orientation

  // Font information (from PDF extraction)
  std::string fontName = ""; ///< Font family name
  double fontSize = 0.0;     ///< Font size in points
  bool isBold = false;       ///< Whether font is bold
  bool isItalic = false;     ///< Whether font is italic
};

/**
 * @brief Result of OCR analysis containing extracted text and metadata
 */
struct OCRResult {
  std::string fullText;            ///< Complete extracted text
  std::vector<TextRegion> regions; ///< Individual text regions
  double processingTimeMs;         ///< Processing time in milliseconds
  bool success;                    ///< Whether OCR was successful
  std::string errorMessage;        ///< Error message if failed
};

/**
 * @brief Configuration options for OCR processing
 */
struct OCRConfig {
  std::string language = "eng"; ///< Language code (e.g., "eng", "deu", "fra")
  tesseract::PageSegMode pageSegMode =
      tesseract::PSM_AUTO;     ///< Page segmentation mode
  bool preprocessImage = true; ///< Apply preprocessing (grayscale, threshold)
  int minConfidence = 0;       ///< Minimum confidence threshold (0-100)
  std::string tessDataPath =
      ""; ///< Path to tessdata directory (empty = default)
};

/**
 * @brief Main class for OCR analysis using OpenCV and Tesseract
 *
 * This class provides functionality to extract text from images using
 * Tesseract OCR with OpenCV for image preprocessing.
 *
 * Example usage:
 * @code
 * ocr::OCRAnalysis analyzer;
 * if (analyzer.initialize()) {
 *     auto result = analyzer.analyzeImage("document.png");
 *     if (result.success) {
 *         std::cout << result.fullText << std::endl;
 *     }
 * }
 * @endcode
 */
class OCRAnalysis {
public:
  /**
   * @brief Default constructor
   */
  OCRAnalysis();

  /**
   * @brief Constructor with custom configuration
   * @param config OCR configuration options
   */
  explicit OCRAnalysis(const OCRConfig &config);

  /**
   * @brief Destructor
   */
  ~OCRAnalysis();

  // Disable copy operations (Tesseract API is not copyable)
  OCRAnalysis(const OCRAnalysis &) = delete;
  OCRAnalysis &operator=(const OCRAnalysis &) = delete;

  // Enable move operations
  OCRAnalysis(OCRAnalysis &&other) noexcept;
  OCRAnalysis &operator=(OCRAnalysis &&other) noexcept;

  /**
   * @brief Initialize the OCR engine
   * @return true if initialization was successful, false otherwise
   */
  bool initialize();

  /**
   * @brief Check if the OCR engine is initialized
   * @return true if initialized, false otherwise
   */
  bool isInitialized() const;

  /**
   * @brief Analyze an image file and extract text
   * @param imagePath Path to the image file
   * @return OCRResult containing extracted text and metadata
   */
  OCRResult analyzeImage(const std::string &imagePath);

  /**
   * @brief Analyze an OpenCV Mat image and extract text
   * @param image OpenCV Mat image to analyze
   * @return OCRResult containing extracted text and metadata
   */
  OCRResult analyzeImage(const cv::Mat &image);

  /**
   * @brief Extraction level for PDF text extraction
   */
  enum class PDFExtractionLevel {
    Word, ///< Extract individual words (default)
    Line  ///< Group words into lines based on position
  };

  /**
   * @brief Extract text from a PDF file using Poppler
   *
   * This method extracts text directly from a PDF file without OCR.
   * It uses the Poppler library to parse the PDF and extract embedded text.
   * For scanned PDFs (images without embedded text), use analyzeImage()
   * instead.
   *
   * @param pdfPath Path to the PDF file
   * @param level Extraction level - Word (default) or Line
   * @return OCRResult containing extracted text with position and orientation
   */
  OCRResult
  extractTextFromPDF(const std::string &pdfPath,
                     PDFExtractionLevel level = PDFExtractionLevel::Word);

  /**
   * @brief Represents a graphic/image extracted from a PDF
   */
  struct PDFGraphic {
    cv::Mat image;  ///< The rendered image data
    int pageNumber; ///< 1-indexed page number
    int width;      ///< Width in pixels
    int height;     ///< Height in pixels
    double dpi;     ///< Resolution used for rendering
  };

  /**
   * @brief Result of PDF graphics extraction
   */
  struct PDFGraphicsResult {
    bool success = false;          ///< Whether extraction succeeded
    std::string errorMessage;      ///< Error message if failed
    std::vector<PDFGraphic> pages; ///< Rendered page images
    double processingTimeMs = 0;   ///< Processing time in milliseconds
  };

  /**
   * @brief Extract graphics from the first page of a PDF file by rendering as
   * an image
   *
   * This method renders the first page of the PDF as an image at the specified
   * resolution. This is useful for:
   * - Extracting visual content from PDFs
   * - Performing OCR on scanned documents
   * - Processing PDF graphics with image analysis
   *
   * @param pdfPath Path to the PDF file
   * @param dpi Resolution for rendering (default: 150 DPI)
   * @return PDFGraphicsResult containing rendered page image
   */
  PDFGraphicsResult extractGraphicsFromPDF(const std::string &pdfPath,
                                           double dpi = 150.0);

  /**
   * @brief Represents an embedded image extracted from a PDF
   */
  struct PDFEmbeddedImage {
    cv::Mat image;        ///< The image data
    int pageNumber;       ///< 1-indexed page number
    int imageIndex;       ///< Index of this image on the page (0-indexed)
    int width;            ///< Width in pixels
    int height;           ///< Height in pixels
    double x;             ///< X position on page (PDF coordinates)
    double y;             ///< Y position on page (PDF coordinates)
    double displayWidth;  ///< Display width on page
    double displayHeight; ///< Display height on page
    double rotationAngle; ///< Rotation angle in radians (counterclockwise)
    std::string type;     ///< Image type (e.g., "JPEG", "PNG", "raw")
  };

  /**
   * @brief Result of embedded image extraction
   */
  struct PDFEmbeddedImagesResult {
    bool success = false;                 ///< Whether extraction succeeded
    std::string errorMessage;             ///< Error message if failed
    std::vector<PDFEmbeddedImage> images; ///< Extracted embedded images
    double processingTimeMs = 0;          ///< Processing time in milliseconds
  };

  /**
   * @brief Extract embedded images from the first page of a PDF file
   *
   * This method uses Poppler's lower-level API to extract the actual
   * embedded image objects from the first page of a PDF, rather than rendering
   * pages. This is useful for:
   * - Extracting original quality images without re-rendering
   * - Getting images in their native resolution
   * - Accessing image metadata and positions
   *
   * @param pdfPath Path to the PDF file
   * @return PDFEmbeddedImagesResult containing extracted images
   */
  PDFEmbeddedImagesResult
  extractEmbeddedImagesFromPDF(const std::string &pdfPath);

  /**
   * @brief Represents a rectangle/box found in a PDF
   */
  struct PDFRectangle {
    int pageNumber; ///< 1-indexed page number
    double x;       ///< X position (left edge) in points, origin bottom-left
    double y;       ///< Y position (bottom edge) in points, origin bottom-left
    double width;   ///< Width in points
    double height;  ///< Height in points
    double lineWidth; ///< Stroke line width in points
    bool filled;      ///< Whether the rectangle is filled
    bool stroked;     ///< Whether the rectangle has a stroke (border)
  };

  /**
   * @brief Result of PDF rectangle extraction
   */
  struct PDFRectanglesResult {
    bool success = false;                 ///< Whether extraction succeeded
    std::string errorMessage;             ///< Error message if failed
    std::vector<PDFRectangle> rectangles; ///< Extracted rectangles
    double processingTimeMs = 0;          ///< Processing time in milliseconds
  };

  /**
   * @brief Extract rectangles (boxes) from the first page of a PDF file
   *
   * This method uses Poppler's lower-level API to extract rectangular
   * paths from the first page of a PDF. This includes:
   * - Box borders and frames
   * - Table cells
   * - Form field borders
   * - Any closed rectangular paths
   *
   * @param pdfPath Path to the PDF file
   * @param minSize Minimum size (width or height) in points to include
   * (default: 5)
   * @return PDFRectanglesResult containing extracted rectangles
   */
  PDFRectanglesResult extractRectanglesFromPDF(const std::string &pdfPath,
                                               double minSize = 5.0);

  /**
   * @brief Represents a line segment found in a PDF
   */
  struct PDFLine {
    int pageNumber;    ///< 1-indexed page number
    double x1;         ///< Start X position in points, origin bottom-left
    double y1;         ///< Start Y position in points, origin bottom-left
    double x2;         ///< End X position in points, origin bottom-left
    double y2;         ///< End Y position in points, origin bottom-left
    double lineWidth;  ///< Line width in points
    double length;     ///< Length of the line in points
    bool isHorizontal; ///< True if line is approximately horizontal
    bool isVertical;   ///< True if line is approximately vertical
  };

  /**
   * @brief Result of PDF line extraction
   */
  struct PDFLinesResult {
    bool success = false;        ///< Whether extraction succeeded
    std::string errorMessage;    ///< Error message if failed
    std::vector<PDFLine> lines;  ///< Extracted lines
    double processingTimeMs = 0; ///< Processing time in milliseconds

    // Bounding box containing all lines
    double boundingBoxX = 0;      ///< X position of bounding box (left edge)
    double boundingBoxY = 0;      ///< Y position of bounding box (top edge)
    double boundingBoxWidth = 0;  ///< Width of bounding box
    double boundingBoxHeight = 0; ///< Height of bounding box
  };

  /**
   * @brief Extract lines from the first page of a PDF file
   *
   * This method uses Poppler's lower-level API to extract line
   * segments from the first page of a PDF. This includes:
   * - Horizontal and vertical rules
   * - Table borders
   * - Underlines
   * - Any stroked line paths
   *
   * @param pdfPath Path to the PDF file
   * @param minLength Minimum line length in points to include (default: 5)
   * @return PDFLinesResult containing extracted lines
   */
  PDFLinesResult extractLinesFromPDF(const std::string &pdfPath,
                                     double minLength = 5.0);

  /**
   * @brief Combined result of extracting all PDF elements
   */
  struct PDFElements {
    bool success = false;        ///< Whether extraction succeeded
    std::string errorMessage;    ///< Error message if failed
    double processingTimeMs = 0; ///< Total processing time in milliseconds

    // Text extraction results (lines with orientation)
    std::string fullText; ///< Full text content
    std::vector<TextRegion>
        textLines; ///< Text lines with positions and orientation

    // Embedded images
    std::vector<PDFEmbeddedImage> images; ///< Actual embedded images

    // Vector graphics (drawn paths)
    std::vector<PDFRectangle> rectangles; ///< Rectangular paths/boxes
    std::vector<PDFLine> graphicLines;    ///< Drawn line segments

    // Statistics
    int pageCount = 0;        ///< Number of pages processed
    int textLineCount = 0;    ///< Number of text lines found
    int imageCount = 0;       ///< Number of images found
    int rectangleCount = 0;   ///< Number of rectangles found
    int graphicLineCount = 0; ///< Number of drawn lines found

    // Interior bounding box (largest box inside the found lines)
    double linesBoundingBoxX = 0;      ///< X position of interior box
    double linesBoundingBoxY = 0;      ///< Y position of interior box
    double linesBoundingBoxWidth = 0;  ///< Width of interior box
    double linesBoundingBoxHeight = 0; ///< Height of interior box

    // Page crop box (defines the visible area of the page)
    double pageX = 0;      ///< Page crop box X origin in points
    double pageY = 0;      ///< Page crop box Y origin in points
    double pageWidth = 0;  ///< Page width in points
    double pageHeight = 0; ///< Page height in points
  };

  /**
   * @brief Extract all elements from the first page of a PDF file
   *
   * This method extracts all available elements from the first page of a PDF in
   * a single call:
   * - Text with position and orientation information
   * - Embedded images (actual image objects, not rendered pages)
   * - Rectangles (boxes, frames, borders)
   * - Lines (rules, underlines, table borders)
   *
   * @param pdfPath Path to the PDF file
   * @param minRectSize Minimum rectangle size in points (default: 5)
   * @param minLineLength Minimum line length in points (default: 5)
   * @return PDFElements containing all extracted elements
   */
  PDFElements extractPDFElements(const std::string &pdfPath,
                                 double minRectSize = 5.0,
                                 double minLineLength = 5.0);

  /**
   * @brief Strip bleed marks from a PDF file
   *
   * This method removes connected rectangles on horizontal lines from a PDF,
   * including any lines forming those rectangles and any fills. This is useful
   * for removing crop marks, registration marks, and other printing marks.
   *
   * The method:
   * - Identifies all rectangles and lines on the first page
   * - Finds rectangles that are all on the same horizontal line (Y-coordinate)
   * - Removes these rectangles and their associated lines
   * - Returns the filtered PDF elements that can be rendered
   *
   * @param pdfPath Path to the input PDF file
   * @return PDFElements containing the filtered elements (without bleed marks)
   */
  PDFElements stripBleedMarks(const std::string &pdfPath);

  /**
   * @brief Structure to hold rendered element with pixel coordinates
   */
  struct RenderedElement {
    enum Type { TEXT, IMAGE, RECTANGLE, LINE };

    Type type;
    int pixelX; ///< X coordinate in pixels (top-left for most, start point for
                ///< lines)
    int pixelY; ///< Y coordinate in pixels (top-left for most, start point for
                ///< lines)
    int pixelWidth;  ///< Width in pixels
    int pixelHeight; ///< Height in pixels

    // Text-specific fields
    std::string text;      ///< Text content (for TEXT type)
    std::string fontName;  ///< Font family name (for TEXT type)
    double fontSize = 0.0; ///< Font size in points (for TEXT type)
    bool isBold = false;   ///< Whether font is bold (for TEXT type)
    bool isItalic = false; ///< Whether font is italic (for TEXT type)

    // Image-specific fields
    cv::Mat image; ///< Image data (for IMAGE type), properly rotated
    double rotationAngle = 0.0; ///< Rotation angle in radians (for IMAGE type)

    // Line-specific fields
    int pixelX2 = 0; ///< End X coordinate in pixels (for LINE type)
    int pixelY2 = 0; ///< End Y coordinate in pixels (for LINE type)
  };

  /**
   * @brief Mode for determining rendering bounds
   */
  enum class RenderBoundsMode {
    USE_CROP_MARKS,       ///< Use crop marks to determine rendering bounds
    USE_LARGEST_RECTANGLE ///< Use largest rectangle to determine rendering
                          ///< bounds
  };

  /**
   * @brief Result of PNG rendering operation
   */
  struct PNGRenderResult {
    bool success = false;
    std::string errorMessage;
    std::string outputPath;
    int imageWidth = 0;
    int imageHeight = 0;
    std::vector<RenderedElement> elements;
  };

  /**
   * @brief Render extracted PDF elements to a PNG image with pixel coordinates
   *
   * Creates a PNG image with all extracted elements rendered using original
   * fonts and images. Returns pixel coordinates of all rendered elements.
   * Properly handles coordinate system conversion:
   * - PDF elements (images, rectangles, lines) use bottom-left origin
   * - Text elements use top-left origin (from Poppler)
   * - Output PNG uses top-left origin
   *
   * Optionally, if markToFile is provided, loads that image file and draws
   * unfilled bounding boxes over it to visualize where elements are mapped,
   * saving the result with a "_marked" suffix.
   *
   * @param elements The extracted PDF elements to render
   * @param pdfPath Original PDF path (used for output filename)
   * @param dpi Resolution in dots per inch (default: 300)
   * @param outputDir Directory to save the PNG (default: "images")
   * @param boundsMode Mode for determining rendering bounds (default:
   * USE_CROP_MARKS)
   * @param markToFile Optional path to an image file to mark with element
   * bounding boxes (default: empty string = no marking)
   * @return PNGRenderResult containing success status, output path, and pixel
   * coordinates
   */
  PNGRenderResult renderElementsToPNG(
      const PDFElements &elements, const std::string &pdfPath,
      double dpi = 300.0, const std::string &outputDir = "images",
      RenderBoundsMode boundsMode = RenderBoundsMode::USE_CROP_MARKS,
      const std::string &markToFile = "");

  /**
   * @brief Sort rendered elements by position (top to bottom, left to right)
   *
   * Sorts the elements vector in a PNGRenderResult by their position,
   * ordered from top to bottom, then left to right (assuming origin at
   * top-left). Elements on the same horizontal line (similar Y coordinates) are
   * sorted by X coordinate. This is useful for reading order processing.
   *
   * @param result Reference to PNGRenderResult whose elements will be sorted
   */
  static void sortByPosition(PNGRenderResult &result);

  /**
   * @brief Align elements using OCR and create marked image with adjusted boxes
   *
   * Uses Tesseract OCR to locate the first text element in the rendered image,
   * calculates the offset between the expected position and OCR-detected
   * position, adjusts all element positions by this offset, verifies alignment
   * with OCR, and creates a marked image on the original image with blue
   * bounding boxes showing the adjusted positions.
   *
   * @param renderedImagePath Path to the rendered image to analyze with OCR
   * @param originalImagePath Path to the original image to mark up with boxes
   * @param renderResult The render result containing element positions
   * @param outputPath Path where the marked image with adjusted boxes will be
   * saved
   * @return true if successful, false otherwise
   */
  bool alignAndMarkElements(const std::string &renderedImagePath,
                            const std::string &originalImagePath,
                            const PNGRenderResult &renderResult,
                            const std::string &outputPath);

  /**
   * @brief Get text from a specific region of an image
   * @param image OpenCV Mat image
   * @param roi Region of interest rectangle
   * @return Extracted text from the region
   */
  std::string getTextFromRegion(const cv::Mat &image, const cv::Rect &roi);

  /**
   * @brief Detect text regions in an image
   * @param image OpenCV Mat image
   * @return Vector of detected text regions with bounding boxes
   */
  std::vector<TextRegion> detectTextRegions(const cv::Mat &image);

  /**
   * @brief Identify all text line regions in an image without full OCR
   *
   * This method locates all lines of text in any orientation (horizontal,
   * vertical, or rotated) and returns their bounding boxes and orientations.
   * Unlike detectTextRegions, this focuses on line-level detection and
   * is optimized for finding text regions rather than recognizing content.
   *
   * @param image OpenCV Mat image
   * @return Vector of detected text line regions with orientations
   */
  std::vector<TextRegion> identifyTextRegions(const cv::Mat &image);

  /**
   * @brief Mask non-text regions (logos, graphics) with white boxes
   *
   * This method analyzes the image to detect regions that appear to be
   * logos, graphics, or other non-text elements and covers them with
   * solid white rectangles. This improves OCR accuracy by removing
   * visual noise that could interfere with text detection.
   *
   * @param image OpenCV Mat image to process
   * @return A copy of the image with non-text regions masked in white
   */
  cv::Mat maskNonTextRegions(const cv::Mat &image);

  /**
   * @brief Set the OCR language
   * @param language Language code (e.g., "eng", "deu+eng" for multiple)
   * @return true if language was set successfully
   */
  bool setLanguage(const std::string &language);

  /**
   * @brief Set the page segmentation mode
   * @param mode Tesseract page segmentation mode
   */
  void setPageSegMode(tesseract::PageSegMode mode);

  /**
   * @brief Get the current configuration
   * @return Current OCR configuration
   */
  const OCRConfig &getConfig() const;

  /**
   * @brief Set a new configuration (requires re-initialization)
   * @param config New OCR configuration
   */
  void setConfig(const OCRConfig &config);

  /**
   * @brief Get the Tesseract version string
   * @return Tesseract version
   */
  static std::string getTesseractVersion();

  /**
   * @brief Get available languages
   * @return Vector of available language codes
   */
  std::vector<std::string> getAvailableLanguages() const;

private:
  /**
   * @brief Preprocess image for better OCR results
   * @param image Input image
   * @return Preprocessed image
   */
  cv::Mat preprocessImage(const cv::Mat &image);

  /**
   * @brief Convert OpenCV Mat to Tesseract-compatible format
   * @param image OpenCV Mat image
   */
  void setImage(const cv::Mat &image);

  /**
   * @brief Find the best rotation for an image by trying all 4 orientations
   * @param image Input image
   * @return Rotation code (-1 = no rotation, or cv::ROTATE_* constant)
   */
  int findBestRotation(const cv::Mat &image);

  /**
   * @brief Determine if a contour region is likely a graphic rather than text
   * @param image The source image
   * @param contour The contour to analyze
   * @param boundingRect The bounding rectangle of the contour
   * @return true if the region appears to be a graphic/logo
   */
  bool isLikelyGraphic(const cv::Mat &image,
                       const std::vector<cv::Point> &contour,
                       const cv::Rect &boundingRect);

  std::unique_ptr<tesseract::TessBaseAPI>
      m_tesseract;    ///< Tesseract API instance
  OCRConfig m_config; ///< Current configuration
  bool m_initialized; ///< Initialization state
};

} // namespace ocr

#endif // OCR_ANALYSIS_HPP
