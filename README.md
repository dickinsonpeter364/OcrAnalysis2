# OCR Analysis

A C++ library for Optical Character Recognition (OCR) using OpenCV and Tesseract.

## Features

- Text extraction from images using Tesseract OCR
- Image preprocessing with OpenCV for improved recognition accuracy
- Detection of text regions with bounding boxes and confidence scores
- Support for multiple languages
- Configurable page segmentation modes
- Simple and intuitive C++ API

## Prerequisites

Before building this project, ensure you have the following installed:

### macOS (using Homebrew)

```bash
brew install opencv tesseract cmake pkg-config
```

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y libopencv-dev libtesseract-dev libleptonica-dev cmake pkg-config
```

### Language Data

Tesseract requires language data files. On macOS with Homebrew, the English data is typically installed automatically. For additional languages:

```bash
# macOS
brew install tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr-eng tesseract-ocr-deu  # Add languages as needed
```

## Building

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc)
```

## Usage

### Command Line Demo

```bash
# Basic usage
./ocr_demo path/to/image.png

# With options
./ocr_demo document.png -l eng+deu -r --confidence 80
```

Options:
- `-l, --language <lang>`: Set OCR language (default: eng)
- `-r, --regions`: Show detected text regions with bounding boxes
- `-c, --confidence <val>`: Minimum confidence threshold (0-100)
- `-h, --help`: Show help message

### Library Usage

```cpp
#include "OCRAnalysis.hpp"

int main() {
    // Create with default configuration
    ocr::OCRAnalysis analyzer;
    
    // Or with custom configuration
    ocr::OCRConfig config;
    config.language = "eng";
    config.preprocessImage = true;
    config.minConfidence = 60;
    ocr::OCRAnalysis analyzer(config);
    
    // Initialize the OCR engine
    if (!analyzer.initialize()) {
        std::cerr << "Failed to initialize OCR" << std::endl;
        return 1;
    }
    
    // Analyze an image file
    auto result = analyzer.analyzeImage("document.png");
    
    if (result.success) {
        std::cout << "Extracted text: " << result.fullText << std::endl;
        std::cout << "Processing time: " << result.processingTimeMs << " ms" << std::endl;
        
        // Access individual text regions
        for (const auto& region : result.regions) {
            std::cout << "Text: " << region.text 
                      << " (confidence: " << region.confidence << "%)" << std::endl;
        }
    }
    
    // Or analyze an OpenCV Mat directly
    cv::Mat image = cv::imread("document.png");
    result = analyzer.analyzeImage(image);
    
    // Extract text from a specific region
    cv::Rect roi(100, 100, 200, 50);
    std::string regionText = analyzer.getTextFromRegion(image, roi);
    
    return 0;
}
```

## API Reference

### OCRAnalysis Class

#### Constructors

- `OCRAnalysis()` - Default constructor
- `OCRAnalysis(const OCRConfig& config)` - Constructor with custom configuration

#### Methods

- `bool initialize()` - Initialize the OCR engine
- `bool isInitialized()` - Check if initialized
- `OCRResult analyzeImage(const std::string& imagePath)` - Analyze image from file
- `OCRResult analyzeImage(const cv::Mat& image)` - Analyze OpenCV Mat
- `std::string getTextFromRegion(const cv::Mat& image, const cv::Rect& roi)` - Extract text from region
- `std::vector<TextRegion> detectTextRegions(const cv::Mat& image)` - Detect text regions
- `bool setLanguage(const std::string& language)` - Set OCR language
- `void setPageSegMode(tesseract::PageSegMode mode)` - Set page segmentation mode
- `static std::string getTesseractVersion()` - Get Tesseract version
- `std::vector<std::string> getAvailableLanguages()` - Get available languages

### Configuration

```cpp
struct OCRConfig {
    std::string language = "eng";           // Language code
    tesseract::PageSegMode pageSegMode;     // Page segmentation mode
    bool preprocessImage = true;            // Apply preprocessing
    int minConfidence = 0;                  // Minimum confidence (0-100)
    std::string tessDataPath = "";          // Path to tessdata
};
```

## License

MIT License
