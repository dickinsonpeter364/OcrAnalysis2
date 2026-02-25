#include "OCRAnalysis.hpp"
#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
  std::cout << "=== Test sortByPosition ===" << std::endl << std::endl;

  // Create a sample PNGRenderResult with unsorted elements
  ocr::OCRAnalysis::PNGRenderResult result;
  result.success = true;
  result.imageWidth = 800;
  result.imageHeight = 600;

  // Add elements in random order
  // Line 1 (Y~0.083): "Hello" at X~0.125, "World" at X~0.375
  // Line 2 (Y~0.167): "This" at X~0.0625, "is" at X~0.1875, "a" at X~0.25,
  //   "test" at X~0.3125
  // Line 3 (Y~0.25):  "Sorted" at X~0.125, "text" at X~0.3125

  ocr::OCRAnalysis::RenderedElement elem;

  // Add in random order
  elem.type = ocr::OCRAnalysis::RenderedElement::TEXT;
  elem.text = "test";
  elem.relativeX = 250.0 / 800.0;
  elem.relativeY = 100.0 / 600.0;
  result.elements.push_back(elem);

  elem.text = "Hello";
  elem.relativeX = 100.0 / 800.0;
  elem.relativeY = 50.0 / 600.0;
  result.elements.push_back(elem);

  elem.text = "text";
  elem.relativeX = 250.0 / 800.0;
  elem.relativeY = 150.0 / 600.0;
  result.elements.push_back(elem);

  elem.text = "World";
  elem.relativeX = 300.0 / 800.0;
  elem.relativeY = 50.0 / 600.0;
  result.elements.push_back(elem);

  elem.text = "This";
  elem.relativeX = 50.0 / 800.0;
  elem.relativeY = 100.0 / 600.0;
  result.elements.push_back(elem);

  elem.text = "Sorted";
  elem.relativeX = 100.0 / 800.0;
  elem.relativeY = 150.0 / 600.0;
  result.elements.push_back(elem);

  elem.text = "is";
  elem.relativeX = 150.0 / 800.0;
  elem.relativeY = 100.0 / 600.0;
  result.elements.push_back(elem);

  elem.text = "a";
  elem.relativeX = 200.0 / 800.0;
  elem.relativeY = 100.0 / 600.0;
  result.elements.push_back(elem);

  std::cout << "Before sorting:" << std::endl;
  std::cout << std::string(60, '-') << std::endl;
  for (size_t i = 0; i < result.elements.size(); i++) {
    const auto &e = result.elements[i];
    std::cout << std::setw(2) << i << ": \"" << std::setw(10) << std::left
              << e.text << "\" at rel(" << std::fixed << std::setprecision(4)
              << e.relativeX << ", " << e.relativeY << ")" << std::endl;
  }

  // Sort by position
  ocr::OCRAnalysis::sortByPosition(result);

  std::cout << std::endl
            << "After sorting (top to bottom, left to right):" << std::endl;
  std::cout << std::string(60, '-') << std::endl;
  for (size_t i = 0; i < result.elements.size(); i++) {
    const auto &e = result.elements[i];
    std::cout << std::setw(2) << i << ": \"" << std::setw(10) << std::left
              << e.text << "\" at rel(" << std::fixed << std::setprecision(4)
              << e.relativeX << ", " << e.relativeY << ")" << std::endl;
  }

  std::cout << std::endl << "Expected reading order:" << std::endl;
  std::cout << "  Line 1 (Y~0.083): Hello World" << std::endl;
  std::cout << "  Line 2 (Y~0.167): This is a test" << std::endl;
  std::cout << "  Line 3 (Y~0.250): Sorted text" << std::endl;

  return 0;
}
