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
  // Line 1 (Y=50): "Hello" at X=100, "World" at X=300
  // Line 2 (Y=100): "This" at X=50, "is" at X=150, "a" at X=200, "test" at
  // X=250 Line 3 (Y=150): "Sorted" at X=100, "text" at X=250

  ocr::OCRAnalysis::RenderedElement elem;

  // Add in random order
  elem.type = ocr::OCRAnalysis::RenderedElement::TEXT;
  elem.text = "test";
  elem.pixelX = 250;
  elem.pixelY = 100;
  result.elements.push_back(elem);

  elem.text = "Hello";
  elem.pixelX = 100;
  elem.pixelY = 50;
  result.elements.push_back(elem);

  elem.text = "text";
  elem.pixelX = 250;
  elem.pixelY = 150;
  result.elements.push_back(elem);

  elem.text = "World";
  elem.pixelX = 300;
  elem.pixelY = 50;
  result.elements.push_back(elem);

  elem.text = "This";
  elem.pixelX = 50;
  elem.pixelY = 100;
  result.elements.push_back(elem);

  elem.text = "Sorted";
  elem.pixelX = 100;
  elem.pixelY = 150;
  result.elements.push_back(elem);

  elem.text = "is";
  elem.pixelX = 150;
  elem.pixelY = 100;
  result.elements.push_back(elem);

  elem.text = "a";
  elem.pixelX = 200;
  elem.pixelY = 100;
  result.elements.push_back(elem);

  std::cout << "Before sorting:" << std::endl;
  std::cout << std::string(50, '-') << std::endl;
  for (size_t i = 0; i < result.elements.size(); i++) {
    const auto &e = result.elements[i];
    std::cout << std::setw(2) << i << ": \"" << std::setw(10) << std::left
              << e.text << "\" at (" << std::setw(3) << e.pixelX << ", "
              << std::setw(3) << e.pixelY << ")" << std::endl;
  }

  // Sort by position
  ocr::OCRAnalysis::sortByPosition(result);

  std::cout << std::endl
            << "After sorting (top to bottom, left to right):" << std::endl;
  std::cout << std::string(50, '-') << std::endl;
  for (size_t i = 0; i < result.elements.size(); i++) {
    const auto &e = result.elements[i];
    std::cout << std::setw(2) << i << ": \"" << std::setw(10) << std::left
              << e.text << "\" at (" << std::setw(3) << e.pixelX << ", "
              << std::setw(3) << e.pixelY << ")" << std::endl;
  }

  std::cout << std::endl << "Expected reading order:" << std::endl;
  std::cout << "  Line 1 (Y=50):  Hello World" << std::endl;
  std::cout << "  Line 2 (Y=100): This is a test" << std::endl;
  std::cout << "  Line 3 (Y=150): Sorted text" << std::endl;

  return 0;
}
