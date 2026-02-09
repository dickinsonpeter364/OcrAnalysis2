#include <iostream>
#include <memory>
#include <poppler-document.h>
#include <poppler-page.h>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <pdf_file>" << std::endl;
    return 1;
  }

  std::string pdfPath = argv[1];
  std::cout << "Loading PDF: " << pdfPath << std::endl;

  try {
    std::unique_ptr<poppler::document> doc(
        poppler::document::load_from_file(pdfPath));

    if (!doc) {
      std::cerr << "Failed to load PDF file" << std::endl;
      return 1;
    }

    std::cout << "PDF loaded successfully!" << std::endl;
    std::cout << "Number of pages: " << doc->pages() << std::endl;
    std::cout << "Is locked: " << (doc->is_locked() ? "yes" : "no")
              << std::endl;

    // Try to get first page
    if (doc->pages() > 0) {
      std::cout << "Getting first page..." << std::endl;
      std::unique_ptr<poppler::page> page(doc->create_page(0));

      if (!page) {
        std::cerr << "Failed to create page" << std::endl;
        return 1;
      }

      std::cout << "Page created successfully!" << std::endl;

      // Try to get text list
      std::cout << "Getting text list..." << std::endl;
      std::vector<poppler::text_box> textBoxes =
          page->text_list(poppler::page::text_list_include_font);

      std::cout << "Found " << textBoxes.size() << " text boxes" << std::endl;

      // Print first few
      int count = 0;
      for (auto &textBox : textBoxes) {
        if (count++ >= 5)
          break;
        poppler::byte_array textBytes = textBox.text().to_utf8();
        std::string text(textBytes.begin(), textBytes.end());
        std::cout << "  Text: " << text << std::endl;
      }
    }

    std::cout << "Test completed successfully!" << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception" << std::endl;
    return 1;
  }
}
