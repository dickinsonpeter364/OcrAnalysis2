#include "OCRAnalysis.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Argument parsing helpers
static bool isImageFile(const std::string &s) {
  auto lo = s;
  for (auto &c : lo) c = static_cast<char>(std::tolower((unsigned char)c));
  return lo.ends_with(".png") || lo.ends_with(".jpg") ||
         lo.ends_with(".jpeg") || lo.ends_with(".bmp");
}

static bool isPdfFile(const std::string &s) {
  auto lo = s;
  for (auto &c : lo) c = static_cast<char>(std::tolower((unsigned char)c));
  return lo.ends_with(".pdf");
}

int main(int argc, char *argv[]) {
#ifdef NDEBUG
  // Release mode: suppress all diagnostic output from the library
  std::ofstream devNull("NUL");
  std::cerr.rdbuf(devNull.rdbuf());
#endif

  if (argc < 3) {
    std::cerr
        << "Usage: " << argv[0]
        << " <l1_pdf> [<l2_pdf>] <image_file> [<token>=<value> ...]\n"
        << "\n"
        << "  l1_pdf      : PDF file defining the label design (required)\n"
        << "  l2_pdf      : optional second PDF (e.g. L2 insert) merged into\n"
        << "                the relative map\n"
        << "  image_file  : photo of the physical label to validate\n"
        << "  token=value : placeholder substitutions applied before checking\n"
        << "                (e.g. \"<MED>=ADALIMUMAB\" \"<PROT>=HUMIRA\")\n"
        << "\n"
        << "Examples:\n"
        << "  " << argv[0] << " label.pdf photo.jpg\n"
        << "  " << argv[0]
        << " label.pdf photo.jpg \"<MED>=ADALIMUMAB\" \"<PROT>=HUMIRA\"\n"
        << "  " << argv[0]
        << " label.pdf insert.pdf photo.jpg \"<MED>=ADALIMUMAB\"\n";
    return 1;
  }

  try {
    std::string l1PdfPath;
    std::string l2PdfPath;
    std::string imagePath;
    std::vector<std::pair<std::string, std::string>> placeholders;

    // Flexible argument parsing: detect by extension / presence of '='
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];

      if (arg.find('=') != std::string::npos) {
        // Placeholder substitution: split on first '='
        auto eq = arg.find('=');
        std::string token = arg.substr(0, eq);
        std::string value = arg.substr(eq + 1);
        placeholders.emplace_back(token, value);
      } else if (isPdfFile(arg)) {
        if (l1PdfPath.empty())
          l1PdfPath = arg;
        else
          l2PdfPath = arg;
      } else if (isImageFile(arg)) {
        imagePath = arg;
      } else {
        std::cerr << "Warning: unrecognised argument \"" << arg
                  << "\" – ignored\n";
      }
    }

    if (l1PdfPath.empty()) {
      std::cerr << "Error: no PDF file specified\n";
      return 1;
    }
    if (imagePath.empty()) {
      std::cerr << "Error: no image file specified\n";
      return 1;
    }

    // ── Extract L1 elements ───────────────────────────────────────────────
#ifndef NDEBUG
    std::cout << "Extracting elements from L1 PDF: " << l1PdfPath << "\n";
#endif
    ocr::OCRAnalysis analyzer;
    auto l1Elements = analyzer.extractPDFElements(l1PdfPath);
    if (!l1Elements.success) {
      std::cerr << "Error extracting L1 PDF: " << l1Elements.errorMessage
                << "\n";
      return 1;
    }
#ifndef NDEBUG
    std::cout << "  Text lines: " << l1Elements.textLineCount
              << "  Images: " << l1Elements.imageCount << "\n";
#endif

    // ── Load photo ────────────────────────────────────────────────────────
#ifndef NDEBUG
    std::cout << "Loading image: " << imagePath << "\n";
#endif
    cv::Mat photo = cv::imread(imagePath);
    if (photo.empty()) {
      std::cerr << "Error: could not load image: " << imagePath << "\n";
      return 1;
    }
#ifndef NDEBUG
    std::cout << "  " << photo.cols << " x " << photo.rows << " px\n";
#endif

    // ── Build relative map (no marking at this stage) ─────────────────────
#ifndef NDEBUG
    std::cout << "\nBuilding relative map…\n";
#endif
    auto relMap = analyzer.createRelativeMap(
        l1Elements, photo, imagePath, /*markImage=*/false,
        l1PdfPath, 300.0, l2PdfPath);

    if (!relMap.success) {
      std::cerr << "Error building relative map: " << relMap.errorMessage
                << "\n";
      return 1;
    }

#ifndef NDEBUG
    std::cout << "  Bounds: (" << relMap.boundsX << ", " << relMap.boundsY
              << ")  " << relMap.boundsWidth << " x " << relMap.boundsHeight
              << " pt\n";
    std::cout << "  Elements: " << relMap.elements.size() << "\n\n";

    // Print placeholders being applied
    if (!placeholders.empty()) {
      std::cout << "Placeholders:\n";
      for (const auto &[token, value] : placeholders)
        std::cout << "  " << token << " -> \"" << value << "\"\n";
      std::cout << "\n";
    }

    std::cout << "Checking image…\n\n";
#endif

    // ── Run check ─────────────────────────────────────────────────────────
    auto t0 = std::chrono::steady_clock::now();
    bool ok = analyzer.checkImage(photo, placeholders);
    auto t1 = std::chrono::steady_clock::now();
    double checkMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── Save annotated output ─────────────────────────────────────────────
    std::filesystem::path imgPath(imagePath);
    std::string outputPath = imgPath.parent_path().string() + "/" +
                             imgPath.stem().string() + "_checked" +
                             imgPath.extension().string();
    cv::imwrite(outputPath, photo);

    // ── Result ────────────────────────────────────────────────────────────
#ifdef NDEBUG
    std::cout << checkMs << " ms  " << (ok ? "PASS" : "FAIL") << "\n";
#else
    std::cout << "\ncheckImage took " << checkMs << " ms\n";
    std::cout << "\nAnnotated image saved: " << outputPath << "\n";
    std::cout << "\n=== Result: " << (ok ? "PASS" : "FAIL") << " ===\n";
#endif
    return ok ? 0 : 2; // exit 0 = pass, 2 = check failed (1 = usage/error)

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }
}
