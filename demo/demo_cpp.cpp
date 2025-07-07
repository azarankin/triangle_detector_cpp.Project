#include <cstdlib>
#include <filesystem>
#include <iostream>

#include <shape_detector_common.h>
#include <shape_detector_logic.h>

namespace fs = std::filesystem;


int main() {
    // Unset GTK_PATH to avoid display issues with OpenCV on some systems
    unsetenv("GTK_PATH");

    // ---- Define file paths relative to this source file ----
    fs::path source_file = __FILE__;
    fs::path dir = source_file.parent_path();

    fs::path triangle_path = dir / "out/triangle.png";
    fs::path other_triangle_path = dir / "other_triangle.png";
    fs::path output_with_contour_path = dir / "out/triangle_with_contour.png";
    fs::path output_detected_path = dir / "out/contours_found.png";

    try
    {
        // ---- Create a new triangle image ----
        std::cout << "Create a new triangle shape\n";
        create_triangle_image(triangle_path);

        // ---- Load triangle image and check if loaded successfully ----
        cv::Mat triangle_img = cv::imread(triangle_path);
        if (triangle_img.empty()) throw std::runtime_error("Failed to load triangle image!");

        // ---- Find the main contour in the triangle image ----
        auto template_contour = find_shape_contour(triangle_path);

        // ---- Draw the contour on the triangle image ----
        cv::Mat triangle_with_contour_img = triangle_img.clone();
        draw_contour_on_image(triangle_with_contour_img, template_contour);
        cv::imwrite(output_with_contour_path, triangle_with_contour_img);

        // ---- Load another triangle image for comparison ----
        cv::Mat other_img = cv::imread(other_triangle_path);
        if (other_img.empty()) throw std::runtime_error("Failed to load other_triangle image!");

        // ---- Find similar contours in the second image ----
        auto detected_contours = contour_compare(other_img, template_contour);

        // ---- Draw found contours on the new image ----
        cv::Mat other_with_detected = other_img.clone();
        draw_contour_on_image(other_with_detected, detected_contours);
        cv::imwrite(output_detected_path, other_with_detected);

        std::cout << "Contours found and drawn.\n";
        std::cout << "Contours found: " << detected_contours.size() << std::endl;

        // ---- Display results ----
        imshow("Triangle with contour", triangle_with_contour_img);
        imshow("Detected contours", other_with_detected);

    }
    catch (const std::exception& e) 
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
