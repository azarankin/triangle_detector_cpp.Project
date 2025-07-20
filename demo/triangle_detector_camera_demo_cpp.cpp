#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
 
#include <shape_detector_common.h>
#include <shape_detector_logic.h>

namespace fs = std::filesystem;

int main(int argc, char** argv) {

    
    fs::path source_file = __FILE__;
    fs::path dir = source_file.parent_path();
    fs::path template_image_path = dir / "triangle.png";

    int camid = 0;

    try {
        // מוצא קונטור מהתמונה (כמו בפייטון)
        std::vector<cv::Point> template_contour = find_shape_contour(template_image_path.string());

        // מריץ מצלמה עם הקונטור
        run_camera(template_contour, camid, 1280, 720);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }

    return 0;
}