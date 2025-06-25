#include "shape_detector.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>

int main() {
    unsetenv("GTK_PATH");       // unset GTK_PATH  //in Visual Studio Code
    namespace fs = std::filesystem;

    // Current cpp path
    fs::path source_file = __FILE__;
    fs::path dir = source_file.parent_path();




    


    try {
        std::cout << "Create a new triangle shape" << std::endl;
        fs::path triangle_file = dir / "triangle.png";
        create_triangle_image(triangle_file);

        cv::Mat triangle_image = cv::imread(triangle_file);

        auto template_contour = find_shape_contour(triangle_file);
        auto triangle_with_contour_img = draw_contour_on_image(triangle_image, template_contour);

        // std::cout << "Contour found and drawn.\n";
        // //cv::imwrite("triangle_with_contour.png", drawn_img);
        // imshow("triangle with contour", triangle_with_contour_img);


        fs::path other_triangle_file = dir / "other_triangle.png";
        cv::Mat other_triangle_frame_img = cv::imread(other_triangle_file);
        //imshow("", other_triangle_frame_img);
        auto detected_contours = contour_compare(other_triangle_frame_img, template_contour);
        auto other_triangle_with_detected_contours_img = draw_contour_on_image(other_triangle_frame_img, detected_contours);
        std::cout << "Contours found and drawn.\n";
        //cv::imwrite("contours_found.png", other_triangle_frame_img);  
        std::cout << "Contours found: " << detected_contours.size() << std::endl;
        imshow("other triangle with contour", other_triangle_with_detected_contours_img);

    }
    catch (const std::exception& e) 
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }


    // unset GTK_PATH
    //cv::Mat img = cv::imread(output_triangle_file);
    //imshow("triangle", img);
    
    return 0;
}
