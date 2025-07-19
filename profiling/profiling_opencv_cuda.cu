#include "profiling_logic.h"
#include <shape_detector_common.h>  // פונקציות הרצה



int main(int argc, char* argv[])
{
    
#ifndef COMPILED_AS_RELEASE
    if(is_debugger_attached()) { std::cerr << "Error: Debugger detected on Release binary. Exiting.\n"; return 1; /*exit(1);*/ }
#endif

    // ProfilingArgs args;
    // if (!parse_and_validate_args(argc, argv, args)) 
    // {
    //     return 1;
    // }



    std::cout << BG_WHITE << CYAN;
    triangle_print_message();
    std::cout << RESET << BG_WHITE << BLACK << "Profilling:" << std::endl;
    //std::cout << "Profiling OpenCV CUDA Triangle Detector application" << std::endl;
    auto template_contour = find_shape_contour(triangle_img);
    cv::Mat other_triangle_image = cv::imread(other_triangle_img, cv::IMREAD_COLOR);
    auto detected_contours = contour_compare(other_triangle_image, template_contour);
    auto result = draw_contour_on_image(other_triangle_image, detected_contours);




    // אפשר לשים לולאה אם רוצים למדוד עומס/ביצועים
    image_save(output_result_img, result);
    std::cout << std::endl;
    std::cout << "Saved result image to: " << output_result_img.string() << std::endl;
    std::cout << GREEN << "✓ Done." << RESET << std::endl;
    return 0;
}