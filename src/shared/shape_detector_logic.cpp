
#include <shape_detector_logic.h>



void imshow(const std::string& title, const cv::Mat& img) 
{
    cv::Mat to_show;
    if (img.channels() == 4) {
        cv::cvtColor(img, to_show, cv::COLOR_BGRA2RGBA);
    } else if (img.channels() == 3) {
        cv::cvtColor(img, to_show, cv::COLOR_BGR2RGB);
    } else {
        to_show = img.clone();
    }
    cv::imshow(title, to_show);
    cv::waitKey(0);  // waiting for key pressing
}


void run_camera(
    const std::vector<cv::Point>& template_contour,
    int camid,
    int display_width,
    int display_height
) {
    cv::VideoCapture cap(camid);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot access the camera." << std::endl;
        return;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Failed to grab frame" << std::endl;
            break;
        }

        // עיבוד זיהוי משולשים

        double match_threshold = 0.09; // Adjusted for better performance
        double min_area_ratio = 0.0005; // Adjusted for better performance
        double max_area_ratio = 0.80; // Adjusted for better performance

        auto contours = contour_compare(frame, template_contour, match_threshold, min_area_ratio, max_area_ratio);
        auto frame_with_contours = draw_contour_on_image(frame, contours);

        // הגבלת גודל תצוגה – שמירה על יחס גובה/רוחב
        cv::Mat display_frame;
        double aspect = static_cast<double>(frame_with_contours.cols) / frame_with_contours.rows;
        int new_width, new_height;
        if (frame_with_contours.cols > display_width || frame_with_contours.rows > display_height) {
            if (aspect > 1.0) { // תמונה רחבה
                new_width = display_width;
                new_height = static_cast<int>(display_width / aspect);
            } else {
                new_height = display_height;
                new_width = static_cast<int>(display_height * aspect);
            }
            cv::resize(frame_with_contours, display_frame, cv::Size(new_width, new_height));
        } else {
            display_frame = frame_with_contours;
        }

        cv::imshow("Live Feed", display_frame);

        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q') { // Esc או q
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}