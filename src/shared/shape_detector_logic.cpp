
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

void run_camera(const std::vector<cv::Point>& template_contour, int camid) 
{
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

        // קריאה לפונקציית ההשוואה
        auto contours = contour_compare(frame, template_contour);
        auto frame_with_contours = draw_contour_on_image(frame, contours);
        // (הפונקציה אמורה לצייר את הקונטורים על frame עצמו)

        cv::imshow("Live Feed", frame_with_contours);

        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q') { // Esc או q
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}


