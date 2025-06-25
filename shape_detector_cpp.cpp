#include "shape_detector_cpp.h"


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




std::vector<cv::Point> find_shape_contour(const std::string& address)
{
    // טען תמונה
    cv::Mat templat_img = cv::imread(address);
    if (templat_img.empty())
        throw std::runtime_error("Failed to load image: " + address);

    // המרה לאפור
    cv::Mat templat_img_gray;
    cv::cvtColor(templat_img, templat_img_gray, cv::COLOR_BGR2GRAY);

    // טשטוש
    cv::Mat blured;
    cv::GaussianBlur(templat_img_gray, blured, cv::Size(5, 5), 0);

    // סף
    cv::Mat thresh1;
    cv::threshold(blured, thresh1, 127, 255, 0);

    // מציאת קונטורים
    std::vector<std::vector<cv::Point>> contours_template;
    cv::findContours(thresh1, contours_template, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // מיון לפי שטח, יורד
    std::sort(contours_template.begin(), contours_template.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) > cv::contourArea(b);
        });

    if (contours_template.size() < 2)
        throw std::runtime_error("Template image must contain at least two contours.");

    // הקונטור השני בגודלו (כמו בפייתון)
    return contours_template[1];
}



cv::Mat draw_contour_on_image(cv::Mat& img, const std::vector<cv::Point>& contour, const cv::Scalar& color, int thickness)
{
    if (img.empty())
        throw std::runtime_error("Failed to load image");

    std::vector<std::vector<cv::Point>> contours{contour};
    cv::drawContours(img, contours, 0, color, thickness);
    return img;
}

cv::Mat draw_contour_on_image(cv::Mat& img, const std::vector<std::vector<cv::Point>>& contours, const cv::Scalar& color, int thickness) 
{
    if (img.empty())
        throw std::runtime_error("Failed to load image");
    cv::drawContours(img, contours, -1, color, thickness);
    return img;
}


std::vector<std::vector<cv::Point>> contour_compare(
    cv::Mat& target_frame, 
    const std::vector<cv::Point>& template_contour,
    double match_threshold, 
    double min_area_ratio, 
    double max_area_ratio)
{
    // אפור
    cv::Mat target_gray;
    cv::cvtColor(target_frame, target_gray, cv::COLOR_BGR2GRAY);

    // טשטוש
    cv::Mat blured;
    cv::GaussianBlur(target_gray, blured, cv::Size(5, 5), 0);

    // סף אדפטיבי
    cv::Mat thresh2;
    cv::adaptiveThreshold(blured, thresh2, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 5);

    // מציאת קונטורים
    std::vector<std::vector<cv::Point>> contours_target;
    cv::findContours(thresh2, contours_target, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // הגדרות סף שטח
    double frame_area = static_cast<double>(target_frame.rows * target_frame.cols);
    double min_area = frame_area * min_area_ratio;
    double max_area = frame_area * max_area_ratio;

    std::vector<std::vector<cv::Point>> closest_contours;

    // חיפוש קונטורים דומים
    for (const auto& c : contours_target) {
        double match = cv::matchShapes(template_contour, c, cv::CONTOURS_MATCH_I3, 0.0);
        if (match < match_threshold) {
            double area = cv::contourArea(c);
            if (area < min_area || area > max_area)
                continue; // קטן/גדול מדי = רעש
            closest_contours.push_back(c);
        }
    }


    return closest_contours;
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


void create_triangle_image(const std::string& filename, cv::Size size, int margin) 
{
    cv::Mat img(size, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::Point pt1(size.width / 2, margin);                 // top
    cv::Point pt2(margin, size.height - margin);           // left
    cv::Point pt3(size.width - margin, size.height - margin); // right

    std::vector<cv::Point> triangle_cnt = {pt1, pt2, pt3};

    cv::drawContours(img, std::vector<std::vector<cv::Point>>{triangle_cnt}, 0, cv::Scalar(0, 0, 0), cv::FILLED);

    cv::imwrite(filename, img);
    std::cout << "Triangle image saved as " << filename << std::endl;
}
