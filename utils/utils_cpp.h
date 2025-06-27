void drawEquilateralTriangl(cv::Mat& img, cv::Size size, int margin)
{
    cv::Point pt1(size.width / 2, margin);                 // top
    cv::Point pt2(margin, size.height - margin);           // left
    cv::Point pt3(size.width - margin, size.height - margin); // right

    std::vector<cv::Point> triangle_cnt = {pt1, pt2, pt3};

    cv::drawContours(img, std::vector<std::vector<cv::Point>>{triangle_cnt}, 0, cv::Scalar(0, 0, 0), cv::FILLED);
}
