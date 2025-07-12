// #include <string>
// #include <functional>
// #include <iostream>
// #include <opencv2/cudaarithm.hpp>
// #include <opencv2/cudafilters.hpp>
// #include <opencv2/cudaimgproc.hpp>

#include "utils.cuh"



struct FilterImpl {
    std::string name;
    std::function<void(const cv::Mat&, cv::Mat&)> run;
};

void benchmark_and_compare(
    const cv::Mat& input,
    const std::vector<FilterImpl>& impls,
    const std::string& test_name)
{
    std::vector<cv::Mat> outputs(impls.size());
    std::vector<double> times(impls.size());

    // הרצה ומדידת זמן
    for (size_t i = 0; i < impls.size(); ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        impls[i].run(input, outputs[i]);
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // השוואת תוצאות (נניח הראשונה היא ה-reference)
    for (size_t i = 1; i < outputs.size(); ++i) {
        double diff = cv::norm(outputs[0], outputs[i], cv::NORM_L2);
        std::cout << "[DIFF] " << impls[0].name << " vs " << impls[i].name << ": " << diff << std::endl;
    }
    for (size_t i = 0; i < impls.size(); ++i) {
        std::cout << "[TIME] " << impls[i].name << ": " << times[i] << " ms" << std::endl;
    }
}



int main() {
    cv::Mat input = cv::imread("your_image.png", cv::IMREAD_COLOR);

    std::vector<FilterImpl> gray_filters = {
        {"CPU", gray_filter_cpu},
        {"CUDA_OpenCV", gray_filter_cuda_opencv},
     //   {"PURE_CUDA", gray_filter_pure_cuda}
    };

    benchmark_and_compare(input, gray_filters, "gray_filter");
}