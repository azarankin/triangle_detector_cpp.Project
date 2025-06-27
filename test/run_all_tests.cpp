#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

void run_test_binary(const std::string& binary_name) {
    fs::path exec_path = fs::current_path() / binary_name;
    std::cout << "Running: " << exec_path << std::endl;
    int ret = system(exec_path.string().c_str());
    if (ret != 0) {
        std::cerr << "Test failed: " << binary_name << " (code: " << ret << ")\n";
    }
}

int main() {
    run_test_binary("triangle_detector_cpp_tests");
    run_test_binary("triangle_cuda_opencv_test");
    return 0;
}