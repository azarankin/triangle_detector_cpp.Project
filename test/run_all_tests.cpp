#include <cstdlib>
#include <filesystem>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

namespace fs = std::filesystem;


void run_test_binary(const std::string& binary_name) {
    fs::path exec_path = fs::current_path() / binary_name;
    std::cout << "Running: " << exec_path << std::endl;
    int ret = system(exec_path.string().c_str());
    if (ret != 0) {
        std::cerr << "Test failed: " << binary_name << " (code: " << ret << ")\n";
    }
}


std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream tokenStream(s);
    std::string token;
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}


int main() {
    // run_test_binary("triangle_detector_cpp_tests");
    // run_test_binary("triangle_cuda_opencv_test");
#ifdef ALL_TEST_BINARIES
    std::string binaries_str = ALL_TEST_BINARIES;
    auto binaries = split(binaries_str, ',');
    for (const auto& bin : binaries) {
        std::cout << "Running test: " << bin << std::endl;
        run_test_binary(bin);
    }
#else
    std::cerr << "No test binaries defined." << std::endl;
#endif
    return 0;
}


