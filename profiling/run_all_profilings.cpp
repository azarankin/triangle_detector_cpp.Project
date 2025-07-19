#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <sstream>

void run_profiling(const std::string& binary, const std::string& argstr) {
    std::vector<std::string> args;
    std::istringstream iss(argstr);
    std::string token;
    while (iss >> token)
        args.push_back(token);


    std::string cmd = "./" + binary;
    for (const auto& arg : args) {
        cmd += " " + arg;
    }
    std::cout << "Running: " << cmd << std::endl;
    int ret = system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Profiling run failed (exit code: " << ret << ")\n";
    }
}

// דוגמה שימוש
int main() {
    // תוכל לייצר כל סט של פרמטרים
    std::vector<std::string> runs = {
        "--name arthur --target shape_detector --runs 500 --cuda_streams 4 --multithreading --input img1.png --profiling_output aaa1.bbb",
        "--name arthur2 --target shape_detector --runs 500 --cuda_streams 1 --input img1.png --profiling_output aaa2.bbb"
    };

    for (const auto& argstr : runs) {

        run_profiling("opencv_cuda_profiling", argstr);
    }


    return 0;
}
