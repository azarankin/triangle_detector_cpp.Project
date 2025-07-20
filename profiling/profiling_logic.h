
#include <unistd.h>
#include <sys/ptrace.h>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

#ifdef ENABLE_IMAGE_DEBUG
    #error "ENABLE_IMAGE_DEBUG is defined, but this file should not be compiled with it."
#endif

#ifndef ENABLE_PROFILING
    #error "ENABLE_PROFILING is not defined, but this file should be compiled with it."
#endif


#ifndef COMPILED_AS_RELEASE
    #error "COMPILED_AS_RELEASE is not defined"
#endif

#ifndef COMPILED_WITH_O3
    #error "COMPILED_WITH_O3 is not defined"
#endif


#define GREEN   "\033[32m"
#define RESET   "\033[0m"
#define CYAN    "\033[36m"
#define BLACK   "\033[30m"
#define BG_WHITE   "\033[47m"

fs::path dir = fs::path(__FILE__).parent_path();
fs::path inpuy_path = dir / "input";
fs::path output_path = dir / "output";
fs::path triangle_img = inpuy_path / "triangle.png";
fs::path other_triangle_img = inpuy_path / "other_triangle.png";
fs::path output_result_img = output_path / "result.png";

bool is_debugger_attached() {
    if(ptrace(PTRACE_TRACEME, 0, 1, 0) == -1)
    {
        //std::cerr << "Error: Debugger detected on Release binary. Exiting.\n";
        return true;
    }
    return false;
}


struct ProfilingArgs {
    std::string name = "";
    std::string target = "";
    int num_runs = -1;
    int num_streams = -1;
    bool multithreading = false;
    std::string input_path = "";
    std::string profiling_output = "";
};


// Extracts and validates command line arguments
bool parse_and_validate_args(int argc, char* argv[], ProfilingArgs& args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--name" && i + 1 < argc) args.name = argv[++i];
        else if (arg == "--target" && i + 1 < argc) args.target = argv[++i];
        else if (arg == "--runs" && i + 1 < argc) args.num_runs = std::stoi(argv[++i]);
        else if (arg == "--cuda_streams" && i + 1 < argc) args.num_streams = std::stoi(argv[++i]);
        else if (arg == "--multithreading") args.multithreading = true;
        else if (arg == "--input" && i + 1 < argc) args.input_path = argv[++i];
        else if (arg == "--profiling_output" && i + 1 < argc) args.profiling_output = argv[++i];
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --runs <number>               Number of runs to perform\n"
                      << "  --cuda_streams <number>       Number of CUDA streams to use\n"
                      << "  --multithreading              Enable multithreading\n"
                      << "  --input <path>                Path to input image\n"
                      << "  --profiling_output <path>     Path to save profiling output\n";
            return false;  // help - לא רץ הלאה
        }
    }

    bool error = false;

    if (args.name.empty()) {
        std::cerr << "ERROR: --name <name> is required!\n";
        error = true;
    }
    if (args.target.empty()) {
        std::cerr << "ERROR: --target <name> is required!\n";
        error = true;
    }
    if (args.num_runs < 0) {
        std::cerr << "ERROR: --runs <number> is required!\n";
        error = true;
    }
    if (args.num_streams < 0) {
        std::cerr << "ERROR: --cuda_streams <number> is required!\n";
        error = true;
    }
    if (args.profiling_output.empty()) {
        std::cerr << "ERROR: --profiling_output <path> is required!\n";
        error = true;
    }


    return !error;
}


void image_save(const fs::path& path, const cv::Mat& img)
{
    if (path.empty()) return;
    if (img.empty()) return;

    std::filesystem::create_directories(path.parent_path());
    //std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
    cv::imwrite(path, img);
}

