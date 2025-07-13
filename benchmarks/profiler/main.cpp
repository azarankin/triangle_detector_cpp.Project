#include "profile_function.h"
#include <iostream>

void my_function() {
    // כאן תכניס את הקוד שלך
    for (volatile int i = 0; i < 10000000; ++i);
}

void print_profile_stats(const ProfileStats& stats) {
    std::cout << "\n=== Profile Results for: " << stats.label << " ===\n";
    std::cout << "Runs: " << stats.num_runs << "\n";
    std::cout << "Runtime (ms): min=" << stats.runtime_ms.min 
              << ", max=" << stats.runtime_ms.max
              << ", mean=" << stats.runtime_ms.mean
              << ", stddev=" << stats.runtime_ms.stddev << "\n";

    std::cout << "CPU usage (%): mean=" << stats.cpu_usage_percent.summary.mean
              << " [baseline mean: " << stats.cpu_usage_percent.baseline.mean
              << ", adjusted: " << stats.cpu_usage_percent.adjusted.mean << "]\n";
    std::cout << "RAM usage (MB): mean=" << stats.ram_usage_mb.summary.mean
              << " [baseline mean: " << stats.ram_usage_mb.baseline.mean
              << ", adjusted: " << stats.ram_usage_mb.adjusted.mean << "]\n";
    std::cout << "==============================\n";
}

int main() {
    int N = 50;
    ProfileStats stats = profile_function("My Heavy Loop", my_function, N);
    print_profile_stats(stats);
    return 0;
}
