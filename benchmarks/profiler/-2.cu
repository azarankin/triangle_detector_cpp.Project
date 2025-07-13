#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <functional>
#include <filesystem>
#include <numeric>

// Console coloring
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

namespace fs = std::filesystem;

// ---- עזרי מדידת usage ----
double get_cpu_usage() {
    static long last_idle = 0, last_total = 0;
    std::ifstream f("/proc/stat");
    std::string line; std::getline(f, line);
    std::istringstream iss(line);
    std::string cpu; long user, nice, system, idle, iowait, irq, softirq, steal;
    iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
    long idle_now = idle + iowait;
    long non_idle = user + nice + system + irq + softirq + steal;
    long total = idle_now + non_idle;
    double percent = 0.0;
    if (last_total != 0) {
        long totald = total - last_total;
        long idled = idle_now - last_idle;
        percent = (100.0 * (totald - idled)) / totald;
    }
    last_total = total; last_idle = idle_now;
    return percent;
}

double get_ram_usage_mb() {
    std::ifstream f("/proc/meminfo");
    std::string s; long mem_total = 0, mem_avail = 0;
    while (f >> s) {
        if (s == "MemTotal:") f >> mem_total;
        else if (s == "MemAvailable:") f >> mem_avail;
    }
    return (mem_total - mem_avail) / 1024.0;
}

// ---- סטטיסטיקות ----
struct StatSummary {
    double min = 0.0, max = 0.0, mean = 0.0, stddev = 0.0, baseline = 0.0;
};
StatSummary compute_stats(const std::vector<double>& v, double baseline=0.0) {
    StatSummary s;
    if (v.empty()) return s;
    s.baseline = baseline;
    std::vector<double> v_adj;
    for (double x : v) v_adj.push_back(x-baseline);
    s.min = *std::min_element(v_adj.begin(), v_adj.end());
    s.max = *std::max_element(v_adj.begin(), v_adj.end());
    double sum = 0.0, sq_sum = 0.0;
    for (auto x : v_adj) sum += x, sq_sum += x*x;
    s.mean = sum / v_adj.size();
    s.stddev = sqrt(sq_sum/v_adj.size() - s.mean*s.mean);
    return s;
}

struct ResourceTimeline { std::vector<double> values; StatSummary summary; };
struct ProfileStats {
    std::string label;
    int num_runs = 0;
    StatSummary runtime_ms;
    std::vector<double> runtimes_all;
    ResourceTimeline cpu_usage_percent;
    ResourceTimeline ram_usage_mb;
    // ניתן להוסיף GPU
};

// ---- דגימת baseline ----
void sample_baseline(double& cpu_base, double& ram_base, int N=5, int interval_ms=30) {
    std::vector<double> cpu, ram;
    for (int i=0;i<N;++i) {
        cpu.push_back(get_cpu_usage());
        ram.push_back(get_ram_usage_mb());
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
    cpu_base = std::accumulate(cpu.begin(), cpu.end(), 0.0) / cpu.size();
    ram_base = std::accumulate(ram.begin(), ram.end(), 0.0) / ram.size();
}

// ---- דגימת usage במהלך ריצה ----
void sample_resources(std::atomic<bool>& running, ResourceTimeline& cpu, ResourceTimeline& ram, int interval_ms = 20)
{
    while (running.load()) {
        cpu.values.push_back(get_cpu_usage());
        ram.values.push_back(get_ram_usage_mb());
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
}

// ---- פונקציית פרופיילינג ----
ProfileStats profile_function(const std::string& label, std::function<void()> fn, int N)
{
    ProfileStats stats;
    stats.label = label;
    stats.num_runs = N;

    // דגום baseline
    double cpu_base=0.0, ram_base=0.0;
    sample_baseline(cpu_base, ram_base, 5);

    std::atomic<bool> running(true);
    ResourceTimeline cpu, ram;

    // Thread דגימה
    std::thread sampler([&]{
        sample_resources(running, cpu, ram, 20);
    });

    // מדידת זמן והרצת N פעמים
    for (int i = 0; i < N; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        fn();
        auto t2 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(t2-t1).count();
        stats.runtimes_all.push_back(dt);
    }

    running = false;
    sampler.join();

    // חישוב summary (תוקן להורדת baseline)
    stats.runtime_ms = compute_stats(stats.runtimes_all); // זמן – בלי baseline!
    cpu.summary = compute_stats(cpu.values, cpu_base);
    ram.summary = compute_stats(ram.values, ram_base);
    stats.cpu_usage_percent = cpu;
    stats.ram_usage_mb = ram;
    return stats;
}

// ---- הדפסת summary צבעונית ----
void print_profile_stats(const ProfileStats& stats)
{
    auto& c = stats.cpu_usage_percent.summary;
    auto& r = stats.ram_usage_mb.summary;
    std::cout << CYAN << "\n=== " << stats.label << " Profile Stats ===" << RESET << std::endl;
    std::cout << BOLD << "⏱️  Runs: " << stats.num_runs << RESET << std::endl;
    std::cout << "Time (ms): mean " << c.mean << " | min " << c.min << " | max " << c.max << " | std " << c.stddev << std::endl;
    std::cout << GREEN << "CPU %:    mean " << c.mean << " | min " << c.min << " | max " << c.max << " | baseline " << c.baseline << RESET << std::endl;
    std::cout << YELLOW << "RAM MB:   mean " << r.mean << " | min " << r.min << " | max " << r.max << " | baseline " << r.baseline << RESET << std::endl;
    std::cout << CYAN << "===========================================\n" << RESET << std::endl;
}

// ---- דוגמה לשימוש ----
void function_to_profile() {
    // פעולה לדוגמה (למשל sleep, עיבוד תמונה, וכו')
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
}

int main() {
    int N = 100;
    auto stats = profile_function("Sample", function_to_profile, N);
    print_profile_stats(stats);
    return 0;
}
