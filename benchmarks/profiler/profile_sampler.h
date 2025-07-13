#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <unistd.h>
#include "profile_stats.h"

// --- דגימה בודדת
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

// --- דגימת baseline
void sample_baseline(ResourceTimeline& cpu, ResourceTimeline& ram, int samples = 5, int interval_ms = 50)
{
    cpu.values.clear(); ram.values.clear();
    for (int i = 0; i < samples; ++i) {
        cpu.values.push_back(get_cpu_usage());
        ram.values.push_back(get_ram_usage_mb());
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
    cpu.baseline = compute_stats(cpu.values);
    ram.baseline = compute_stats(ram.values);
}

// --- דגימה חיה ברקע
void sample_resources(std::atomic<bool>& running, ResourceTimeline& cpu, ResourceTimeline& ram, int interval_ms = 20)
{
    cpu.values.clear(); ram.values.clear();
    while (running.load()) {
        cpu.values.push_back(get_cpu_usage());
        ram.values.push_back(get_ram_usage_mb());
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
    cpu.summary = compute_stats(cpu.values);
    ram.summary = compute_stats(ram.values);
}
