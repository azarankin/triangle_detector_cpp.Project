#pragma once
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

struct StatSummary {
    double min = 0.0, max = 0.0, mean = 0.0, stddev = 0.0;
};
inline StatSummary compute_stats(const std::vector<double>& v) {
    StatSummary s;
    if (v.empty()) return s;
    s.min = *std::min_element(v.begin(), v.end());
    s.max = *std::max_element(v.begin(), v.end());
    double sum = 0.0, sq_sum = 0.0;
    for (auto x : v) sum += x, sq_sum += x*x;
    s.mean = sum / v.size();
    s.stddev = sqrt(sq_sum/v.size() - s.mean*s.mean);
    return s;
}

struct ResourceTimeline {
    std::vector<double> values;
    StatSummary summary;
    StatSummary baseline; // חדש! baseline
    StatSummary adjusted; // summary פחות baseline.mean
};

struct ProfileStats {
    std::string label;
    int num_runs = 0;
    StatSummary runtime_ms;
    std::vector<double> runtimes_all;
    ResourceTimeline cpu_usage_percent;
    ResourceTimeline ram_usage_mb;
    // הרחב לפי הצורך: GPU וכו'
    std::string cpu_model;
    int cpu_cores = 0;
    double ram_total_gb = 0.0;
    std::string notes;
};
