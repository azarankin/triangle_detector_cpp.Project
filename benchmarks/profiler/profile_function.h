#pragma once
#include <functional>
#include <thread>
#include <atomic>
#include "profile_stats.h"
#include "profile_sampler.h"

// --- ביצוע N פעמים, עם baseline לפני, וחיסור baseline בתוצאות הסופיות
ProfileStats profile_function(const std::string& label, std::function<void()> fn, int N)
{
    ProfileStats stats;
    stats.label = label;
    stats.num_runs = N;

    // --- שלב 1: דגום baseline (לפני ריצה)
    ResourceTimeline cpu_base, ram_base;
    sample_baseline(cpu_base, ram_base, 5, 50);

    // --- שלב 2: הפעל דגימה ברקע תוך כדי ריצה
    std::atomic<bool> running(true);
    ResourceTimeline cpu_run, ram_run;
    std::thread sampler([&]{ sample_resources(running, cpu_run, ram_run, 20); });

    // --- שלב 3: הרץ את הפונקציה N פעמים, שמור זמני ריצה
    for (int i = 0; i < N; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        fn();
        auto t2 = std::chrono::high_resolution_clock::now();
        stats.runtimes_all.push_back(std::chrono::duration<double, std::milli>(t2-t1).count());
    }
    running = false;
    sampler.join();

    // --- שלב 4: חשב סטטיסטיקות
    stats.runtime_ms = compute_stats(stats.runtimes_all);

    // --- דחוף ל־ResourceTimeline
    cpu_run.summary = compute_stats(cpu_run.values);
    ram_run.summary = compute_stats(ram_run.values);

    cpu_run.baseline = cpu_base.baseline;
    ram_run.baseline = ram_base.baseline;

    // החסרת baseline מהמדדים (mean בלבד)
    cpu_run.adjusted = cpu_run.summary;
    ram_run.adjusted = ram_run.summary;
    cpu_run.adjusted.mean -= cpu_base.baseline.mean;
    ram_run.adjusted.mean -= ram_base.baseline.mean;

    stats.cpu_usage_percent = cpu_run;
    stats.ram_usage_mb = ram_run;

    // (אם תרצה – הוסף GPU בהתאם)
    return stats;
}
