#include "onnx_policy.hpp"

#ifdef XZ1_HAS_ONNXRUNTIME
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

double process_cpu_seconds() {
  return static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}

double percentile(const std::vector<double>& sorted, double percent) {
  const double index = percent / 100.0 * (sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(index);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  const double fraction = index - lower;
  return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

int parse_int(const char* text, const char* name, int minimum) {
  std::size_t parsed = 0;
  const int value = std::stoi(text, &parsed);
  if (text[parsed] != '\0' || value < minimum) {
    throw std::invalid_argument(std::string(name) + " must be >= " +
                                std::to_string(minimum));
  }
  return value;
}

double parse_hz(const char* text) {
  std::size_t parsed = 0;
  const double value = std::stod(text, &parsed);
  if (text[parsed] != '\0' || !std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument("HZ must be finite and >= 0");
  }
  return value;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 5) {
    std::cerr << "Usage: " << argv[0]
              << " policy.onnx [WARMUP=100] [ITERATIONS=1000] [HZ=0]\n"
              << "HZ=0 runs continuously; HZ=50 measures a paced 50 Hz loop.\n";
    return 2;
  }

  try {
    const int warmup = argc >= 3 ? parse_int(argv[2], "WARMUP", 0) : 100;
    const int iterations = argc >= 4 ? parse_int(argv[3], "ITERATIONS", 1)
                                     : 1000;
    const double hz = argc >= 5 ? parse_hz(argv[4]) : 0.0;

    const auto load_start = Clock::now();
    nazarite::xz1::him::OnnxPolicy policy(argv[1]);
    const double load_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - load_start)
            .count();

    std::array<float, nazarite::xz1::him::kObservationDim> observation{};
    const auto first_start = Clock::now();
    const auto first_action = policy.infer(observation);
    const double first_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - first_start)
            .count();

    for (int i = 0; i < warmup; ++i) {
      (void)policy.infer(observation);
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    const auto loop_start = Clock::now();
    const double cpu_start = process_cpu_seconds();
    std::size_t late_cycles = 0;
    const auto period = hz > 0.0
                            ? std::chrono::duration<double>(1.0 / hz)
                            : std::chrono::duration<double>::zero();

    for (int i = 0; i < iterations; ++i) {
      const auto start = Clock::now();
      (void)policy.infer(observation);
      samples.push_back(
          std::chrono::duration<double, std::milli>(Clock::now() - start)
              .count());

      if (hz > 0.0) {
        const auto deadline = loop_start + period * (i + 1);
        const auto now = Clock::now();
        if (now < deadline) {
          std::this_thread::sleep_until(deadline);
        } else {
          ++late_cycles;
        }
      }
    }

    const double wall_seconds =
        std::chrono::duration<double>(Clock::now() - loop_start).count();
    const double cpu_seconds = process_cpu_seconds() - cpu_start;
    std::sort(samples.begin(), samples.end());
    double sum_ms = 0.0;
    for (const double sample : samples) sum_ms += sample;
    const double mean_ms = sum_ms / samples.size();
    const double budget_ms = 20.0;
    const auto over_budget = static_cast<std::size_t>(std::count_if(
        samples.begin(), samples.end(),
        [budget_ms](double sample) { return sample > budget_ms; }));
    const long logical_cpus = ::sysconf(_SC_NPROCESSORS_ONLN);
    const double cpu_one_core = cpu_seconds / wall_seconds * 100.0;

    std::cout << std::fixed << std::setprecision(4)
              << "input=282 output=12 input_values=all_zeros\n"
              << "model_load_ms=" << load_ms
              << " first_inference_ms=" << first_ms << '\n'
              << "warmup=" << warmup << " iterations=" << iterations
              << " requested_hz=" << hz
              << " actual_hz=" << iterations / wall_seconds << '\n'
              << "mean_ms=" << mean_ms << " min_ms=" << samples.front()
              << " p50_ms=" << percentile(samples, 50)
              << " p95_ms=" << percentile(samples, 95)
              << " p99_ms=" << percentile(samples, 99)
              << " max_ms=" << samples.back() << '\n'
              << "over_20ms=" << over_budget << "/" << iterations
              << " late_cycles=" << late_cycles << '\n'
              << "wall_seconds=" << wall_seconds
              << " process_cpu_seconds=" << cpu_seconds
              << " cpu_percent_one_core=" << cpu_one_core;
    if (logical_cpus > 0) {
      std::cout << " cpu_percent_whole_machine="
                << cpu_one_core / logical_cpus
                << " logical_cpus=" << logical_cpus;
    }
    std::cout << "\naction=";
    for (const float value : first_action) std::cout << value << ' ';
    std::cout << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ONNX test failed: " << error.what() << '\n';
    return 1;
  }
}
#else
int main() { return 2; }
#endif
