#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace nazarite::xz1::motor {

// Smooth output-shaft velocity reference; acceleration is zero at both ends.
class VelocityRamp {
 public:
  VelocityRamp(std::array<float, 3> start, float target, double max_acceleration)
      : start_(start), target_(target) {
    if (!std::isfinite(target) || !std::isfinite(max_acceleration) ||
        max_acceleration <= 0) {
      throw std::invalid_argument("invalid velocity ramp target/acceleration");
    }
    double distance = 0;
    for (float speed : start) {
      if (!std::isfinite(speed)) throw std::invalid_argument("invalid initial speed");
      distance = std::max(distance, std::abs(double(target) - speed));
    }
    duration_ = std::max(0.5, 1.5 * distance / max_acceleration);
  }

  double duration() const { return duration_; }

  std::array<float, 3> sample(double elapsed) const {
    if (!std::isfinite(elapsed)) throw std::invalid_argument("invalid ramp time");
    const double u = std::clamp(elapsed / duration_, 0.0, 1.0);
    const double blend = u * u * (3.0 - 2.0 * u);
    std::array<float, 3> speeds{};
    for (std::size_t i = 0; i < speeds.size(); ++i) {
      speeds[i] = static_cast<float>(start_[i] + (target_ - start_[i]) * blend);
    }
    return speeds;
  }

 private:
  std::array<float, 3> start_;
  float target_;
  double duration_;
};

}  // namespace nazarite::xz1::motor
