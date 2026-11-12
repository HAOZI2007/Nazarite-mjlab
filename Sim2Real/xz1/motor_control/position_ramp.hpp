#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace nazarite::xz1::motor {

// Output-shaft absolute angles. The cubic ramp starts and ends at zero speed.
class PositionRamp {
 public:
  PositionRamp(std::array<float, 3> start, float target, double max_speed,
               double max_acceleration = 0.5)
      : start_(start), target_(target) {
    if (!std::isfinite(target) || !std::isfinite(max_speed) || max_speed <= 0 ||
        !std::isfinite(max_acceleration) || max_acceleration <= 0) {
      throw std::invalid_argument("invalid ramp target/speed/acceleration");
    }
    double distance = 0;
    for (float position : start) {
      if (!std::isfinite(position)) {
        throw std::invalid_argument("invalid starting position");
      }
      distance = std::max(distance, std::abs(double(target) - position));
    }
    // Peak speed = 1.5 * distance / T; peak acceleration = 6 * distance / T^2.
    // Both requested limits constrain duration; the motion need not reach either.
    duration_ = std::max({0.5, 1.5 * distance / max_speed,
                          std::sqrt(6.0 * distance / max_acceleration)});
    peak_speed_ = 1.5 * distance / duration_;
  }

  double duration() const { return duration_; }
  double peak_speed() const { return peak_speed_; }

  std::array<float, 3> sample(double elapsed) const {
    if (!std::isfinite(elapsed)) {
      throw std::invalid_argument("invalid ramp time");
    }
    const double u = std::clamp(elapsed / duration_, 0.0, 1.0);
    const double blend = u * u * (3.0 - 2.0 * u);
    std::array<float, 3> positions{};
    for (std::size_t i = 0; i < positions.size(); ++i) {
      positions[i] = static_cast<float>(start_[i] +
                                       (target_ - start_[i]) * blend);
    }
    return positions;
  }

 private:
  std::array<float, 3> start_;
  float target_;
  double duration_;
  double peak_speed_;
};

}  // namespace nazarite::xz1::motor
