#pragma once

#include <array>
#include <cstddef>

namespace nazarite::xz1::him {

constexpr std::size_t kActionDim = 12;
constexpr std::size_t kFrameDim = 47;
constexpr std::size_t kHistorySize = 6;
constexpr std::size_t kObservationDim = kFrameDim * kHistorySize;

struct RobotState {
  std::array<float, 3> angular_velocity_rad_s{};
  std::array<float, 4> quaternion_body_to_world_wxyz{1, 0, 0, 0};
  std::array<float, 12> joint_position_rad{};
  std::array<float, 12> joint_velocity_rad_s{};
};

struct JointTargets {
  std::array<float, 12> position_rad{};
  std::array<float, 12> kp{};
  std::array<float, 12> kd{};
};

class HimObservation {
 public:
  HimObservation();
  void reset();
  std::array<float, kObservationDim> build(
      const RobotState& state, const std::array<float, 3>& command,
      const std::array<float, 12>& previous_action);
  void advance();
  const std::array<float, 12>& action_offset() const noexcept;
  const std::array<float, 12>& action_scale() const noexcept;
  const std::array<float, 12>& default_joint_position() const noexcept;
  JointTargets targets(const std::array<float, 12>& raw_action) const;

 private:
  std::array<std::array<float, kFrameDim>, kHistorySize> history_{};
  float phase_{0};
  bool initialized_{false};
};

}  // namespace nazarite::xz1::him
