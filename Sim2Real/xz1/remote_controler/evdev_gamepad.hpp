#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nazarite::xz1::controller {

struct TwistCommand {
  float vx{0};
  float vy{0};
  float yaw_rate{0};
  bool reset_requested{false};
};

std::vector<std::string> list_gamepads();

class EvdevGamepad {
 public:
  EvdevGamepad(const std::string& path, float max_vx = 1.0F,
               float max_vy = 1.0F, float max_yaw = 1.0F,
               float deadzone = 0.08F);
  ~EvdevGamepad();
  EvdevGamepad(const EvdevGamepad&) = delete;
  EvdevGamepad& operator=(const EvdevGamepad&) = delete;
  TwistCommand poll();
  const std::string& path() const noexcept { return path_; }

 private:
  int fd_{-1};
  std::string path_;
  float max_vx_, max_vy_, max_yaw_, deadzone_;
  int axis_vx_{-1}, axis_vy_{-1}, axis_yaw_{-1};
  int min_vx_{0}, max_axis_vx_{0}, min_vy_{0}, max_axis_vy_{0},
      min_yaw_{0}, max_axis_yaw_{0};
  int value_vx_{0}, value_vy_{0}, value_yaw_{0};
};

}  // namespace nazarite::xz1::controller
