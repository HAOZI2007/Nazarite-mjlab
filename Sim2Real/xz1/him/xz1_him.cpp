#include "xz1_him.hpp"
#include "deployment_config.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nazarite::xz1::him {
namespace {
constexpr float kPi = 3.14159265358979323846F;
void require_finite(float value, const char* name) {
  if (!std::isfinite(value)) throw std::invalid_argument(name);
}
}  // namespace

HimObservation::HimObservation() { reset(); }
void HimObservation::reset() {
  for (auto& frame : history_) frame.fill(0);
  phase_ = 0;
  initialized_ = false;
}
const std::array<float, 12>& HimObservation::action_offset() const noexcept { return config::kActionOffset; }
const std::array<float, 12>& HimObservation::action_scale() const noexcept { return config::kActionScale; }
const std::array<float, 12>& HimObservation::default_joint_position() const noexcept { return config::kDefaultJointPosition; }

std::array<float, kObservationDim> HimObservation::build(
    const RobotState& state, const std::array<float, 3>& command,
    const std::array<float, 12>& previous_action) {
  std::array<float, 4> q = state.quaternion_body_to_world_wxyz;
  float norm_squared = 0;
  for (float value : q) { require_finite(value, "quaternion must be finite"); norm_squared += value * value; }
  const float norm = std::sqrt(norm_squared);
  if (norm < .9F || norm > 1.1F) throw std::invalid_argument("invalid IMU quaternion norm");
  for (auto& value : q) value /= norm;
  const float w=q[0], x=q[1], y=q[2], z=q[3];
  std::array<float, kFrameDim> frame{};
  for(std::size_t i=0;i<3;++i){require_finite(state.angular_velocity_rad_s[i],"gyro must be finite");frame[i]=state.angular_velocity_rad_s[i];}
  frame[3] = 2*(w*y-x*z);
  frame[4] = -2*(y*z+w*x);
  frame[5] = 2*(x*x+y*y)-1;
  for (std::size_t i=0;i<3;++i) { require_finite(command[i], "command must be finite"); if(std::abs(command[i])>1) throw std::invalid_argument("command outside [-1,1]"); frame[6+i]=command[i]; }
  if (std::hypot(command[0], std::hypot(command[1],command[2])) >= .1F) {
    frame[9]=std::sin(2*kPi*phase_); frame[10]=std::cos(2*kPi*phase_);
  }
  for(std::size_t i=0;i<12;++i) {
    require_finite(state.joint_position_rad[i], "joint position must be finite");
    require_finite(state.joint_velocity_rad_s[i], "joint velocity must be finite");
    require_finite(previous_action[i], "previous action must be finite");
    frame[11+i]=state.joint_position_rad[i]-config::kDefaultJointPosition[i];
    frame[23+i]=state.joint_velocity_rad_s[i];
    frame[35+i]=std::clamp(previous_action[i],-10.0F,10.0F);
  }
  for(float& value:frame) { require_finite(value,"observation must be finite"); value=std::clamp(value,-100.0F,100.0F); }
  for(std::size_t i=35;i<47;++i) frame[i]=std::clamp(frame[i],-10.0F,10.0F);
  if(!initialized_) { for(auto& old:history_) old=frame; initialized_=true; }
  else { for(std::size_t i=kHistorySize-1;i>0;--i) history_[i]=history_[i-1]; history_[0]=frame; }
  std::array<float,kObservationDim> result{};
  for(std::size_t i=0;i<kHistorySize;++i) std::copy(history_[i].begin(),history_[i].end(),result.begin()+static_cast<std::ptrdiff_t>(i*kFrameDim));
  return result;
}
void HimObservation::advance() { phase_=std::fmod(phase_+config::kControlDt/config::kPhasePeriod,1.0F); }

JointTargets HimObservation::targets(const std::array<float,12>& raw_action) const {
  JointTargets result;
  for(std::size_t i=0;i<12;++i) {
    require_finite(raw_action[i],"policy action must be finite");
    const float action=std::clamp(raw_action[i],-10.0F,10.0F);
    result.position_rad[i]=config::kActionOffset[i]+config::kActionScale[i]*action;
    if(result.position_rad[i]<config::kJointLower[i] || result.position_rad[i]>config::kJointUpper[i])
      throw std::out_of_range("policy joint target exceeds XZ1 MJCF joint range");
    result.kp[i]=config::kKp[i]; result.kd[i]=config::kKd[i];
  }
  return result;
}
}  // namespace nazarite::xz1::him
