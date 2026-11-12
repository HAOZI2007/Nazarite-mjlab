#include "deployment_config.hpp"
#include "xz1_him.hpp"

#include <array>
#include <iomanip>
#include <iostream>

int main() {
  using namespace nazarite::xz1::him;
  constexpr std::array<const char*, 12> joints{
      "FL_hip", "FL_thigh", "FL_shank", "FR_hip", "FR_thigh", "FR_shank",
      "RR_hip", "RR_thigh", "RR_shank", "RL_hip", "RL_thigh", "RL_shank"};
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "control_dt_s=" << config::kControlDt << " ("
            << 1.0F / config::kControlDt << " Hz)\n"
            << "phase_period_s=" << config::kPhasePeriod << '\n';
  std::cout << "index,joint,default_q_rad,action_offset_rad,action_scale,kp,kd,lower_rad,upper_rad\n";
  for (std::size_t i = 0; i < kActionDim; ++i) {
    std::cout << i << ',' << joints[i] << ','
              << config::kDefaultJointPosition[i] << ',' << config::kActionOffset[i]
              << ',' << config::kActionScale[i] << ',' << config::kKp[i] << ','
              << config::kKd[i] << ',' << config::kJointLower[i] << ','
              << config::kJointUpper[i] << '\n';
  }
}
