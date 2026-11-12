#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <string>

namespace nazarite::xz1::motor {

struct JointTarget;

struct Command {
  std::uint8_t motor_id{1};
  bool enabled{false};
  bool timeout_protection{false};
  float torque_nm{0};
  float speed_rad_s{0};
  float position_rad{0};
  float kp_nm_rad{0};
  float kd_nm_per_rad_s{0};
};

struct Feedback {
  std::uint8_t motor_id{0};
  bool enabled{false};
  bool timeout_triggered{false};
  std::int8_t driver_temp_c{0};
  std::uint8_t winding_temp_c{0};
  float voltage_v{0};
  float torque_nm{0};
  float speed_rad_s{0};
  float position_rad{0};
  std::uint32_t error{0};
  std::uint8_t warning{0};
};

std::uint32_t crc32_mpeg2_words(const std::uint8_t* data, std::size_t size);
std::array<std::uint8_t, 20> encode(const Command& command);
Command position_command(std::uint8_t motor_id, float position_rad,
                         float kp_nm_rad, float kd_nm_per_rad_s);
Command velocity_command(std::uint8_t motor_id, float speed_rad_s,
                         float kd_nm_per_rad_s);
Feedback decode(const std::array<std::uint8_t, 26>& frame,
                std::uint8_t expected_motor_id = 0xFF);

class Connection {
 public:
  Connection(const std::string& device, int baud, std::uint8_t motor_id,
             int timeout_ms = 100);
  ~Connection();
  Connection(const Connection&) = delete;
  Connection& operator=(const Connection&) = delete;
  Feedback transact(const Command& command);
  void stop();
  void stop(std::uint8_t motor_id);

 private:
  int fd_{-1};
  std::uint8_t motor_id_;
  int timeout_ms_;
  std::array<std::uint8_t, 512> rx_{};
  std::size_t rx_size_{0};
};

}  // namespace nazarite::xz1::motor
