#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nazarite::xz1::imu {

struct ImuSample {
  std::array<float, 3> angular_velocity_rad_s{};
  std::array<float, 4> quaternion_body_to_world_wxyz{1.0F, 0.0F, 0.0F, 0.0F};
  bool has_angular_velocity{false};
  bool has_quaternion{false};
  std::uint64_t timestamp_ns{0};
};

// sensor_to_body maps sensor-frame vectors into the trained XZ1 body frame.
ImuSample transform_to_body(const ImuSample& sensor_sample,
                            const std::array<float, 4>& sensor_to_body_wxyz);

class DmImuParser {
 public:
  explicit DmImuParser(float gyro_scale_to_rad_s = 0.017453292519943295F);
  std::vector<ImuSample> feed(const std::uint8_t* data, std::size_t size,
                              std::uint64_t timestamp_ns);
  std::uint64_t crc_errors() const noexcept { return crc_errors_; }
  std::uint64_t discarded_bytes() const noexcept { return discarded_bytes_; }

 private:
  float gyro_scale_to_rad_s_;
  std::vector<std::uint8_t> buffer_;
  std::uint64_t crc_errors_{0};
  std::uint64_t discarded_bytes_{0};
};

class DmImuSerial {
 public:
  DmImuSerial(const std::string& device, int baud = 921600,
              float gyro_scale_to_rad_s = 0.017453292519943295F);
  ~DmImuSerial();
  DmImuSerial(const DmImuSerial&) = delete;
  DmImuSerial& operator=(const DmImuSerial&) = delete;
  ImuSample read_sample(int timeout_ms = 50);
  const std::string& device() const noexcept { return device_; }

 private:
  int fd_{-1};
  std::string device_;
  DmImuParser parser_;
  ImuSample last_gyro_{};
  ImuSample last_quaternion_{};
};

}  // namespace nazarite::xz1::imu
