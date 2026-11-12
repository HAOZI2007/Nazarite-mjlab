#include "dm_imu.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <system_error>
#include <termios.h>
#include <unistd.h>

namespace nazarite::xz1::imu {
namespace {

constexpr std::array<std::uint8_t, 2> kHeader{0x55, 0xAA};

std::uint16_t crc16(const std::uint8_t* data, std::size_t size) {
  std::uint16_t crc = 0xFFFF;
  for (std::size_t i = 0; i < size; ++i) {
    crc ^= static_cast<std::uint16_t>(data[i]) << 8;
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x8000U) ? static_cast<std::uint16_t>((crc << 1) ^ 0x1021U)
                            : static_cast<std::uint16_t>(crc << 1);
    }
  }
  return crc;
}

float read_float_le(const std::uint8_t* data) {
  const std::uint32_t bits = static_cast<std::uint32_t>(data[0]) |
                             (static_cast<std::uint32_t>(data[1]) << 8) |
                             (static_cast<std::uint32_t>(data[2]) << 16) |
                             (static_cast<std::uint32_t>(data[3]) << 24);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::array<float, 4> multiply(const std::array<float, 4>& a,
                              const std::array<float, 4>& b) {
  return {a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
          a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
          a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
          a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]};
}

std::array<float, 3> rotate(const std::array<float, 4>& q,
                            const std::array<float, 3>& v) {
  const std::array<float, 4> pure{0, v[0], v[1], v[2]};
  const std::array<float, 4> conjugate{q[0], -q[1], -q[2], -q[3]};
  const auto out = multiply(multiply(q, pure), conjugate);
  return {out[1], out[2], out[3]};
}

speed_t baud_constant(int baud) {
  switch (baud) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
#ifdef B230400
    case 230400: return B230400;
#endif
#ifdef B460800
    case 460800: return B460800;
#endif
#ifdef B921600
    case 921600: return B921600;
#endif
    default: throw std::invalid_argument("unsupported IMU baud rate");
  }
}

std::uint64_t monotonic_ns() {
  timespec ts{};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    throw std::system_error(errno, std::generic_category(), "clock_gettime");
  }
  return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ULL +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

}  // namespace

DmImuParser::DmImuParser(float gyro_scale_to_rad_s)
    : gyro_scale_to_rad_s_(gyro_scale_to_rad_s) {
  if (!std::isfinite(gyro_scale_to_rad_s_) || gyro_scale_to_rad_s_ <= 0.0F) {
    throw std::invalid_argument("gyro scale must be finite and positive");
  }
}

ImuSample transform_to_body(const ImuSample& sensor_sample,
                            const std::array<float, 4>& sensor_to_body_wxyz) {
  auto q = sensor_to_body_wxyz;
  float norm2 = 0;
  for (float value : q) {
    if (!std::isfinite(value)) throw std::invalid_argument("mount quaternion must be finite");
    norm2 += value * value;
  }
  const float norm = std::sqrt(norm2);
  if (norm < .9F || norm > 1.1F)
    throw std::invalid_argument("invalid sensor mounting quaternion");
  for (auto& value : q) value /= norm;
  ImuSample result = sensor_sample;
  if (sensor_sample.has_angular_velocity)
    result.angular_velocity_rad_s = rotate(q, sensor_sample.angular_velocity_rad_s);
  if (sensor_sample.has_quaternion) {
    const std::array<float, 4> conjugate{q[0], -q[1], -q[2], -q[3]};
    result.quaternion_body_to_world_wxyz =
        multiply(sensor_sample.quaternion_body_to_world_wxyz, conjugate);
  }
  return result;
}

std::vector<ImuSample> DmImuParser::feed(const std::uint8_t* data,
                                         std::size_t size,
                                         std::uint64_t timestamp_ns) {
  if (size && data == nullptr) throw std::invalid_argument("null IMU input");
  buffer_.insert(buffer_.end(), data, data + size);
  std::vector<ImuSample> output;
  while (true) {
    const auto header = std::search(buffer_.begin(), buffer_.end(),
                                    kHeader.begin(), kHeader.end());
    if (header == buffer_.end()) {
      const bool retain = !buffer_.empty() && buffer_.back() == kHeader.front();
      const auto discard = buffer_.size() - (retain ? 1U : 0U);
      discarded_bytes_ += discard;
      buffer_.erase(buffer_.begin(), buffer_.begin() +
                                           static_cast<std::ptrdiff_t>(discard));
      break;
    }
    if (header != buffer_.begin()) {
      const auto count = static_cast<std::size_t>(header - buffer_.begin());
      discarded_bytes_ += count;
      buffer_.erase(buffer_.begin(), header);
    }
    if (buffer_.size() < 4) break;
    const auto type = buffer_[3];
    const std::size_t length = (type == 0x04) ? 23 :
        ((type == 0x01 || type == 0x02 || type == 0x03 || type == 0x07) ? 19 : 0);
    if (length == 0) {
      ++discarded_bytes_;
      buffer_.erase(buffer_.begin());
      continue;
    }
    if (buffer_.size() < length) break;
    const auto received = static_cast<std::uint16_t>(buffer_[length - 3]) |
                          (static_cast<std::uint16_t>(buffer_[length - 2]) << 8);
    if (buffer_[length - 1] != 0x0A ||
        crc16(buffer_.data(), length - 3) != received) {
      ++crc_errors_;
      ++discarded_bytes_;
      buffer_.erase(buffer_.begin());
      continue;
    }

    ImuSample sample;
    sample.timestamp_ns = timestamp_ns;
    if (type == 0x02) {
      for (std::size_t i = 0; i < 3; ++i) {
        sample.angular_velocity_rad_s[i] =
            read_float_le(buffer_.data() + 4 + 4 * i) * gyro_scale_to_rad_s_;
      }
      sample.has_angular_velocity = true;
    } else {
      for (std::size_t i = 0; i < 4; ++i) {
        sample.quaternion_body_to_world_wxyz[i] =
            read_float_le(buffer_.data() + 4 + 4 * i);
      }
      const auto& q = sample.quaternion_body_to_world_wxyz;
      const float norm = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
      if (std::isfinite(norm) && norm > 1.0e-6F) {
        for (auto& value : sample.quaternion_body_to_world_wxyz) value /= norm;
        sample.has_quaternion = true;
      }
    }
    if ((type == 0x02 && std::all_of(sample.angular_velocity_rad_s.begin(),
                                     sample.angular_velocity_rad_s.end(),
                                     [](float value) { return std::isfinite(value); })) ||
        sample.has_quaternion) {
      output.push_back(sample);
    }
    buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(length));
  }
  return output;
}

DmImuSerial::DmImuSerial(const std::string& device, int baud,
                         float gyro_scale_to_rad_s)
    : device_(device), parser_(gyro_scale_to_rad_s) {
  fd_ = ::open(device.c_str(), O_RDONLY | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "open IMU");
  termios tty{};
  if (tcgetattr(fd_, &tty) != 0) {
    const int saved = errno; ::close(fd_); fd_ = -1;
    throw std::system_error(saved, std::generic_category(), "tcgetattr IMU");
  }
  cfmakeraw(&tty);
  const auto speed = baud_constant(baud);
  cfsetispeed(&tty, speed);
  cfsetospeed(&tty, speed);
  tty.c_cflag |= CLOCAL | CREAD;
  tty.c_cflag &= static_cast<tcflag_t>(~(CSTOPB | PARENB | CSIZE));
  tty.c_cflag |= CS8;
  if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
    const int saved = errno; ::close(fd_); fd_ = -1;
    throw std::system_error(saved, std::generic_category(), "tcsetattr IMU");
  }
  tcflush(fd_, TCIFLUSH);
}

DmImuSerial::~DmImuSerial() { if (fd_ >= 0) ::close(fd_); }

ImuSample DmImuSerial::read_sample(int timeout_ms) {
  if (timeout_ms < 0) throw std::invalid_argument("timeout must be nonnegative");
  const auto deadline = monotonic_ns() + static_cast<std::uint64_t>(timeout_ms) * 1000000ULL;
  std::array<std::uint8_t, 256> bytes{};
  while (true) {
    const auto now = monotonic_ns();
    if (now >= deadline) break;
    pollfd item{fd_, POLLIN, 0};
    const int remaining = static_cast<int>((deadline - now) / 1000000ULL);
    const int ready = ::poll(&item, 1, std::max(remaining, 1));
    if (ready < 0 && errno == EINTR) continue;
    if (ready < 0) throw std::system_error(errno, std::generic_category(), "poll IMU");
    if (ready == 0) break;
    if (item.revents & (POLLERR | POLLHUP | POLLNVAL))
      throw std::runtime_error("IMU serial disconnected");
    const auto count = ::read(fd_, bytes.data(), bytes.size());
    if (count < 0 && (errno == EAGAIN || errno == EINTR)) continue;
    if (count < 0) throw std::system_error(errno, std::generic_category(), "read IMU");
    auto samples = parser_.feed(bytes.data(), static_cast<std::size_t>(count), monotonic_ns());
    for (const auto& sample : samples) {
      if (sample.has_angular_velocity && sample.has_quaternion) return sample;
      // Pair adjacent sensor frames by caching the most recent partial sample.
      if (sample.has_angular_velocity) last_gyro_ = sample;
      if (sample.has_quaternion) last_quaternion_ = sample;
      const auto now = sample.timestamp_ns;
      if (last_gyro_.has_angular_velocity && now > last_gyro_.timestamp_ns &&
          now - last_gyro_.timestamp_ns > 50000000ULL) last_gyro_ = {};
      if (last_quaternion_.has_quaternion && now > last_quaternion_.timestamp_ns &&
          now - last_quaternion_.timestamp_ns > 50000000ULL) last_quaternion_ = {};
      if (last_gyro_.has_angular_velocity && last_quaternion_.has_quaternion) {
        last_gyro_.quaternion_body_to_world_wxyz = last_quaternion_.quaternion_body_to_world_wxyz;
        last_gyro_.has_quaternion = true;
        last_gyro_.timestamp_ns = std::max(last_gyro_.timestamp_ns, last_quaternion_.timestamp_ns);
        return last_gyro_;
      }
    }
  }
  throw std::runtime_error("timed out waiting for IMU gyro and quaternion frames");
}

}  // namespace nazarite::xz1::imu
