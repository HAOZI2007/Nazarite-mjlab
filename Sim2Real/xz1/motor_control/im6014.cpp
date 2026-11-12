#include "im6014.hpp"

#include <algorithm>
#include <asm/termbits.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <thread>
#include <system_error>
#include <unistd.h>

namespace nazarite::xz1::motor {
namespace {
constexpr float kGear = 38.0F / 3.0F;
constexpr float kTorqueScale = 2560.0F;
constexpr float kSpeedScale = 64.0F / (2.0F * 3.14159265358979323846F);
constexpr float kPositionScale = 32768.0F / (2.0F * 3.14159265358979323846F);
constexpr std::uint32_t kPolynomial = 0x04C11DB7U;

void write_u16(std::uint8_t* out, std::uint16_t value) {
  out[0] = static_cast<std::uint8_t>(value);
  out[1] = static_cast<std::uint8_t>(value >> 8);
}
void write_u32(std::uint8_t* out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i) out[i] = static_cast<std::uint8_t>(value >> (8 * i));
}
std::uint16_t read_u16(const std::uint8_t* in) {
  return static_cast<std::uint16_t>(in[0]) |
         (static_cast<std::uint16_t>(in[1]) << 8);
}
std::uint32_t read_u32(const std::uint8_t* in) {
  return static_cast<std::uint32_t>(in[0]) |
         (static_cast<std::uint32_t>(in[1]) << 8) |
         (static_cast<std::uint32_t>(in[2]) << 16) |
         (static_cast<std::uint32_t>(in[3]) << 24);
}
std::int16_t read_i16(const std::uint8_t* in) {
  return static_cast<std::int16_t>(read_u16(in));
}
std::int32_t read_i32(const std::uint8_t* in) {
  return static_cast<std::int32_t>(read_u32(in));
}
std::int16_t signed_scaled(const char* name, float value, float scale) {
  if (!std::isfinite(value)) throw std::invalid_argument(std::string(name) + " not finite");
  const double raw = std::round(static_cast<double>(value) * scale);
  if (raw < -32768 || raw > 32767)
    throw std::out_of_range(std::string(name) + " outside motor protocol range");
  return static_cast<std::int16_t>(raw);
}
std::int32_t position_scaled(float value) {
  if (!std::isfinite(value)) throw std::invalid_argument("position not finite");
  const double raw = std::round(static_cast<double>(value) * kPositionScale * kGear);
  if (raw < -2147483648.0 || raw > 2147483647.0)
    throw std::out_of_range("position outside motor protocol range");
  return static_cast<std::int32_t>(raw);
}
std::uint16_t gain_scaled(const char* name, float value, float scale) {
  if (!std::isfinite(value) || value < 0)
    throw std::invalid_argument(std::string(name) + " must be finite and nonnegative");
  const double raw = std::round(static_cast<double>(value) * scale);
  if (raw > 32767) throw std::out_of_range(std::string(name) + " outside protocol range");
  return static_cast<std::uint16_t>(raw);
}
void configure_baud(int fd, int baud) {
  if (baud != 4000000 && baud != 6000000)
    throw std::invalid_argument("IM6014 baud must be 4000000 or 6000000");
  termios2 tty{};
  if (ioctl(fd, TCGETS2, &tty) != 0)
    throw std::system_error(errno, std::generic_category(), "TCGETS2");
  tty.c_iflag = 0;
  tty.c_oflag = 0;
  tty.c_lflag = 0;
  tty.c_cflag &= ~(CBAUD | CSIZE | PARENB | CSTOPB | CRTSCTS);
  tty.c_cflag |= BOTHER | CS8 | CLOCAL | CREAD;
  tty.c_ispeed = static_cast<unsigned int>(baud);
  tty.c_ospeed = static_cast<unsigned int>(baud);
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 1;
  if (ioctl(fd, TCSETS2, &tty) != 0)
    throw std::system_error(errno, std::generic_category(), "TCSETS2");
  if (ioctl(fd, TCFLSH, 2) != 0)
    throw std::system_error(errno, std::generic_category(), "flush motor serial");
}
}  // namespace

std::uint32_t crc32_mpeg2_words(const std::uint8_t* data, std::size_t size) {
  if (size % 4 != 0) throw std::invalid_argument("CRC input must be 4-byte aligned");
  std::uint32_t crc = 0xFFFFFFFFU;
  for (std::size_t offset = 0; offset < size; offset += 4) {
    std::uint32_t word = static_cast<std::uint32_t>(data[offset]) |
                         (static_cast<std::uint32_t>(data[offset + 1]) << 8) |
                         (static_cast<std::uint32_t>(data[offset + 2]) << 16) |
                         (static_cast<std::uint32_t>(data[offset + 3]) << 24);
    std::uint32_t mask = 1U << 31;
    for (int bit = 0; bit < 32; ++bit) {
      const bool top = (crc & 0x80000000U) != 0;
      crc <<= 1;
      if (top) crc ^= kPolynomial;
      if (word & mask) crc ^= kPolynomial;
      mask >>= 1;
    }
  }
  return crc;
}

std::array<std::uint8_t, 20> encode(const Command& command) {
  if (command.motor_id > 14) throw std::invalid_argument("motor ID must be in [0,14]");
  std::array<std::uint8_t, 20> frame{};
  frame[0] = 0xFE; frame[1] = 0xEE;
  frame[2] = command.motor_id | (command.enabled ? 0x10 : 0) |
             (command.timeout_protection ? 0x80 : 0);
  write_u16(&frame[4], static_cast<std::uint16_t>(signed_scaled(
      "torque", command.torque_nm, kTorqueScale / kGear)));
  write_u16(&frame[6], static_cast<std::uint16_t>(signed_scaled(
      "speed", command.speed_rad_s, kSpeedScale * kGear)));
  write_u32(&frame[8], static_cast<std::uint32_t>(position_scaled(command.position_rad)));
  write_u16(&frame[12], gain_scaled("kp", command.kp_nm_rad,
                                    12800.0F / (kGear * kGear)));
  write_u16(&frame[14], gain_scaled("kd", command.kd_nm_per_rad_s,
                                    51200.0F / (kGear * kGear)));
  write_u32(&frame[16], crc32_mpeg2_words(frame.data(), 16));
  return frame;
}

Command position_command(std::uint8_t motor_id, float position_rad,
                         float kp_nm_rad, float kd_nm_per_rad_s) {
  Command command;
  command.motor_id = motor_id;
  command.enabled = true;
  command.timeout_protection = true;
  command.position_rad = position_rad;
  command.kp_nm_rad = kp_nm_rad;
  command.kd_nm_per_rad_s = kd_nm_per_rad_s;
  (void)encode(command);
  return command;
}

Command velocity_command(std::uint8_t motor_id, float speed_rad_s,
                         float kd_nm_per_rad_s) {
  if (!std::isfinite(kd_nm_per_rad_s) || kd_nm_per_rad_s <= 0) {
    throw std::invalid_argument("velocity control requires positive finite kd");
  }
  Command command;
  command.motor_id = motor_id;
  command.enabled = true;
  command.timeout_protection = true;
  command.speed_rad_s = speed_rad_s;
  command.kd_nm_per_rad_s = kd_nm_per_rad_s;
  // Kp, position and feedforward torque stay zero: motor-side velocity feedback.
  (void)encode(command);
  return command;
}

Feedback decode(const std::array<std::uint8_t, 26>& frame,
                std::uint8_t expected_motor_id) {
  if (frame[0] != 0xFC || frame[1] != 0xEE)
    throw std::runtime_error("invalid IM6014 feedback header");
  if (read_u32(&frame[22]) != crc32_mpeg2_words(&frame[2], 20))
    throw std::runtime_error("IM6014 feedback CRC mismatch");
  const auto mode = frame[2];
  const auto id = static_cast<std::uint8_t>(mode & 0x0F);
  if (expected_motor_id != 0xFF && id != expected_motor_id)
    throw std::runtime_error("unexpected IM6014 motor ID");
  Feedback result;
  result.motor_id = id;
  result.enabled = (mode & 0x70) != 0;
  result.timeout_triggered = (mode & 0x80) != 0;
  result.driver_temp_c = static_cast<std::int8_t>(frame[3]);
  result.winding_temp_c = frame[4];
  result.voltage_v = frame[5] / 2.0F;
  result.torque_nm = (read_i16(&frame[6]) / kTorqueScale) * kGear;
  result.speed_rad_s = (read_i16(&frame[8]) / kSpeedScale) / kGear;
  result.position_rad = (static_cast<float>(read_i32(&frame[10])) / kPositionScale) / kGear;
  result.error = read_u32(&frame[14]);
  result.warning = static_cast<std::uint8_t>((read_u16(&frame[18]) >> 13) & 0x07);
  return result;
}

Connection::Connection(const std::string& device, int baud, std::uint8_t motor_id,
                       int timeout_ms)
    : motor_id_(motor_id), timeout_ms_(timeout_ms) {
  if (motor_id > 14 || timeout_ms <= 0) throw std::invalid_argument("invalid motor settings");
  fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "open motor serial");
  try { configure_baud(fd_, baud); }
  catch (...) { ::close(fd_); fd_ = -1; throw; }
}

Connection::~Connection() { if (fd_ >= 0) ::close(fd_); }

Feedback Connection::transact(const Command& command) {
  if (command.motor_id > 14) throw std::invalid_argument("invalid motor ID");
  const auto frame = encode(command);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms_);
  std::size_t sent = 0;
  while (sent < frame.size()) {
    const auto written = ::write(fd_, frame.data() + sent, frame.size() - sent);
    if (written > 0) { sent += static_cast<std::size_t>(written); continue; }
    if (written < 0 && errno == EINTR) continue;
    if (written < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
      throw std::system_error(errno, std::generic_category(), "write motor command");
    pollfd writable{fd_, POLLOUT, 0};
    if (::poll(&writable, 1, 2) < 0 && errno != EINTR)
      throw std::system_error(errno, std::generic_category(), "poll motor write");
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error("motor command write timeout");
  }
  while (std::chrono::steady_clock::now() < deadline) {
    pollfd pfd{fd_, POLLIN, 0};
    const int ready = ::poll(&pfd, 1, 5);
    if (ready < 0 && errno == EINTR) continue;
    if (ready < 0) throw std::system_error(errno, std::generic_category(), "poll motor");
    if (ready == 0) continue;
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) throw std::runtime_error("motor disconnected");
    std::array<std::uint8_t, 128> input{};
    const auto count = ::read(fd_, input.data(), input.size());
    if (count < 0 && (errno == EAGAIN || errno == EINTR)) continue;
    if (count < 0) throw std::system_error(errno, std::generic_category(), "read motor");
    for (ssize_t i = 0; i < count; ++i) {
      if (rx_size_ == rx_.size()) { std::move(rx_.begin() + 1, rx_.end(), rx_.begin()); --rx_size_; }
      rx_[rx_size_++] = input[static_cast<std::size_t>(i)];
    }
    for (std::size_t i = 0; i + 26 <= rx_size_;) {
      if (rx_[i] != 0xFC || rx_[i + 1] != 0xEE) { ++i; continue; }
      std::array<std::uint8_t, 26> candidate{};
      std::copy_n(rx_.begin() + static_cast<std::ptrdiff_t>(i), 26, candidate.begin());
      try {
        auto feedback = decode(candidate, command.motor_id);
        const auto consumed = i + 26;
        std::move(rx_.begin() + static_cast<std::ptrdiff_t>(consumed),
                  rx_.begin() + static_cast<std::ptrdiff_t>(rx_size_), rx_.begin());
        rx_size_ -= consumed;
        return feedback;
      } catch (const std::runtime_error&) { ++i; }
    }
    if (rx_size_ > 25) {
      const auto keep = std::min<std::size_t>(25, rx_size_);
      std::move(rx_.begin() + static_cast<std::ptrdiff_t>(rx_size_ - keep),
                rx_.begin() + static_cast<std::ptrdiff_t>(rx_size_), rx_.begin());
      rx_size_ = keep;
    }
  }
  throw std::runtime_error("IM6014 response timeout");
}

void Connection::stop() {
  stop(motor_id_);
}

void Connection::stop(std::uint8_t motor_id) {
  if (motor_id > 14) throw std::invalid_argument("invalid motor ID");
  Command command{};
  command.motor_id = motor_id;
  for (int attempt = 0; attempt < 3; ++attempt) {
    try { (void)transact(command); } catch (...) {}
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace nazarite::xz1::motor
