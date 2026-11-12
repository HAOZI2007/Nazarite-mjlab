#pragma once

#include "im6014.hpp"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace nazarite::xz1::motor {

constexpr std::size_t kXz1MotorCount = 12;
constexpr std::size_t kXz1LegCount = 4;
constexpr std::size_t kMotorsPerLeg = 3;
using MotorCommands = std::array<Command, kXz1MotorCount>;
using LegDevices = std::array<std::string, kXz1LegCount>;
using MotorIdMap = std::array<std::uint8_t, kXz1MotorCount>;
inline constexpr MotorIdMap kActionIndexMotorIds{
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
inline constexpr MotorIdMap kPerLegMotorIds{
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};

// Commands are indexed by HIM action. IDs may repeat across the four isolated
// UARTs, but must be unique within each three-motor leg bus.
void validate_motor_commands(const MotorCommands& commands,
                             const MotorIdMap& motor_ids);

struct AsyncSystemSnapshot {
  std::array<bool, kXz1LegCount> active_legs{};
  std::array<Feedback, kXz1MotorCount> feedback{};
  std::array<bool, kXz1MotorCount> has_feedback{};
  std::array<std::chrono::steady_clock::time_point, kXz1MotorCount>
      feedback_time{};
  std::array<std::uint64_t, kXz1LegCount> completed_leg_cycles{};
  std::array<std::uint64_t, kXz1LegCount> missed_deadlines{};
  // Serial work only, excluding the worker's wait for the next cycle.
  std::array<double, kXz1LegCount> last_leg_work_ms{};
  std::array<double, kXz1LegCount> max_leg_work_ms{};
  bool running{false};
  bool faulted{false};
  std::string last_error;
};

// Owns up to four independent half-duplex RS-485 serial ports, one per leg.
// An empty device path disables that leg; at least one leg must be active.
// Each worker addresses only the three action indices belonging to its leg.
class AsyncMotorSystem {
 public:
  AsyncMotorSystem(const LegDevices& devices, int baud,
                   MotorIdMap motor_ids = kActionIndexMotorIds,
                   int response_timeout_ms = 10,
                   std::chrono::milliseconds control_period =
                       std::chrono::milliseconds(20),
                   std::chrono::milliseconds stale_target_after =
                       std::chrono::milliseconds(100));
  ~AsyncMotorSystem();
  AsyncMotorSystem(const AsyncMotorSystem&) = delete;
  AsyncMotorSystem& operator=(const AsyncMotorSystem&) = delete;

  // Array order is FL, FR, RR, RL; each leg is hip, thigh, shank.
  // The newest complete command array replaces any older pending array.
  void submit_commands(const MotorCommands& commands);
  AsyncSystemSnapshot snapshot() const;
  void stop() noexcept;

 private:
  void run_leg(std::size_t leg) noexcept;
  void send_stop_leg(std::size_t leg) noexcept;
  void record_fault(const std::string& message) noexcept;

  const MotorIdMap motor_ids_;
  std::array<std::unique_ptr<Connection>, kXz1LegCount> connections_{};
  const std::chrono::milliseconds control_period_;
  const std::chrono::milliseconds stale_target_after_;
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  MotorCommands latest_commands_{};
  bool has_target_set_{false};
  std::chrono::steady_clock::time_point target_time_{};
  bool stopping_{false};
  std::size_t workers_remaining_{0};
  AsyncSystemSnapshot state_{};
  std::array<std::thread, kXz1LegCount> workers_{};
};

}  // namespace nazarite::xz1::motor
