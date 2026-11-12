#include "async_motor_system.hpp"

#include <exception>
#include <stdexcept>

namespace nazarite::xz1::motor {

AsyncMotorSystem::AsyncMotorSystem(
    const LegDevices& devices, int baud, MotorIdMap motor_ids,
    int response_timeout_ms,
    std::chrono::milliseconds control_period,
    std::chrono::milliseconds stale_target_after)
    : motor_ids_(motor_ids),
      control_period_(control_period),
      stale_target_after_(stale_target_after) {
  if (control_period_ <= std::chrono::milliseconds::zero() ||
      stale_target_after_ <= std::chrono::milliseconds::zero()) {
    throw std::invalid_argument("motor system periods must be positive");
  }

  MotorCommands initial_commands{};
  for (std::size_t action = 0; action < kXz1MotorCount; ++action) {
    initial_commands[action].motor_id = motor_ids_[action];
  }
  validate_motor_commands(initial_commands, motor_ids_);

  for (std::size_t leg = 0; leg < kXz1LegCount; ++leg) {
    if (devices[leg].empty()) continue;
    for (std::size_t other = 0; other < leg; ++other) {
      if (devices[leg] == devices[other]) {
        throw std::invalid_argument("active legs must use separate serial devices");
      }
    }
    const auto first_action = leg * kMotorsPerLeg;
    connections_[leg] = std::make_unique<Connection>(
        devices[leg], baud, motor_ids_[first_action], response_timeout_ms);
    state_.active_legs[leg] = true;
    ++workers_remaining_;
  }
  if (workers_remaining_ == 0) {
    throw std::invalid_argument("at least one motor UART must be configured");
  }

  state_.running = true;
  try {
    for (std::size_t leg = 0; leg < kXz1LegCount; ++leg) {
      if (connections_[leg]) {
        workers_[leg] = std::thread(&AsyncMotorSystem::run_leg, this, leg);
      }
    }
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) worker.join();
    }
    throw;
  }
}

AsyncMotorSystem::~AsyncMotorSystem() { stop(); }

void validate_motor_commands(const MotorCommands& commands,
                             const MotorIdMap& motor_ids) {
  for (std::size_t index = 0; index < commands.size(); ++index) {
    const auto& command = commands[index];
    if (motor_ids[index] > 14 || command.motor_id != motor_ids[index]) {
      throw std::invalid_argument(
          "each action command must use its configured motor ID in [0, 14]");
    }
    if (command.enabled && !command.timeout_protection) {
      throw std::invalid_argument(
          "enabled motor commands require timeout protection");
    }
    for (std::size_t other = index - (index % kMotorsPerLeg);
         other < index; ++other) {
      if (motor_ids[other] == motor_ids[index]) {
        throw std::invalid_argument(
            "motor IDs must be unique within each leg's RS-485 bus");
      }
    }
    (void)encode(command);
  }
}

void AsyncMotorSystem::submit_commands(const MotorCommands& commands) {
  validate_motor_commands(commands, motor_ids_);
  for (std::size_t action = 0; action < commands.size(); ++action) {
    if (commands[action].enabled && !connections_[action / kMotorsPerLeg]) {
      throw std::invalid_argument("cannot enable a motor on an inactive UART");
    }
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!state_.running || stopping_ || state_.faulted) {
      throw std::runtime_error("motor system is stopped or faulted");
    }
    latest_commands_ = commands;
    has_target_set_ = true;
    target_time_ = std::chrono::steady_clock::now();
  }
  wake_.notify_all();
}

AsyncSystemSnapshot AsyncMotorSystem::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

void AsyncMotorSystem::stop() noexcept {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_all();
  for (auto& worker : workers_) {
    if (worker.joinable()) worker.join();
  }
}

void AsyncMotorSystem::record_fault(const std::string& message) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  state_.faulted = true;
  if (state_.last_error.empty()) state_.last_error = message;
  stopping_ = true;
  wake_.notify_all();
}

void AsyncMotorSystem::send_stop_leg(std::size_t leg) noexcept {
  const auto first_action = leg * kMotorsPerLeg;
  for (std::size_t offset = 0; offset < kMotorsPerLeg; ++offset) {
    const auto action = first_action + offset;
    Command command{};
    command.motor_id = motor_ids_[action];
    try {
      const auto feedback = connections_[leg]->transact(command);
      std::lock_guard<std::mutex> lock(mutex_);
      state_.feedback[action] = feedback;
      state_.has_feedback[action] = true;
      state_.feedback_time[action] = std::chrono::steady_clock::now();
    } catch (const std::exception& error) {
      record_fault("STOP on UART " + std::to_string(leg + 1) +
                   ", motor ID " + std::to_string(command.motor_id) +
                   " failed: " + error.what());
    } catch (...) {
      record_fault("STOP on UART " + std::to_string(leg + 1) +
                   " failed with an unknown error");
    }
  }
}

void AsyncMotorSystem::run_leg(std::size_t leg) noexcept {
  auto next_cycle = std::chrono::steady_clock::now();
  const auto first_action = leg * kMotorsPerLeg;
  try {
    while (true) {
      MotorCommands commands{};
      bool use_target = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait_until(lock, next_cycle, [this] { return stopping_; });
        if (stopping_) break;
        if (has_target_set_ &&
            std::chrono::steady_clock::now() - target_time_ > stale_target_after_) {
          throw std::runtime_error("motor command update expired");
        }
        use_target = has_target_set_;
        if (use_target) commands = latest_commands_;
      }

      const auto work_start = std::chrono::steady_clock::now();
      bool abort_cycle = false;
      for (std::size_t offset = 0; offset < kMotorsPerLeg; ++offset) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (stopping_) {
            abort_cycle = true;
            break;
          }
        }
        const auto action = first_action + offset;
        Command command{};
        command.motor_id = motor_ids_[action];
        if (use_target) command = commands[action];

        const auto feedback = connections_[leg]->transact(command);
        {
          std::lock_guard<std::mutex> lock(mutex_);
          state_.feedback[action] = feedback;
          state_.has_feedback[action] = true;
          state_.feedback_time[action] = std::chrono::steady_clock::now();
        }
        if (feedback.error != 0 ||
            (use_target && feedback.timeout_triggered)) {
          throw std::runtime_error("motor ID " +
                                   std::to_string(command.motor_id) +
                                   " reported error/timeout");
        }
      }
      if (abort_cycle) break;

      {
        std::lock_guard<std::mutex> lock(mutex_);
        ++state_.completed_leg_cycles[leg];
        const double work_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - work_start)
                                   .count();
        state_.last_leg_work_ms[leg] = work_ms;
        if (work_ms > state_.max_leg_work_ms[leg]) {
          state_.max_leg_work_ms[leg] = work_ms;
        }
      }
      next_cycle += control_period_;
      const auto now = std::chrono::steady_clock::now();
      if (next_cycle <= now) {
        const auto missed = static_cast<std::uint64_t>(
                                (now - next_cycle) / control_period_) +
                            1;
        std::lock_guard<std::mutex> lock(mutex_);
        state_.missed_deadlines[leg] += missed;
        // A slow serial sweep already consumed the period. Start the next
        // sweep immediately rather than adding another full-period delay.
        next_cycle = now;
      }
    }
  } catch (const std::exception& error) {
    record_fault("UART " + std::to_string(leg + 1) + ": " + error.what());
  } catch (...) {
    record_fault("UART " + std::to_string(leg + 1) +
                 ": unknown motor worker error");
  }

  send_stop_leg(leg);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (workers_remaining_ > 0) --workers_remaining_;
    if (workers_remaining_ == 0) state_.running = false;
  }
}

}  // namespace nazarite::xz1::motor
