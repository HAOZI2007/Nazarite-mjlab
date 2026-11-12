#include "motor_control/async_motor_system.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace motor = nazarite::xz1::motor;
using Clock = std::chrono::steady_clock;

void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    motor::AsyncMotorSystem bus({argv[1], "", "", ""}, 6000000);
    const auto deadline = Clock::now() + std::chrono::seconds(2);
    auto state = bus.snapshot();
    while ((!state.has_feedback[0] || !state.has_feedback[1] ||
            !state.has_feedback[2]) && !state.faulted && Clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      state = bus.snapshot();
    }
    check(!state.faulted, state.last_error.c_str());
    check(state.has_feedback[0] && state.has_feedback[1] && state.has_feedback[2],
          "missing initial feedback");
    check(state.active_legs == std::array<bool, 4>{true, false, false, false},
          "inactive legs were opened");

    motor::MotorCommands commands{};
    for (std::size_t i = 0; i < commands.size(); ++i) {
      commands[i].motor_id = motor::kActionIndexMotorIds[i];
    }
    commands[3] = motor::position_command(3, 0, 1, 0.2F);
    bool rejected = false;
    try { bus.submit_commands(commands); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "enabled command to an inactive leg was accepted");
    commands[3] = motor::Command{};
    commands[3].motor_id = 3;
    for (std::size_t i = 0; i < 3; ++i) {
      commands[i] = motor::position_command(static_cast<std::uint8_t>(i),
                                           0.1F, 1, 0.2F);
    }
    bus.submit_commands(commands);
    // Deliberately stop publishing, while serial feedback remains healthy.
    const auto stale_deadline = Clock::now() + std::chrono::seconds(2);
    do {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      state = bus.snapshot();
    } while (!state.faulted && Clock::now() < stale_deadline);
    check(state.faulted && state.last_error.find("update expired") !=
                              std::string::npos,
          "expired commands must latch a fault");
    rejected = false;
    try { bus.submit_commands(commands); }
    catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "faulted bus accepted a new command");
    bus.stop();
    state = bus.snapshot();
    check(!state.running, "worker did not stop");
    for (std::size_t i = 3; i < state.has_feedback.size(); ++i) {
      check(!state.has_feedback[i], "inactive leg has feedback");
    }
    std::cout << "single-UART stale-command fault test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
