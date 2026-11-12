#include "async_motor_system.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

int main(int argc, char** argv) {
  if (argc < 6 || argc > 8) {
    std::cerr << "Usage: " << argv[0]
              << " UART1 UART2 UART3 UART4 BAUD [action|per-leg] [SECONDS]\n"
              << "Sends STOP commands using action-index IDs 0-11 by default.\n";
    return 2;
  }
  try {
    nazarite::xz1::motor::LegDevices devices{
        argv[1], argv[2], argv[3], argv[4]};
    const int baud = std::stoi(argv[5]);
    auto motor_ids = nazarite::xz1::motor::kActionIndexMotorIds;
    double seconds = 2.0;
    if (argc >= 7) {
      const std::string layout = argv[6];
      if (layout == "action") {
        motor_ids = nazarite::xz1::motor::kActionIndexMotorIds;
      } else if (layout == "per-leg") {
        motor_ids = nazarite::xz1::motor::kPerLegMotorIds;
      } else {
        seconds = std::stod(layout);
      }
    }
    if (argc == 8) seconds = std::stod(argv[7]);
    if (!(seconds > 0.0 && seconds <= 30.0)) {
      throw std::invalid_argument("SECONDS must be in (0, 30]");
    }

    nazarite::xz1::motor::AsyncMotorSystem system(devices, baud, motor_ids);
    // Workers send STOP until the first motion target is submitted.

    const auto end = std::chrono::steady_clock::now() +
                     std::chrono::duration<double>(seconds);
    auto next_print = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() < end) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= next_print) {
        const auto state = system.snapshot();
        for (std::size_t action = 0; action < state.feedback.size(); ++action) {
          if (!state.has_feedback[action]) {
            std::cout << "action=" << action << " no feedback yet\n";
            continue;
          }
          const auto& feedback = state.feedback[action];
          std::cout << "uart=" << action / 3 + 1
                    << " action=" << action << " id="
                    << static_cast<unsigned>(feedback.motor_id)
                    << " pos=" << feedback.position_rad
                    << " rad speed=" << feedback.speed_rad_s
                    << " rad/s error=0x" << std::hex << feedback.error
                    << std::dec << '\n';
        }
        if (state.faulted) {
          throw std::runtime_error("motor system fault: " + state.last_error);
        }
        next_print = now + std::chrono::milliseconds(200);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    system.stop();
    const auto state = system.snapshot();
    if (state.faulted) {
      throw std::runtime_error("motor system fault: " + state.last_error);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "motor system probe error: " << error.what() << '\n';
    return 1;
  }
}
