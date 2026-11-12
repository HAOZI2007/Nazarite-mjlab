#include "async_motor_system.hpp"
#include "position_ramp.hpp"
#include "velocity_ramp.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
namespace motor = nazarite::xz1::motor;
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }

struct Options {
  std::string device;
  std::string mode{"--probe"};
  double speed{1.0};
  double acceleration{0.5};
  float kp{2.0F};
  float kd{0.3F};
  double hold_seconds{3.0};
  double max_start_rad{100.0};
};

double parse_number(const std::string& text) {
  std::size_t used = 0;
  const double value = std::stod(text, &used);
  if (used != text.size() || !std::isfinite(value)) {
    throw std::invalid_argument("invalid numeric argument: " + text);
  }
  return value;
}

Options parse_options(int argc, char** argv) {
  Options options;
  options.device = argv[1];
  if (argc >= 3) options.mode = argv[2];
  const bool velocity_mode = options.mode == "--velocity" ||
                             options.mode == "--velocity-preview";
  if (velocity_mode) {
    options.speed = 10.0;
    options.acceleration = 2.0;
    options.kp = 0.0F;
    options.kd = 1.0F;
  }
  if (options.mode != "--probe" && options.mode != "--zero" &&
      options.mode != "--run" &&
      options.mode != "--preview" && !velocity_mode) {
    throw std::invalid_argument("unknown mode; use --help");
  }
  for (int i = 3; i < argc; i += 2) {
    const std::string key = argv[i];
    if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
    const double value = parse_number(argv[i + 1]);
    if (key == "--speed") options.speed = value;
    else if (key == "--accel") options.acceleration = value;
    else if (key == "--kp" && !velocity_mode) options.kp = static_cast<float>(value);
    else if (key == "--kd") options.kd = static_cast<float>(value);
    else if (key == "--hold" && velocity_mode) options.hold_seconds = value;
    else if (key == "--max-start-rad" && !velocity_mode) options.max_start_rad = value;
    else throw std::invalid_argument("unknown option: " + key);
  }
  if (!(options.max_start_rad > 0 && options.max_start_rad <= 1000)) {
    throw std::invalid_argument("max-start-rad must be in (0,1000] output rad");
  }
  if (!(options.speed >= 0.1 && options.speed <= (velocity_mode ? 10.0 : 5.0)) ||
      !(options.acceleration >= 0.1 && options.acceleration <= 2.0) ||
      (!velocity_mode && !(options.kp >= 0.1F && options.kp <= 10.0F)) ||
      !(options.kd >= 0.0F && options.kd <= 1.0F) ||
      (velocity_mode && (options.kd <= 0 || options.hold_seconds < 0.5 ||
                         options.hold_seconds > 10))) {
    if (velocity_mode) {
      throw std::invalid_argument(
          "velocity limits: speed in [0.1,10] rad/s, accel in [0.1,2] rad/s^2, "
          "kd in (0,1], hold in [0.5,10] seconds; kp is fixed at zero");
    }
    throw std::invalid_argument(
        "test limits: speed in [0.1,5] rad/s, accel in [0.1,2] rad/s^2, "
        "kp in [0.1,10], kd in [0,1]");
  }
  return options;
}

void check_state(const motor::AsyncSystemSnapshot& state) {
  if (interrupted) throw std::runtime_error("interrupted; stopping motors");
  if (state.faulted) throw std::runtime_error(state.last_error);
  if (!state.running) throw std::runtime_error("motor worker stopped");
  const auto now = Clock::now();
  for (std::size_t i = 0; i < 3; ++i) {
    if (!state.has_feedback[i] ||
        now - state.feedback_time[i] > std::chrono::milliseconds(100)) {
      throw std::runtime_error("missing/stale feedback for ID " +
                               std::to_string(i));
    }
    const auto& feedback = state.feedback[i];
    if (feedback.error || feedback.warning ||
        !std::isfinite(feedback.position_rad) ||
        !std::isfinite(feedback.speed_rad_s)) {
      throw std::runtime_error("motor fault/warning for ID " +
                               std::to_string(i));
    }
  }
}

void print_state(const motor::AsyncSystemSnapshot& state,
                 const std::array<float, 3>* commands = nullptr,
                 bool velocity_commands = false) {
  for (std::size_t i = 0; i < 3; ++i) {
    const auto& feedback = state.feedback[i];
    std::cout << " id=" << i << " q=" << feedback.position_rad
              << " v=" << feedback.speed_rad_s
              << " torque=" << feedback.torque_nm;
    if (commands) {
      std::cout << (velocity_commands ? " v_cmd=" : " cmd=") << (*commands)[i]
                << (velocity_commands ? " v_err=" : " err=")
                << (*commands)[i] - (velocity_commands ? feedback.speed_rad_s
                                                       : feedback.position_rad);
    }
  }
  std::cout << " cycles=" << state.completed_leg_cycles[0]
            << " missed=" << state.missed_deadlines[0]
            << " serial_ms=" << state.last_leg_work_ms[0]
            << " max_serial_ms=" << state.max_leg_work_ms[0] << std::endl;
}

void print_usb_latency(const std::string& device) {
  // Reading driver settings is independent of opening/commanding the motor bus.
  char resolved[4096];
  const char* canonical = ::realpath(device.c_str(), resolved);
  const std::string path = canonical ? canonical : device;
  const auto basename = path.substr(path.find_last_of('/') + 1);
  const std::string setting = "/sys/bus/usb-serial/devices/" + basename +
                              "/latency_timer";
  std::ifstream input(setting);
  int latency_ms = 0;
  if (input >> latency_ms) {
    std::cout << "usb_latency_timer_ms=" << latency_ms
              << " setting=" << setting << std::endl;
    if (latency_ms > 1) {
      std::cout << "USB short-frame latency may limit the 50 Hz serial sweep; "
                   "see README for the driver setting.\n";
    }
  }
}

void probe_timing(motor::AsyncMotorSystem& bus) {
  const auto initial = bus.snapshot();
  const auto start = Clock::now();
  const auto end = start + std::chrono::seconds(2);
  while (Clock::now() < end) {
    check_state(bus.snapshot());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  const auto final_state = bus.snapshot();
  check_state(final_state);
  const double seconds = std::chrono::duration<double>(Clock::now() - start)
                             .count();
  const double sweep_hz = (final_state.completed_leg_cycles[0] -
                           initial.completed_leg_cycles[0]) / seconds;
  std::cout << "STOP-only serial_sweep_hz=" << sweep_hz << " requested_hz=50";
  print_state(final_state);
}

motor::AsyncSystemSnapshot wait_for_feedback(motor::AsyncMotorSystem& bus) {
  const auto deadline = Clock::now() + std::chrono::seconds(2);
  while (Clock::now() < deadline) {
    const auto state = bus.snapshot();
    if (interrupted || state.faulted) {
      throw std::runtime_error(interrupted ? "interrupted" : state.last_error);
    }
    if (state.has_feedback[0] && state.has_feedback[1] &&
        state.has_feedback[2]) {
      check_state(state);
      return state;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  throw std::runtime_error("initial feedback timed out");
}

void move_to(motor::AsyncMotorSystem& bus, float target,
             const Options& options) {
  auto state = bus.snapshot();
  check_state(state);
  std::array<float, 3> initial{};
  for (std::size_t i = 0; i < 3; ++i) {
    initial[i] = state.feedback[i].position_rad;
    if (std::abs(initial[i]) > options.max_start_rad) {
      throw std::runtime_error(
          "ID " + std::to_string(i) + " initial absolute position " +
          std::to_string(initial[i]) + " rad exceeds --max-start-rad " +
          std::to_string(options.max_start_rad) +
          "; verify the encoder reference and free multi-turn travel, then "
          "explicitly increase --max-start-rad if needed");
    }
  }
  const motor::PositionRamp ramp(initial, target, options.speed,
                                  options.acceleration);
  std::cout << "stage target=" << target << " output_rad ramp_seconds="
            << ramp.duration() << " peak_target_speed=" << ramp.peak_speed()
            << std::endl;

  motor::MotorCommands commands{};
  for (std::size_t i = 0; i < commands.size(); ++i) {
    commands[i].motor_id = motor::kActionIndexMotorIds[i];
  }
  const auto start = Clock::now();
  const auto stage_start_cycles = state.completed_leg_cycles[0];
  auto next = start;
  auto next_print = start;
  auto settled_since = Clock::time_point{};
  auto previous_command = initial;
  double max_submit_us = 0;

  while (true) {
    const auto now = Clock::now();
    const double elapsed = std::chrono::duration<double>(now - start).count();
    state = bus.snapshot();
    check_state(state);
    bool at_target = true;
    for (std::size_t i = 0; i < 3; ++i) {
      const auto& feedback = state.feedback[i];
      if (std::abs(feedback.position_rad - previous_command[i]) > 0.75F ||
          std::abs(feedback.speed_rad_s) > 2 * options.speed + 0.5) {
        throw std::runtime_error("tracking/speed limit exceeded for ID " +
                                 std::to_string(i));
      }
      at_target = at_target &&
                  std::abs(feedback.position_rad - target) <= 0.05F &&
                  std::abs(feedback.speed_rad_s) <= 0.1F;
    }

    const auto positions = ramp.sample(elapsed);
    for (std::size_t i = 0; i < 3; ++i) {
      commands[i] = motor::position_command(
          static_cast<std::uint8_t>(i), positions[i], options.kp, options.kd);
    }
    const auto submit_start = Clock::now();
    bus.submit_commands(commands);
    max_submit_us = std::max(max_submit_us,
        std::chrono::duration<double, std::micro>(Clock::now() - submit_start)
            .count());
    previous_command = positions;

    if (now >= next_print) {
      const double sweep_hz = (state.completed_leg_cycles[0] -
                               stage_start_cycles) / std::max(elapsed, 1e-6);
      std::cout << "t=" << elapsed << " target=" << target
                << " serial_sweep_hz=" << sweep_hz;
      print_state(state, &positions);
      next_print = now + std::chrono::milliseconds(500);
    }
    if (elapsed >= ramp.duration() && at_target) {
      if (settled_since == Clock::time_point{}) settled_since = now;
      if (now - settled_since >= std::chrono::seconds(1)) {
        std::cout << "reached=" << target
                  << " held_seconds=1 max_submit_us=" << max_submit_us
                  << std::endl;
        return;
      }
    } else {
      settled_since = Clock::time_point{};
    }
    if (elapsed > ramp.duration() + 15.0) {
      std::cerr << "settle failure: position tolerance=0.05 rad, "
                   "speed tolerance=0.1 rad/s; final feedback:";
      print_state(state, &positions);
      throw std::runtime_error("motors did not settle at stage target; "
                               "use --zero to tune homing first");
    }
    next += std::chrono::milliseconds(20);
    if (next < Clock::now()) next = Clock::now();
    std::this_thread::sleep_until(next);
  }
}
void run_velocity_stage(motor::AsyncMotorSystem& bus, float target,
                         double hold_seconds, const Options& options,
                         std::array<float, 3>& reference) {
  auto state = bus.snapshot();
  check_state(state);
  // Carry the previous reference across stages. Feedback may have steady-state
  // error; using it as the next starting target would cause a command step.
  const auto initial = reference;
  const motor::VelocityRamp ramp(initial, target, options.acceleration);
  std::cout << "velocity_stage target_rad_s=" << target
            << " ramp_seconds=" << ramp.duration()
            << " hold_seconds=" << hold_seconds << std::endl;
  motor::MotorCommands commands{};
  for (std::size_t i = 0; i < commands.size(); ++i) {
    commands[i].motor_id = motor::kActionIndexMotorIds[i];
  }
  const auto start = Clock::now();
  const auto start_cycles = state.completed_leg_cycles[0];
  auto next = start;
  auto next_print = start;
  auto settled_since = Clock::time_point{};
  auto tracking_failure_since = Clock::time_point{};
  auto previous_command = initial;
  double max_submit_us = 0;
  const double tolerance = target == 0 ? 0.1 : std::max(0.5, 0.1 * double(target));

  while (true) {
    const auto now = Clock::now();
    const double elapsed = std::chrono::duration<double>(now - start).count();
    state = bus.snapshot();
    check_state(state);
    bool at_target = true;
    bool tracking_bad = false;
    for (std::size_t i = 0; i < 3; ++i) {
      const double speed = state.feedback[i].speed_rad_s;
      if (std::abs(speed) > options.speed + 2.0) {
        throw std::runtime_error("velocity overspeed for ID " + std::to_string(i));
      }
      tracking_bad = tracking_bad || std::abs(speed - previous_command[i]) > 3.0;
      at_target = at_target && std::abs(speed - target) <= tolerance;
    }
    if (tracking_bad) {
      if (tracking_failure_since == Clock::time_point{}) tracking_failure_since = now;
      if (now - tracking_failure_since > std::chrono::seconds(1)) {
        throw std::runtime_error("velocity tracking error > 3 rad/s for 1 second");
      }
    } else {
      tracking_failure_since = Clock::time_point{};
    }

    const auto speeds = ramp.sample(elapsed);
    for (std::size_t i = 0; i < 3; ++i) {
      commands[i] = motor::velocity_command(static_cast<std::uint8_t>(i),
                                            speeds[i], options.kd);
    }
    const auto submit_start = Clock::now();
    bus.submit_commands(commands);
    max_submit_us = std::max(max_submit_us,
        std::chrono::duration<double, std::micro>(Clock::now() - submit_start)
            .count());
    previous_command = speeds;
    reference = speeds;
    if (now >= next_print) {
      std::cout << "t=" << elapsed << " target_rad_s=" << target
                << " serial_sweep_hz="
                << (state.completed_leg_cycles[0] - start_cycles) /
                       std::max(elapsed, 1e-6);
      print_state(state, &speeds, true);
      next_print = now + std::chrono::milliseconds(500);
    }
    if (elapsed >= ramp.duration() && at_target) {
      if (settled_since == Clock::time_point{}) settled_since = now;
      if (std::chrono::duration<double>(now - settled_since).count() >= hold_seconds) {
        std::cout << "reached_velocity=" << target
                  << " tolerance_rad_s=" << tolerance
                  << " held_seconds=" << hold_seconds
                  << " max_submit_us=" << max_submit_us << std::endl;
        return;
      }
    } else {
      settled_since = Clock::time_point{};
    }
    if (elapsed > ramp.duration() + hold_seconds + 15.0) {
      throw std::runtime_error("motors did not settle at target velocity");
    }
    next += std::chrono::milliseconds(20);
    if (next < Clock::now()) next = Clock::now();
    std::this_thread::sleep_until(next);
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]) == "--help") {
    std::cout << "Usage: " << argv[0]
              << " DEVICE [--probe|--zero|--preview|--run]"
                 " [--speed 1] [--accel 0.5] [--kp 2] [--kd 0.3]"
                 " [--max-start-rad 100]\n"
              << "Channel 1 only, IDs 0/1/2, 6000000 baud, 50 Hz.\n"
              << "--probe (default): STOP and feedback only, measure UART for 2 s.\n"
              << "--zero: move to absolute 0 rad only, then STOP.\n"
              << "--preview: print the plan without opening any serial port.\n"
              << "--run: move unloaded output shafts to absolute 0 -> 50 -> 0 rad.\n";
    std::cout << "Limits: speed 0.1-5 rad/s, accel 0.1-2 rad/s^2, "
                 "kp 0.1-10 Nm/rad, "
                 "kd 0-1 Nm/(rad/s).\n";
    std::cout << "--max-start-rad: allowed absolute starting output angle, "
                 "default 100, range (0,1000] rad; position modes only. "
                 "Does not reset the encoder.\n";
    std::cout << "Velocity test: DEVICE --velocity [--speed 10] [--accel 2] "
                 "[--kd 1] [--hold 3]\n"
              << "--velocity: output speed 0 -> requested speed -> 0 rad/s, kp=0.\n"
              << "--velocity-preview: print speed plan without serial access.\n"
              << "Velocity limits: speed 0.1-10, accel 0.1-2, kd (0,1], "
                 "hold 0.5-10 seconds.\n";
    return argc < 2 ? 2 : 0;
  }
  try {
    const auto options = parse_options(argc, argv);
    std::cout << std::fixed << std::setprecision(4)
              << "device=" << options.device
              << " baud=6000000 ids=0,1,2 hz=50 speed=" << options.speed
              << " accel=" << options.acceleration
              << " kp=" << options.kp << " kd=" << options.kd;
    if (options.mode != "--velocity" && options.mode != "--velocity-preview") {
      std::cout << " max_start_rad=" << options.max_start_rad;
    }
    std::cout << std::endl;
    if (options.mode == "--velocity-preview") {
      std::array<float, 3> speeds{};
      for (float target : {0.0F, static_cast<float>(options.speed), 0.0F}) {
        const motor::VelocityRamp ramp(speeds, target, options.acceleration);
        std::cout << "velocity_target_rad_s=" << target
                  << " ramp_seconds=" << ramp.duration()
                  << " hold_seconds=" << (target == 0 ? 1.0 : options.hold_seconds)
                  << '\n';
        speeds = ramp.sample(ramp.duration());
      }
      return 0;
    }
    if (options.mode == "--preview") {
      std::array<float, 3> positions{};
      for (float target : {0.0F, 50.0F, 0.0F}) {
        const motor::PositionRamp ramp(positions, target, options.speed,
                                        options.acceleration);
        std::cout << "target=" << target
                  << " ramp_seconds=" << ramp.duration()
                  << " peak_target_speed=" << ramp.peak_speed()
                  << " hold_seconds=1 (assumes initial 0 rad)\n";
        positions = ramp.sample(ramp.duration());
      }
      return 0;
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    print_usb_latency(options.device);
    motor::LegDevices devices{options.device, "", "", ""};
    motor::AsyncMotorSystem bus(devices, 6000000,
                                motor::kActionIndexMotorIds, 20);
    try {
      const auto initial = wait_for_feedback(bus);
      std::cout << "initial STOP feedback:";
      print_state(initial);
      if (options.mode == "--velocity") {
        std::array<float, 3> reference{};
        for (std::size_t i = 0; i < 3; ++i) {
          reference[i] = initial.feedback[i].speed_rad_s;
        }
        for (float target : {0.0F, static_cast<float>(options.speed), 0.0F}) {
          run_velocity_stage(bus, target,
                             target == 0 ? 1.0 : options.hold_seconds, options,
                             reference);
        }
      } else if (options.mode == "--run") {
        for (float target : {0.0F, 50.0F, 0.0F}) {
          move_to(bus, target, options);
        }
      } else if (options.mode == "--zero") {
        move_to(bus, 0.0F, options);
      } else {
        probe_timing(bus);
      }
      bus.stop();
      const auto final_state = bus.snapshot();
      if (final_state.faulted) throw std::runtime_error(final_state.last_error);
      std::cout << "STOP completed for IDs 0,1,2\n";
      return 0;
    } catch (...) {
      bus.stop();
      throw;
    }
  } catch (const std::exception& error) {
    std::cerr << "motor sequence failed: " << error.what() << '\n';
    return 1;
  }
}
