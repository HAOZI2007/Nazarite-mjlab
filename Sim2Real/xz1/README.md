# XZ1 C++ deployment modules

Linux/C++17 reusable modules for the first XZ1-HIM deployment prototype.
Hardware adapters are separated from the HIM observation and action contract.

## Build and host-side tests

```bash
cmake -S Sim2Real/xz1 -B Sim2Real/xz1/build
cmake --build Sim2Real/xz1/build -j
ctest --test-dir Sim2Real/xz1/build --output-on-failure
```

ONNX Runtime C++ support is enabled when `onnxruntime_cxx_api.h` and the
`onnxruntime` library are discoverable. Set `ONNXRUNTIME_ROOT` to the C++ SDK
directory when needed. The Python wheel usually includes the shared library
but does not include C++ headers; the C++ SDK is separate from `onnxruntime`.
With the SDK installed, run a 30-second zero-input 50 Hz benchmark as
`Sim2Real/xz1/build/xz1_onnx_test /path/to/policy.onnx 100 1500 50`.
Arguments after the model are warmup count, measured iterations and requested
frequency; set frequency to `0` for continuous inference. This measures the
policy process only, without sensors or motor communication. Run
`Sim2Real/xz1/build/xz1_him_config` to print the compiled HIM parameters.

## Module responsibilities

- `imu/`: DM-BMI088 USB serial protocol parser. Reads frame `0x02` as xyz gyro
  and `0x04` as wxyz quaternion, validates CRC16, and converts gyro units to
  rad/s. Default gyro input is degrees/s. Confirm output frame configuration,
  quaternion order/direction and sensor axis convention against the exact IMU
  setup before using it for control. `transform_to_body()` applies the configured
  sensor-to-body mounting quaternion to gyro and orientation. The reference
  program documents Euler frames but not these calibration details.
- `motor_control/`: IM6014/N6014 RS-485 protocol. Includes command/feedback
  CRC and output-side unit conversion, STOP/probe and bounded position-motion
  test. `--move` actuates hardware; run unloaded with the power switch reachable.
  The standalone move test limits the relative step to 0.2 rad, Kp to 5,
  Kd to 1, and duration to 5 seconds.
  The full HIM PD gains and output scaling have not been validated for this
  motor and must not be applied to a robot until the actuator model is confirmed.
  `position_command()` maps one HIM position/Kp/Kd triplet to a protected
  per-motor command; one serial `Connection` can address multiple IDs on a bus
  and `stop(id)` sends a stop command to a selected motor. `AsyncMotorSystem`
  opens up to four serial ports and runs one worker per active leg; each worker serially
  communicates with that leg's three motors. The command array stays in HIM
  action order, while a configurable ID map supports either IDs 0-11 or the
  module manual's per-channel IDs 0-2. The control thread submits targets and
  reads feedback without waiting for serial replies. Verify cycle timing and
  motor IDs on real hardware.
- `remote_controler/`: nonblocking Linux evdev sticks; left Y maps to forward
  velocity, left X to lateral velocity, right X to yaw rate, BTN_START requests
  reset. Axis max velocities are configuration and must be selected for the
  robot. Pass an event path or allow auto-selection when exactly one pad exists.
- `him/`: exact 47 x 6 current-first XZ1-HIM input history, gravity projection,
  phase and action-offset mapping. Optional ONNX Runtime C++ inference checks
  tensor dimensions and exported metadata.

Joint order is FL, FR, RR, RL; each leg is hip, thigh, shank. Internal units are
radians and radians/second. IMU data must be expressed in the trained robot body
frame with a body-to-world wxyz quaternion. Joint driver IDs, signs and zero
offsets are hardware-specific and intentionally not guessed here.

## Programs

```bash
Sim2Real/xz1/build/xz1_deploy_test
Sim2Real/xz1/build/xz1_him_config
Sim2Real/xz1/build/xz1_imu_reader /dev/serial/by-id/DEVICE [gyro-scale]
Sim2Real/xz1/build/xz1_motor_test /dev/ttyUSB0 6000000 1 --probe
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --probe
Sim2Real/xz1/build/xz1_motor_system_probe \
  /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyUSB2 /dev/ttyUSB3 6000000 action 2
Sim2Real/xz1/build/xz1_gamepad_test --list
Sim2Real/xz1/build/xz1_gamepad_test /dev/input/eventX
```

Use `--move DELTA_RAD KP KD SECONDS` only for the single-motor test, with the
motor unloaded. A 12-motor asynchronous RS-485 transport is now available, but
it is not yet connected to the HIM policy loop. Per-joint directions, encoder
zeros, actuator limits and emergency-stop wiring still need hardware
calibration and verification.

### Asynchronous four-channel motor control

`AsyncMotorSystem` keeps blocking RS-485 transactions off the policy/control
thread. Pass Linux serial device paths in leg order; an empty path disables
that leg, and at least one leg must be active. Submit a complete,
fresh 12-command array each control iteration in the action order described
above; each worker keeps only the newest targets for its three motors and
exposes recent feedback in `snapshot()`:

```cpp
nazarite::xz1::motor::LegDevices ports{
    "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyUSB3"};
nazarite::xz1::motor::AsyncMotorSystem bus(
    ports, 6000000, nazarite::xz1::motor::kActionIndexMotorIds);
nazarite::xz1::motor::MotorCommands commands{};
for (std::size_t i = 0; i < commands.size(); ++i) {
  commands[i] = nazarite::xz1::motor::position_command(
      static_cast<std::uint8_t>(i), calibrated_target[i], kp[i], kd[i]);
}
bus.submit_commands(commands);  // returns without waiting for 12 replies
const auto state = bus.snapshot();
```

`calibrated_target` must already include each joint's URDF direction and zero
offset. The module manual shows IDs 0, 1, 2 on each independent leg channel; use
`kPerLegMotorIds` if configured that way. The example above uses globally unique
IDs 0-11. Enabled commands require the motor's 1-second timeout protection. If
no fresh command arrives for 100 ms or a communication/motor fault occurs, all
four workers attempt STOP on their three motors. `stop()` waits for their STOP
sweeps; keep a physical power disconnect available.
Expired commands latch a fault; new commands cannot restart a faulted instance.
`xz1_motor_system_probe` sends only STOP and displays feedback; it is a
communications check, not a motion test.

The user has already verified this 4-channel converter with the existing
IM6014 control program. The C++ interface defaults to 6 Mbps, matching the
reference program, and also accepts 4 Mbps. Reuse the baud rate and UART mapping
that were successful in that test; the converter manual's nominal rate is not a
deployment blocker for this verified setup. Motor IDs default to the global
HIM action indices 0-11; use `per-leg` only if the motors are configured with
repeated IDs 0-2 on their separate UARTs.

### Single-channel three-motor sequence

`xz1_motor_sequence` uses the same async transport with only channel 1 active,
IDs 0/1/2, 6 Mbps, and a 50 Hz command publisher. It requires three unloaded
motors with shafts free to turn multiple revolutions. Angles are absolute
gearbox output radians in the existing encoder reference; it does not reset
the encoder or calibrate mechanical zero.

```bash
# STOP feedback only; no motion. Confirm the serial path belongs to channel 1.
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --probe
# No serial access; show the plan (assumes initial zero).
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --preview
# Home only when tuning; never enters the 50-rad stage.
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --zero --speed 0.5 --kp 5 --kd 0.5
# Actuate all three: measured positions -> 0 -> 50 -> 0 output radians.
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --run
```

Default peak target speed is 1 rad/s, target acceleration at most 0.5 rad/s²,
Kp=2 and Kd=0.3, with no velocity or torque feedforward. Each 50-rad leg takes
about 75 seconds. Each stage waits until all three positions are within 0.05 rad
and speeds within 0.1 rad/s for one second. `--speed`, `--kp` and `--kd` can be
set explicitly, along with `--accel`; see `--help` and the Chinese guide for
details and limits. Speed accepts 0.1-5 rad/s; acceleration accepts 0.1-2 rad/s².

Position modes reject initial absolute output angles beyond 100 rad by default.
Velocity tests accumulate multi-turn angles. To return from about 102 rad with
unloaded, freely rotating shafts, explicitly raise the starting-angle limit:

```bash
# From the Sim2Real directory, rebuild to enable the new option.
cmake --build xz1/build -j2
./xz1/build/xz1_motor_sequence /dev/ttyUSB0 --zero \
  --max-start-rad 200 --speed 5 --accel 2 --kp 10 --kd 0.5
```

`--max-start-rad` accepts (0,1000] output radians and applies only to position
modes. The ramp starts at measured positions and physically returns to the
existing absolute zero, without resetting the encoder or wrapping angles.
From 102 rad, the ramp takes about 31 seconds at speed 5, or 306 seconds at
speed 0.5, plus settling. Existing tracking and STOP checks remain active.

For a 5 rad/s peak target speed on each 50-rad leg:

```bash
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --run \
  --speed 5 --accel 2 --kp 10 --kd 0.5
```

Each 50-rad leg takes about 15 seconds, with smooth acceleration and deceleration.
The complete sequence takes about 34 seconds when starting near zero. Setting
only `--speed 5` keeps the default acceleration limit, which yields a 24.5-second
leg and a peak target speed of about 3.06 rad/s. Preview and stage logs print
`peak_target_speed`; measured speed still comes from motor feedback.

Tracking error above 0.75 rad, actual speed above `2*speed+0.5` rad/s, stale
feedback (100 ms), motor errors/warnings and failure to settle within 15 seconds
after the ramp abort the sequence. Ctrl+C attempts STOP on all three motors;
it does not return them to zero. Final STOP releases position holding. Keep the
physical power disconnect reachable. Host-side tests cover the trajectory and
use pseudo terminals to check worker communication, fragmented replies,
inactive channels, command expiry and stopping. Hardware validation is pending.

The first hardware test established communication and motion on IDs 0/1/2,
but default Kp=2 left 0.15-0.24 rad homing errors, so the sequence correctly
stopped before the 50-rad stage. `--zero` helps tune homing independently; it
prints each motor's target, error and measured output torque. Kp=5 may help but
does not guarantee the existing 0.05-rad tolerance. The measured FTDI driver
latency timer was 16 ms; see the Chinese guide for setting it to 1 ms and
rechecking timing. `--probe` measures STOP-only serial sweeps for 2 seconds.
`serial_sweep_hz` measures completed three-motor sweeps, while `serial_ms` and
`max_serial_ms` measure serial work per sweep. Overruns no longer add an extra
full-period sleep. End-to-end hardware validation remains pending.

A later Kp=5 hardware run held approximately 0.061/0.088/0.061 rad of error,
with 50 Hz sweeps, no missed deadlines and about 0.35-0.48 Nm static feedback
torque. The sequence program now accepts Kp up to 10 (default still 2) to test
whether higher stiffness reduces the residual error; use `--zero --speed 0.5
--kp 10 --kd 0.5` before trying the full sequence. Arrival conditions remain
unchanged. The single-motor bounded test and HIM configuration are unchanged.

### Three-motor velocity loop test

```bash
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --velocity \
  --speed 10 --accel 2 --kd 1 --hold 3
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --velocity-preview
```

This uses the same async UART, IDs 0/1/2, 6 Mbps and a 50 Hz publisher. Commands
use Kp=0, zero position/feedforward torque, and a motor-side velocity feedback
gain Kd: `torque = Kd * (target_velocity - measured_velocity)`. No host-side
integrator is added. `velocity_command()` exposes this protocol mapping.

The sequence establishes zero velocity for one second, smoothly accelerates
to 10 output rad/s, holds the measured velocity in tolerance for three seconds,
then decelerates to zero and holds for one second before STOP. At 2 rad/s² peak
target acceleration the 10-rad/s ramps take about 7.5 seconds each, with a total
duration near 20-21 seconds. Final position is not constrained. Use only unloaded
shafts that can rotate freely, with motors fixed and power disconnect reachable.

Options are `--speed` in [0.1,10], `--accel` in [0.1,2], `--kd` in (0,1] and
`--hold` in [0.5,10]; defaults are 10, 2, 1 and 3 respectively. `--kp` is rejected
in velocity mode. Positive-speed acceptance tolerance is max(0.5 rad/s, 10% of
target); zero-speed tolerance is 0.1 rad/s. Telemetry includes `v_cmd`, actual
`v`, `v_err`, output torque, position and UART timing. Overspeed above requested
peak+2 rad/s, tracking error above 3 rad/s for one second, stale feedback,
faults/warnings or stage timeouts abort with STOP. Ctrl+C attempts immediate
STOP rather than continuing the deceleration trajectory. The emulated-UART
test runs the complete 0 -> 10 -> 0 velocity command sequence and checks that
position stiffness remains zero. Hardware velocity performance is not yet measured.
