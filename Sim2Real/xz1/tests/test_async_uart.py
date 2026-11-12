"""Exercise real serial worker threads against an emulated three-motor UART.

Uses Linux pseudo terminals only; never opens a real motor device.
"""

import errno
import math
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import threading
import time
import unittest


SEQUENCE, CLIENT = sys.argv[1:3]
del sys.argv[1:3]


def crc_words(data):
    crc = 0xFFFFFFFF
    for offset in range(0, len(data), 4):
        word = int.from_bytes(data[offset:offset + 4], "little")
        for bit in range(31, -1, -1):
            top = crc & 0x80000000
            crc = (crc << 1) & 0xFFFFFFFF
            if top:
                crc ^= 0x04C11DB7
            if word & (1 << bit):
                crc ^= 0x04C11DB7
    return crc


class MotorUart:
    def __init__(self, drop_id=None, warning=False, fragments=False, delay=0,
                 velocity_ratio=1.0, initial_positions=None):
        self.master, self.slave = pty.openpty()
        self.device = os.ttyname(self.slave)
        os.set_blocking(self.master, False)
        self.records = []
        self.positions = dict.fromkeys(range(3), 0)
        if initial_positions is not None:
            self.positions = {i: round(q * (38 / 3) * 32768 / (2 * math.pi))
                              for i, q in enumerate(initial_positions)}
        self.speeds = dict.fromkeys(range(3), 0)
        self.drop_id = drop_id
        self.warning = warning
        self.fragments = fragments
        self.delay = delay
        self.velocity_ratio = velocity_ratio
        self.errors = []
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.serve, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.closed.set()
        self.thread.join(timeout=2)
        os.close(self.master)
        os.close(self.slave)

    def serve(self):
        buffer = bytearray()
        try:
            while not self.closed.is_set():
                ready, _, _ = select.select([self.master], [], [], 0.01)
                if not ready:
                    continue
                try:
                    buffer.extend(os.read(self.master, 4096))
                except OSError as error:
                    if error.errno in (errno.EIO, errno.EAGAIN):
                        continue
                    raise
                while len(buffer) >= 20:
                    if buffer[:2] != b"\xfe\xee":
                        del buffer[0]
                        continue
                    frame = bytes(buffer[:20])
                    del buffer[:20]
                    if crc_words(frame[:16]) != struct.unpack_from("<I", frame, 16)[0]:
                        raise AssertionError("invalid command CRC")
                    motor_id = frame[2] & 0x0F
                    if motor_id not in self.positions:
                        raise AssertionError("command sent to inactive motor ID")
                    enabled = bool(frame[2] & 0x10)
                    if enabled and not frame[2] & 0x80:
                        raise AssertionError("motion missing timeout protection")
                    self.records.append((motor_id, enabled, frame))
                    if enabled:
                        if struct.unpack_from("<H", frame, 12)[0]:
                            self.positions[motor_id] = struct.unpack_from("<i", frame, 8)[0]
                            self.speeds[motor_id] = 0
                        else:
                            self.speeds[motor_id] = round(
                                struct.unpack_from("<h", frame, 6)[0] *
                                self.velocity_ratio)
                            self.positions[motor_id] += round(self.speeds[motor_id] * 512 * 0.02)
                    else:
                        self.speeds[motor_id] = 0
                    if motor_id == self.drop_id:
                        continue
                    self.closed.wait(self.delay)
                    reply = bytearray(26)
                    reply[:2] = b"\xfc\xee"
                    reply[2] = motor_id | (0x10 if enabled else 0)
                    reply[3:6] = bytes((25, 25, 48))
                    struct.pack_into("<h", reply, 8, self.speeds[motor_id])
                    struct.pack_into("<i", reply, 10, self.positions[motor_id])
                    if self.warning and enabled:
                        struct.pack_into("<H", reply, 18, 1 << 13)
                    struct.pack_into("<I", reply, 22, crc_words(reply[2:22]))
                    if self.fragments:
                        # A partial frame after noise must survive buffer compaction.
                        os.write(self.master, b"\x00" * 8 + reply[:20])
                        self.closed.wait(0.001)
                        os.write(self.master, reply[20:])
                    else:
                        os.write(self.master, reply)
        except Exception as error:
            self.errors.append(str(error))


class AsyncUartTests(unittest.TestCase):
    def assert_stopped(self, uart):
        self.assertFalse(uart.errors, uart.errors)
        self.assertTrue(uart.records)
        last = {}
        for motor_id, enabled, _ in uart.records:
            last[motor_id] = enabled
        self.assertEqual(last, {0: False, 1: False, 2: False})

    def test_fragmented_stop_probe(self):
        with MotorUart(fragments=True) as uart:
            result = subprocess.run([SEQUENCE, uart.device, "--probe"],
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(any(enabled for _, enabled, _ in uart.records))
            self.assert_stopped(uart)

    def test_stale_command_fault_and_inactive_channels(self):
        with MotorUart() as uart:
            result = subprocess.run([CLIENT, uart.device], capture_output=True,
                                    text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(any(enabled for _, enabled, _ in uart.records))
            self.assert_stopped(uart)

    def test_slow_uart_sweep_does_not_add_an_extra_period(self):
        with MotorUart(delay=0.009) as uart:
            result = subprocess.run([SEQUENCE, uart.device, "--probe"],
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            match = re.search(r"serial_sweep_hz=([0-9.]+)", result.stdout)
            self.assertIsNotNone(match, result.stdout)
            # 3*9 ms > 20 ms; no extra 20-ms sleep after each slow sweep.
            self.assertGreater(float(match.group(1)), 28, result.stdout)
            self.assert_stopped(uart)

    def test_zero_only_never_starts_the_50_rad_stage(self):
        with MotorUart() as uart:
            result = subprocess.run([SEQUENCE, uart.device, "--zero", "--kp", "10",
                                     "--kd", "0.5", "--speed", "0.5"],
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("reached=0.0000", result.stdout)
            self.assertNotIn("target=50.", result.stdout)
            enabled_frames = [frame for _, enabled, frame in uart.records if enabled]
            self.assertTrue(enabled_frames)
            for frame in enabled_frames:
                self.assertEqual(struct.unpack_from("<i", frame, 8)[0], 0)
                self.assertEqual(struct.unpack_from("<H", frame, 12)[0],
                                 round(10 * 12800 / ((38 / 3) ** 2)))
            self.assert_stopped(uart)

    def test_sequence_gain_limit_is_still_enforced(self):
        result = subprocess.run([SEQUENCE, "/no/such/uart", "--preview",
                                 "--kp", "11"],
                                capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("kp in [0.1,10]", result.stderr)

    def test_large_initial_angle_requires_explicit_limit(self):
        for angle in (102.0228, -102.0228):
            with self.subTest(angle=angle), MotorUart(
                    initial_positions=(angle, 98.6479, 98.9174)) as uart:
                result = subprocess.run([SEQUENCE, uart.device, "--zero"],
                                        capture_output=True, text=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ID 0 initial absolute position", result.stderr)
                self.assertIn("--max-start-rad", result.stderr)
                self.assertFalse(any(active for _, active, _ in uart.records))
                self.assert_stopped(uart)

    def test_large_angle_return_starts_smoothly_and_interrupt_stops(self):
        initial = (102.0228, 98.6479, -101.0)
        with MotorUart(initial_positions=initial) as uart:
            process = subprocess.Popen(
                [SEQUENCE, uart.device, "--zero", "--max-start-rad", "200",
                 "--speed", "5", "--accel", "2", "--kp", "10", "--kd", "0.5"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 3
                while sum(active for _, active, _ in uart.records) < 45:
                    self.assertIsNone(process.poll(), "homing exited early")
                    self.assertLess(time.monotonic(), deadline, "no motion commands")
                    time.sleep(0.01)
                process.send_signal(signal.SIGINT)
                output, _ = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 1, output.decode())
                self.assertIn(b"interrupted", output)
                self.assertNotIn(b"stage target=50.", output)
                for motor_id, start in enumerate(initial):
                    positions = [struct.unpack_from("<i", frame, 8)[0] *
                                 (2 * math.pi) / (32768 * (38 / 3))
                                 for i, active, frame in uart.records
                                 if active and i == motor_id]
                    self.assertAlmostEqual(positions[0], start, delta=0.002)
                    self.assertLess(abs(positions[-1]), abs(start))
                    self.assertTrue(all(abs(q - start) < 1 for q in positions))
                self.assert_stopped(uart)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_start_angle_limit_validation(self):
        for value in ("0", "-200", "1001", "nan", "inf"):
            with self.subTest(value=value):
                result = subprocess.run(
                    [SEQUENCE, "/no/such/uart", "--preview",
                     "--max-start-rad", value],
                    capture_output=True, text=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("open serial", result.stderr)
        result = subprocess.run(
            [SEQUENCE, "/no/such/uart", "--preview", "--max-start-rad", "1000"],
            capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ctrl_c_stops_all_three_during_second_stage(self):
        with MotorUart() as uart:
            process = subprocess.Popen([SEQUENCE, uart.device, "--run",
                                        "--speed", "5", "--accel", "2",
                                        "--kp", "10", "--kd", "0.5"],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)
            prefix = b""
            try:
                # Initial zero stage must settle before the 50-rad stage begins.
                while True:
                    ready, _, _ = select.select([process.stdout], [], [], 5)
                    self.assertTrue(ready, "no progress from sequence")
                    chunk = os.read(process.stdout.fileno(), 4096)
                    prefix += chunk
                    self.assertTrue(chunk, "sequence exited before second stage")
                    if b"stage target=50." in prefix:
                        break
                time.sleep(0.2)
                process.send_signal(signal.SIGINT)
                output, _ = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 1, (prefix + output).decode())
                self.assertIn(b"interrupted", output)
                self.assert_stopped(uart)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_missing_motor_aborts_without_motion(self):
        with MotorUart(drop_id=1) as uart:
            result = subprocess.run([SEQUENCE, uart.device, "--run"],
                                    capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("timeout", result.stderr)
            self.assertFalse(any(enabled for _, enabled, _ in uart.records))
            self.assert_stopped(uart)

    def test_motor_warning_aborts_motion(self):
        with MotorUart(warning=True) as uart:
            result = subprocess.run([SEQUENCE, uart.device, "--run"],
                                    capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("warning", result.stderr)
            self.assert_stopped(uart)

    def test_preview_never_opens_device(self):
        result = subprocess.run([SEQUENCE, "/no/such/uart", "--preview"],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("target=0.0000"), 2)
        self.assertIn("target=50.0000", result.stdout)
        result = subprocess.run([SEQUENCE, "/no/such/uart", "--run", "--speed", "nan"],
                                capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid numeric", result.stderr)

    def test_fast_preview_and_parameter_limits(self):
        result = subprocess.run([SEQUENCE, "/no/such/uart", "--preview",
                                 "--speed", "5", "--accel", "2"],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("ramp_seconds=15.0000"), 2)
        self.assertEqual(result.stdout.count("peak_target_speed=5.0000"), 2)
        for flag, value in (("--speed", "6"), ("--accel", "3"),
                            ("--accel", "0")):
            result = subprocess.run([SEQUENCE, "/no/such/uart", "--preview",
                                     flag, value],
                                    capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("test limits", result.stderr)

    def test_full_velocity_sequence(self):
        # Model a small steady velocity error; phase changes must not step the reference.
        with MotorUart(velocity_ratio=0.95) as uart:
            result = subprocess.run([SEQUENCE, uart.device, "--velocity",
                                     "--speed", "10", "--accel", "2",
                                     "--kd", "1", "--hold", "0.5"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.count("reached_velocity=0.0000"), 2)
            self.assertIn("reached_velocity=10.0000", result.stdout)
            enabled = [frame for _, active, frame in uart.records if active]
            self.assertTrue(enabled)
            for frame in enabled:
                self.assertEqual(struct.unpack_from("<h", frame, 4)[0], 0)
                self.assertEqual(struct.unpack_from("<i", frame, 8)[0], 0)
                self.assertEqual(struct.unpack_from("<H", frame, 12)[0], 0)
            for motor_id in range(3):
                speeds = [struct.unpack_from("<h", frame, 6)[0]
                          for i, active, frame in uart.records
                          if i == motor_id and active]
                self.assertGreaterEqual(max(speeds), 1290)
                self.assertEqual(speeds[-1], 0)
                self.assertLessEqual(max(abs(b - a) for a, b in
                                         zip(speeds, speeds[1:])), 30)
            self.assert_stopped(uart)

    def test_velocity_preview_and_limits(self):
        result = subprocess.run([SEQUENCE, "/no/such/uart", "--velocity-preview"],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("ramp_seconds=7.5000"), 2)
        for flag, value in (("--speed", "11"), ("--kd", "0"),
                            ("--hold", "0"), ("--kp", "1")):
            result = subprocess.run([SEQUENCE, "/no/such/uart", "--velocity-preview",
                                     flag, value],
                                    capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
