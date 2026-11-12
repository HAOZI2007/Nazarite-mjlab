#include "dm_imu.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc == 2 && (std::string(argv[1]) == "--help" ||
                      std::string(argv[1]) == "-h")) {
      std::cout << "Usage: " << argv[0] << " DEVICE [GYRO_SCALE]\n"
                << "GYRO_SCALE converts device units to rad/s; default is deg/s.\n";
      return 0;
    }
    if (argc < 2 || argc > 3) {
      std::cerr << "Usage: " << argv[0] << " DEVICE [GYRO_SCALE]\n"
                << "GYRO_SCALE converts device units to rad/s; default is deg/s.\n";
      return 2;
    }
    const float scale = argc == 3 ? std::strtof(argv[2], nullptr)
                                  : 0.017453292519943295F;
    nazarite::xz1::imu::DmImuSerial imu(argv[1], 921600, scale);
    std::cout << "gyro_x,gyro_y,gyro_z,qw,qx,qy,qz\n";
    while (true) {
      const auto sample = imu.read_sample(100);
      const auto& g = sample.angular_velocity_rad_s;
      const auto& q = sample.quaternion_body_to_world_wxyz;
      std::cout << std::fixed << std::setprecision(6) << g[0] << ',' << g[1]
                << ',' << g[2] << ',' << q[0] << ',' << q[1] << ',' << q[2]
                << ',' << q[3] << '\n' << std::flush;
    }
  } catch (const std::exception& error) {
    std::cerr << "IMU error: " << error.what() << '\n';
    return 1;
  }
}
