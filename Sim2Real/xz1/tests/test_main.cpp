#include "../him/xz1_him.hpp"
#include "../imu/dm_imu.hpp"
#include "../motor_control/im6014.hpp"
#include "../motor_control/async_motor_system.hpp"
#include "../motor_control/position_ramp.hpp"
#include "../motor_control/velocity_ramp.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void append_float(std::vector<std::uint8_t>& bytes,float value){
  std::uint32_t bits;std::memcpy(&bits,&value,4);
  for(int i=0;i<4;++i)bytes.push_back(static_cast<std::uint8_t>(bits>>(8*i)));
}
std::uint16_t crc16(const std::vector<std::uint8_t>& bytes){
  std::uint16_t crc=0xFFFF;
  for(auto byte:bytes){crc^=static_cast<std::uint16_t>(byte)<<8;for(int i=0;i<8;++i)crc=(crc&0x8000)?static_cast<std::uint16_t>((crc<<1)^0x1021):static_cast<std::uint16_t>(crc<<1);}
  return crc;
}
std::vector<std::uint8_t> imu_frame(std::uint8_t type,const std::vector<float>& values){
  std::vector<std::uint8_t> frame{0x55,0xAA,1,type};for(float v:values)append_float(frame,v);
  const auto crc=crc16(frame);frame.push_back(crc&0xFF);frame.push_back(crc>>8);frame.push_back(0x0A);return frame;
}
void test_imu_parser(){
  nazarite::xz1::imu::DmImuParser parser;
  auto gyro=imu_frame(2,{180,-90,0});auto quat=imu_frame(4,{1,0,0,0});
  std::vector<std::uint8_t> stream{0x00};stream.insert(stream.end(),gyro.begin(),gyro.end());stream.insert(stream.end(),quat.begin(),quat.end());
  auto first=parser.feed(stream.data(),7,123);check(first.empty(),"partial IMU frame accepted");
  auto output=parser.feed(stream.data()+7,stream.size()-7,456);
  check(output.size()==2,"gyro/quaternion frame count");
  check(output[0].has_angular_velocity&&std::abs(output[0].angular_velocity_rad_s[0]-3.14159265F)<1e-5F,"gyro deg/s conversion");
  check(output[1].has_quaternion&&output[1].quaternion_body_to_world_wxyz[0]==1,"quaternion parsing");
  const auto body=nazarite::xz1::imu::transform_to_body(
      output[0],{.70710678F,0,0,.70710678F});
  check(std::abs(body.angular_velocity_rad_s[0]-1.5707963F)<1e-5F&&
        std::abs(body.angular_velocity_rad_s[1]-3.14159265F)<1e-5F,
        "IMU mounting transform");
}
void test_him_contract(){
  using namespace nazarite::xz1::him;
  HimObservation builder;RobotState state;state.joint_position_rad=builder.default_joint_position();
  std::array<float,12> previous{};auto obs=builder.build(state,{0,0,0},previous);
  check(obs.size()==282&&obs[5]==-1,"gravity/input shape");
  check(obs[47+5]==-1&&obs[235+5]==-1,"initial history is current-frame repeated");
  builder.advance();auto moving=builder.build(state,{.2F,0,0},previous);
  check(std::abs(moving[9]-std::sin(2*3.14159265F/30))<1e-5F&&
        std::abs(moving[10]-std::cos(2*3.14159265F/30))<1e-5F,"HIM phase advance");
  std::array<float,12> action{};auto targets=builder.targets(action);
  check(std::abs(targets.position_rad[1]-.8483F)<1e-6F&&targets.kp[2]==35,"action target contract");
  action[0]=10;bool rejected=false;try{(void)builder.targets(action);}catch(const std::out_of_range&){rejected=true;}
  check(rejected,"joint-limit violation rejected");
}
void test_motor_protocol(){
  using namespace nazarite::xz1::motor;
  Command stop{};stop.motor_id=1;auto frame=encode(stop);
  check(frame[0]==0xFE&&frame[1]==0xEE&&frame[2]==1,"STOP header and mode");
  const std::array<std::uint8_t,4> expected_crc{0xF2,0xA7,0x2F,0xA4};
  check(std::equal(expected_crc.begin(),expected_crc.end(),frame.begin()+16),
        "IM6014 CRC matches Python reference vector");
  const auto mapped=position_command(3,.5F,32,1.9F);
  check(mapped.enabled&&mapped.timeout_protection&&mapped.motor_id==3&&
        mapped.position_rad==.5F&&mapped.kp_nm_rad==32,
        "HIM joint target maps to enabled protected motor command");
  const auto speed_command = velocity_command(2, 10, 1);
  const auto speed_frame = encode(speed_command);
  check(speed_command.enabled && speed_command.timeout_protection &&
        speed_command.speed_rad_s == 10 && speed_command.kp_nm_rad == 0 &&
        speed_command.position_rad == 0 && speed_command.torque_nm == 0 &&
        speed_command.kd_nm_per_rad_s == 1,
        "velocity control must disable position feedback and enable speed damping");
  check(speed_frame[12] == 0 && speed_frame[13] == 0 &&
        speed_frame[6] == 0x0A && speed_frame[7] == 0x05,
        "10 output rad/s must encode as rotor-side speed raw 1290");
  bool zero_kd_rejected = false;
  try { (void)velocity_command(0, 10, 0); }
  catch (const std::invalid_argument&) { zero_kd_rejected = true; }
  check(zero_kd_rejected, "velocity mode requires positive damping");
  check(crc32_mpeg2_words(frame.data(),16)==
        (static_cast<std::uint32_t>(frame[16])|(static_cast<std::uint32_t>(frame[17])<<8)|
         (static_cast<std::uint32_t>(frame[18])<<16)|(static_cast<std::uint32_t>(frame[19])<<24)),"command CRC");
  bool rejected=false;try{stop.motor_id=15;(void)encode(stop);}catch(const std::invalid_argument&){rejected=true;}
  check(rejected,"broadcast motor ID rejected");
}
void test_async_motor_command_mapping(){
  using namespace nazarite::xz1::motor;
  MotorCommands commands{};
  for(std::size_t index=0;index<commands.size();++index){
    commands[index].motor_id=kActionIndexMotorIds[index];
  }
  validate_motor_commands(commands,kActionIndexMotorIds);
  commands[6].motor_id=5;
  bool rejected=false;
  try{validate_motor_commands(commands,kActionIndexMotorIds);}catch(const std::invalid_argument&){rejected=true;}
  check(rejected,"action command must use the configured motor ID");
  for(std::size_t index=0;index<commands.size();++index){
    commands[index].motor_id=kPerLegMotorIds[index];
  }
  validate_motor_commands(commands,kPerLegMotorIds);
  commands[0].enabled=true;
  rejected=false;
  try{validate_motor_commands(commands,kPerLegMotorIds);}catch(const std::invalid_argument&){rejected=true;}
  check(rejected,"enabled bus commands require timeout protection");
  commands[0].timeout_protection=true;
  commands[0].position_rad=40000;
  rejected=false;
  try{validate_motor_commands(commands,kPerLegMotorIds);}catch(const std::exception&){rejected=true;}
  check(rejected,"invalid motor command range rejected before submission");
  commands[0].position_rad=0;
  commands[1].motor_id=0;
  rejected=false;
  try{validate_motor_commands(commands,kPerLegMotorIds);}catch(const std::invalid_argument&){rejected=true;}
  check(rejected,"motor IDs must be unique within one leg bus");
}

void test_output_position_ramp() {
  using namespace nazarite::xz1::motor;
  std::array<float, 3> previous{0.3F, -0.4F, 0.1F};
  for (float target : {0.0F, 50.0F, 0.0F}) {
    const PositionRamp ramp(previous, target, 1.0);
    check(ramp.sample(0) == previous, "ramp must start at measured positions");
    const auto end = ramp.sample(ramp.duration());
    for (std::size_t i = 0; i < 3; ++i) {
      check(std::abs(end[i] - target) < 1e-5F, "absolute output ramp endpoint");
    }
    auto last = previous;
    const double dt = 0.02;
    for (double t = dt; t <= ramp.duration() + dt; t += dt) {
      const auto positions = ramp.sample(t);
      for (std::size_t i = 0; i < 3; ++i) {
        check(std::abs(positions[i] - last[i]) <= 0.02001F,
              "output trajectory must obey 1 rad/s speed bound");
        check(positions[i] >= std::min(previous[i], target) - 1e-5F &&
              positions[i] <= std::max(previous[i], target) + 1e-5F,
              "trajectory must not overshoot target");
        (void)encode(position_command(static_cast<std::uint8_t>(i),
                                     positions[i], 2.0F, 0.3F));
      }
      last = positions;
    }
    previous = end;
  }
  bool rejected = false;
  try { AsyncMotorSystem bus(LegDevices{}, 6000000); }
  catch (const std::invalid_argument&) { rejected = true; }
  check(rejected, "all-disabled serial configuration must be rejected");

  const PositionRamp fast({0, 0, 0}, 50, 5, 2);
  check(std::abs(fast.duration() - 15) < 1e-6 &&
        std::abs(fast.peak_speed() - 5) < 1e-6,
        "50-rad trajectory should reach 5 rad/s with the selected acceleration");
  auto last = fast.sample(0);
  double previous_speed = 0;
  for (int tick = 1; tick <= 750; ++tick) {
    const auto position = fast.sample(tick * 0.02);
    const double speed = (position[0] - last[0]) / 0.02;
    check(speed >= 0 && speed <= 5.001,
          "fast trajectory violates speed or direction bound");
    check(std::abs(speed - previous_speed) / 0.02 <= 2.02,
          "fast trajectory violates acceleration bound");
    previous_speed = speed;
    last = position;
  }
  check(std::abs(last[0] - 50) < 1e-6,
        "fast trajectory must stop at 50 output radians");
  const PositionRamp limited({0, 0, 0}, 50, 5);
  check(limited.duration() > 24 && limited.peak_speed() < 3.1,
        "default acceleration limit must still constrain a high requested speed");

  const PositionRamp home({102.0228F, 98.6479F, -101.0F}, 0, 5, 2);
  check(std::abs(home.duration() - 30.60684) < 1e-4,
        "multi-turn homing duration must account for the full absolute distance");
  auto previous_home = home.sample(0);
  for (double t = 0.02; t < home.duration() + 0.02; t += 0.02) {
    const auto positions = home.sample(t);
    for (std::size_t i = 0; i < 3; ++i) {
      check(std::abs(positions[i] - previous_home[i]) <= 0.10002F,
            "multi-turn homing must obey the 5 rad/s speed bound");
      check(std::abs(positions[i]) <= std::abs(previous_home[i]),
            "multi-turn homing must approach absolute zero without wrapping");
    }
    previous_home = positions;
  }
  check(home.sample(home.duration()) == std::array<float, 3>{0, 0, 0},
        "multi-turn homing must end at absolute zero");
}

void test_velocity_ramp() {
  using nazarite::xz1::motor::VelocityRamp;
  std::array<float, 3> speeds{};
  for (float target : {0.0F, 10.0F, 0.0F}) {
    const VelocityRamp ramp(speeds, target, 2);
    auto last = speeds;
    for (double t = 0.02; t <= ramp.duration() + 0.02; t += 0.02) {
      const auto values = ramp.sample(t);
      for (std::size_t i = 0; i < 3; ++i) {
        check(std::abs(values[i] - last[i]) <= 0.04001F,
              "velocity profile exceeds 2 rad/s^2 acceleration");
        check(values[i] >= 0 && values[i] <= 10,
              "velocity profile reverses or overshoots");
      }
      last = values;
    }
    speeds = ramp.sample(ramp.duration());
    for (float speed : speeds) {
      check(std::abs(speed - target) < 1e-6F, "velocity ramp endpoint mismatch");
    }
  }
}
}

int main(){
  try{test_imu_parser();test_him_contract();test_motor_protocol();test_async_motor_command_mapping();test_output_position_ramp();test_velocity_ramp();std::cout<<"XZ1 deployment tests passed\n";return 0;}
  catch(const std::exception& error){std::cerr<<"Test failed: "<<error.what()<<'\n';return 1;}
}
