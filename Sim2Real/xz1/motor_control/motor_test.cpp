#include "im6014.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

int main(int argc,char** argv){
  if(argc<4){
    std::cerr<<"Usage: "<<argv[0]<<" DEVICE BAUD MOTOR_ID [--probe | --move RAD KP KD SECONDS]\n"
             <<"--probe sends STOP only; --move actively moves one unloaded motor.\n";
    return 2;
  }
  try{
    const std::string device=argv[1];
    const int baud=std::stoi(argv[2]);
    const int id=std::stoi(argv[3]);
    if(id<0||id>14)throw std::invalid_argument("motor ID must be in [0,14]");
    nazarite::xz1::motor::Connection motor(device,baud,static_cast<std::uint8_t>(id));
    nazarite::xz1::motor::Command stop{}; stop.motor_id=static_cast<std::uint8_t>(id);
    auto feedback=motor.transact(stop);
    std::cout<<"probe id="<<unsigned(feedback.motor_id)<<" pos="<<feedback.position_rad
             <<" rad, voltage="<<feedback.voltage_v<<" V, error=0x"<<std::hex<<feedback.error<<std::dec<<'\n';
    if(argc==5&&std::string(argv[4])=="--probe")return 0;
    if(argc!=9||std::string(argv[4])!="--move")
      throw std::invalid_argument("expected --probe or --move RAD KP KD SECONDS");
    const float target=feedback.position_rad+std::stof(argv[5]);
    const float kp=std::stof(argv[6]),kd=std::stof(argv[7]);
    const float seconds=std::stof(argv[8]);
    if(!(seconds>0&&seconds<=5))throw std::invalid_argument("duration must be in (0,5] seconds");
    if(!std::isfinite(target)||std::abs(std::stof(argv[5]))>.2F||
       !std::isfinite(kp)||kp<0||kp>5||!std::isfinite(kd)||kd<0||kd>1)
      throw std::invalid_argument("test limits: |delta|<=0.2 rad, kp<=5, kd<=1");
    auto command=nazarite::xz1::motor::position_command(
        static_cast<std::uint8_t>(id),target,kp,kd);
    const auto end=std::chrono::steady_clock::now()+std::chrono::duration<float>(seconds);
    auto next=std::chrono::steady_clock::now();
    try{
      while(std::chrono::steady_clock::now()<end){
        feedback=motor.transact(command);
        if(feedback.error||feedback.timeout_triggered)throw std::runtime_error("motor fault/timeout protection");
        next+=std::chrono::milliseconds(10);
        std::this_thread::sleep_until(next);
      }
    }catch(...){motor.stop();throw;}
    motor.stop();
    return 0;
  }catch(const std::exception& error){std::cerr<<"motor test error: "<<error.what()<<'\n';return 1;}
}
