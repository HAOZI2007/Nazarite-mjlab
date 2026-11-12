#include "evdev_gamepad.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

int main(int argc,char** argv){
  try{
    std::string path;
    if(argc==2&&std::string(argv[1])=="--list"){
      for(const auto& item:nazarite::xz1::controller::list_gamepads())std::cout<<item<<'\n';
      return 0;
    }
    if(argc==2)path=argv[1];
    else if(argc==1){
      const auto paths=nazarite::xz1::controller::list_gamepads();
      if(paths.size()!=1){
        std::cerr<<"Found "<<paths.size()<<" gamepad candidates; run "<<argv[0]<<" --list and pass one event path.\n";
        return 2;
      }
      path=paths.front();
    }else{std::cerr<<"Usage: "<<argv[0]<<" [--list | /dev/input/eventX]\n";return 2;}
    nazarite::xz1::controller::EvdevGamepad pad(path);
    std::cout<<"vx,vy,yaw_rate,reset\n";
    while(true){
      const auto cmd=pad.poll();
      std::cout<<cmd.vx<<','<<cmd.vy<<','<<cmd.yaw_rate<<','<<cmd.reset_requested<<'\n';
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }catch(const std::exception& error){std::cerr<<"gamepad error: "<<error.what()<<'\n';return 1;}
}
