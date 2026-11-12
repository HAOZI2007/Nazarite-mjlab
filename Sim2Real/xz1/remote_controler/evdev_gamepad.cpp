#include "evdev_gamepad.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <filesystem>
#include <fcntl.h>
#include <linux/input.h>
#include <stdexcept>
#include <system_error>
#include <sys/ioctl.h>
#include <unistd.h>

namespace nazarite::xz1::controller {
namespace {
constexpr std::size_t kWordBits = 8 * sizeof(unsigned long);
constexpr std::size_t kAbsWords = (ABS_MAX + kWordBits) / kWordBits;

float normalize(int value, int minimum, int maximum, float deadzone) {
  if (maximum <= minimum) return 0;
  const float center = (minimum + maximum) * .5F;
  const float radius = std::max(center - minimum, maximum - center);
  const float axis = std::clamp((value - center) / radius, -1.0F, 1.0F);
  const float magnitude = std::abs(axis);
  if (magnitude <= deadzone) return 0;
  return std::copysign((magnitude-deadzone)/(1-deadzone),axis);
}
bool has_axis(int fd, unsigned code) {
  unsigned long bits[kAbsWords]{};
  if (ioctl(fd, EVIOCGBIT(EV_ABS, sizeof(bits)), bits) < 0) return false;
  return (bits[code/(8*sizeof(unsigned long))] & (1UL << (code%(8*sizeof(unsigned long))))) != 0;
}
void read_axis(int fd,int code,int& minimum,int& maximum,int& value) {
  input_absinfo info{};
  if (ioctl(fd,EVIOCGABS(code),&info)<0) throw std::system_error(errno,std::generic_category(),"read gamepad axis range");
  minimum=info.minimum; maximum=info.maximum; value=info.value;
}
int select_axis(int fd, int preferred, int fallback) {
  if(has_axis(fd,static_cast<unsigned>(preferred))) return preferred;
  if(has_axis(fd,static_cast<unsigned>(fallback))) return fallback;
  return -1;
}
}  // namespace

std::vector<std::string> list_gamepads() {
  std::vector<std::string> paths;
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator("/dev/input", error)) {
    const auto name = entry.path().filename().string();
    if (name.rfind("event", 0) != 0) continue;
    const int fd = ::open(entry.path().c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) continue;
    const bool has_sticks = has_axis(fd, ABS_X) && has_axis(fd, ABS_Y);
    ::close(fd);
    if (has_sticks) paths.push_back(entry.path().string());
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

EvdevGamepad::EvdevGamepad(const std::string& path,float max_vx,float max_vy,
                           float max_yaw,float deadzone)
    : path_(path),max_vx_(max_vx),max_vy_(max_vy),max_yaw_(max_yaw),deadzone_(deadzone) {
  if(!std::isfinite(max_vx_)||!std::isfinite(max_vy_)||!std::isfinite(max_yaw_)||
     !std::isfinite(deadzone_)||max_vx_<0||max_vy_<0||max_yaw_<0||deadzone_<0||deadzone_>=1)
    throw std::invalid_argument("invalid gamepad scaling/deadzone");
  fd_=::open(path.c_str(),O_RDONLY|O_NONBLOCK|O_CLOEXEC);
  if(fd_<0) throw std::system_error(errno,std::generic_category(),"open gamepad");
  axis_vx_=select_axis(fd_,ABS_Y,ABS_RY);
  axis_vy_=select_axis(fd_,ABS_X,ABS_RX);
  axis_yaw_=select_axis(fd_,ABS_RX,ABS_RZ);
  if(axis_vx_<0||axis_vy_<0){::close(fd_);fd_=-1;throw std::runtime_error("gamepad must expose two analog stick axes");}
  try {
    read_axis(fd_,axis_vx_,min_vx_,max_axis_vx_,value_vx_);
    read_axis(fd_,axis_vy_,min_vy_,max_axis_vy_,value_vy_);
    if(axis_yaw_>=0) read_axis(fd_,axis_yaw_,min_yaw_,max_axis_yaw_,value_yaw_);
  } catch (...) {::close(fd_);fd_=-1;throw;}
}
EvdevGamepad::~EvdevGamepad(){if(fd_>=0)::close(fd_);}

TwistCommand EvdevGamepad::poll(){
  input_event events[32];
  bool reset=false;
  while(true){
    const auto bytes=::read(fd_,events,sizeof(events));
    if(bytes<0&&(errno==EAGAIN||errno==EWOULDBLOCK))break;
    if(bytes<0&&errno==EINTR)continue;
    if(bytes<0)throw std::system_error(errno,std::generic_category(),"read gamepad");
    if(bytes==0)throw std::runtime_error("gamepad disconnected");
    const auto count=static_cast<std::size_t>(bytes)/sizeof(input_event);
    for(std::size_t i=0;i<count;++i){
      const auto& event=events[i];
      if(event.type==EV_ABS){
        if(event.code==axis_vx_)value_vx_=event.value;
        if(event.code==axis_vy_)value_vy_=event.value;
        if(event.code==axis_yaw_)value_yaw_=event.value;
      }else if(event.type==EV_KEY&&event.code==BTN_START&&event.value==1)reset=true;
    }
  }
  return TwistCommand{-normalize(value_vx_,min_vx_,max_axis_vx_,deadzone_)*max_vx_,
                      normalize(value_vy_,min_vy_,max_axis_vy_,deadzone_)*max_vy_,
                      axis_yaw_>=0?normalize(value_yaw_,min_yaw_,max_axis_yaw_,deadzone_)*max_yaw_:0,
                      reset};
}
}  // namespace nazarite::xz1::controller
