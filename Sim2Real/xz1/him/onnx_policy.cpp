#include "onnx_policy.hpp"

#ifdef XZ1_HAS_ONNXRUNTIME
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nazarite::xz1::him {
OnnxPolicy::OnnxPolicy(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING,"xz1-him") {
  options_.SetIntraOpNumThreads(1);
  options_.SetInterOpNumThreads(1);
  session_=std::make_unique<Ort::Session>(env_,model_path.c_str(),options_);
  Ort::AllocatorWithDefaultOptions allocator;
  if(session_->GetInputCount()!=1||session_->GetOutputCount()!=1)
    throw std::runtime_error("XZ1-HIM ONNX must have one input and one output");
  auto input=session_->GetInputNameAllocated(0,allocator);
  auto output=session_->GetOutputNameAllocated(0,allocator);
  input_name_=input.get(); output_name_=output.get();
  const auto input_type=session_->GetInputTypeInfo(0);
  const auto output_type=session_->GetOutputTypeInfo(0);
  const auto input_info=input_type.GetTensorTypeAndShapeInfo();
  const auto output_info=output_type.GetTensorTypeAndShapeInfo();
  const auto in_shape=input_info.GetShape(),out_shape=output_info.GetShape();
  if(input_info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||in_shape.size()!=2||in_shape[1]!=282||
     output_info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||out_shape.size()!=2||out_shape[1]!=12){
    const auto show=[](const std::vector<std::int64_t>& shape){std::string result="[";for(std::size_t i=0;i<shape.size();++i){if(i)result+=",";result+=std::to_string(shape[i]);}return result+"]";};
    throw std::runtime_error("XZ1-HIM expected float [batch,282]->[batch,12], got type "+
      std::to_string(input_info.GetElementType())+" "+show(in_shape)+" -> type "+
      std::to_string(output_info.GetElementType())+" "+show(out_shape));
  }
  const auto metadata=session_->GetModelMetadata();
  for(const auto& item:std::array<std::pair<const char*,const char*>,7>{{
      {"him_version","1"},{"him_frame_size","47"},{"him_history_size","6"},
      {"him_export_history_order","frame_major_current_first"},
      {"him_last_action","clipped_policy_action_before_joint_scale_and_offset"},
      {"joint_names","FL_hip_joint,FL_thigh_joint,FL_shank_joint,FR_hip_joint,FR_thigh_joint,FR_shank_joint,RR_hip_joint,RR_thigh_joint,RR_shank_joint,RL_hip_joint,RL_thigh_joint,RL_shank_joint"},
      {"observation_names","base_ang_vel,projected_gravity,command,phase,joint_pos,joint_vel,actions"}}}){
    auto value=metadata.LookupCustomMetadataMapAllocated(item.first,allocator);
    if(!value||item.second!=std::string(value.get()))throw std::runtime_error(std::string("XZ1-HIM metadata mismatch: ")+item.first);
  }
  const std::array<std::pair<const char*,std::array<float,12>>,4> contracts{{
      {"action_scale",{.25F,.25F,.25F,.25F,.25F,.25F,.25F,.25F,.25F,.25F,.25F,.25F}},
      {"joint_stiffness",{32,32,35,32,32,35,32,32,35,32,32,35}},
      {"joint_damping",{1.9F,1.9F,1.8F,1.9F,1.9F,1.8F,1.9F,1.9F,1.8F,1.9F,1.9F,1.8F}},
      {"default_joint_pos",{0,.83F,-1.7404F,0,.83F,-1.7404F,0,.83F,-1.7404F,0,.83F,-1.7404F}}}};
  for(const auto& contract:contracts){
    auto value=metadata.LookupCustomMetadataMapAllocated(contract.first,allocator);
    if(!value)throw std::runtime_error(std::string("missing HIM metadata: ")+contract.first);
    std::string text(value.get());std::size_t begin=0;
    for(std::size_t i=0;i<12;++i){
      const auto end=text.find(',',begin);
      const float parsed=std::stof(text.substr(begin,end-begin));
      if(std::abs(parsed-contract.second[i])>5.1e-4F)
        throw std::runtime_error(std::string("HIM metadata mismatch: ")+contract.first);
      begin=end==std::string::npos?text.size():end+1;
    }
    if(begin<text.size())throw std::runtime_error(std::string("invalid HIM metadata: ")+contract.first);
  }
}
std::array<float,kActionDim> OnnxPolicy::infer(const std::array<float,kObservationDim>& observation){
  for(float value:observation)if(!std::isfinite(value))throw std::invalid_argument("non-finite HIM observation");
  const std::array<std::int64_t,2> shape{1,static_cast<std::int64_t>(kObservationDim)};
  auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
  auto input=Ort::Value::CreateTensor<float>(memory,const_cast<float*>(observation.data()),
      observation.size(),shape.data(),shape.size());
  const char* input_name=input_name_.c_str();const char* output_name=output_name_.c_str();
  auto result=session_->Run(Ort::RunOptions{nullptr},&input_name,&input,1,&output_name,1);
  if(result.size()!=1||!result[0].IsTensor())throw std::runtime_error("invalid HIM ONNX output");
  const auto info=result[0].GetTensorTypeAndShapeInfo();
  if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||info.GetElementCount()!=kActionDim)
    throw std::runtime_error("HIM ONNX output must contain 12 float actions");
  std::array<float,kActionDim> action{};
  const float* data=result[0].GetTensorData<float>();
  for(std::size_t i=0;i<action.size();++i){
    if(!std::isfinite(data[i]))throw std::runtime_error("non-finite HIM action");
    action[i]=std::clamp(data[i],-10.0F,10.0F);
  }
  return action;
}
}  // namespace nazarite::xz1::him
#endif
