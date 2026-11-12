#pragma once

#include "xz1_him.hpp"

#ifdef XZ1_HAS_ONNXRUNTIME
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>

namespace nazarite::xz1::him {
class OnnxPolicy {
 public:
  explicit OnnxPolicy(const std::string& model_path);
  std::array<float, kActionDim> infer(
      const std::array<float, kObservationDim>& observation);

 private:
  Ort::Env env_;
  Ort::SessionOptions options_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
};
}  // namespace nazarite::xz1::him
#endif
