#pragma once
#include "interfaces.h"
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>

/**
 * @class OnnxExecutor
 * @brief Concrete IExecutor leveraging Microsoft's ONNX Runtime C++ API.
 *        Supports both single-image [1,3,224,224] and dynamic-batch
 * [N,3,224,224] inference.
 */
class OnnxExecutor : public IExecutor {
private:
  Ort::Env env;
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memory_info;

public:
  OnnxExecutor(const std::string &model_path);

  /// Single-image. Returns heap-allocated float[1000].
  float *run(float *input_tensor_ptr) override;

  /// Batch: input is [N,3,224,224]. Returns heap-allocated float[N*1000].
  float *runBatch(float *input_tensor_ptr, int batch_size) override;
};
