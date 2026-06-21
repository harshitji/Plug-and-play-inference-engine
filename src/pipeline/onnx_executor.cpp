#include "onnx_executor.h"
#include <iostream>

OnnxExecutor::OnnxExecutor(const std::string &model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "HybridEngine"),
      memory_info(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

  // Allow ONNX to use multiple cores for maximum throughput
  session_options.SetIntraOpNumThreads(4);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session =
      std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
}

// ---------------------------------------------------------------------------
// Single-image inference [1 x 3 x 224 x 224]
// ---------------------------------------------------------------------------
float *OnnxExecutor::run(float *input_tensor_ptr) {
  std::vector<int64_t> input_shape = {1, 3, 224, 224};
  std::vector<int64_t> output_shape = {1, 1000};
  const size_t output_size = 1000;

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_tensor_ptr, 1 * 3 * 224 * 224, input_shape.data(),
      input_shape.size());

  float *output_ptr = new float[output_size];
  Ort::Value output_tensor =
      Ort::Value::CreateTensor<float>(memory_info, output_ptr, output_size,
                                      output_shape.data(), output_shape.size());

  const char *input_names[] = {"input"};
  const char *output_names[] = {"output"};

  std::cout << "[OnnxExecutor] Running Zero-Copy Tensor Math..." << std::endl;
  session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
               output_names, &output_tensor, 1);

  return output_ptr;
}

// ---------------------------------------------------------------------------
// Batch inference [N x 3 x 224 x 224]
// Returns heap-allocated float[N * 1000].
// ---------------------------------------------------------------------------
float *OnnxExecutor::runBatch(float *input_tensor_ptr, int batch_size) {
  const size_t input_size = static_cast<size_t>(batch_size) * 3 * 224 * 224;
  const size_t output_size = static_cast<size_t>(batch_size) * 1000;

  std::vector<int64_t> input_shape = {batch_size, 3, 224, 224};
  std::vector<int64_t> output_shape = {batch_size, 1000};

  Ort::Value input_tensor =
      Ort::Value::CreateTensor<float>(memory_info, input_tensor_ptr, input_size,
                                      input_shape.data(), input_shape.size());

  float *output_ptr = new float[output_size];
  Ort::Value output_tensor =
      Ort::Value::CreateTensor<float>(memory_info, output_ptr, output_size,
                                      output_shape.data(), output_shape.size());

  const char *input_names[] = {"input"};
  const char *output_names[] = {"output"};

  std::cout << "[OnnxExecutor] Running Batched Tensor Math [N=" << batch_size
            << "]..." << std::endl;
  session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
               output_names, &output_tensor, 1);

  return output_ptr;
}
