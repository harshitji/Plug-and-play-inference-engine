#include "inference_core.h"
#include <iostream>

InferenceCore::InferenceCore(std::unique_ptr<MemoryPool> memory_pool,
                             std::unique_ptr<IPreProcessor> pre_processor,
                             std::unique_ptr<IExecutor> executor,
                             std::unique_ptr<IPostProcessor> post_processor)
    : memory_pool(std::move(memory_pool)),
      pre_processor(std::move(pre_processor)), executor(std::move(executor)),
      post_processor(std::move(post_processor)) {}

InferenceCore::~InferenceCore() = default;

// ---------------------------------------------------------------------------
// Single-image predict (gRPC path — unchanged)
// ---------------------------------------------------------------------------
PredictionResult
InferenceCore::predict(const std::vector<unsigned char> &image_bytes) {
  PredictionResult result;
  result.success = false;
  float *buffer_ptr = nullptr;

  try {
    buffer_ptr = memory_pool->acquire();
    pre_processor->process(image_bytes, buffer_ptr);
    float *output_ptr = executor->run(buffer_ptr);
    result.json_response = post_processor->format(output_ptr);
    delete[] output_ptr;
    result.success = true;
  } catch (const std::exception &e) {
    result.json_response = std::string("{\"error\": \"") + e.what() + "\"}";
  }

  if (buffer_ptr != nullptr) {
    memory_pool->release(buffer_ptr);
  }

  return result;
}

// ---------------------------------------------------------------------------
// Batch predict (Kafka path)
// Allocates one large contiguous buffer: batch_size * 3 * 224 * 224 floats.
// ---------------------------------------------------------------------------
std::vector<PredictionResult> InferenceCore::predictBatch(
    const std::vector<std::vector<unsigned char>> &batch) {
  const int N = static_cast<int>(batch.size());
  const size_t single_size = 3 * 224 * 224;
  const size_t batch_buf_size = N * single_size;

  std::vector<PredictionResult> results(N);
  float *batch_buffer = nullptr;
  float *output_ptr = nullptr;

  try {
    // Single heap allocation for entire batch
    batch_buffer = new float[batch_buf_size];

    std::cout << "[InferenceCore] Pre-processing batch of " << N << " images..."
              << std::endl;
    pre_processor->processBatch(batch, batch_buffer);

    std::cout << "[InferenceCore] Running batched ONNX inference [" << N
              << " x 3 x 224 x 224]..." << std::endl;
    output_ptr = executor->runBatch(batch_buffer, N);

    std::cout << "[InferenceCore] Formatting " << N << " results..."
              << std::endl;
    std::vector<std::string> json_results =
        post_processor->formatBatch(output_ptr, N);

    for (int i = 0; i < N; ++i) {
      results[i].success = true;
      results[i].json_response = json_results[i];
    }
  } catch (const std::exception &e) {
    for (int i = 0; i < N; ++i) {
      results[i].success = false;
      results[i].json_response =
          std::string("{\"error\": \"") + e.what() + "\"}";
    }
  }

  delete[] batch_buffer;
  delete[] output_ptr;
  return results;
}
