#pragma once
#include "interfaces.h"
#include "memory_pool.h"
#include <memory>
#include <string>
#include <vector>

struct PredictionResult {
  bool success;
  std::string json_response;
};

class InferenceCore {
private:
  std::unique_ptr<MemoryPool> memory_pool;
  std::unique_ptr<IPreProcessor> pre_processor;
  std::unique_ptr<IExecutor> executor;
  std::unique_ptr<IPostProcessor> post_processor;

public:
  InferenceCore(std::unique_ptr<MemoryPool> memory_pool,
                std::unique_ptr<IPreProcessor> pre_processor,
                std::unique_ptr<IExecutor> executor,
                std::unique_ptr<IPostProcessor> post_processor);
  ~InferenceCore();

  /// Single-image inference (gRPC path).
  PredictionResult predict(const std::vector<unsigned char> &image_bytes);

  /// Batch inference (Kafka path). Returns one PredictionResult per image.
  std::vector<PredictionResult>
  predictBatch(const std::vector<std::vector<unsigned char>> &batch);
};
