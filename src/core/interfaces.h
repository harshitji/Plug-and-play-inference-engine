#pragma once
#include <string>
#include <vector>

/**
 * @brief Strategy Interface for evaluating raw bytes into native math buffers.
 *        Both single-image and batch variants are supported.
 */
class IPreProcessor {
public:
  virtual ~IPreProcessor() = default;
  /// Single-image path (used by gRPC adapter).
  virtual void process(const std::vector<unsigned char> &data_bytes,
                       float *buffer_ptr) = 0;
  /// Batch path: each image is written contiguously into buffer_ptr.
  /// buffer_ptr must have capacity of batch_size * 3 * 224 * 224 floats.
  virtual void
  processBatch(const std::vector<std::vector<unsigned char>> &batch,
               float *buffer_ptr) = 0;
};

/**
 * @brief Strategy Interface for Native Acceleration Engines (ONNX, TensorRT,
 * etc).
 */
class IExecutor {
public:
  virtual ~IExecutor() = default;
  /// Single-image inference. Returns heap-allocated float[1000].
  virtual float *run(float *input_tensor_ptr) = 0;
  /// Batch inference. Input shape: [N, 3, 224, 224]. Returns heap-allocated
  /// float[N*1000].
  virtual float *runBatch(float *input_tensor_ptr, int batch_size) = 0;
};

/**
 * @brief Strategy Interface mapping Tensor Mathematics into JSON/String
 * responses.
 */
class IPostProcessor {
public:
  virtual ~IPostProcessor() = default;
  /// Single-image result.
  virtual std::string format(float *output_tensor_ptr) = 0;
  /// Batch result: returns one JSON string per image.
  virtual std::vector<std::string> formatBatch(float *output_tensor_ptr,
                                               int batch_size) = 0;
};

/**
 * @brief Strategy Interface ensuring all networking Adapters can identically be
 * booted.
 */
class IAdapter {
public:
  virtual ~IAdapter() = default;
  virtual void start() = 0;
  virtual void stop() = 0;
};
