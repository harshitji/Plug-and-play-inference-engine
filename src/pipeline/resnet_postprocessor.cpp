#include "resnet_postprocessor.h"
#include <iostream>
#include <string>

// ---------------------------------------------------------------------------
// Single-image (gRPC path)
// ---------------------------------------------------------------------------
std::string ResNetPostProcessor::format(float *output_tensor_ptr) {
  int max_index = 0;
  float max_val = output_tensor_ptr[0];
  for (int i = 1; i < 1000; ++i) {
    if (output_tensor_ptr[i] > max_val) {
      max_val = output_tensor_ptr[i];
      max_index = i;
    }
  }
  std::cout << "[ResNetPostProcessor] Result Parsed!" << std::endl;
  return "{\"status\": \"success\", \"class_id\": " +
         std::to_string(max_index) +
         ", \"confidence\": " + std::to_string(max_val) + "}";
}

// ---------------------------------------------------------------------------
// Batch path — argmax over each row of N*1000 output tensor
// ---------------------------------------------------------------------------
std::vector<std::string>
ResNetPostProcessor::formatBatch(float *output_tensor_ptr, int batch_size) {
  std::vector<std::string> results;
  results.reserve(batch_size);

  for (int n = 0; n < batch_size; ++n) {
    float *row = output_tensor_ptr + n * 1000;
    int max_index = 0;
    float max_val = row[0];
    for (int i = 1; i < 1000; ++i) {
      if (row[i] > max_val) {
        max_val = row[i];
        max_index = i;
      }
    }
    results.push_back(
        "{\"status\": \"success\", \"class_id\": " + std::to_string(max_index) +
        ", \"confidence\": " + std::to_string(max_val) + "}");
  }

  std::cout << "[ResNetPostProcessor] Batch of " << batch_size
            << " results parsed." << std::endl;
  return results;
}
