#include "image_preprocessor.h"
#include "inference_core.h"
#include "memory_pool.h"
#include "onnx_executor.h"
#include "resnet_postprocessor.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

std::vector<unsigned char> read_file_bytes(const std::string &filepath) {
  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> buffer(size);
  file.read(reinterpret_cast<char *>(buffer.data()), size);
  return buffer;
}

int main() {
  std::cout << "=== C++ Benchmark ===" << std::endl;

  size_t resnet_input_elements = 1 * 3 * 224 * 224;
  auto pool = std::make_unique<MemoryPool>(10, resnet_input_elements);
  auto preprocessor = std::make_unique<ImagePreProcessor>();
  auto executor = std::make_unique<OnnxExecutor>("assets/resnet50.onnx");
  auto postprocessor = std::make_unique<ResNetPostProcessor>();

  InferenceCore engine(std::move(pool), std::move(preprocessor),
                       std::move(executor), std::move(postprocessor));
  std::vector<unsigned char> image_bytes =
      read_file_bytes("assets/test_image.jpg");

  // 1. Warmup the C++ OS memory links
  std::cout.setstate(std::ios_base::failbit); // Disable deep pipeline couts
  for (int i = 0; i < 5; ++i) {
    engine.predict(image_bytes);
  }

  // 2. Perform 100 continuous executions identically testing the pipeline
  // limits End-to-End
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) {
    engine.predict(image_bytes);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::cout.clear(); // Re-enable cout

  double total_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "C++ Average Latency: " << (total_ms / 100.0) << " ms"
            << std::endl;

  return 0;
}
