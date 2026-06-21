#include "image_preprocessor.h"
#include "inference_core.h"
#include "memory_pool.h"
#include "onnx_executor.h"
#include "resnet_postprocessor.h"
#include <fstream>
#include <iostream>
#include <vector>

std::vector<unsigned char> read_file_bytes(const std::string &filepath) {
  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Could not open file: " + filepath);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> buffer(size);
  if (file.read(reinterpret_cast<char *>(buffer.data()), size)) {
    return buffer;
  }
  throw std::runtime_error("Failed to read file.");
}

int main() {
  std::cout << "=== Hybrid Inference Engine Complete OpenCV Test ==="
            << std::endl;

  // Core Engine Composition (Dependency Injection)
  size_t resnet_input_elements = 1 * 3 * 224 * 224;
  auto pool = std::make_unique<MemoryPool>(10, resnet_input_elements);
  auto preprocessor = std::make_unique<ImagePreProcessor>();
  auto executor = std::make_unique<OnnxExecutor>("assets/resnet50.onnx");
  auto postprocessor = std::make_unique<ResNetPostProcessor>();

  // Pass implementations to the generic core
  InferenceCore engine(std::move(pool), std::move(preprocessor),
                       std::move(executor), std::move(postprocessor));

  try {
    std::cout << "Reading test_image.jpg from disk..." << std::endl;
    std::vector<unsigned char> image_bytes =
        read_file_bytes("assets/test_image.jpg");
    std::cout << "\n[Engine] Received " << image_bytes.size()
              << " bytes. Running Prediction Sequence..." << std::endl;

    PredictionResult result = engine.predict(image_bytes);

    if (result.success) {
      std::cout << "[Engine] Prediction Sequence Successful!\n[Engine] Final "
                   "JSON Response: "
                << result.json_response << std::endl;
    } else {
      std::cerr << "[Engine] Prediction Failed: " << result.json_response
                << std::endl;
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
