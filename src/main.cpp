#include "grpc_adapter.h"
#include "image_preprocessor.h"
#include "inference_core.h"
#include "kafka_adapter.h"
#include "memory_pool.h"
#include "onnx_executor.h"
#include "resnet_postprocessor.h"
#include <iostream>
#include <memory>
#include <thread>

int main() {
  std::cout << "=== Advanced Hybrid Inference Engine Boot Sequence ==="
            << std::endl;

  // 1. Pre-Allocate the 32-bit floats logically during boot up
  size_t resnet_input_elements = 1 * 3 * 224 * 224;
  auto pool = std::make_unique<MemoryPool>(10, resnet_input_elements);

  // 2. Define the exact components forming the mathematical framework pipeline
  auto preprocessor = std::make_unique<ImagePreProcessor>();
  auto executor = std::make_unique<OnnxExecutor>("assets/resnet50.onnx");
  auto postprocessor = std::make_unique<ResNetPostProcessor>();

  // 3. Consolidate memory and logic inside the orchestrator model perfectly
  // cleanly
  InferenceCore engine(std::move(pool), std::move(preprocessor),
                       std::move(executor), std::move(postprocessor));

  // 4. Bind the C++ core execution engine backwards mathematically onto HTTP/2
  // and raw TCP protocols!
  GrpcAdapter grpc_server(engine, "0.0.0.0:50051");
  // Kafka uses port 9092 natively
  KafkaAdapter kafka_server(engine, "localhost:9092", "inference_requests",
                            "inference_results");

  std::cout << "\nDual-Protocol Engine Deployment Success!" << std::endl;
  std::cout << "[Orchestrator] Spawning Kafka Consumer globally onto "
               "independent CPU thread..."
            << std::endl;

  // Isolate massive-throughput polling onto a concurrent background hardware
  // thread natively
  std::thread kafka_thread([&kafka_server]() {
    try {
      kafka_server.start();
    } catch (const std::exception &e) {
      std::cerr << "Kafka Adapter fatally crashed: " << e.what() << std::endl;
    }
  });

  // Isolate synchronous traffic identically onto the main thread (blocking
  // gracefully)
  grpc_server.start();

  // 5. Clean teardown operations whenever the blocking thread releases
  std::cout << "[Orchestrator] Sending interrupt to Kafka Worker Thread..."
            << std::endl;
  kafka_server.stop();
  if (kafka_thread.joinable()) {
    kafka_thread.join();
  }

  return 0;
}
