# Setup Guide: Hybrid C++ Inference Engine

This guide provides step-by-step instructions for setting up the production-grade Hybrid Inference Engine on a macOS (arm64) environment.

---

## 1. Prerequisites

Ensure you have the following installed on your system:
- **Homebrew:** For managing dependencies.
- **CMake (3.20+):** Build system generator.
- **Apple Clang / GCC:** C++17 compliant compiler.
- **Python 3.9+:** For running verification clients.

---

## 2. Install Dependencies

Run the following commands to install the required C++ libraries via Homebrew:

```bash
# Core Dependencies
brew install opencv
brew install onnxruntime

# Networking & Communication
brew install grpc
brew install protobuf
brew install librdkafka
brew install re2
```

---

## 3. Kafka Infrastructure Setup

The engine uses Apache Kafka for asynchronous processing. Start the Kafka and Zookeeper services natively:

```bash
# Start Kafka (automatically starts Zookeeper as a dependency)
brew services start kafka

# Verify services are running
brew services list | grep kafka

# Ensure topics exist (KafkaAdapter will auto-create if configured, but manual is safer)
kafka-topics --create --topic inference_requests --bootstrap-server localhost:9092 --if-not-exists
kafka-topics --create --topic inference_results --bootstrap-server localhost:9092 --if-not-exists
```

---

## 4. Build the Project

Clone the repository and build using CMake:

```bash
# Navigate to project root
cd /Users/harshitdubey/WorkSpace/cpp-inference-engine/

# Generate build files and compile
cmake -B build
cmake --build build
```

The build process will generate the following executables in the `build/` directory:
- `hybrid_engine_server`: The main multi-protocol inference server.
- `hybrid_engine_benchmark`: C++ performance benchmarking tool.
- `hybrid_engine_cv_test`: OpenCV integration test.

---

## 4.5 Asset Preparation

The engine expects model and test assets in the `assets/` directory:

1.  **Model Export:** If `resnet50.onnx` is missing, run the export script (requires `torch`, `torchvision`, `onnx`):
    ```bash
    python3 tests/export_model.py
    mv resnet50.onnx assets/
    ```
2.  **Test Image:** Ensure a valid JPEG exists at `assets/test_image.jpg`.

---

## 5. Running the Inference Engine

Start the server to listen for gRPC (Port 50051) and Kafka (Port 9092) requests:

```bash
./build/hybrid_engine_server
```

---

## 6. Verification & Testing

Use the provided Python clients to verify the engine's functionality.

### Build and Activate Virtual Environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install grpcio grpcio-tools confluent-kafka opencv-python
```

### Run Tests:
| Test Type | Command | Description |
|---|---|---|
| **gRPC** | `python tests/test_client.py` | Synchronous real-time inference. |
| **Kafka** | `python tests/kafka_client.py` | Asynchronous single-image inference. |
| **Batching**| `python tests/batch_client.py`| High-throughput dynamic batching test. |

---

## 7. Project Structure

- `src/core/`: Orchestrator (`InferenceCore`) and Interfaces.
- `src/pipeline/`: ONNX, OpenCV, and Post-processing logic.
- `src/adapters/`: gRPC and Kafka network adapters.
- `proto/`: Protobuf service definitions.
- `assets/`: Model files (`.onnx`) and test images.
- `tests/`: Benchmarking and verification scripts.

---

## 8. Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| **Kafka Connection Failed** | Broker not running. | `brew services restart kafka` |
| **gRPC Port Busy (50051)** | Previous server still running. | `lsof -ti:50051 | xargs kill -9` |
| **ONNX Runtime Error** | Model not found in `assets/`. | Run `export_model.py` and move to `assets/`. |
| **OpenCV cv::imdecode fails** | Corrupt/missing image. | Check `assets/test_image.jpg`. |
| **Kafka Timeout (Python)** | Topic offset misalignment. | Use `latest` vs `earliest` in `kafka_client.py`. |
