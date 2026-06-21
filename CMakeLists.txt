cmake_minimum_required(VERSION 3.10)
project(HybridInferenceEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

set(CMAKE_PREFIX_PATH "/opt/homebrew" ${CMAKE_PREFIX_PATH})

find_package(OpenCV REQUIRED)
find_package(gRPC REQUIRED)

find_library(ONNXRUNTIME_LIB onnxruntime HINTS /opt/homebrew/lib)
find_library(RDKAFKA_LIB rdkafka++ HINTS /opt/homebrew/lib)

include_directories(
    /opt/homebrew/include 
    /opt/homebrew/include/onnxruntime 
    ${OpenCV_INCLUDE_DIRS}
    src/core
    src/pipeline
    src/adapters
    src/
)

# Protobuf C++ Headers mapped statically
set(PROTO_SRC src/adapters/inference.pb.cc src/adapters/inference.grpc.pb.cc)

# Production Dual-Protocol Network Adapter Binary
add_executable(hybrid_engine_server
    src/main.cpp
    src/core/inference_core.cpp
    src/core/memory_pool.cpp
    src/pipeline/image_preprocessor.cpp
    src/pipeline/onnx_executor.cpp
    src/pipeline/resnet_postprocessor.cpp
    src/adapters/grpc_adapter.cpp
    src/adapters/kafka_adapter.cpp
    ${PROTO_SRC}
)

target_link_libraries(hybrid_engine_server
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    ${RDKAFKA_LIB}
    gRPC::grpc++
    protobuf::libprotobuf
)

# Benchmark Executable
add_executable(hybrid_engine_benchmark
    tests/benchmark_cpp.cpp
    src/core/inference_core.cpp
    src/core/memory_pool.cpp
    src/pipeline/image_preprocessor.cpp
    src/pipeline/onnx_executor.cpp
    src/pipeline/resnet_postprocessor.cpp
)

target_link_libraries(hybrid_engine_benchmark
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
)

# Test Executable
add_executable(hybrid_engine_cv_test
    tests/main_test.cpp
    src/core/inference_core.cpp
    src/core/memory_pool.cpp
    src/pipeline/image_preprocessor.cpp
    src/pipeline/onnx_executor.cpp
    src/pipeline/resnet_postprocessor.cpp
)

target_link_libraries(hybrid_engine_cv_test
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
)
