#include "memory_pool.h"

MemoryPool::MemoryPool(size_t pool_size, size_t buffer_elements)
    : pool_size(pool_size), buffer_elements(buffer_elements) {

  // During boot initialization, we ask the OS for memory ONCE.
  // We allocate hundreds of huge blocks simultaneously so we never have to
  // ask for memory during the time-critical inference stage.
  for (size_t i = 0; i < pool_size; ++i) {
    float *buffer = new float[buffer_elements];
    available_buffers.push(buffer);
    all_allocated_buffers.push_back(buffer);
  }
}

MemoryPool::~MemoryPool() {
  // We iterate over the master list of all allocated blocks to prevent memory
  // leaks even if some buffers were currently loaned out and missing from the
  // idle queue.
  for (float *buffer : all_allocated_buffers) {
    delete[] buffer;
  }
}

float *MemoryPool::acquire() {
  // Thread safety is strictly required because Kafka and gRPC listeners might
  // simultaneously ask for memory from different processor threads.
  std::lock_guard<std::mutex> lock(mtx);
  if (available_buffers.empty()) {
    throw std::runtime_error("Memory Pool Exhausted! Too many concurrent "
                             "requests competing for buffers.");
  }

  // Grab the first available block in array
  float *ptr = available_buffers.front();
  available_buffers.pop();
  return ptr;
}

void MemoryPool::release(float *ptr) {
  std::lock_guard<std::mutex> lock(mtx);
  // Note: We deliberately bypass clearing `memset(ptr, 0, size)` here for pure
  // performance. Instead, we fundamentally trust the OpenCV PreProcessor to
  // perfectly overwrite all pixels inside this buffer linearly when it begins
  // decoding a new image.
  available_buffers.push(ptr);
}
