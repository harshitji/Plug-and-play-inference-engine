#pragma once
#include <mutex>
#include <queue>
#include <stdexcept>
#include <vector>

/**
 * @class MemoryPool
 * @brief A high-performance Zero-Copy Memory Pool.
 *
 * Pre-allocates a fixed number of massive float arrays (tensors) dynamically on
 * the heap. Prevents OS `malloc` and `free` latency overhead during real-time
 * inference routing.
 */
class MemoryPool {
private:
  std::queue<float *>
      available_buffers; // Queue holding currently idle memory blocks
  std::vector<float *>
      all_allocated_buffers; // Master list of all blocks to delete accurately
                             // on destruction
  std::mutex mtx;            // Mutex for thread-safety during gRPC concurrency
  size_t pool_size;          // Maximum size of the pool
  size_t buffer_elements;    // Length of the float array for each buffer

public:
  /**
   * @brief Initializes the MemoryPool with pre-calculated tensor buffers.
   * @param pool_size Number of buffers to pre-allocate (e.g., maximum
   * concurrent connections).
   * @param buffer_elements Total continuous capacity per block (e.g., 3 * 224 *
   * 224 floats).
   */
  MemoryPool(size_t pool_size, size_t buffer_elements);

  /**
   * @brief Cleans up the pool, freeing the huge chunks of pre-allocated memory.
   */
  ~MemoryPool();

  /**
   * @brief Acquires a free memory block from the queue in thread-safe manner.
   * @return float* A pointer to the 32-bit float array.
   * @throws std::runtime_error If the pool is exhausted and no buffers remain.
   */
  float *acquire();

  /**
   * @brief Returns a memory block back into the available pool for future
   * consumption.
   * @param ptr The array pointer originally lent out by acquire().
   */
  void release(float *ptr);
};
