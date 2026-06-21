#pragma once
#include "interfaces.h"
#include <vector>

/**
 * @class ImagePreProcessor
 * @brief Concrete IPreProcessor utilizing OpenCV. Supports both single-image
 *        and batch preprocessing into a contiguous float buffer.
 */
class ImagePreProcessor : public IPreProcessor {
public:
  void process(const std::vector<unsigned char> &image_bytes,
               float *buffer_ptr) override;

  /// Batch: writes N images contiguously [N * 3 * 224 * 224] into buffer_ptr.
  void processBatch(const std::vector<std::vector<unsigned char>> &batch,
                    float *buffer_ptr) override;
};
