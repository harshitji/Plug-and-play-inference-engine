#pragma once
#include "interfaces.h"
#include <string>
#include <vector>

/**
 * @class ResNetPostProcessor
 * @brief Concrete IPostProcessor for 1000-class ImageNet outputs.
 *        Supports single-image argmax and batch argmax over N rows.
 */
class ResNetPostProcessor : public IPostProcessor {
public:
  std::string format(float *output_tensor_ptr) override;
  std::vector<std::string> formatBatch(float *output_tensor_ptr,
                                       int batch_size) override;
};
