#include "image_preprocessor.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>

// Reusable single-image preprocessing into a float buffer offset
static void preprocess_one(const std::vector<unsigned char> &image_bytes,
                           float *dst) {
  cv::Mat img = cv::imdecode(image_bytes, cv::IMREAD_COLOR);
  if (img.empty()) {
    throw std::runtime_error("Failed to decode image bytes.");
  }

  cv::Mat resized;
  cv::resize(img, resized, cv::Size(224, 224));

  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

  cv::Scalar mean(0.485, 0.456, 0.406);
  cv::Scalar std_dev(0.229, 0.224, 0.225);
  cv::subtract(rgb, mean, rgb);
  cv::divide(rgb, std_dev, rgb);

  std::vector<cv::Mat> channels(3);
  cv::split(rgb, channels);

  const size_t channel_size = 224 * 224;
  for (int c = 0; c < 3; ++c) {
    std::memcpy(dst + c * channel_size, channels[c].data,
                channel_size * sizeof(float));
  }
}

// ---------------------------------------------------------------------------
// Single-image (gRPC path)
// ---------------------------------------------------------------------------
void ImagePreProcessor::process(const std::vector<unsigned char> &image_bytes,
                                float *buffer_ptr) {
  std::cout << "[ImagePreProcessor] Decoding image of size "
            << image_bytes.size() << " bytes..." << std::endl;
  preprocess_one(image_bytes, buffer_ptr);
  std::cout
      << "[ImagePreProcessor] OpenCV Image Decode and Standardize Finished."
      << std::endl;
}

// ---------------------------------------------------------------------------
// Batch path (Kafka)
// Writes N images contiguously: image0[3*224*224] | image1[3*224*224] | ...
// ---------------------------------------------------------------------------
void ImagePreProcessor::processBatch(
    const std::vector<std::vector<unsigned char>> &batch, float *buffer_ptr) {
  const size_t single_size = 3 * 224 * 224;
  for (size_t i = 0; i < batch.size(); ++i) {
    std::cout << "[ImagePreProcessor] Batch[" << i << "] decoding "
              << batch[i].size() << " bytes..." << std::endl;
    preprocess_one(batch[i], buffer_ptr + i * single_size);
  }
  std::cout << "[ImagePreProcessor] Batch preprocessing complete." << std::endl;
}
