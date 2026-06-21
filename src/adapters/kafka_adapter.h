#pragma once
#include "../core/inference_core.h"
#include "../core/interfaces.h"
#include <atomic>
#include <librdkafka/rdkafkacpp.h>
#include <memory>
#include <string>

/**
 * @class KafkaAdapter
 * @brief Implements IAdapter over Apache Kafka with dynamic batching.
 *
 * Collects up to MAX_BATCH_SIZE messages from `consume_topic` (or fires after
 * BATCH_TIMEOUT_MS) and passes the entire batch to
 * InferenceCore::predictBatch() in a single ONNX session call, significantly
 * improving CPU throughput.
 */
class KafkaAdapter : public IAdapter {
public:
  static constexpr int MAX_BATCH_SIZE = 8; ///< Max images per ONNX call
  static constexpr int BATCH_TIMEOUT_MS =
      200; ///< Max wait before partial flush

private:
  InferenceCore &core_ref;
  std::string brokers;
  std::string consume_topic;
  std::string produce_topic;
  std::unique_ptr<RdKafka::KafkaConsumer> consumer;
  std::unique_ptr<RdKafka::Producer> producer;
  std::atomic<bool> is_running;

public:
  KafkaAdapter(InferenceCore &core, const std::string &brokers,
               const std::string &in_topic, const std::string &out_topic);
  ~KafkaAdapter();

  void start() override;
  void stop() override;
};
