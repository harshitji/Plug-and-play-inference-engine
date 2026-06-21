#include "kafka_adapter.h"
#include <chrono>
#include <iostream>
#include <vector>

KafkaAdapter::KafkaAdapter(InferenceCore &core, const std::string &brokers,
                           const std::string &in_topic,
                           const std::string &out_topic)
    : core_ref(core), brokers(brokers), consume_topic(in_topic),
      produce_topic(out_topic), is_running(false) {

  std::string errstr;

  // Consumer configuration
  RdKafka::Conf *conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
  conf->set("metadata.broker.list", brokers, errstr);
  conf->set("group.id", "inference_engine_cg", errstr);
  conf->set("auto.offset.reset", "latest", errstr);
  consumer.reset(RdKafka::KafkaConsumer::create(conf, errstr));
  if (!consumer)
    throw std::runtime_error("Failed to create consumer: " + errstr);
  delete conf;

  // Producer configuration
  RdKafka::Conf *p_conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
  p_conf->set("metadata.broker.list", brokers, errstr);
  producer.reset(RdKafka::Producer::create(p_conf, errstr));
  if (!producer)
    throw std::runtime_error("Failed to create producer: " + errstr);
  delete p_conf;
}

KafkaAdapter::~KafkaAdapter() { stop(); }

void KafkaAdapter::stop() {
  is_running = false;
  if (consumer)
    consumer->close();
}

void KafkaAdapter::start() {
  std::string errstr;
  consumer->subscribe({consume_topic});
  std::cout << "[KafkaAdapter] Listening on topic {" << consume_topic
            << "} | max_batch=" << MAX_BATCH_SIZE
            << " | timeout=" << BATCH_TIMEOUT_MS << "ms" << std::endl;
  is_running = true;

  // Batch state
  std::vector<std::vector<unsigned char>> batch_images;
  std::vector<std::string> batch_keys;
  batch_images.reserve(MAX_BATCH_SIZE);
  batch_keys.reserve(MAX_BATCH_SIZE);

  auto batch_start = std::chrono::steady_clock::now();

  while (is_running) {
    // Poll with short timeout so we can check max-batch and deadline
    std::unique_ptr<RdKafka::Message> msg(consumer->consume(50));

    if (msg->err() == RdKafka::ERR_NO_ERROR) {
      const auto *raw = static_cast<const unsigned char *>(msg->payload());
      batch_images.emplace_back(raw, raw + msg->len());
      batch_keys.push_back(msg->key() ? *msg->key() : "unknown");

      std::cout << "[KafkaAdapter] Queued message " << batch_images.size()
                << "/" << MAX_BATCH_SIZE << std::endl;
    }

    // Decide whether to flush the batch
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - batch_start)
                          .count();

    bool batch_full = (int)batch_images.size() >= MAX_BATCH_SIZE;
    bool timeout_hit =
        (elapsed_ms >= BATCH_TIMEOUT_MS) && !batch_images.empty();

    if (batch_full || timeout_hit) {
      const int N = (int)batch_images.size();
      std::cout << "\n[KafkaAdapter] Firing batch inference: N=" << N
                << std::endl;

      std::vector<PredictionResult> results =
          core_ref.predictBatch(batch_images);

      // Publish each result with its original request key
      for (int i = 0; i < N; ++i) {
        producer->produce(produce_topic, RdKafka::Topic::PARTITION_UA,
                          RdKafka::Producer::RK_MSG_COPY,
                          const_cast<char *>(results[i].json_response.c_str()),
                          results[i].json_response.size(),
                          batch_keys[i].c_str(), batch_keys[i].size(), 0,
                          nullptr);
      }
      producer->flush(1000);

      std::cout << "[KafkaAdapter] Batch of " << N << " results published to {"
                << produce_topic << "}" << std::endl;

      // Reset batch state
      batch_images.clear();
      batch_keys.clear();
      batch_start = std::chrono::steady_clock::now();
    }
  }
}
