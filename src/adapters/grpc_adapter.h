#pragma once
#include "inference.grpc.pb.h"
#include "inference_core.h"
#include "interfaces.h"
#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>

/**
 * @class InferenceServiceImpl
 * @brief Directly inherits from the auto-generated Protobuf RPC Service schema.
 */
class InferenceServiceImpl final : public inference::InferenceService::Service {
private:
  InferenceCore &core_ref;

public:
  InferenceServiceImpl(InferenceCore &core) : core_ref(core) {}

  grpc::Status Predict(grpc::ServerContext *context,
                       const inference::PredictRequest *request,
                       inference::PredictResponse *response) override;
};

/**
 * @class GrpcAdapter
 * @brief Binds the InferenceServiceImpl to the Operating System's HTTP/2 TCP
 * Port.
 */
class GrpcAdapter : public IAdapter {
private:
  InferenceCore &core_ref;
  std::string server_address;
  std::unique_ptr<grpc::Server> server;
  InferenceServiceImpl service;

public:
  GrpcAdapter(InferenceCore &core, const std::string &address);
  ~GrpcAdapter();

  void start() override;
  void stop() override;
};
