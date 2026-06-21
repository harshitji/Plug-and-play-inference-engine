#include "grpc_adapter.h"
#include <iostream>

grpc::Status
InferenceServiceImpl::Predict(grpc::ServerContext *context,
                              const inference::PredictRequest *request,
                              inference::PredictResponse *response) {
  // 1. Convert the exact remote Protobuf byte stream into standard C++ vectors
  const std::string &raw_bytes = request->image_data();
  std::vector<unsigned char> image_data(raw_bytes.begin(), raw_bytes.end());

  std::cout << "[gRPC] Received inference network request: "
            << image_data.size() << " bytes." << std::endl;

  // 2. Drop the bytes directly into the heavily optimized Hexagonal Core
  PredictionResult core_result = core_ref.predict(image_data);

  // 3. Serialize back over the wire
  response->set_success(core_result.success);
  response->set_json_response(core_result.json_response);

  if (core_result.success) {
    return grpc::Status::OK;
  } else {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Inference Core Internal Error");
  }
}

GrpcAdapter::GrpcAdapter(InferenceCore &core, const std::string &address)
    : core_ref(core), server_address(address), service(core) {}

GrpcAdapter::~GrpcAdapter() { stop(); }

void GrpcAdapter::start() {
  grpc::ServerBuilder builder;

  // We bind over insecure HTTP/2 for raw speed locally.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  // Engage the network port bindings
  server = builder.BuildAndStart();
  std::cout << "[GrpcAdapter] Synchronous RPC TCP Server listening natively on "
            << server_address << std::endl;

  // In Phase 2, this blocks the main thread to constantly poll inbound network
  // requests!
  server->Wait();
}

void GrpcAdapter::stop() {
  if (server) {
    server->Shutdown();
    server.reset();
  }
}
