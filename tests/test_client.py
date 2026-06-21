import grpc
import sys
import time
import inference_pb2
import inference_pb2_grpc

def run():
    print("=== Python gRPC Client ===")
    
    # 1. Read binary image from disk
    try:
        with open("assets/test_image.jpg", "rb") as f:
            image_data = f.read()
            print(f"Loaded {len(image_data)} bytes from test_image.jpg")
    except FileNotFoundError:
        print("Error: Could not find assets/test_image.jpg")
        sys.exit(1)

    # 2. Open HTTP/2 TCP Port connection to our C++ Inference Engine
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        
        # 3. Formulate Protobuf Request
        request = inference_pb2.PredictRequest(image_data=image_data)
        
        print("\nSending bytes over synchronous gRPC...")
        start_time = time.time()
        
        # 4. Trigger synchronous network call!
        response = stub.Predict(request)
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        
        print(f"\n[Response Received in {latency:.2f} ms]")
        print(f"Success: {response.success}")
        print(f"JSON Payload: {response.json_response}")
        
    except grpc.RpcError as e:
        print(f"gRPC failed: {e.details()}")

if __name__ == '__main__':
    run()
