import time
import cv2
import numpy as np
import onnxruntime as ort

def benchmark():
    print("=== Python Benchmark ===")
    
    # 1. Setup ONNX (identical to our C++ thread configuration)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    session = ort.InferenceSession("assets/resnet50.onnx", session_options)
    
    with open("assets/test_image.jpg", "rb") as f:
        image_bytes = f.read()

    # 2. Warm up the GPU / CPU bindings (load caches)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    for _ in range(5):
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        resized = cv2.resize(img, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - mean) / std
        chw = np.transpose(rgb, (2, 0, 1))
        tensor = np.expand_dims(chw, axis=0)
        _ = session.run(["output"], {"input": tensor})
    
    # 3. Time 100 sequential inferences covering the ENTIRE pipeline (decode -> resize -> infer)
    start_time = time.time()
    for _ in range(100):
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        resized = cv2.resize(img, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - mean) / std
        chw = np.transpose(rgb, (2, 0, 1))
        tensor = np.expand_dims(chw, axis=0)
        _ = session.run(["output"], {"input": tensor})
    end_time = time.time()
    
    avg_latency = ((end_time - start_time) / 100) * 1000
    print(f"Python Average Latency: {avg_latency:.2f} ms")

if __name__ == "__main__":
    benchmark()
