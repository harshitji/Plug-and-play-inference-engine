"""
Kafka batch validation: sends 4 images rapidly, expects 4 results back
from a single batched ONNX inference call on the C++ server.
"""
import time, threading
from confluent_kafka import Producer, Consumer

NUM_MESSAGES = 4
results = []
lock = threading.Lock()

def listen():
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'batch_val_' + str(int(time.time())),
        'auto.offset.reset': 'earliest'
    }
    c = Consumer(conf)
    c.subscribe(['inference_results'])
    deadline = time.time() + 15
    while time.time() < deadline:
        msg = c.poll(0.3)
        if msg and not msg.error():
            with lock:
                results.append(msg.value().decode())
                if len(results) >= NUM_MESSAGES:
                    break
    c.close()

def run():
    print(f"=== Batch Kafka Validation: sending {NUM_MESSAGES} images ===\n")
    with open("assets/test_image.jpg", "rb") as f:
        image_data = f.read()

    t = threading.Thread(target=listen, daemon=True)
    t.start()
    time.sleep(1.5)   # let consumer register

    p = Producer({'bootstrap.servers': 'localhost:9092'})
    t_start = time.time()
    for i in range(NUM_MESSAGES):
        p.produce('inference_requests', value=image_data, key=f"img_{i}")
    p.flush()
    print(f"[Producer] {NUM_MESSAGES} messages sent.\n")

    t.join(timeout=15)
    elapsed = (time.time() - t_start) * 1000

    print(f"Results received: {len(results)}/{NUM_MESSAGES}")
    for i, r in enumerate(results):
        print(f"  [{i}] {r}")
    if len(results) == NUM_MESSAGES:
        print(f"\n✅ Batch round-trip complete in {elapsed:.0f} ms  ({elapsed/NUM_MESSAGES:.0f} ms/image avg)")
    else:
        print("\n❌ Not all results received within timeout.")

if __name__ == '__main__':
    run()
