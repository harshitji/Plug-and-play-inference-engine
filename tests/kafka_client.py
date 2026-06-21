"""
Kafka end-to-end validation:
1. Subscribe consumer with 'earliest' offset FIRST
2. Wait for group registration
3. Produce the image
4. Poll for result
"""
import time
import threading
from confluent_kafka import Producer, Consumer

result_holder = []

def listen_for_response():
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'py_val_' + str(int(time.time())),
        'auto.offset.reset': 'earliest'  # Read ANY message including ones produced just now
    }
    consumer = Consumer(conf)
    consumer.subscribe(['inference_results'])

    deadline = time.time() + 12.0
    while time.time() < deadline:
        msg = consumer.poll(0.5)
        if msg is None:
            continue
        if msg.error():
            continue
        result_holder.append(msg.value().decode('utf-8'))
        break
    consumer.close()

def run_test():
    print("=== Kafka End-to-End Validation ===\n")

    with open("assets/test_image.jpg", "rb") as f:
        image_data = f.read()
    print(f"Image loaded: {len(image_data)} bytes")

    # Start consumer thread first and give it 2s to register
    t = threading.Thread(target=listen_for_response, daemon=True)
    t.start()
    time.sleep(2)

    # Now produce
    p = Producer({'bootstrap.servers': 'localhost:9092'})
    t_start = time.time()
    p.produce('inference_requests', value=image_data, key="e2e_test")
    p.flush()
    print(f"[Producer] Image sent to 'inference_requests'")

    # Wait up to 12s
    t.join(timeout=12)

    if result_holder:
        ms = (time.time() - t_start) * 1000
        print(f"\n✅ SUCCESS — Full Kafka round-trip in {ms:.0f} ms")
        print(f"   Response: {result_holder[0]}")
    else:
        print("\n❌ No response within timeout.")

if __name__ == '__main__':
    run_test()
