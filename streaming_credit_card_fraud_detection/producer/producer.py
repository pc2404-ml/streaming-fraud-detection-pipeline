from kafka import KafkaProducer
import json
import time
import csv
import os

print("Starting Fraud Detection Kafka Producer...")

# Wait for Kafka to be ready
print("Waiting for Kafka to be ready...")
time.sleep(30)

# Simple producer
producer = KafkaProducer(
    bootstrap_servers=['kafka:9092'],  # Use internal Docker network
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

print("Kafka Producer connected successfully!")


print("Script file path:", os.path.abspath(__file__))

# Update CSV path to your fraud data file
csv_path = os.path.join(os.path.dirname(__file__), "data", "credit_card_fraud_raw_data.csv")
print("Fraud data file path:", csv_path)

messages = []

try:
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert string values to correct types for fraud data
            row['transaction_amount'] = float(row['transaction_amount'])
            row['transaction_hour'] = int(row['transaction_hour'])
            row['is_fraud'] = int(row['is_fraud'])
            messages.append(row)

    print(f"Loaded {len(messages)} fraud detection records from CSV")

    # Show sample record
    if messages:
        sample = messages[0]
        print(
            f"Sample record: User={sample['username']}, Amount=${sample['transaction_amount']}, Fraud={sample['is_fraud']}")

except FileNotFoundError:
    print(f"Error: Could not find file {csv_path}")
    print("Make sure you have generated the fraud data CSV file first!")
    exit(1)

print(f"Sending {len(messages)} fraud detection messages to Kafka topic 'fraud-topic'...")

# Send messages with progress indicator
for i, msg in enumerate(messages, 1):
    producer.send('fraud-topic', msg)  # Changed topic name to 'fraud-topic'

    # Show progress for every 100 messages
    if i % 100 == 0:
        print(f"  ✓ Sent {i}/{len(messages)} transactions...")

    # Optional: Add small delay to simulate real-time streaming
    time.sleep(0.01)  # 10ms delay

# Show final few messages for verification
print(f"\nLast few transactions sent:")
for msg in messages[-3:]:
    fraud_status = "FRAUD" if msg['is_fraud'] == 1 else "NORMAL"
    print(f"  {fraud_status}: {msg['username']} - ${msg['transaction_amount']} at {msg['merchant']}")

producer.flush()
producer.close()

print(f"\nAll {len(messages)} fraud detection messages sent successfully!")
print("Producer finished. Fraud transactions are now streaming in Kafka.")
print("Topic: fraud-topic")
print("Ready for real-time fraud detection processing!")