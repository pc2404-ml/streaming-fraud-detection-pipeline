import subprocess
import time
import os


class KafkaPipeline:
    def __init__(self):
        self.kafka_container = "streaming_credit_card_fraud_detection-kafka-1"

    def cleanup_and_start_docker(self):
        """Step 1: Clean up and start Docker services"""
        print("=== Kafka Pipeline ===")
        print()

        print("Cleaning up any existing containers...")
        subprocess.run("docker-compose down", shell=True)

        print("Starting all services...")
        subprocess.run("docker-compose up -d", shell=True)

        print("Waiting 45 seconds for all services to initialize...")
        time.sleep(45)

        print("Checking running containers...")
        subprocess.run('docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}"', shell=True)

    def generate_synthetic_data(self):
        """Step 2: Generate synthetic fraud data"""
        print()
        print("Generating fraud data...")

        if os.path.exists("/data/data_generate.py"):
            subprocess.run("python data/data_generate.py", shell=True)
            print("Data generated!")
        else:
            print("data_generate.py not found, using existing data")

    def create_kafka_topics(self):
        """Step 3: Create Kafka topics"""
        print()
        print("Creating Kafka topics...")

        # Create fraud-topic
        subprocess.run(
            f"docker exec {self.kafka_container} kafka-topics --create --topic fraud-topic --bootstrap-server kafka:9092 --partitions 1 --replication-factor 1 --if-not-exists",
            shell=True)

        print("Topics created! Verifying...")
        subprocess.run(f"docker exec {self.kafka_container} kafka-topics --list --bootstrap-server kafka:9092",
                       shell=True)

    def run_producer(self):
        """Step 4: Start fraud detection producer using docker-compose"""
        print()
        print("Starting fraud detection producer...")

        # Use docker-compose to run producer (your working method)
        subprocess.run("docker-compose run --rm producer", shell=True)

    def run_consumer(self):
        """Step 5: Starting Spark fraud consumer (your existing consumer.py)"""
        print()
        print("Starting Spark fraud consumer...")

        # Use your existing consumer with docker-compose (like in run.sh)
        subprocess.run(
            "docker-compose run --rm spark spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 consumer.py",
            shell=True)

    def show_results(self):
        """Step 6: Show what was produced and consumed"""
        print()
        print("Pipeline Results:")

        print("Check your output files:")
        print("   ls -la ./output/")
        subprocess.run("ls -la ./output/", shell=True)

        print()
        print("View CSV content:")
        subprocess.run("ls ./output/training_data/part-*.csv 2>/dev/null | head -1 | xargs cat | head -10", shell=True)
    def model_training(self):
        """Step 7: Run training the model"""
        print()
        print("Running training the model...")
        try:
            subprocess.run("python model/model_train.py", shell=True, check=True)
            print("Model training completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Model training failed: {e}")

    def step8_prediction(self):
        """Step 8: Run prediction using best model"""
        print()
        print("Running predictions using best model...")


        # Run your prediction.py which uses the best model
        try:
            subprocess.run("python model/prediction.py", shell=True, check=True)
            print("Predictions completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Prediction failed: {e}")

    def step9_cleanup(self):
        """Step 7: Cleanup"""
        print()
        print("Cleaning up...")
        subprocess.run("docker-compose down", shell=True)
        print("All services stopped")

    def run_kafka_pipeline(self):
        """Run basic Kafka pipeline"""
        try:
            self.cleanup_and_start_docker()
            self.generate_synthetic_data()
            self.create_kafka_topics()
            self.run_producer()
            self.run_consumer()
            self.show_results()
            self.model_training()
            self.step8_prediction()
            self.step9_cleanup()

        except Exception as e:
            print(f"Pipeline failed: {e}")
        finally:
            self.step9_cleanup()


def main():
    pipeline = KafkaPipeline()
    pipeline.run_kafka_pipeline()


if __name__ == "__main__":
    main()