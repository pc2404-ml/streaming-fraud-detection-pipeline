from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import time

print("Starting Raw Data Consumer...")

# Create Spark
spark = SparkSession.builder.appName("RawDataConsumer").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Original schema - no modifications
fraud_schema = StructType([
    StructField("username", StringType()),
    StructField("full_name", StringType()),
    StructField("email", StringType()),
    StructField("phone", StringType()),
    StructField("transaction_amount", DoubleType()),
    StructField("merchant", StringType()),
    StructField("card_last_4", StringType()),
    StructField("card_provider", StringType()),
    StructField("country", StringType()),
    StructField("address", StringType()),
    StructField("city", StringType()),
    StructField("state", StringType()),
    StructField("zipcode", StringType()),
    StructField("transaction_date", StringType()),
    StructField("transaction_hour", IntegerType()),
    StructField("is_fraud", IntegerType())
])

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "fraud-topic") \
    .option("startingOffsets", "earliest") \
    .load()

# Parse JSON only - no feature engineering
parsed_df = df.select(
    from_json(col("value").cast("string"), fraud_schema).alias("data")
).select("data.*")
#drop duplicates
parsed_df = parsed_df.dropDuplicates()
print("JSON parsing configured!")

# Write raw data to CSV - UPDATED: Fixed for no duplicates
query = parsed_df.coalesce(1).writeStream \
    .outputMode("append") \
    .format("csv") \
    .option("path", "/output/training_data") \
    .option("header", "true") \
    .option("checkpointLocation", "/tmp/checkpoint") \
    .trigger(processingTime='5 seconds') \
    .start()

print(" Reading all data at once...")
print("Output: /output/training_data/")
print("Processing all available data...")

# UPDATED: Wait for completion instead of fixed time
query.awaitTermination(timeout=300)

print("Processing completed!")

# Check results
print("Checking results...")
try:
    result = spark.read.csv("/output/training_data", header=True, inferSchema=True)
    total = result.count()

    print(f"SUCCESS!")
    print(f"Total records written: {total}")

except Exception as e:
    print(f" Error: {e}")

spark.stop()
print("Raw data extraction complete!")