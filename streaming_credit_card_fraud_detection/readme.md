# Streaming Credit Card Fraud Detection Pipeline

A complete end-to-end machine learning pipeline for real-time fraud detection using Kafka streaming, Spark processing, and trained ML models.

## Project Overview

This system processes credit card transactions in real-time through a streaming pipeline that:
1. Generates synthetic transaction data
2. Streams data through Kafka
3. Processes transactions with Spark
4. Applies feature engineering and ML models for fraud prediction
5. Provides both batch and real-time prediction capabilities

## Architecture

```
Data Generation → Kafka Producer → Kafka Topic → Spark Consumer → Feature Engineering → ML Model → Predictions
```

## Project Structure

```
streaming_credit_card_fraud_detection/
├── consumer/
│   └── consumer.py                      # Spark consumer for reading Kafka streams
├── data/
│   ├── credit_card_fraud_data.csv       # Raw training data
│   ├── credit_card_fraud_fe_updated.csv # Feature-engineered training data
│   └── data_generate.py                 # Synthetic data generation
├── eda_notebooks/
│   ├── EDA_notebook.ipynb               # Exploratory data analysis
│   └── feature_engineering.ipynb       # Feature engineering development
├── model/
│   ├── feature_engineering.py           # Feature transformation pipeline
│   ├── model_train.py                   # Model training script
│   ├── prediction.py                    # Batch prediction script
│   ├── fraud_model_logistic_regression.pkl  # Trained model
│   ├── fraud_smote_sampler.pkl          # SMOTE sampler for imbalanced data
│   ├── remaining_label_encoders.pkl     # Label encoders for categorical features
│   ├── feature_names.txt                # Model feature names
│   ├── model_performance.json           # Model comparison and best model selection
│   └── predict_transactions.csv         # Sample prediction data
|   └── batch_predictions.csv            # Sample fraud predicted data     
├── output/
│   └── training_data/                   # Spark consumer output
├── producer/
│   ├── data/                           # Producer data directory
│   └── producer.py                     # Kafka producer script
├── docker-compose.yml                  # Docker services configuration
├── pipeline.py                         # Main pipeline orchestration
└── README.md
```

## Components

### 1. Data Generation (`data/data_generate.py`)
Generates synthetic credit card transaction data with realistic patterns and fraud indicators.

### 2. Kafka Producer (`producer/producer.py`)
Streams transaction data to Kafka topics using Docker containers.

### 3. Kafka Consumer (`consumer/consumer.py`)
Spark-based consumer that reads from Kafka streams and saves data for ML processing.

### 4. Feature Engineering (`model/feature_engineering.py`)
Transforms raw transaction data into ML-ready features including:
- Transaction amount normalization
- Temporal features (hour, day patterns)
- Merchant and location encoding
- Risk indicators

### 5. ML Model Training (`model/model_train.py`)
Trains multiple models and selects the best performer:
- Logistic Regression
- Random Forest
- XGBoost
- Handles class imbalance with SMOTE

### 6. Prediction System (`model/prediction.py`)
Batch prediction system that processes CSV files and outputs fraud predictions.

### 7. Pipeline Orchestrator (`pipeline.py`)
Main script that coordinates the entire workflow.

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Required Python packages (see `requirement.txt`)

## Quick Start

### 1. Start the Complete Pipeline

```bash
python pipeline.py
```

This will automatically:
- Clean up existing containers
- Generate synthetic transaction data
- Start Kafka and Spark services
- Create Kafka topics
- Run producer to stream data
- Run Spark consumer to process data
- Train ML models (if needed)
- Generate batch predictions

### 2. Manual Step-by-Step Execution

#### Start Docker Services
```bash
docker-compose up -d
```

#### Generate Training Data
```bash
python data/data_generate.py
```

#### Create Kafka Topics
```bash
docker exec streaming_credit_card_fraud_detection-kafka-1 kafka-topics --create --topic fraud-topic --bootstrap-server kafka:9092 --partitions 1 --replication-factor 1
```

#### Run Producer
```bash
docker-compose run --rm producer
```

#### Run Spark Consumer
```bash
docker-compose run --rm spark spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 consumer.py
```

#### Train Models
```bash
cd model && python model_train.py
```

#### Run Batch Predictions
```bash
cd model && python prediction.py predict_transactions.csv
```

## Configuration

### Docker Services
- **Zookeeper**: Port 2181
- **Kafka**: Port 9092
- **Spark**: Bitnami Spark 3.4.0

### Model Configuration
The system automatically selects the best performing model based on metrics stored in `model/model_performance.json`.

## Usage Examples

### Batch Prediction
```bash
cd model
python prediction.py your_transactions.csv output_predictions.csv
```

### Real-time Streaming
```bash
# Start pipeline for continuous processing
python pipeline.py
```

## Monitoring

### Check Kafka Topics
```bash
docker exec streaming_credit_card_fraud_detection-kafka-1 kafka-topics --list --bootstrap-server kafka:9092
```

### View Kafka Messages
```bash
docker exec streaming_credit_card_fraud_detection-kafka-1 kafka-console-consumer --topic fraud-topic --bootstrap-server kafka:9092
```

### Check Processing Results
```bash
ls -la ./output/training_data/
cat ./output/training_data/part-*.csv
```

## Model Performance

The system compares multiple algorithms and automatically selects the best performer. Current models include:
- Logistic Regression (baseline)
- Random Forest
- XGBoost

Performance metrics are tracked in `model/model_performance.json`.

## Output

### Prediction Results
Batch predictions are saved with the following fields:
- `transaction_id`: Unique identifier
- `is_fraud`: Boolean prediction
- `fraud_probability`: Probability score (0-1)
- `recommendation`: APPROVE/REVIEW/BLOCK
- `risk_level`: LOW/MEDIUM/HIGH

### Processing Summary
- Total transactions processed
- Fraud detection rate
- Processing time metrics

## Troubleshooting

### Common Issues

1. **Kafka Connection Errors**
   - Ensure Docker services are running
   - Wait 45+ seconds after starting services
   - Check container logs: `docker logs streaming_credit_card_fraud_detection-kafka-1`

2. **Model Loading Errors**
   - Verify model files exist in `model/` directory
   - Check `model_performance.json` for correct model names
   - Retrain models if necessary

3. **Empty Prediction Results**
   - Verify input CSV format matches expected schema
   - Check feature engineering output
   - Ensure model features align with input data

### Cleanup
```bash
docker-compose down
docker system prune -f
```

## Development

### Adding New Features
1. Modify `model/feature_engineering.py`
2. Retrain models with `python model/model_train.py`
3. Update prediction pipeline

### Testing New Models
1. Add model to `model/model_train.py`
2. Run training comparison
3. Best model will be automatically selected

## License

This project is for educational and development purposes.
