import pandas as pd
import random
from faker import Faker

# Create Faker instance
fake = Faker()

# Set random seed for consistent results
Faker.seed(42)
random.seed(42)


def generate_fraud_data_fixed(num_records=1000):
    """
    Generate fraud detection data with NO missing values
    """

    print(f"Generating {num_records} transactions with NO missing values...")

    # Create lists to store data (this ensures no missing values)
    usernames = []
    full_names = []
    emails = []
    phones = []
    amounts = []
    merchants = []
    card_numbers = []
    card_providers = []
    countries = []
    addresses = []
    cities = []
    states = []
    zipcodes = []
    dates = []
    hours = []
    fraud_flags = []

    # Define lists for consistent data
    normal_merchants = ['Amazon', 'Walmart', 'Target', 'Starbucks', 'Shell', 'McDonalds']
    fraud_merchants = ['Unknown Store', 'Suspicious Site', 'Foreign ATM']
    card_types = ['Visa', 'Mastercard', 'American Express', 'Discover']

    # Generate each record
    for i in range(num_records):

        # First decide if fraud (2% chance)
        is_fraud = random.choice([True, False, False, False, False])  # 20% fraud for better testing

        # Generate all fields - NO MISSING VALUES
        username = fake.user_name()
        full_name = fake.name()
        email = fake.email()
        phone = fake.phone_number()

        # Transaction amount
        if is_fraud:
            amount = round(random.uniform(1500, 5000), 2)  # High amounts for fraud
            merchant = random.choice(fraud_merchants)
            country = fake.country()  # Foreign country
            hour = random.choice([1, 2, 3, 23, 0])  # Late night
        else:
            amount = round(random.uniform(10, 800), 2)  # Normal amounts
            merchant = random.choice(normal_merchants)
            country = "United States"  # Domestic
            hour = random.randint(8, 20)  # Business hours

        # Card info
        card_last_4 = f"{random.randint(1000, 9999)}"
        card_provider = random.choice(card_types)

        # Address info
        address = fake.street_address()
        city = fake.city()
        state = fake.state()
        zipcode = fake.zipcode()

        # Date
        transaction_date = fake.date_this_year().strftime('%Y-%m-%d')

        # Convert fraud boolean to integer
        fraud_flag = 1 if is_fraud else 0

        # Add to lists
        usernames.append(username)
        full_names.append(full_name)
        emails.append(email)
        phones.append(phone)
        amounts.append(amount)
        merchants.append(merchant)
        card_numbers.append(card_last_4)
        card_providers.append(card_provider)
        countries.append(country)
        addresses.append(address)
        cities.append(city)
        states.append(state)
        zipcodes.append(zipcode)
        dates.append(transaction_date)
        hours.append(hour)
        fraud_flags.append(fraud_flag)

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_records} records")

    # Create DataFrame from lists
    df = pd.DataFrame({
        'username': usernames,
        'full_name': full_names,
        'email': emails,
        'phone': phones,
        'transaction_amount': amounts,
        'merchant': merchants,
        'card_last_4': card_numbers,
        'card_provider': card_providers,
        'country': countries,
        'address': addresses,
        'city': cities,
        'state': states,
        'zipcode': zipcodes,
        'transaction_date': dates,
        'transaction_hour': hours,
        'is_fraud': fraud_flags
    })

    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"\n Checking for missing values:")
    if missing_values.sum() == 0:
        print("NO missing values found!")
    else:
        print("Missing values found:")
        print(missing_values[missing_values > 0])

    # Save to CSV
    filename = 'data/credit_card_fraud_data.csv'
    df.to_csv(filename, index=False)

    # Print detailed summary
    print(f"\n DATASET SUMMARY")
    print("=" * 30)
    print(f"Total records: {len(df):,}")
    print(f"Fraud transactions: {df['is_fraud'].sum():,}")
    print(f"Normal transactions: {(df['is_fraud'] == 0).sum():,}")
    print(f"Fraud rate: {df['is_fraud'].mean() * 100:.1f}%")
    print(f"Average amount: ${df['transaction_amount'].mean():.2f}")

    # Show fraud vs normal comparison
    fraud_data = df[df['is_fraud'] == 1]
    normal_data = df[df['is_fraud'] == 0]

    print(f"\n AMOUNT COMPARISON")
    print(f"Fraud avg amount: ${fraud_data['transaction_amount'].mean():.2f}")
    print(f"Normal avg amount: ${normal_data['transaction_amount'].mean():.2f}")

    print(f"\n File saved: {filename}")
    print(f" Dataset shape: {df.shape}")

    return df


def check_data_quality(df):
    """
    Check the quality of generated data
    """
    print(f"\n DATA QUALITY CHECK")
    print("=" * 30)

    # Check each column
    for col in df.columns:
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        print(f"{col:20s}: {null_count:3d} missing, {unique_count:4d} unique")

    # Show sample data
    print(f"\nSAMPLE DATA (First 3 rows):")
    print(df.head(3).to_string())

    # Show fraud examples
    print(f"\n FRAUD EXAMPLES:")
    fraud_examples = df[df['is_fraud'] == 1].head(2)
    if not fraud_examples.empty:
        for idx, row in fraud_examples.iterrows():
            print(f"\nFraud Transaction {idx}:")
            print(f"  User: {row['username']} (${row['transaction_amount']})")
            print(f"  Merchant: {row['merchant']}")
            print(f"  Country: {row['country']}")
            print(f"  Hour: {row['transaction_hour']}")

    # Show normal examples
    print(f"\nNORMAL EXAMPLES:")
    normal_examples = df[df['is_fraud'] == 0].head(2)
    for idx, row in normal_examples.iterrows():
        print(f"\nNormal Transaction {idx}:")
        print(f"  User: {row['username']} (${row['transaction_amount']})")
        print(f"  Merchant: {row['merchant']}")
        print(f"  Country: {row['country']}")
        print(f"  Hour: {row['transaction_hour']}")


# Main execution
if __name__ == "__main__":

    try:
        print(" Starting fraud data generation...")

        # Generate data
        fraud_df = generate_fraud_data_fixed(1000)

        # Check quality
        check_data_quality(fraud_df)

        # Show value counts for is_fraud column
        print(f"\n FRAUD DISTRIBUTION:")
        print(fraud_df['is_fraud'].value_counts())
        print(f"\nFraud percentage: {fraud_df['is_fraud'].mean() * 100:.1f}%")

        print(f"\nSUCCESS! Dataset generated with no missing values!")

    except ImportError:
        print(" Error: Faker library not found!")
        print(" Install it with: pip install faker")
    except Exception as e:
        print(f" Error occurred: {e}")
        print("Please check your Python environment")