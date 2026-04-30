"""
KALESS — Demo Seed Data Generator
Generates realistic customer satisfaction and psychological scale data for demo purposes.
Output: .csv and .sav (SPSS) formats.
"""

import os
import random
import pandas as pd
import numpy as np
from faker import Faker
import pyreadstat

# Initialize
fake = Faker()
Faker.seed(42)
np.random.seed(42)

OUTPUT_DIR = "public/mock-data"
NUM_ROWS = 1000

def generate_data():
    print(f"Generating {NUM_ROWS} rows of realistic data...")
    
    data = []
    
    # Define Likert scale questions
    # Q1: Satisfaction, Q2: Ease of Use, Q3: Recommendation, Q4: Value for Money, Q5: Reliability
    
    for i in range(NUM_ROWS):
        age = random.randint(18, 75)
        gender = random.choice([1, 2, 3]) # 1: Male, 2: Female, 3: Non-binary/Other
        income = round(np.random.normal(55000, 15000), 2)
        if income < 12000: income = 12000
        
        # Correlate some responses with age/income for "realistic" patterns
        # Younger people might find it easier to use (Ease of Use)
        ease_bias = 4 if age < 35 else (3 if age < 55 else 2)
        
        row = {
            "ID": i + 1,
            "Age": age,
            "Gender": gender,
            "Income": income,
            "Region": random.choice([1, 2, 3, 4]), # 1: North, 2: South, 3: East, 4: West
            "Q1_Satisfaction": np.clip(random.randint(1, 5), 1, 5),
            "Q2_EaseOfUse": np.clip(random.randint(1, 5) + (ease_bias - 3), 1, 5),
            "Q3_Recommendation": np.clip(random.randint(1, 5), 1, 5),
            "Q4_Value": np.clip(random.randint(1, 5), 1, 5),
            "Q5_Reliability": np.clip(random.randint(1, 5), 1, 5),
            "SignupDate": fake.date_between(start_date='-2y', end_date='today').isoformat()
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # --- Save as CSV ---
    csv_path = os.path.join(OUTPUT_DIR, "customer_satisfaction_demo.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # --- Save as SAV (SPSS) ---
    sav_path = os.path.join(OUTPUT_DIR, "customer_satisfaction_demo.sav")
    
    # SPSS Variable Labels
    var_labels = {
        "ID": "Respondent Identification Number",
        "Age": "Age of Respondent",
        "Gender": "Gender Identity",
        "Income": "Annual Household Income (USD)",
        "Region": "Geographic Region",
        "Q1_Satisfaction": "Overall Satisfaction with the Service",
        "Q2_EaseOfUse": "Ease of Navigating the Platform",
        "Q3_Recommendation": "Likelihood to Recommend to a Peer",
        "Q4_Value": "Perceived Value for the Price Paid",
        "Q5_Reliability": "Platform Reliability and Uptime",
        "SignupDate": "Date of Account Creation"
    }
    
    # SPSS Value Labels
    value_labels = {
        "Gender": {1: "Male", 2: "Female", 3: "Other"},
        "Region": {1: "North", 2: "South", 3: "East", 4: "West"},
        "Q1_Satisfaction": {1: "Very Dissatisfied", 2: "Dissatisfied", 3: "Neutral", 4: "Satisfied", 5: "Very Satisfied"},
        "Q2_EaseOfUse": {1: "Very Difficult", 2: "Difficult", 3: "Neutral", 4: "Easy", 5: "Very Easy"},
        "Q3_Recommendation": {1: "Very Unlikely", 2: "Unlikely", 3: "Neutral", 4: "Likely", 5: "Very Likely"},
        "Q4_Value": {1: "Poor Value", 2: "Fair Value", 3: "Good Value", 4: "Very Good Value", 5: "Excellent Value"},
        "Q5_Reliability": {1: "Very Unreliable", 2: "Unreliable", 3: "Neutral", 4: "Reliable", 5: "Highly Reliable"}
    }
    
    pyreadstat.write_sav(
        df, 
        sav_path, 
        column_labels=var_labels, 
        variable_value_labels=value_labels,
        variable_display_width={"ID": 8, "Age": 4},
        compress=True
    )
    print(f"SPSS (.sav) saved: {sav_path}")
    print("\nDemo data generation complete. Ready for production import.")

if __name__ == "__main__":
    generate_data()
