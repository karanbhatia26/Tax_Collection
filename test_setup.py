import pandas as pd
import numpy as np
print("Testing data loading...")

try:
    print("Loading sample from arrears_query.csv...")
    arrears_sample = pd.read_csv('arrears_query.csv', nrows=100)
    print(f"✓ Arrears sample loaded: {len(arrears_sample)} rows")
    print("Columns:", list(arrears_sample.columns))
    print("Sample data:")
    print(arrears_sample.head())
    
    print("\nLoading sample from bill_query.csv...")
    bills_sample = pd.read_csv('bill_query.csv', nrows=100)
    print(f"✓ Bills sample loaded: {len(bills_sample)} rows")
    print("Columns:", list(bills_sample.columns))
    
    print("\nLoading sample from collection_query.csv...")
    collections_sample = pd.read_csv('collection_query.csv', nrows=100)
    print(f"✓ Collections sample loaded: {len(collections_sample)} rows") 
    print("Columns:", list(collections_sample.columns))
    
    print("\n✓ All data files loaded successfully!")
    print("✓ Setup is working correctly!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("Please check if the CSV files are in the current directory.")
