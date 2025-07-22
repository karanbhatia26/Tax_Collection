#!/usr/bin/env python3
"""
Demo script for bulk tax collection predictions
"""

from prediction_deployment import TaxCollectionPredictor
import pandas as pd

def demo_bulk_predictions():
    """Demonstrate bulk predictions on sample data"""
    
    print("="*60)
    print("BULK TAX COLLECTION PREDICTIONS DEMO")
    print("="*60)
    
    # Initialize predictor
    predictor = TaxCollectionPredictor()
    
    # Load trained models
    try:
        predictor.load_trained_models()
        print("✓ Models loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return
    
    # Load sample data
    print("\nLoading sample properties data...")
    sample_data = pd.read_csv('sample_properties.csv')
    print("Sample data:")
    print(sample_data)
    
    # Make predictions
    print("\nMaking predictions...")
    try:
        results = predictor.predict_from_csv('sample_properties.csv', 'sample_predictions.csv')
        
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        
        # Show results
        display_cols = ['ULBID', 'PROPID', 'TAX_CATEGORY', 'BILLS_SUM', 'ARREARS_SUM', 
                       'predicted_total_collection', 'predicted_collection_rate', 'predicted_default_risk']
        print(results[display_cols].round(2))
        
        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        total_bills = results['BILLS_SUM'].sum()
        total_predicted = results['predicted_total_collection'].sum()
        avg_collection_rate = results['predicted_collection_rate'].mean()
        high_risk_count = results['predicted_default_risk'].sum()
        
        print(f"Total Bills Amount: ₹{total_bills:,.2f}")
        print(f"Total Predicted Collection: ₹{total_predicted:,.2f}")
        print(f"Overall Collection Rate: {avg_collection_rate:.2%}")
        print(f"High Risk Properties: {high_risk_count} out of {len(results)}")
        
        # Category-wise analysis
        print("\n" + "="*60)
        print("CATEGORY-WISE ANALYSIS")
        print("="*60)
        
        category_summary = results.groupby('TAX_CATEGORY').agg({
            'BILLS_SUM': 'sum',
            'predicted_total_collection': 'sum',
            'predicted_collection_rate': 'mean',
            'predicted_default_risk': 'mean'
        }).round(2)
        
        print(category_summary)
        
        print(f"\n✓ Detailed predictions saved to 'sample_predictions.csv'")
        
    except Exception as e:
        print(f"✗ Error making predictions: {e}")

def demo_single_predictions():
    """Demonstrate single property predictions"""
    
    print("\n" + "="*60)
    print("SINGLE PROPERTY PREDICTIONS DEMO") 
    print("="*60)
    
    predictor = TaxCollectionPredictor()
    predictor.load_trained_models()
    
    # Test different scenarios
    scenarios = [
        {
            'name': 'High-value General Tax Property',
            'ulbid': 100, 'propid': 1001, 'tax_category': 'General Tax',
            'bills_sum': 100000, 'arrears_sum': 20000
        },
        {
            'name': 'Small Water Tax Property',
            'ulbid': 200, 'propid': 2001, 'tax_category': 'Water Tax', 
            'bills_sum': 15000, 'arrears_sum': 3000
        },
        {
            'name': 'High-risk Penalty Case',
            'ulbid': 300, 'propid': 3001, 'tax_category': 'Penalty',
            'bills_sum': 50000, 'arrears_sum': 45000
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print("-" * 40)
        
        prediction = predictor.predict_collection(
            ulbid=scenario['ulbid'],
            propid=scenario['propid'], 
            tax_category=scenario['tax_category'],
            bills_sum=scenario['bills_sum'],
            arrears_sum=scenario['arrears_sum']
        )
        
        collection_rate = prediction['total_collection'] / scenario['bills_sum'] if scenario['bills_sum'] > 0 else 0
        default_risk = "HIGH" if collection_rate < 0.5 else "LOW"
        
        print(f"Bills Amount: ₹{scenario['bills_sum']:,}")
        print(f"Arrears Amount: ₹{scenario['arrears_sum']:,}")
        print(f"Predicted Collection: ₹{prediction['total_collection']:,.2f}")
        print(f"Collection Rate: {collection_rate:.1%}")
        print(f"Default Risk: {default_risk}")

if __name__ == "__main__":
    demo_bulk_predictions()
    demo_single_predictions()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED!")
    print("="*60)
    print("You can now use the prediction system for your own data!")
    print("1. Create a CSV with your property data")
    print("2. Use predictor.predict_from_csv() for bulk predictions")
    print("3. Use predictor.predict_collection() for single predictions")
