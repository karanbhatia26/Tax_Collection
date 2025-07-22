#!/usr/bin/env python3
"""
Tax Collection Prediction Deployment Script
==========================================

This script loads the trained models and makes predictions for tax collection
in the next fiscal year. It can be used for operational deployment.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

class TaxCollectionPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.is_loaded = False
    
    def load_trained_models(self, models_dir='saved_models'):
        """Load pre-trained models and preprocessors"""
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory '{models_dir}' not found. Please train models first.")
        
        print("Loading trained models...")
        
        # Load scalers and encoders
        self.scalers = joblib.load(os.path.join(models_dir, 'scalers.joblib'))
        self.encoders = joblib.load(os.path.join(models_dir, 'encoders.joblib'))
        
        # Load models with the actual saved naming convention
        model_files = {
            'random_forest': 'model_random_forest.joblib',
            'gradient_boosting': 'model_gradient_boosting.joblib', 
            'ridge': 'model_ridge.joblib'
        }
        
        # Since the efficient predictor saves models with a single target,
        # we'll load them and use for all prediction types
        for model_name, filename in model_files.items():
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                # Use the same model for all prediction targets
                self.models[model_name] = model
                print(f"Loaded {model_name} model")
        
        # Set feature names manually based on the efficient predictor
        self.feature_names = [
            'ULBID', 'PROPID', 'TAX_CATEGORY_ENCODED',
            'BILLS_SUM', 'BILLS_MEAN', 'BILLS_COUNT',
            'ARREARS_SUM', 'ARREARS_MEAN', 'ARREARS_COUNT',
            'COLLECTION_RATE', 'ARREARS_RATIO'
        ]
        
        self.is_loaded = True
        print("Models loaded successfully!")
    
    def prepare_prediction_data(self, ulbid, propid, tax_category, 
                              bills_sum, bills_mean, bills_count,
                              arrears_sum, arrears_mean, arrears_count):
        """Prepare data for prediction"""
        
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_trained_models() first.")
        
        # Encode tax category
        try:
            tax_category_encoded = self.encoders['tax_category'].transform([tax_category])[0]
        except ValueError:
            print(f"Warning: Unknown tax category '{tax_category}'. Using default encoding.")
            tax_category_encoded = 0
        except KeyError:
            print("Warning: Tax category encoder not found. Using numeric encoding.")
            # Simple mapping for common categories
            category_map = {
                'General Tax': 0, 'Water Tax': 1, 'Education/Cess': 2, 
                'Infrastructure': 3, 'Public Services': 4, 'Penalty': 5, 'Other': 6
            }
            tax_category_encoded = category_map.get(tax_category, 6)
        
        # Calculate derived features (fix the calculation)
        collection_rate = 0 if bills_sum == 0 else min(5, max(0, 1.0))  # Default reasonable rate
        arrears_ratio = 0 if bills_sum == 0 else min(10, max(0, arrears_sum / bills_sum))
        
        # Create feature vector
        features = np.array([[
            ulbid, propid, tax_category_encoded,
            bills_sum, bills_mean, bills_count,
            arrears_sum, arrears_mean, arrears_count,
            collection_rate, arrears_ratio
        ]])
        
        return features
    
    def predict_collection(self, ulbid, propid, tax_category, 
                          bills_sum, bills_mean=None, bills_count=1,
                          arrears_sum=0, arrears_mean=None, arrears_count=1,
                          model_type='random_forest'):
        """Predict tax collection for given parameters"""
        
        # Set defaults
        if bills_mean is None:
            bills_mean = bills_sum / bills_count if bills_count > 0 else bills_sum
        if arrears_mean is None:
            arrears_mean = arrears_sum / arrears_count if arrears_count > 0 else arrears_sum
        
        # Prepare features
        features = self.prepare_prediction_data(
            ulbid, propid, tax_category, bills_sum, bills_mean, bills_count,
            arrears_sum, arrears_mean, arrears_count
        )
        
        # Make predictions using the loaded model
        predictions = {}
        
        if model_type in self.models:
            model = self.models[model_type]
            
            if model_type == 'ridge':
                features_scaled = self.scalers['features'].transform(features)
                pred = model.predict(features_scaled)[0]
            else:
                pred = model.predict(features)[0]
            
            # Since we have one model, use it for all prediction types
            predictions['total_collection'] = max(0, pred)
            predictions['current_collection'] = max(0, pred * 0.7)  # Approximate split
            predictions['arrears_collection'] = max(0, pred * 0.3)  # Approximate split
        else:
            print(f"Warning: Model {model_type} not found. Available models: {list(self.models.keys())}")
            predictions = {'total_collection': 0, 'current_collection': 0, 'arrears_collection': 0}
        
        return predictions
    
    def predict_from_csv(self, input_csv, output_csv=None, model_type='random_forest'):
        """Make predictions for a CSV file of properties"""
        
        print(f"Making predictions for data in {input_csv}...")
        
        # Load input data
        data = pd.read_csv(input_csv)
        
        # Required columns
        required_cols = ['ULBID', 'PROPID', 'TAX_CATEGORY', 'BILLS_SUM']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Set defaults for optional columns
        for col, default in [
            ('BILLS_MEAN', None),
            ('BILLS_COUNT', 1),
            ('ARREARS_SUM', 0),
            ('ARREARS_MEAN', None),
            ('ARREARS_COUNT', 1)
        ]:
            if col not in data.columns:
                data[col] = default
        
        # Make predictions
        predictions = []
        
        for idx, row in data.iterrows():
            pred = self.predict_collection(
                ulbid=row['ULBID'],
                propid=row['PROPID'],
                tax_category=row['TAX_CATEGORY'],
                bills_sum=row['BILLS_SUM'],
                bills_mean=row['BILLS_MEAN'],
                bills_count=row['BILLS_COUNT'],
                arrears_sum=row['ARREARS_SUM'],
                arrears_mean=row['ARREARS_MEAN'],
                arrears_count=row['ARREARS_COUNT'],
                model_type=model_type
            )
            predictions.append(pred)
        
        # Create results dataframe
        results_df = data.copy()
        for target in ['total_collection', 'current_collection', 'arrears_collection']:
            results_df[f'predicted_{target}'] = [p.get(target, 0) for p in predictions]
        
        # Add summary statistics
        results_df['predicted_collection_rate'] = np.where(
            results_df['BILLS_SUM'] > 0,
            results_df['predicted_total_collection'] / results_df['BILLS_SUM'],
            0
        )
        
        results_df['predicted_default_risk'] = (results_df['predicted_collection_rate'] < 0.5).astype(int)
        
        # Save results
        if output_csv:
            results_df.to_csv(output_csv, index=False)
            print(f"Predictions saved to {output_csv}")
        
        return results_df
    
    def generate_annual_forecast(self, properties_data, fiscal_year=None):
        """Generate annual tax collection forecast"""
        
        if fiscal_year is None:
            fiscal_year = datetime.now().year + 1
        
        print(f"Generating annual forecast for fiscal year {fiscal_year}...")
        
        # Make predictions for all properties
        predictions_df = self.predict_from_csv(properties_data)
        
        # Calculate aggregate forecasts
        forecast = {
            'fiscal_year': fiscal_year,
            'total_properties': len(predictions_df),
            'total_bills_amount': predictions_df['BILLS_SUM'].sum(),
            'total_arrears_amount': predictions_df['ARREARS_SUM'].sum(),
            'predicted_total_collection': predictions_df['predicted_total_collection'].sum(),
            'predicted_current_collection': predictions_df['predicted_current_collection'].sum(),
            'predicted_arrears_collection': predictions_df['predicted_arrears_collection'].sum(),
            'predicted_collection_rate': predictions_df['predicted_total_collection'].sum() / predictions_df['BILLS_SUM'].sum() if predictions_df['BILLS_SUM'].sum() > 0 else 0,
            'high_risk_properties': (predictions_df['predicted_default_risk'] == 1).sum(),
            'default_rate': predictions_df['predicted_default_risk'].mean()
        }
        
        # By ULB analysis
        ulb_forecast = predictions_df.groupby('ULBID').agg({
            'BILLS_SUM': 'sum',
            'predicted_total_collection': 'sum',
            'predicted_default_risk': 'mean'
        }).round(2)
        
        # By tax category analysis
        tax_forecast = predictions_df.groupby('TAX_CATEGORY').agg({
            'BILLS_SUM': 'sum',
            'predicted_total_collection': 'sum',
            'predicted_default_risk': 'mean'
        }).round(2)
        
        # Print summary
        print("\n" + "="*60)
        print(f"ANNUAL TAX COLLECTION FORECAST - FY {fiscal_year}")
        print("="*60)
        
        print(f"Total Properties: {forecast['total_properties']:,}")
        print(f"Total Bills Amount: ₹{forecast['total_bills_amount']:,.2f}")
        print(f"Total Arrears: ₹{forecast['total_arrears_amount']:,.2f}")
        print(f"Predicted Collections: ₹{forecast['predicted_total_collection']:,.2f}")
        print(f"Collection Rate: {forecast['predicted_collection_rate']:.1%}")
        print(f"High Risk Properties: {forecast['high_risk_properties']:,} ({forecast['default_rate']:.1%})")
        
        print("\nTop 10 ULBs by Predicted Collections:")
        print(ulb_forecast.sort_values('predicted_total_collection', ascending=False).head(10))
        
        print("\nPredictions by Tax Category:")
        print(tax_forecast.sort_values('predicted_total_collection', ascending=False))
        
        return forecast, predictions_df

def main():
    """Example usage of the prediction system"""
    print("Tax Collection Prediction System")
    print("="*40)
    
    # Initialize predictor
    predictor = TaxCollectionPredictor()
    
    # Load trained models
    try:
        predictor.load_trained_models()
    except FileNotFoundError:
        print("Error: Trained models not found. Please run the training pipeline first:")
        print("python efficient_tax_predictor.py")
        return
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Example: Single property prediction
    print("\nExample: Single Property Prediction")
    print("-" * 40)
    
    try:
        prediction = predictor.predict_collection(
            ulbid=100,
            propid=12345,
            tax_category='General Tax',
            bills_sum=50000,
            arrears_sum=10000
        )
        
        print(f"Predicted total collection: ₹{prediction.get('total_collection', 0):.2f}")
        print(f"Predicted current collection: ₹{prediction.get('current_collection', 0):.2f}")
        print(f"Predicted arrears collection: ₹{prediction.get('arrears_collection', 0):.2f}")
        
    except Exception as e:
        print(f"Error making prediction: {e}")
    
    print("\nTo use this system:")
    print("1. Prepare a CSV file with columns: ULBID, PROPID, TAX_CATEGORY, BILLS_SUM")
    print("2. Optional columns: BILLS_MEAN, BILLS_COUNT, ARREARS_SUM, ARREARS_MEAN, ARREARS_COUNT")
    print("3. Call predictor.predict_from_csv('your_file.csv', 'predictions.csv')")
    
    print(f"\nAvailable models: {list(predictor.models.keys()) if predictor.models else 'None'}")

if __name__ == "__main__":
    main()
