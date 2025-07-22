#!/usr/bin/env python3
"""
Efficient Tax Collection Prediction Pipeline
===========================================

Optimized version for large datasets with chunked processing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

class EfficientTaxPredictor:
    def __init__(self, sample_size=50000):
        self.sample_size = sample_size
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def load_and_sample_data(self):
        """Load data with sampling for efficiency"""
        print(f"Loading data with sample size: {self.sample_size}")
        
        # Load samples from each file
        self.arrears_df = pd.read_csv('arrears_query.csv', nrows=self.sample_size)
        self.bills_df = pd.read_csv('bill_query.csv', nrows=self.sample_size) 
        self.collections_df = pd.read_csv('collection_query.csv', nrows=self.sample_size)
        
        print(f"Loaded: {len(self.arrears_df)} arrears, {len(self.bills_df)} bills, {len(self.collections_df)} collections")
        
        return self.arrears_df, self.bills_df, self.collections_df
    
    def clean_and_standardize_data(self):
        """Clean and standardize the data"""
        print("Cleaning and standardizing data...")
        
        # Standardize column names
        self.arrears_df.columns = ['ULBID', 'PROPID', 'TAXID', 'TAX_NAME', 'ARREARS_AMOUNT']
        self.bills_df.columns = ['ULBID', 'PROPID', 'TAXID', 'TAX_NAME', 'BILL_AMOUNT']
        self.collections_df.columns = ['ULBID', 'PROPID', 'TAXID', 'TAX_NAME', 'COLL_ARREARS', 'COLL_CURRENT', 'COLL_TOTAL']
        
        # Clean tax names and create categories
        for df in [self.arrears_df, self.bills_df, self.collections_df]:
            df['TAX_NAME'] = df['TAX_NAME'].fillna('Unknown').astype(str)
            df['TAX_CATEGORY'] = df['TAX_NAME'].apply(self._categorize_tax)
        
        # Convert amounts to numeric
        self.arrears_df['ARREARS_AMOUNT'] = pd.to_numeric(self.arrears_df['ARREARS_AMOUNT'], errors='coerce').fillna(0)
        self.bills_df['BILL_AMOUNT'] = pd.to_numeric(self.bills_df['BILL_AMOUNT'], errors='coerce').fillna(0)
        self.collections_df['COLL_TOTAL'] = pd.to_numeric(self.collections_df['COLL_TOTAL'], errors='coerce').fillna(0)
        self.collections_df['COLL_CURRENT'] = pd.to_numeric(self.collections_df['COLL_CURRENT'], errors='coerce').fillna(0)
        self.collections_df['COLL_ARREARS'] = pd.to_numeric(self.collections_df['COLL_ARREARS'], errors='coerce').fillna(0)
        
    def _categorize_tax(self, tax_name):
        """Categorize tax names into broader categories"""
        tax_name = str(tax_name).lower()
        
        if any(word in tax_name for word in ['general', 'property']):
            return 'General Tax'
        elif any(word in tax_name for word in ['water', 'पाणी', 'jallabha', 'mallabha']):
            return 'Water Tax'
        elif any(word in tax_name for word in ['education', 'cess']):
            return 'Education/Cess'
        elif any(word in tax_name for word in ['road', 'street', 'light']):
            return 'Infrastructure'
        elif any(word in tax_name for word in ['health', 'fire', 'solid', 'waste']):
            return 'Public Services'
        elif any(word in tax_name for word in ['penalty', 'illegal']):
            return 'Penalty'
        else:
            return 'Other'
    
    def create_ml_dataset(self):
        """Create dataset for machine learning"""
        print("Creating ML dataset...")
        
        # Aggregate data by property and tax category
        arrears_agg = self.arrears_df.groupby(['ULBID', 'PROPID', 'TAX_CATEGORY']).agg({
            'ARREARS_AMOUNT': ['sum', 'mean', 'count']
        }).reset_index()
        arrears_agg.columns = ['ULBID', 'PROPID', 'TAX_CATEGORY', 'ARREARS_SUM', 'ARREARS_MEAN', 'ARREARS_COUNT']
        
        bills_agg = self.bills_df.groupby(['ULBID', 'PROPID', 'TAX_CATEGORY']).agg({
            'BILL_AMOUNT': ['sum', 'mean', 'count']
        }).reset_index()
        bills_agg.columns = ['ULBID', 'PROPID', 'TAX_CATEGORY', 'BILLS_SUM', 'BILLS_MEAN', 'BILLS_COUNT']
        
        collections_agg = self.collections_df.groupby(['ULBID', 'PROPID', 'TAX_CATEGORY']).agg({
            'COLL_TOTAL': 'sum',
            'COLL_CURRENT': 'sum',
            'COLL_ARREARS': 'sum'
        }).reset_index()
        
        # Merge datasets
        ml_data = collections_agg.merge(bills_agg, on=['ULBID', 'PROPID', 'TAX_CATEGORY'], how='left')
        ml_data = ml_data.merge(arrears_agg, on=['ULBID', 'PROPID', 'TAX_CATEGORY'], how='left')
        
        # Fill missing values
        numeric_cols = ml_data.select_dtypes(include=[np.number]).columns
        ml_data[numeric_cols] = ml_data[numeric_cols].fillna(0)
        
        # Create derived features
        ml_data['COLLECTION_RATE'] = np.where(ml_data['BILLS_SUM'] > 0, 
                                            ml_data['COLL_TOTAL'] / ml_data['BILLS_SUM'], 0)
        ml_data['ARREARS_RATIO'] = np.where(ml_data['BILLS_SUM'] > 0,
                                          ml_data['ARREARS_SUM'] / ml_data['BILLS_SUM'], 0)
        ml_data['DEFAULT_RISK'] = (ml_data['COLLECTION_RATE'] < 0.5).astype(int)
        
        # Cap extreme values
        ml_data['COLLECTION_RATE'] = np.clip(ml_data['COLLECTION_RATE'], 0, 3)
        ml_data['ARREARS_RATIO'] = np.clip(ml_data['ARREARS_RATIO'], 0, 5)
        
        self.ml_data = ml_data
        print(f"ML dataset created with {len(ml_data)} records")
        
        return ml_data
    
    def prepare_features_targets(self):
        """Prepare features and target variables"""
        print("Preparing features and targets...")
        
        # Encode categorical variables
        le_tax = LabelEncoder()
        self.ml_data['TAX_CATEGORY_ENCODED'] = le_tax.fit_transform(self.ml_data['TAX_CATEGORY'])
        self.encoders['tax_category'] = le_tax
        
        # Define features
        feature_columns = [
            'ULBID', 'PROPID', 'TAX_CATEGORY_ENCODED',
            'BILLS_SUM', 'BILLS_MEAN', 'BILLS_COUNT',
            'ARREARS_SUM', 'ARREARS_MEAN', 'ARREARS_COUNT',
            'COLLECTION_RATE', 'ARREARS_RATIO'
        ]
        
        # Remove rows with missing targets
        valid_data = self.ml_data[self.ml_data['COLL_TOTAL'].notna()].copy()
        
        self.X = valid_data[feature_columns].fillna(0)
        self.y_total = valid_data['COLL_TOTAL']
        self.y_current = valid_data['COLL_CURRENT']
        self.y_arrears = valid_data['COLL_ARREARS']
        
        print(f"Features prepared: {self.X.shape}")
        
        return self.X, self.y_total, self.y_current, self.y_arrears
    
    def train_models(self):
        """Train prediction models"""
        print("Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_total, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['features'] = scaler
        
        # Train models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name == 'ridge':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {'mae': mae, 'mse': mse, 'rmse': np.sqrt(mse), 'r2': r2}
            self.models[name] = model
            
            print(f"  MAE: ₹{mae:,.2f}, RMSE: ₹{np.sqrt(mse):,.2f}, R²: {r2:.3f}")
        
        self.results = results
        self.X_test = X_test
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test
        
        return results
    
    def predict_next_year(self, model_name='random_forest'):
        """Generate predictions for next year"""
        print(f"\nGenerating next year predictions using {model_name}...")
        
        model = self.models[model_name]
        
        if model_name == 'ridge':
            predictions = model.predict(self.X_test_scaled)
        else:
            predictions = model.predict(self.X_test)
        
        # Calculate aggregated predictions
        total_predicted = np.sum(predictions)
        total_actual = np.sum(self.y_test)
        
        print(f"Predicted total collection: ₹{total_predicted:,.2f}")
        print(f"Actual collection (test): ₹{total_actual:,.2f}")
        print(f"Prediction accuracy: {(1 - abs(total_predicted - total_actual) / total_actual) * 100:.1f}%")
        
        # Analyze by categories
        test_data = self.ml_data.iloc[self.X_test.index].copy()
        test_data['PREDICTED_COLLECTION'] = predictions
        
        category_analysis = test_data.groupby('TAX_CATEGORY').agg({
            'COLL_TOTAL': 'sum',
            'PREDICTED_COLLECTION': 'sum',
            'DEFAULT_RISK': 'mean'
        }).round(2)
        
        print("\nPredictions by Tax Category:")
        print(category_analysis)
        
        return predictions, category_analysis
    
    def generate_business_insights(self):
        """Generate actionable business insights"""
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # 1. Tax category performance
        tax_performance = self.ml_data.groupby('TAX_CATEGORY').agg({
            'COLL_TOTAL': ['sum', 'count'],
            'COLLECTION_RATE': 'mean',
            'DEFAULT_RISK': 'mean'
        }).round(3)
        
        print("\n1. TAX CATEGORY PERFORMANCE:")
        print(tax_performance)
        
        # 2. High-risk properties
        high_risk = self.ml_data[self.ml_data['DEFAULT_RISK'] == 1]
        print(f"\n2. HIGH-RISK PROPERTIES:")
        print(f"   Count: {len(high_risk):,} properties ({len(high_risk)/len(self.ml_data)*100:.1f}%)")
        print(f"   Potential revenue at risk: ₹{high_risk['BILLS_SUM'].sum():,.2f}")
        
        # 3. Best performing ULBs
        ulb_performance = self.ml_data.groupby('ULBID').agg({
            'COLL_TOTAL': 'sum',
            'COLLECTION_RATE': 'mean'
        }).sort_values('COLL_TOTAL', ascending=False)
        
        print(f"\n3. TOP 10 PERFORMING ULBs:")
        print(ulb_performance.head(10))
        
        # 4. Feature importance
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']
            feature_names = ['ULBID', 'PROPID', 'TAX_CATEGORY', 'BILLS_SUM', 'BILLS_MEAN', 'BILLS_COUNT',
                           'ARREARS_SUM', 'ARREARS_MEAN', 'ARREARS_COUNT', 'COLLECTION_RATE', 'ARREARS_RATIO']
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print(f"\n4. MOST IMPORTANT PREDICTION FACTORS:")
            print(importance_df.head(8))
        
        # 5. Revenue optimization recommendations
        print(f"\n5. REVENUE OPTIMIZATION RECOMMENDATIONS:")
        print("   - Focus collection efforts on General Tax (highest revenue)")
        print("   - Target high-arrears properties for immediate action")
        print("   - Implement risk-based collection strategies")
        print("   - Monitor ULB performance and share best practices")
        
    def create_visualizations(self):
        """Create key visualizations"""
        print("\nCreating visualizations...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Collection rate by tax category
        tax_rates = self.ml_data.groupby('TAX_CATEGORY')['COLLECTION_RATE'].mean().sort_values()
        axes[0, 0].barh(tax_rates.index, tax_rates.values, color='skyblue')
        axes[0, 0].set_title('Collection Rate by Tax Category')
        axes[0, 0].set_xlabel('Average Collection Rate')
        
        # 2. Revenue by tax category
        tax_revenue = self.ml_data.groupby('TAX_CATEGORY')['COLL_TOTAL'].sum().sort_values(ascending=False)
        axes[0, 1].bar(tax_revenue.index, tax_revenue.values, color='lightgreen')
        axes[0, 1].set_title('Total Revenue by Tax Category')
        axes[0, 1].set_ylabel('Revenue (₹)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Model performance comparison
        if hasattr(self, 'results'):
            model_names = list(self.results.keys())
            r2_scores = [self.results[model]['r2'] for model in model_names]
            axes[1, 0].bar(model_names, r2_scores, color='orange')
            axes[1, 0].set_title('Model Performance (R² Score)')
            axes[1, 0].set_ylabel('R² Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Default risk analysis
        default_by_category = self.ml_data.groupby('TAX_CATEGORY')['DEFAULT_RISK'].mean().sort_values(ascending=False)
        axes[1, 1].bar(default_by_category.index, default_by_category.values, color='red', alpha=0.7)
        axes[1, 1].set_title('Default Risk by Tax Category')
        axes[1, 1].set_ylabel('Default Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('tax_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'tax_prediction_results.png'")
    
    def save_models(self):
        """Save trained models"""
        print("Saving models...")
        
        os.makedirs('saved_models', exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f'saved_models/model_{name}.joblib')
        
        joblib.dump(self.scalers, 'saved_models/scalers.joblib')
        joblib.dump(self.encoders, 'saved_models/encoders.joblib')
        
        print("Models saved successfully!")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*60)
        print("TAX COLLECTION PREDICTION ANALYSIS")
        print("="*60)
        
        # Step 1: Load and clean data
        self.load_and_sample_data()
        self.clean_and_standardize_data()
        
        # Step 2: Create ML dataset
        self.create_ml_dataset()
        
        # Step 3: Prepare features and targets
        self.prepare_features_targets()
        
        # Step 4: Train models
        self.train_models()
        
        # Step 5: Generate predictions
        predictions, category_analysis = self.predict_next_year()
        
        # Step 6: Business insights
        self.generate_business_insights()
        
        # Step 7: Visualizations
        self.create_visualizations()
        
        # Step 8: Save models
        self.save_models()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Next steps:")
        print("1. Review the generated insights and visualizations")
        print("2. Use saved models for future predictions")
        print("3. Implement recommended collection strategies")
        
        return predictions, category_analysis

# Run the analysis
if __name__ == "__main__":
    predictor = EfficientTaxPredictor(sample_size=100000)  # Adjust sample size as needed
    predictions, analysis = predictor.run_complete_analysis()
