
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
import re
from datetime import datetime, timedelta
import joblib
import os

warnings.filterwarnings('ignore')

class TaxCollectionPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def load_data(self):
        """Load and initial processing of CSV files"""
        print("Loading data files...")
        
        try:
            self.arrears_df = pd.read_csv('arrears_query.csv', low_memory=False)
            print(f"Arrears data loaded: {len(self.arrears_df)} records")
            
            self.bills_df = pd.read_csv('bill_query.csv', low_memory=False)
            print(f"Bills data loaded: {len(self.bills_df)} records")
            
            self.collections_df = pd.read_csv('collection_query.csv', low_memory=False)
            print(f"Collections data loaded: {len(self.collections_df)} records")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def clean_tax_names(self, df, tax_column='VAR_TAX_ENAME'):
        if tax_column not in df.columns:
            return df
            
        df_clean = df.copy()
        
        df_clean[tax_column] = df_clean[tax_column].fillna('Unknown Tax')
        
        df_clean[tax_column] = df_clean[tax_column].astype(str).str.strip()
        
        tax_mapping = {
            'General Tax': 'General Tax',
            'Education Cess Tax': 'Education Tax',
            'WATER TAX': 'Water Tax',
            'Road Tax': 'Road Tax',
            'Street Light Tax': 'Street Light Tax',
            'Fire Tax': 'Fire Tax',
            'Health Tax': 'Health Tax',
            'Employment Guaranty Cess': 'Employment Tax',
            'Solid Waste Tax': 'Waste Tax',
            'Illegal Penalty Tax': 'Penalty Tax'
        }
        
        marathi_mappings = {
            'मळ प्रवाह कर': 'Drainage Tax',
            'पाणी पुरवठा लाभ कर': 'Water Supply Tax',
            'Jallabha Kar': 'Water Tax',
            'Mallabha Tax': 'Water Tax'
        }
        
        for original, standardized in tax_mapping.items():
            df_clean[tax_column] = df_clean[tax_column].str.replace(original, standardized, case=False)
        
        for marathi, english in marathi_mappings.items():
            df_clean[tax_column] = df_clean[tax_column].str.replace(marathi, english, case=False)
        
        df_clean[f'{tax_column}_category'] = df_clean[tax_column].apply(self._categorize_tax)
        
        return df_clean
    
    def _categorize_tax(self, tax_name):
        tax_name = str(tax_name).lower()
        
        if any(word in tax_name for word in ['water', 'पाणी', 'jallabha', 'mallabha']):
            return 'Water Related'
        elif any(word in tax_name for word in ['education', 'cess']):
            return 'Education/Cess'
        elif any(word in tax_name for word in ['road', 'street', 'light']):
            return 'Infrastructure'
        elif any(word in tax_name for word in ['health', 'fire', 'solid', 'waste']):
            return 'Public Services'
        elif any(word in tax_name for word in ['general', 'property']):
            return 'General Tax'
        elif any(word in tax_name for word in ['penalty', 'illegal']):
            return 'Penalty'
        else:
            return 'Other'
    
    def prepare_features(self):
        print("Preparing features...")
        
        self.arrears_df = self.clean_tax_names(self.arrears_df)
        self.bills_df = self.clean_tax_names(self.bills_df)
        self.collections_df = self.clean_tax_names(self.collections_df)
        
        arrears_agg = self.arrears_df.groupby(['NUM_PROP_ULBID', 'NUM_PROPMAS_PROPID', 'VAR_TAX_ENAME_category']).agg({
            'ARREARSAMOUNT': ['sum', 'mean', 'count']
        }).reset_index()
        arrears_agg.columns = ['ULBID', 'PROPID', 'TAX_CATEGORY', 'ARREARS_SUM', 'ARREARS_MEAN', 'ARREARS_COUNT']
        
        bills_agg = self.bills_df.groupby(['NUM_BILL_ULBID', 'NUM_BILL_PROPID', 'VAR_TAX_ENAME_category']).agg({
            'CURRENT_S1_S2': ['sum', 'mean', 'count']
        }).reset_index()
        bills_agg.columns = ['ULBID', 'PROPID', 'TAX_CATEGORY', 'BILLS_SUM', 'BILLS_MEAN', 'BILLS_COUNT']
        
        collections_agg = self.collections_df.groupby(['NUM_REC_ULBID', 'NUM_REC_PROPID', 'VAR_TAX_ENAME_category']).agg({
            'COLL_ARREARS': 'sum',
            'COLL_CURRENT': 'sum',
            'COLL_TOTAL': 'sum'
        }).reset_index()
        collections_agg.columns = ['ULBID', 'PROPID', 'TAX_CATEGORY', 'COLL_ARREARS', 'COLL_CURRENT', 'COLL_TOTAL']
        
        merged_data = collections_agg.copy()
        
        merged_data = merged_data.merge(
            bills_agg, 
            on=['ULBID', 'PROPID', 'TAX_CATEGORY'], 
            how='left'
        )
        
        merged_data = merged_data.merge(
            arrears_agg, 
            on=['ULBID', 'PROPID', 'TAX_CATEGORY'], 
            how='left'
        )
        
        numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
        merged_data[numeric_columns] = merged_data[numeric_columns].fillna(0)
        
        merged_data['COLLECTION_RATE'] = np.where(
            merged_data['BILLS_SUM'] > 0,
            merged_data['COLL_TOTAL'] / merged_data['BILLS_SUM'],
            0
        )
        
        merged_data['ARREARS_RATIO'] = np.where(
            merged_data['BILLS_SUM'] > 0,
            merged_data['ARREARS_SUM'] / merged_data['BILLS_SUM'],
            0
        )
        
        merged_data['DEFAULT_INDICATOR'] = (merged_data['COLLECTION_RATE'] < 0.5).astype(int)
        
        # Cap ratios at reasonable values
        merged_data['COLLECTION_RATE'] = np.clip(merged_data['COLLECTION_RATE'], 0, 5)
        merged_data['ARREARS_RATIO'] = np.clip(merged_data['ARREARS_RATIO'], 0, 10)
        
        self.final_data = merged_data
        print(f"Final dataset shape: {self.final_data.shape}")
        
        return self.final_data
    
    def create_features_and_targets(self):
        """Create feature matrix and target variables"""
        print("Creating feature matrix and targets...")
        
        le_tax = LabelEncoder()
        self.final_data['TAX_CATEGORY_ENCODED'] = le_tax.fit_transform(self.final_data['TAX_CATEGORY'])
        self.encoders['tax_category'] = le_tax
        
        feature_columns = [
            'ULBID', 'PROPID', 'TAX_CATEGORY_ENCODED',
            'BILLS_SUM', 'BILLS_MEAN', 'BILLS_COUNT',
            'ARREARS_SUM', 'ARREARS_MEAN', 'ARREARS_COUNT',
            'COLLECTION_RATE', 'ARREARS_RATIO'
        ]
        
        valid_data = self.final_data[self.final_data['COLL_TOTAL'].notna()].copy()
        
        self.X = valid_data[feature_columns].fillna(0)
        self.y_total = valid_data['COLL_TOTAL']
        self.y_current = valid_data['COLL_CURRENT']
        self.y_arrears = valid_data['COLL_ARREARS']
        self.y_default = valid_data['DEFAULT_INDICATOR']
        
        self.feature_names = feature_columns
        
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Target variables created for total, current, arrears collections and default prediction")
        
        return self.X, self.y_total, self.y_current, self.y_arrears, self.y_default
    
    def train_models(self):
        """Train multiple models for different prediction tasks"""
        print("Training models...")
        
        X_train, X_test, y_total_train, y_total_test = train_test_split(
            self.X, self.y_total, test_size=0.2, random_state=42
        )
        
        _, _, y_current_train, y_current_test = train_test_split(
            self.X, self.y_current, test_size=0.2, random_state=42
        )
        
        _, _, y_arrears_train, y_arrears_test = train_test_split(
            self.X, self.y_arrears, test_size=0.2, random_state=42
        )
        
        _, _, y_default_train, y_default_test = train_test_split(
            self.X, self.y_default, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['features'] = scaler
        
        models_config = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'ridge': Ridge(alpha=1.0)
        }
        
        targets = {
            'total_collection': (y_total_train, y_total_test),
            'current_collection': (y_current_train, y_current_test),
            'arrears_collection': (y_arrears_train, y_arrears_test)
        }
        
        self.results = {}
        
        for target_name, (y_train, y_test) in targets.items():
            print(f"\nTraining models for {target_name}...")
            self.models[target_name] = {}
            self.results[target_name] = {}
            
            for model_name, model in models_config.items():
                if model_name == 'ridge':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.models[target_name][model_name] = model
                self.results[target_name][model_name] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2
                }
                
                print(f"{model_name} - MAE: {mae:.2f}, RMSE: {np.sqrt(mse):.2f}, R²: {r2:.3f}")
        
        self.X_test = X_test
        self.X_test_scaled = X_test_scaled
        self.test_targets = {
            'total_collection': y_total_test,
            'current_collection': y_current_test,
            'arrears_collection': y_arrears_test
        }
    
    def predict_next_year(self, model_type='random_forest'):
        print(f"\nGenerating predictions for next year using {model_type}...")
        
        predictions = {}
        
        for target_name in ['total_collection', 'current_collection', 'arrears_collection']:
            model = self.models[target_name][model_type]
            
            if model_type == 'ridge':
                pred = model.predict(self.X_test_scaled)
            else:
                pred = model.predict(self.X_test)
            
            predictions[target_name] = pred
            
            total_predicted = np.sum(pred)
            print(f"Predicted {target_name.replace('_', ' ')}: ₹{total_predicted:,.2f}")
        
        predicted_default_rate = np.mean(predictions['total_collection'] < (predictions['current_collection'] * 0.5))
        print(f"Predicted default rate: {predicted_default_rate:.2%}")
        
        return predictions
    
    def generate_insights(self):
        print("\n" + "="*50)
        print("BUSINESS INSIGHTS")
        print("="*50)
        
        tax_analysis = self.final_data.groupby('TAX_CATEGORY').agg({
            'COLL_TOTAL': ['sum', 'mean'],
            'COLLECTION_RATE': 'mean',
            'DEFAULT_INDICATOR': 'mean'
        }).round(2)
        
        print("\n1. Tax Category Performance:")
        print(tax_analysis)
        
        high_risk = self.final_data[
            (self.final_data['ARREARS_RATIO'] > 2) & 
            (self.final_data['COLLECTION_RATE'] < 0.3)
        ]
        
        print(f"\n2. High-Risk Properties: {len(high_risk)} properties")
        print(f"   Total arrears at risk: ₹{high_risk['ARREARS_SUM'].sum():,.2f}")
        
        ulb_performance = self.final_data.groupby('ULBID').agg({
            'COLL_TOTAL': 'sum',
            'COLLECTION_RATE': 'mean'
        }).sort_values('COLL_TOTAL', ascending=False).head(10)
        
        print("\n3. Top 10 Performing ULBs by Collection:")
        print(ulb_performance)
        
        best_model = self.models['total_collection']['random_forest']
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n4. Most Important Features for Prediction:")
        print(feature_importance.head(10))
    
    def save_models(self):
        print("\nSaving models...")
        
        os.makedirs('saved_models', exist_ok=True)
        
        for target_name, models in self.models.items():
            for model_name, model in models.items():
                filename = f'saved_models/{target_name}_{model_name}.joblib'
                joblib.dump(model, filename)
        
        joblib.dump(self.scalers, 'saved_models/scalers.joblib')
        joblib.dump(self.encoders, 'saved_models/encoders.joblib')
        joblib.dump(self.feature_names, 'saved_models/feature_names.joblib')
        
        print("Models saved successfully!")
    
    def create_visualizations(self):
        print("\nCreating visualizations...")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        tax_collection = self.final_data.groupby('TAX_CATEGORY')['COLLECTION_RATE'].mean().sort_values()
        axes[0, 0].barh(tax_collection.index, tax_collection.values)
        axes[0, 0].set_title('Average Collection Rate by Tax Category')
        axes[0, 0].set_xlabel('Collection Rate')
        
        sample_data = self.final_data.sample(min(1000, len(self.final_data)))
        axes[0, 1].scatter(sample_data['ARREARS_SUM'], sample_data['COLL_TOTAL'], alpha=0.6)
        axes[0, 1].set_title('Arrears vs Collections')
        axes[0, 1].set_xlabel('Arrears Amount')
        axes[0, 1].set_ylabel('Collection Amount')
        
        model_names = list(self.results['total_collection'].keys())
        r2_scores = [self.results['total_collection'][model]['r2'] for model in model_names]
        axes[1, 0].bar(model_names, r2_scores)
        axes[1, 0].set_title('Model Performance (R² Score)')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        best_model = self.models['total_collection']['random_forest']
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=True).tail(8)
        
        axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'])
        axes[1, 1].set_title('Top Feature Importance')
        axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('tax_prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'tax_prediction_analysis.png'")
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("Starting Tax Collection Prediction Pipeline")
        print("="*50)
        
        self.load_data()
        self.prepare_features()
        self.create_features_and_targets()
        
        self.train_models()
        
        predictions = self.predict_next_year()
        
        self.generate_insights()
        
        self.save_models()
        
        self.create_visualizations()
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return predictions

if __name__ == "__main__":
    predictor = TaxCollectionPredictor()
    predictions = predictor.run_complete_pipeline()
