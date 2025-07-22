# Tax Collection Prediction ML System

This repository contains a comprehensive Machine Learning system for predicting tax collection and default rates for the next fiscal year based on historical data.

## 📊 Data Overview

The system works with three main datasets:
- **arrears_query.csv**: Historical arrears (unpaid taxes) data
- **bill_query.csv**: Bill/demand data 
- **collection_query.csv**: Actual tax collections data

### Data Features
- **Multi-lingual support**: Handles both English and Marathi tax names
- **Large scale**: Processes millions of records efficiently
- **Comprehensive**: Covers multiple tax categories and ULBs

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Quick Data Analysis (Recommended first step)
```bash
python quick_analysis.py
```
This will give you insights into your data structure and patterns.

### 3. Train the Complete ML Pipeline
```bash
python tax_prediction_pipeline.py
```
This will:
- Load and clean all data
- Handle multilingual tax names
- Create features for ML
- Train multiple models
- Generate predictions for next year
- Save trained models
- Create visualizations
- Provide business insights

### 4. Use Trained Models for Predictions
```bash
python prediction_deployment.py
```

## 📁 File Structure

```
internship/
├── arrears_query.csv           # Arrears data
├── bill_query.csv              # Bills data  
├── collection_query.csv        # Collections data
├── tax_prediction_pipeline.py  # Main ML pipeline
├── quick_analysis.py           # Quick data exploration
├── prediction_deployment.py    # Deployment script
├── requirements.txt            # Python dependencies
├── saved_models/              # Trained models (created after training)
├── tax_prediction_analysis.png # Generated visualizations
└── README.md                  # This file
```

## 🔧 Technical Details

### Machine Learning Approach

The system uses an ensemble approach with multiple algorithms:

1. **Random Forest Regressor**: Handles non-linear relationships and feature interactions
2. **Gradient Boosting Regressor**: Captures complex patterns in tax collection behavior
3. **Ridge Regression**: Provides baseline linear predictions with regularization

### Key Features Created

1. **Property-level aggregations**: Sum, mean, count of bills and arrears
2. **Collection efficiency metrics**: Collection rates and arrears ratios
3. **Tax category encoding**: Standardized multilingual tax classifications
4. **Risk indicators**: Default probability based on historical patterns

### Prediction Targets

The system predicts:
- **Total tax collection** for next fiscal year
- **Current year collection** (new assessments)
- **Arrears collection** (historical dues)
- **Default risk** (probability of non-payment)

## 📈 Business Insights Generated

1. **Tax Category Performance**: Which tax types generate most revenue
2. **High-Risk Properties**: Properties with high default probability
3. **ULB Rankings**: Best and worst performing Urban Local Bodies
4. **Feature Importance**: Key factors driving tax collection

## 🎯 Key Predictions

After running the pipeline, you'll get:

- **Annual collection forecast**: Total expected revenue
- **Default rate prediction**: Percentage of properties likely to default
- **Category-wise breakdown**: Revenue by tax type
- **ULB-wise performance**: Collection efficiency by region

## 📊 Sample Output

```
ANNUAL TAX COLLECTION FORECAST - FY 2026
=========================================
Total Properties: 1,250,000
Total Bills Amount: ₹2,500,000,000.00
Predicted Collections: ₹2,100,000,000.00
Collection Rate: 84.0%
High Risk Properties: 125,000 (10.0%)
```

## 🔄 Using the System for New Predictions

### For Single Property
```python
from prediction_deployment import TaxCollectionPredictor

predictor = TaxCollectionPredictor()
predictor.load_trained_models()

prediction = predictor.predict_collection(
    ulbid=100,
    propid=12345,
    tax_category='General Tax',
    bills_sum=50000,
    arrears_sum=10000
)
```

### For Bulk Predictions
Create a CSV file with columns:
- `ULBID`: Urban Local Body ID
- `PROPID`: Property ID  
- `TAX_CATEGORY`: Type of tax
- `BILLS_SUM`: Total bill amount
- `ARREARS_SUM`: Outstanding arrears (optional)

```python
results = predictor.predict_from_csv('input_properties.csv', 'predictions.csv')
```

## 🏗️ Model Architecture

```
Input Data (Arrears + Bills + Collections)
    ↓
Data Cleaning & Multilingual Processing
    ↓
Feature Engineering
    ↓
Model Training (RF + GB + Ridge)
    ↓
Prediction & Evaluation
    ↓
Business Insights & Visualization
```

## 📊 Evaluation Metrics

- **MAE (Mean Absolute Error)**: Average prediction error in rupees
- **RMSE (Root Mean Square Error)**: Penalizes large errors
- **R² Score**: Explained variance (0-1, higher is better)
- **Collection Rate Accuracy**: How well the model predicts collection efficiency

## 🎨 Visualizations

The system generates:
1. **Collection rate by tax category**
2. **Arrears vs Collections scatter plot**
3. **Model performance comparison**
4. **Feature importance ranking**

## ⚠️ Important Notes

1. **Large Files**: CSV files are >50MB, processing may take time
2. **Memory Requirements**: Ensure sufficient RAM (8GB+ recommended)
3. **Multilingual Data**: System handles English and Marathi text
4. **Data Quality**: Zero values and missing data are handled automatically

## 🔧 Customization

### Adding New Tax Categories
Modify the `tax_mapping` and `marathi_mappings` dictionaries in `tax_prediction_pipeline.py`

### Changing Model Parameters
Update the `models_config` section to tune hyperparameters

### Adding New Features
Extend the `prepare_features()` method with additional calculated fields

## 🚀 Production Deployment

For production use:
1. Set up scheduled runs for model retraining
2. Implement data validation checks
3. Add monitoring for model performance drift
4. Create API endpoints using the deployment script

## 📞 Support

For questions or issues:
1. Check the console output for error messages
2. Ensure all CSV files are in the correct format
3. Verify Python dependencies are installed
4. Check available memory for large datasets

## 🎯 Expected Results

- **Accuracy**: 80-90% collection prediction accuracy
- **Insights**: Clear identification of high-risk properties
- **Efficiency**: Automated processing of millions of records
- **Actionability**: Specific recommendations for revenue optimization
