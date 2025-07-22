#!/usr/bin/env python3
"""
FastAPI Tax Collection Prediction Client Examples
================================================

This script demonstrates how to interact with the Tax Collection Prediction API
"""

import requests
import json
import pandas as pd
from pathlib import Path

# API base URL (change this to your deployed API URL)
API_BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if the API is running and healthy"""
    print("🔍 Testing API Health...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API is healthy!")
            print(f"   Predictor loaded: {health_data['predictor_loaded']}")
            print(f"   Available models: {health_data['models_available']}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_single_prediction():
    """Test single property prediction"""
    print("\n🏠 Testing Single Property Prediction...")
    
    # Sample property data
    property_data = {
        "ulbid": 100,
        "propid": 12345,
        "tax_category": "General Tax",
        "bills_sum": 50000,
        "arrears_sum": 10000,
        "model_type": "random_forest"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/single",
            json=property_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"   Property: {result['ulbid']}-{result['propid']}")
            print(f"   Bills Amount: ₹{result['bills_sum']:,.2f}")
            print(f"   Predicted Collection: ₹{result['predicted_total_collection']:,.2f}")
            print(f"   Collection Rate: {result['collection_rate']:.1%}")
            print(f"   Default Risk: {result['default_risk']}")
            return result
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def test_bulk_prediction():
    """Test bulk property prediction"""
    print("\n🏢 Testing Bulk Property Prediction...")
    
    # Sample bulk data
    bulk_data = {
        "properties": [
            {
                "ulbid": 100,
                "propid": 1001,
                "tax_category": "General Tax",
                "bills_sum": 75000,
                "arrears_sum": 15000
            },
            {
                "ulbid": 150,
                "propid": 2001,
                "tax_category": "Water Tax",
                "bills_sum": 25000,
                "arrears_sum": 5000
            },
            {
                "ulbid": 200,
                "propid": 3001,
                "tax_category": "Infrastructure",
                "bills_sum": 60000,
                "arrears_sum": 20000
            }
        ],
        "model_type": "random_forest"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/bulk",
            json=bulk_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Bulk prediction successful!")
            print(f"   Properties processed: {result['summary']['total_properties']}")
            print(f"   Total bills: ₹{result['summary']['total_bills']:,.2f}")
            print(f"   Total predicted: ₹{result['summary']['total_predicted']:,.2f}")
            print(f"   High risk properties: {result['summary']['high_risk_count']}")
            
            print("\n   Individual results:")
            for pred in result['predictions']:
                print(f"   - {pred['ulbid']}-{pred['propid']}: ₹{pred['predicted_total_collection']:,.2f} ({pred['default_risk']} risk)")
            
            return result
        else:
            print(f"❌ Bulk prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def test_database_upload():
    """Test database upload and prediction"""
    print("\n📁 Testing Database Upload...")
    
    # Check if sample file exists
    sample_file = "sample_properties.csv"
    if not Path(sample_file).exists():
        print(f"❌ Sample file {sample_file} not found. Creating one...")
        # Create a sample file
        sample_data = pd.DataFrame({
            'ULBID': [100, 150, 200, 250, 300],
            'PROPID': [1001, 2001, 3001, 4001, 5001],
            'TAX_CATEGORY': ['General Tax', 'Water Tax', 'Education/Cess', 'Infrastructure', 'Public Services'],
            'BILLS_SUM': [50000, 25000, 30000, 45000, 35000],
            'ARREARS_SUM': [10000, 5000, 8000, 12000, 7000]
        })
        sample_data.to_csv(sample_file, index=False)
        print(f"✅ Created sample file: {sample_file}")
    
    try:
        # Upload file
        with open(sample_file, 'rb') as f:
            files = {'file': (sample_file, f, 'text/csv')}
            response = requests.post(f"{API_BASE_URL}/upload/database", files=files)
        
        if response.status_code == 200:
            upload_result = response.json()
            print("✅ Database upload successful!")
            print(f"   Upload ID: {upload_result['upload_id']}")
            print(f"   Records: {upload_result['records_count']}")
            print(f"   Columns: {upload_result['columns']}")
            
            # Make predictions on uploaded data
            upload_id = upload_result['upload_id']
            return test_uploaded_prediction(upload_id)
            
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Upload request failed: {e}")
        return None

def test_uploaded_prediction(upload_id):
    """Test prediction on uploaded database"""
    print(f"\n🔮 Testing Prediction on Upload ID: {upload_id}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/from-upload/{upload_id}",
            params={"model_type": "random_forest"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Predictions completed!")
            print(f"   Records processed: {result['total_records_processed']}")
            print(f"   Total bills: ₹{result['summary']['total_bills_amount']:,.2f}")
            print(f"   Predicted collection: ₹{result['summary']['total_predicted_collection']:,.2f}")
            print(f"   Collection rate: {result['summary']['overall_collection_rate']:.1%}")
            print(f"   High risk properties: {result['summary']['high_risk_properties']}")
            print(f"   Download URL: {API_BASE_URL}{result['download_url']}")
            
            return upload_id
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Prediction request failed: {e}")
        return None

def test_summary_retrieval(upload_id):
    """Test summary retrieval"""
    print(f"\n📊 Testing Summary Retrieval for Upload: {upload_id}")
    
    try:
        response = requests.get(f"{API_BASE_URL}/summary/{upload_id}")
        
        if response.status_code == 200:
            summary = response.json()
            print("✅ Summary retrieved!")
            print(f"   Total properties: {summary['total_properties']}")
            print(f"   Total bills: ₹{summary['total_bills_amount']:,.2f}")
            print(f"   Predicted collection: ₹{summary['total_predicted_collection']:,.2f}")
            print(f"   Default rate: {summary['default_rate']:.1%}")
            
            print(f"\n   Top categories by collection:")
            for category, data in summary['category_breakdown']['predicted_total_collection'].items():
                print(f"   - {category}: ₹{data:,.2f}")
                
            return summary
        else:
            print(f"❌ Summary retrieval failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Summary request failed: {e}")
        return None

def test_model_info():
    """Test model information endpoint"""
    print("\n🤖 Testing Model Information...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        
        if response.status_code == 200:
            models = response.json()
            print("✅ Model information retrieved!")
            print(f"   Available models: {models['available_models']}")
            print(f"   Default model: {models['default_model']}")
            
            print(f"\n   Model descriptions:")
            for model, desc in models['model_descriptions'].items():
                print(f"   - {model}: {desc}")
                
            return models
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Model info request failed: {e}")
        return None

def test_categories():
    """Test tax categories endpoint"""
    print("\n🏷️  Testing Tax Categories...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/categories")
        
        if response.status_code == 200:
            categories = response.json()
            print("✅ Tax categories retrieved!")
            print(f"   Available categories: {categories['tax_categories']}")
            return categories
        else:
            print(f"❌ Categories retrieval failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Categories request failed: {e}")
        return None

def run_complete_api_test():
    """Run complete API test suite"""
    print("=" * 70)
    print("TAX COLLECTION PREDICTION API - COMPLETE TEST SUITE")
    print("=" * 70)
    
    # Test 1: Health check
    if not test_api_health():
        print("\n❌ API is not available. Please start the server first:")
        print("   python fastapi_tax_api.py")
        return
    
    # Test 2: Model and category info
    test_model_info()
    test_categories()
    
    # Test 3: Single prediction
    test_single_prediction()
    
    # Test 4: Bulk prediction
    test_bulk_prediction()
    
    # Test 5: Database upload and prediction
    upload_id = test_database_upload()
    
    if upload_id:
        # Test 6: Summary retrieval
        test_summary_retrieval(upload_id)
    
    print("\n" + "=" * 70)
    print("✅ API TEST SUITE COMPLETED!")
    print("=" * 70)
    print(f"\n🌐 API Documentation: {API_BASE_URL}/docs")
    print(f"🔍 API Health: {API_BASE_URL}/health")
    print(f"📊 Interactive Docs: {API_BASE_URL}/redoc")

if __name__ == "__main__":
    run_complete_api_test()
