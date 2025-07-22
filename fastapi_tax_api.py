#!/usr/bin/env python3
"""
FastAPI Tax Collection Prediction API
=====================================

A REST API for tax collection predictions with database upload capabilities
and JSON query support.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
import io
import json
from datetime import datetime
import uuid
import asyncio
from pathlib import Path

# Import our prediction system
from prediction_deployment import TaxCollectionPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Tax Collection Prediction API",
    description="ML-powered tax collection forecasting and default risk assessment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow web interface access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None
upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)

# Pydantic models for API requests/responses
class PropertyPredictionRequest(BaseModel):
    ulbid: int = Field(..., description="Urban Local Body ID")
    propid: int = Field(..., description="Property ID")
    tax_category: str = Field(..., description="Tax category (e.g., 'General Tax', 'Water Tax')")
    bills_sum: float = Field(..., gt=0, description="Total bill amount")
    bills_mean: Optional[float] = Field(None, description="Average bill amount")
    bills_count: Optional[int] = Field(1, description="Number of bills")
    arrears_sum: Optional[float] = Field(0, description="Total arrears amount")
    arrears_mean: Optional[float] = Field(None, description="Average arrears amount")
    arrears_count: Optional[int] = Field(1, description="Number of arrears records")
    model_type: Optional[str] = Field("random_forest", description="Model to use for prediction")

class PropertyPredictionResponse(BaseModel):
    ulbid: int
    propid: int
    tax_category: str
    bills_sum: float
    arrears_sum: float
    predicted_total_collection: float
    predicted_current_collection: float
    predicted_arrears_collection: float
    collection_rate: float
    default_risk: str
    confidence_score: float

class BulkPredictionRequest(BaseModel):
    properties: List[PropertyPredictionRequest]
    model_type: Optional[str] = "random_forest"

class DatabaseUploadResponse(BaseModel):
    upload_id: str
    filename: str
    records_count: int
    columns: List[str]
    status: str
    message: str

class PredictionSummary(BaseModel):
    total_properties: int
    total_bills_amount: float
    total_predicted_collection: float
    overall_collection_rate: float
    high_risk_properties: int
    default_rate: float
    category_breakdown: Dict[str, Any]
    ulb_breakdown: Dict[str, Any]

# Initialize the predictor
@app.on_event("startup")
async def startup_event():
    """Initialize the ML model on startup"""
    global predictor
    try:
        predictor = TaxCollectionPredictor()
        predictor.load_trained_models()
        print("✓ Tax Collection Predictor loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading predictor: {e}")
        raise

# Root endpoint
@app.get("/")
async def root():
    """API health check and information"""
    return {
        "message": "Tax Collection Prediction API",
        "version": "1.0.0",
        "status": "active",
        "available_models": list(predictor.models.keys()) if predictor else [],
        "endpoints": {
            "predict_single": "/predict/single",
            "predict_bulk": "/predict/bulk",
            "upload_database": "/upload/database",
            "predict_from_upload": "/predict/from-upload/{upload_id}",
            "get_prediction_summary": "/summary/{upload_id}",
            "docs": "/docs"
        }
    }

# Single property prediction
@app.post("/predict/single", response_model=PropertyPredictionResponse)
async def predict_single_property(request: PropertyPredictionRequest):
    """Predict tax collection for a single property"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Make prediction
        prediction = predictor.predict_collection(
            ulbid=request.ulbid,
            propid=request.propid,
            tax_category=request.tax_category,
            bills_sum=request.bills_sum,
            bills_mean=request.bills_mean,
            bills_count=request.bills_count,
            arrears_sum=request.arrears_sum or 0,
            arrears_mean=request.arrears_mean,
            arrears_count=request.arrears_count,
            model_type=request.model_type
        )
        
        # Calculate additional metrics
        collection_rate = prediction['total_collection'] / request.bills_sum if request.bills_sum > 0 else 0
        default_risk = "HIGH" if collection_rate < 0.5 else "MEDIUM" if collection_rate < 0.8 else "LOW"
        confidence_score = min(1.0, max(0.1, collection_rate))  # Simple confidence based on collection rate
        
        return PropertyPredictionResponse(
            ulbid=request.ulbid,
            propid=request.propid,
            tax_category=request.tax_category,
            bills_sum=request.bills_sum,
            arrears_sum=request.arrears_sum or 0,
            predicted_total_collection=prediction['total_collection'],
            predicted_current_collection=prediction['current_collection'],
            predicted_arrears_collection=prediction['arrears_collection'],
            collection_rate=collection_rate,
            default_risk=default_risk,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Bulk property prediction
@app.post("/predict/bulk")
async def predict_bulk_properties(request: BulkPredictionRequest):
    """Predict tax collection for multiple properties"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    if len(request.properties) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 properties per bulk request")
    
    try:
        results = []
        for prop in request.properties:
            prediction = predictor.predict_collection(
                ulbid=prop.ulbid,
                propid=prop.propid,
                tax_category=prop.tax_category,
                bills_sum=prop.bills_sum,
                bills_mean=prop.bills_mean,
                bills_count=prop.bills_count,
                arrears_sum=prop.arrears_sum or 0,
                arrears_mean=prop.arrears_mean,
                arrears_count=prop.arrears_count,
                model_type=request.model_type
            )
            
            collection_rate = prediction['total_collection'] / prop.bills_sum if prop.bills_sum > 0 else 0
            default_risk = "HIGH" if collection_rate < 0.5 else "MEDIUM" if collection_rate < 0.8 else "LOW"
            
            results.append(PropertyPredictionResponse(
                ulbid=prop.ulbid,
                propid=prop.propid,
                tax_category=prop.tax_category,
                bills_sum=prop.bills_sum,
                arrears_sum=prop.arrears_sum or 0,
                predicted_total_collection=prediction['total_collection'],
                predicted_current_collection=prediction['current_collection'],
                predicted_arrears_collection=prediction['arrears_collection'],
                collection_rate=collection_rate,
                default_risk=default_risk,
                confidence_score=min(1.0, max(0.1, collection_rate))
            ))
        
        return {
            "predictions": results,
            "summary": {
                "total_properties": len(results),
                "total_bills": sum(r.bills_sum for r in results),
                "total_predicted": sum(r.predicted_total_collection for r in results),
                "high_risk_count": sum(1 for r in results if r.default_risk == "HIGH")
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bulk prediction error: {str(e)}")

# Database upload endpoint
@app.post("/upload/database", response_model=DatabaseUploadResponse)
async def upload_database(file: UploadFile = File(...)):
    """Upload a CSV database file for batch predictions"""
    
    # Validate file type
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
    
    # Generate unique upload ID
    upload_id = str(uuid.uuid4())
    
    try:
        # Read file content
        content = await file.read()
        
        # Save uploaded file
        file_path = upload_dir / f"{upload_id}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Read and validate the data
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Validate required columns
        required_columns = ['ULBID', 'PROPID', 'TAX_CATEGORY', 'BILLS_SUM']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            os.remove(file_path)  # Clean up
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}. Required: {required_columns}"
            )
        
        return DatabaseUploadResponse(
            upload_id=upload_id,
            filename=file.filename,
            records_count=len(df),
            columns=list(df.columns),
            status="success",
            message=f"Database uploaded successfully. {len(df)} records ready for prediction."
        )
        
    except Exception as e:
        # Clean up on error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Upload error: {str(e)}")

# Predict from uploaded database
@app.post("/predict/from-upload/{upload_id}")
async def predict_from_upload(
    upload_id: str, 
    background_tasks: BackgroundTasks,
    model_type: str = "random_forest"
):
    """Make predictions on uploaded database"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    # Find uploaded file
    upload_files = list(upload_dir.glob(f"{upload_id}_*"))
    if not upload_files:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    file_path = upload_files[0]
    
    try:
        # Read the uploaded data
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Prepare data for prediction
        if 'ARREARS_SUM' not in df.columns:
            df['ARREARS_SUM'] = 0
        
        # Create a temporary CSV for the predictor
        temp_csv = upload_dir / f"temp_{upload_id}.csv"
        df.to_csv(temp_csv, index=False)
        
        # Make predictions
        results_df = predictor.predict_from_csv(
            str(temp_csv), 
            str(upload_dir / f"predictions_{upload_id}.csv"),
            model_type=model_type
        )
        
        # Clean up temp file
        os.remove(temp_csv)
        
        # Calculate summary statistics
        summary = {
            "total_properties": len(results_df),
            "total_bills_amount": float(results_df['BILLS_SUM'].sum()),
            "total_predicted_collection": float(results_df['predicted_total_collection'].sum()),
            "overall_collection_rate": float(results_df['predicted_collection_rate'].mean()),
            "high_risk_properties": int(results_df['predicted_default_risk'].sum()),
            "default_rate": float(results_df['predicted_default_risk'].mean()),
            "category_breakdown": results_df.groupby('TAX_CATEGORY').agg({
                'predicted_total_collection': 'sum',
                'predicted_default_risk': 'mean'
            }).to_dict(),
            "ulb_breakdown": results_df.groupby('ULBID').agg({
                'predicted_total_collection': 'sum',
                'predicted_default_risk': 'mean'
            }).head(10).to_dict()
        }
        
        return {
            "upload_id": upload_id,
            "status": "completed",
            "summary": summary,
            "download_url": f"/download/predictions/{upload_id}",
            "total_records_processed": len(results_df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Download predictions
@app.get("/download/predictions/{upload_id}")
async def download_predictions(upload_id: str):
    """Download prediction results as CSV"""
    predictions_file = upload_dir / f"predictions_{upload_id}.csv"
    
    if not predictions_file.exists():
        raise HTTPException(status_code=404, detail="Predictions not found")
    
    return FileResponse(
        predictions_file,
        media_type='text/csv',
        filename=f"tax_predictions_{upload_id}.csv"
    )

# Get prediction summary
@app.get("/summary/{upload_id}", response_model=PredictionSummary)
async def get_prediction_summary(upload_id: str):
    """Get summary of predictions for an upload"""
    predictions_file = upload_dir / f"predictions_{upload_id}.csv"
    
    if not predictions_file.exists():
        raise HTTPException(status_code=404, detail="Predictions not found")
    
    try:
        df = pd.read_csv(predictions_file)
        
        # Calculate category breakdown
        category_breakdown = df.groupby('TAX_CATEGORY').agg({
            'predicted_total_collection': 'sum',
            'predicted_collection_rate': 'mean',
            'predicted_default_risk': 'mean'
        }).round(2).to_dict()
        
        # Calculate ULB breakdown (top 10)
        ulb_breakdown = df.groupby('ULBID').agg({
            'predicted_total_collection': 'sum',
            'predicted_collection_rate': 'mean'
        }).sort_values('predicted_total_collection', ascending=False).head(10).round(2).to_dict()
        
        return PredictionSummary(
            total_properties=len(df),
            total_bills_amount=float(df['BILLS_SUM'].sum()),
            total_predicted_collection=float(df['predicted_total_collection'].sum()),
            overall_collection_rate=float(df['predicted_collection_rate'].mean()),
            high_risk_properties=int(df['predicted_default_risk'].sum()),
            default_rate=float(df['predicted_default_risk'].mean()),
            category_breakdown=category_breakdown,
            ulb_breakdown=ulb_breakdown
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Summary calculation error: {str(e)}")

# Get available tax categories
@app.get("/categories")
async def get_tax_categories():
    """Get list of supported tax categories"""
    return {
        "tax_categories": [
            "General Tax",
            "Water Tax", 
            "Education/Cess",
            "Infrastructure",
            "Public Services",
            "Penalty",
            "Other"
        ],
        "description": "Supported tax categories for predictions"
    }

# Get model information
@app.get("/models")
async def get_model_info():
    """Get information about available models"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    return {
        "available_models": list(predictor.models.keys()),
        "default_model": "random_forest",
        "model_descriptions": {
            "random_forest": "Best overall performance, handles non-linear relationships",
            "gradient_boosting": "Good for complex patterns, moderate speed",
            "ridge": "Fast linear predictions, good baseline"
        }
    }

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "predictor_loaded": predictor is not None,
        "models_available": list(predictor.models.keys()) if predictor else [],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_tax_api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        title="Tax Collection Prediction API"
    )
