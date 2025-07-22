#!/usr/bin/env python3
"""
Tax Collection Prediction API Server Startup Script
==================================================

This script starts the FastAPI server for tax collection predictions.
"""

import uvicorn
import os
import sys
from pathlib import Path

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting Tax Collection Prediction API Server...")
    print("=" * 60)
    
    # Check if models exist
    models_dir = Path("saved_models")
    if not models_dir.exists() or not any(models_dir.glob("*.joblib")):
        print("❌ Error: Trained models not found!")
        print("   Please run the training pipeline first:")
        print("   python efficient_tax_predictor.py")
        return
    
    print("✅ Models found in saved_models/")
    
    # Create uploads directory
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    print("✅ Uploads directory ready")
    
    print("\n📡 Server Configuration:")
    print("   Host: 0.0.0.0 (accessible from all interfaces)")
    print("   Port: 8000")
    print("   Reload: Enabled (development mode)")
    
    print("\n🌐 API Endpoints will be available at:")
    print("   Main API: http://localhost:8000/")
    print("   Interactive Docs: http://localhost:8000/docs")
    print("   ReDoc: http://localhost:8000/redoc")
    print("   Health Check: http://localhost:8000/health")
    
    print("\n🔧 API Features:")
    print("   ✓ Single property predictions")
    print("   ✓ Bulk property predictions") 
    print("   ✓ Database file upload (CSV/Excel)")
    print("   ✓ Batch predictions on uploaded data")
    print("   ✓ Download prediction results")
    print("   ✓ Summary statistics and analytics")
    
    print("\n" + "=" * 60)
    print("🚀 STARTING SERVER...")
    print("   Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "fastapi_tax_api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            access_log=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == "__main__":
    start_server()
