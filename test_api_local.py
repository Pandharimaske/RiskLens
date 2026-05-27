#!/usr/bin/env python
"""
Test FastAPI application locally (without Docker).
Tests all endpoints to ensure deployment readiness.
"""

import requests
import json
import pandas as pd
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_DATA = {
    "id": "CUST_TEST_001",
    "gender": "Male",
    "age": 35,
    "driving_license": 1,
    "region_code": 28,
    "previously_insured": 0,
    "vehicle_age": "1-2 Year",
    "vehicle_damage": "No",
    "annual_premium": 40000,
    "policy_sales_channel": 26,
    "vintage": 200
}

def test_health():
    """Test health endpoint."""
    print("\n[1] Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        print(f"   Status: {data['status']}")
        print(f"   Model loaded: {data['model_loaded']}")
        print(f"   Model version: {data['model_version']}")
        print("   ✓ PASS")
        return True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False

def test_model_info():
    """Test model info endpoint."""
    print("\n[2] Testing /model-info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model-info", timeout=5)
        assert response.status_code == 200
        data = response.json()
        print(f"   Model: {data['model_name']}")
        print(f"   Version: {data['model_version']}")
        print(f"   Threshold: {data['decision_threshold']}")
        print("   ✓ PASS")
        return True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False

def test_single_prediction():
    """Test single prediction endpoint."""
    print("\n[3] Testing /predict endpoint (single)...")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=TEST_DATA,
            timeout=5
        )
        assert response.status_code == 200
        data = response.json()
        print(f"   Customer ID: {data['id']}")
        print(f"   Probability: {data['probability']:.4f}")
        print(f"   Prediction: {data['prediction']}")
        print(f"   Threshold: {data['threshold']}")
        assert 0 <= data['probability'] <= 1
        assert data['prediction'] in [0, 1]
        print("   ✓ PASS")
        return True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\n[4] Testing /predict-batch endpoint...")
    try:
        # Create test batch
        batch_data = [
            {**TEST_DATA, "id": f"CUST_{i:03d}"} 
            for i in range(1, 11)  # 10 records
        ]
        df = pd.DataFrame(batch_data)
        
        # Save to temporary CSV
        csv_path = Path("/tmp/test_batch.csv")
        df.to_csv(csv_path, index=False)
        
        # Upload
        with open(csv_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{BASE_URL}/predict-batch",
                files=files,
                timeout=10
            )
        
        assert response.status_code == 200
        data = response.json()
        print(f"   Batch size: {data['batch_size']}")
        print(f"   Predictions returned: {len(data['predictions'])}")
        assert data['batch_size'] == 10
        assert len(data['predictions']) == 10
        print("   ✓ PASS")
        return True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False

def test_error_handling():
    """Test error handling."""
    print("\n[5] Testing error handling...")
    try:
        # Invalid data
        bad_data = {"id": "TEST"}  # Missing required fields
        response = requests.post(
            f"{BASE_URL}/predict",
            json=bad_data,
            timeout=5
        )
        assert response.status_code == 422  # Validation error
        print("   Validation error correctly returned")
        print("   ✓ PASS")
        return True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False

def main():
    print("=" * 70)
    print("RISKLENS API TEST SUITE")
    print("=" * 70)
    print(f"\nBase URL: {BASE_URL}")
    print("Testing API endpoints...")
    
    # Run tests
    results = {
        "Health": test_health(),
        "Model Info": test_model_info(),
        "Single Prediction": test_single_prediction(),
        "Batch Prediction": test_batch_prediction(),
        "Error Handling": test_error_handling(),
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! API is ready for deployment.")
        return True
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed.")
        return False

if __name__ == "__main__":
    print("\n⚠️  Make sure the API is running before starting tests!")
    print("   Run: uvicorn app:app --reload")
    input("\nPress Enter to continue with tests (Ctrl+C to cancel)...")
    
    success = main()
    exit(0 if success else 1)
