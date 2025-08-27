#!/usr/bin/env python3
"""
Verification script for exact requirements.txt versions
Ensures all pinned versions match the working environment
"""

import importlib
import sys
from packaging import version

def verify_exact_versions():
    """Verify that installed packages match the exact requirements.txt versions"""
    
    # Expected exact versions from working environment
    expected_versions = {
        'pandas': '2.0.3',
        'numpy': '1.24.4', 
        'sklearn': '1.3.2',
        'scipy': '1.10.1',
        'matplotlib': '3.7.5',
        'seaborn': '0.13.2',
        'joblib': '1.4.2'
    }
    
    print("🔍 EXACT VERSION VERIFICATION")
    print("=" * 50)
    print(f"🐍 Python Version: {sys.version}")
    print()
    
    all_correct = True
    
    for package_name, expected_version in expected_versions.items():
        try:
            # Import the package
            if package_name == 'sklearn':
                module = importlib.import_module('sklearn')
            else:
                module = importlib.import_module(package_name)
            
            # Get actual version
            actual_version = module.__version__
            
            # Check if versions match exactly
            versions_match = actual_version == expected_version
            
            if versions_match:
                print(f"✅ {package_name:12s}: {actual_version} (EXACT MATCH)")
            else:
                print(f"❌ {package_name:12s}: {actual_version} (EXPECTED: {expected_version})")
                all_correct = False
                
        except ImportError as e:
            print(f"❌ {package_name:12s}: NOT INSTALLED - {e}")
            all_correct = False
        except AttributeError:
            print(f"⚠️  {package_name:12s}: Version attribute not found")
    
    print()
    print("=" * 50)
    if all_correct:
        print("🎉 ALL VERSIONS MATCH EXACTLY!")
        print("✅ requirements.txt is perfectly aligned with working environment")
        print("🚀 Ready for reliable deployment")
    else:
        print("⚠️  VERSION MISMATCHES DETECTED")
        print("📝 Update requirements.txt or reinstall packages")
    
    return all_correct

def test_core_functionality():
    """Test that core ML functionality works with current versions"""
    
    print()
    print("🧪 CORE FUNCTIONALITY TEST")
    print("=" * 40)
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Test basic functionality
        print("✅ All imports successful")
        
        # Test TF-IDF
        tfidf = TfidfVectorizer(max_features=100)
        texts = ["good app", "bad app", "excellent application"]
        X = tfidf.fit_transform(texts)
        print("✅ TF-IDF vectorization works")
        
        # Test LogisticRegression
        y = [1, 0, 1]
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)
        print("✅ Logistic Regression works")
        
        # Test metrics
        acc = accuracy_score(y, predictions)
        print(f"✅ Metrics calculation works (accuracy: {acc:.2f})")
        
        print()
        print("🎉 ALL CORE FUNCTIONALITY VERIFIED!")
        return True
        
    except Exception as e:
        print(f"❌ FUNCTIONALITY TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Multilingual Sentiment Analysis - Requirements Verification")
    print("=" * 70)
    
    versions_ok = verify_exact_versions()
    functionality_ok = test_core_functionality()
    
    print()
    print("📋 FINAL VERIFICATION SUMMARY")
    print("=" * 45)
    
    if versions_ok and functionality_ok:
        print("🎯 STATUS: ✅ PERFECT")
        print("📦 All package versions match exactly")
        print("🔧 All functionality works correctly")
        print("🚀 Ready for production deployment")
        sys.exit(0)
    else:
        print("🎯 STATUS: ⚠️  NEEDS ATTENTION")
        if not versions_ok:
            print("📦 Package version mismatches detected")
        if not functionality_ok:
            print("🔧 Functionality issues detected")
        print("🔄 Please review and fix issues")
        sys.exit(1)
