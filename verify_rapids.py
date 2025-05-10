#!/usr/bin/env python3
"""
RAPIDS Installation Verification Script

This script verifies that RAPIDS components are properly installed and working.
It tests the basic functionality of cuDF, cuML, and other RAPIDS libraries.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

def print_header(message):
    """Print a formatted header message."""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)

def print_success(message):
    """Print a success message."""
    print(f"✅ {message}")

def print_warning(message):
    """Print a warning message."""
    print(f"⚠️ {message}")

def print_error(message):
    """Print an error message."""
    print(f"❌ {message}")

def check_gpu():
    """Check if CUDA-capable GPU is available."""
    try:
        import cupy as cp
        gpu_info = cp.cuda.runtime.getDeviceProperties(0)
        print_success(f"GPU detected: {gpu_info['name'].decode()}")
        print_success(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        print_success(f"GPU memory: {gpu_info['totalGlobalMem'] / (1024**3):.2f} GB")
        return True
    except ImportError:
        print_error("CuPy not installed. Cannot detect GPU information.")
        return False
    except Exception as e:
        print_error(f"Error detecting GPU: {e}")
        return False

def test_cudf():
    """Test cuDF functionality."""
    print_header("Testing cuDF")
    try:
        import cudf
        
        # Create a simple DataFrame
        print("Creating cuDF DataFrame...")
        df = cudf.DataFrame({
            'A': np.random.randint(0, 100, size=1000000),
            'B': np.random.normal(0, 1, size=1000000),
            'C': np.random.choice(['X', 'Y', 'Z'], size=1000000)
        })
        
        # Basic operations
        print("Performing basic operations...")
        start_time = time.time()
        result = df.groupby('C').agg({'A': 'mean', 'B': 'std'})
        gpu_time = time.time() - start_time
        print(f"Result shape: {result.shape}")
        print(f"GPU execution time: {gpu_time:.4f} seconds")
        
        # Compare with pandas
        print("Comparing with pandas...")
        pdf = df.to_pandas()
        start_time = time.time()
        pandas_result = pdf.groupby('C').agg({'A': 'mean', 'B': 'std'})
        cpu_time = time.time() - start_time
        print(f"Pandas execution time: {cpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
        
        print_success("cuDF test completed successfully")
        return True
    except ImportError:
        print_error("cuDF not installed")
        return False
    except Exception as e:
        print_error(f"Error testing cuDF: {e}")
        return False

def test_cuml():
    """Test cuML functionality."""
    print_header("Testing cuML")
    try:
        import cuml
        from cuml.datasets import make_blobs
        from cuml.metrics import accuracy_score
        
        # Generate synthetic data
        print("Generating synthetic data...")
        X, y = make_blobs(n_samples=10000, n_features=20, centers=2, random_state=42)
        
        # Train a model
        print("Training RandomForestClassifier...")
        from cuml.ensemble import RandomForestClassifier
        start_time = time.time()
        model = RandomForestClassifier(n_estimators=100, max_depth=10)
        model.fit(X, y)
        preds = model.predict(X)
        gpu_time = time.time() - start_time
        acc = accuracy_score(y, preds)
        print(f"Accuracy: {acc:.4f}")
        print(f"GPU execution time: {gpu_time:.4f} seconds")
        
        # Compare with scikit-learn
        print("Comparing with scikit-learn...")
        from sklearn.ensemble import RandomForestClassifier as SklearnRF
        X_np = X.get() if hasattr(X, 'get') else X
        y_np = y.get() if hasattr(y, 'get') else y
        
        start_time = time.time()
        sk_model = SklearnRF(n_estimators=100, max_depth=10, random_state=42)
        sk_model.fit(X_np, y_np)
        sk_preds = sk_model.predict(X_np)
        cpu_time = time.time() - start_time
        sk_acc = accuracy_score(y_np, sk_preds)
        print(f"Scikit-learn accuracy: {sk_acc:.4f}")
        print(f"CPU execution time: {cpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
        
        print_success("cuML test completed successfully")
        return True
    except ImportError:
        print_error("cuML not installed")
        return False
    except Exception as e:
        print_error(f"Error testing cuML: {e}")
        return False

def test_dask_cudf():
    """Test dask_cudf functionality."""
    print_header("Testing dask_cudf")
    try:
        import dask_cudf
        import cudf
        
        # Create a dask_cudf DataFrame
        print("Creating dask_cudf DataFrame...")
        df = cudf.DataFrame({
            'A': np.random.randint(0, 100, size=1000000),
            'B': np.random.normal(0, 1, size=1000000)
        })
        
        ddf = dask_cudf.from_cudf(df, npartitions=4)
        print(f"Number of partitions: {ddf.npartitions}")
        
        # Perform operations
        print("Performing operations...")
        # Create a groupby key column to avoid ambiguity
        ddf = ddf.assign(key=ddf.A % 10)
        result = ddf.groupby('key').B.mean().compute()
        print(f"Result shape: {result.shape}")
        
        print_success("dask_cudf test completed successfully")
        return True
    except ImportError:
        print_warning("dask_cudf not installed (optional component)")
        return False
    except Exception as e:
        print_error(f"Error testing dask_cudf: {e}")
        return False

def main():
    """Main function to run all tests."""
    print_header("RAPIDS Installation Verification")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check if GPU is available
    gpu_available = check_gpu()
    if not gpu_available:
        print_warning("No GPU detected or CUDA not properly configured")
        print_warning("RAPIDS requires an NVIDIA GPU with CUDA support")
        return
    
    # Run tests
    tests = [
        ("cuDF", test_cudf),
        ("cuML", test_cuml),
        ("dask_cudf", test_dask_cudf)
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    # Print summary
    print_header("Test Summary")
    for name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")
    
    # Overall assessment
    if all(results.values()):
        print_success("All tests passed! RAPIDS is properly installed and working.")
    elif results["cuDF"] and results["cuML"]:
        print_success("Core components (cuDF, cuML) are working properly.")
        print_warning("Some optional components failed. See details above.")
    else:
        print_error("Some core components failed. RAPIDS installation may be incomplete or incorrect.")
        print_warning("Please check the error messages above and refer to the installation instructions.")

if __name__ == "__main__":
    main()
