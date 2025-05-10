import cudf
import cuml
print(f"cuDF version: {cudf.__version__}")
print(f"cuML version: {cuml.__version__}")

# Test with a simple example
s = cudf.Series([1, 2, 3, 4, 5])
print(f"Mean of series: {s.mean()}")
