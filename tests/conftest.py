import os
import warnings

# Automatically enforce single-threaded OpenMP execution for tests to prevent
# segmentation faults caused by conflicts between Intel OpenMP (PyTorch) and 
# LLVM OpenMP (Scikit-learn).
os.environ["OMP_NUM_THREADS"] = "1"

# The specific RuntimeWarning from threadpoolctl about multiple OpenMP 
# libraries is suppressed in pytest.ini to ensure a clean test output.
