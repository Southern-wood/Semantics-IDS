import os
os.environ["OMP_NUM_THREADS"] = "8"  # OpenMP thread
os.environ["OPENBLAS_NUM_THREADS"] = "8"  # OpenBLAS thread
os.environ["MKL_NUM_THREADS"] = "8"  # MKL thread
os.environ["NUMEXPR_NUM_THREADS"] = "8"  # NumExpr thread