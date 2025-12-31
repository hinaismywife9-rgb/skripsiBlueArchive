@echo off
setlocal enabledelayedexpansion

REM Set environment variables to avoid Intel MKL threading issues
set MKL_THREADING_LAYER=GNU
set KMP_DUPLICATE_LIB_OK=True
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

REM Run training
python train_simple.py

pause
