@echo off
REM Set all threading environment variables to avoid Intel MKL issues
setlocal enabledelayedexpansion

set MKL_THREADING_LAYER=GNU
set KMP_DUPLICATE_LIB_OK=True
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set NUMEXPR_MAX_THREADS=1

echo Training with MKL workarounds...
echo Running: python train_clean.py
python train_clean.py

echo.
echo Training finished!
pause
