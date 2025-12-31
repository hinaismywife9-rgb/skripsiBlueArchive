@echo off
REM Fix NumPy/MKL issue
echo Installing compatible NumPy version...
pip uninstall numpy -y
pip install numpy==1.26.0
echo.
echo Now running training...
python train_quick.py
pause
