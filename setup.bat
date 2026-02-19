@echo off
echo ============================================================
echo   TurboForge - Full Auto Setup
echo ============================================================

:: Create folders and init files
mkdir data 2>nul
mkdir models 2>nul
mkdir utils 2>nul
echo. > data\__init__.py
echo. > models\__init__.py
echo. > utils\__init__.py
echo [OK] Folders created

:: Install packages
echo.
echo Installing packages...
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm anthropic --quiet
echo [OK] Packages installed

:: Clear pycache
if exist models\__pycache__ rmdir /s /q models\__pycache__
if exist data\__pycache__ rmdir /s /q data\__pycache__
if exist utils\__pycache__ rmdir /s /q utils\__pycache__
echo [OK] Cache cleared

:: Generate synthetic data
echo.
echo Generating SCADA data...
python generate_data.py
echo [OK] Data ready

:: Smoke test
echo.
echo Running smoke test...
python main.py --mode train --epochs 2 --skip_gan

echo.
echo ============================================================
echo   Done! Run: python main.py --mode train --epochs 50 --skip_gan
echo ============================================================
pause
