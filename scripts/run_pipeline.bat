@echo off
echo ========================================
echo Melanoma Detection Pipeline Launcher
echo Bittensor Subnet 76 Competition
echo ========================================
echo.

echo Choose an option:
echo 1. Create configuration file
echo 2. Run complete pipeline
echo 3. Prepare dataset only
echo 4. Train model only
echo 5. Show help
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Creating configuration file...
    python scripts\complete_pipeline.py --create_config
    echo.
    echo Configuration file created! Please edit pipeline_config.json with your credentials.
    pause
    goto :eof
)

if "%choice%"=="2" (
    echo.
    echo Running complete pipeline...
    if exist pipeline_config.json (
        python scripts\complete_pipeline.py --config pipeline_config.json
    ) else (
        echo Configuration file not found! Please create one first (option 1).
    )
    pause
    goto :eof
)

if "%choice%"=="3" (
    echo.
    echo Preparing dataset...
    set /p samples="Enter number of samples (default 200): "
    if "%samples%"=="" set samples=200
    python scripts\prepare_dataset.py --action sample --num_samples %samples%
    pause
    goto :eof
)

if "%choice%"=="4" (
    echo.
    echo Training model...
    set /p csv="Enter CSV file path: "
    set /p img_dir="Enter image directory path: "
    set /p epochs="Enter number of epochs (default 10): "
    if "%epochs%"=="" set epochs=10
    python scripts\train_melanoma_model.py --csv_file %csv% --img_dir %img_dir% --epochs %epochs% --convert_onnx
    pause
    goto :eof
)

if "%choice%"=="5" (
    echo.
    echo ========================================
    echo HELP - Melanoma Detection Pipeline
    echo ========================================
    echo.
    echo This pipeline helps you:
    echo 1. Create and prepare melanoma datasets
    echo 2. Train a CNN model for melanoma detection
    echo 3. Convert model to ONNX format
    echo 4. Submit to Bittensor subnet 76 competition
    echo.
    echo Prerequisites:
    echo - Python 3.8+ installed
    echo - Required packages installed (see requirements_training.txt)
    echo - Hugging Face account and token
    echo - Bittensor wallet set up
    echo.
    echo For detailed instructions, see README_TRAINING.md
    echo.
    pause
    goto :eof
)

echo Invalid choice! Please enter 1-5.
pause
goto :eof
