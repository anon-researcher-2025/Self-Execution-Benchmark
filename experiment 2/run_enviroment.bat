@echo off

:: Define the virtual environment folder name
set VENV_DIR=.venv

:: Check if the virtual environment exists
if not exist "%VENV_DIR%" (
    echo Virtual environment not found. Creating one...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo Failed to create virtual environment. Ensure Python is installed and added to PATH.
        exit /b 1
    )
    echo Virtual environment created successfully.
)

:: Activate the virtual environment
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    exit /b 1
)

:: Update pip
echo Updating pip to the latest version...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to update pip.
    exit /b 1
)

:: Install requirements
if exist "requierments.txt" (
    echo Checking for missing requirements...
    for /f "delims=" %%r in ('pip freeze') do (
        echo %%r >> installed_packages.txt
    )
    for /f "delims=" %%p in (requierments.txt) do (
        findstr /i /c:"%%p" installed_packages.txt >nul || (
            echo Installing missing package: %%p
            pip install %%p
        )
    )
    del installed_packages.txt
    echo Requirements check completed.
) else (
    echo requierments.txt not found. Skipping requirements installation.
)

:: Run the virtual environment
echo Virtual environment is now active. You can run your project.
cmd /k