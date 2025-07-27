#!/bin/bash


# Define the virtual environment folder name
VENV_DIR=".venv"

# Detect Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
VENV_PACKAGE="python3-venv"
if [[ -n "$PYTHON_VERSION" ]]; then
    VENV_PACKAGE="python${PYTHON_VERSION}-venv"
fi
echo "Detected Python version: ${PYTHON_VERSION}, will use package: ${VENV_PACKAGE}"

# Check if python3-venv is installed for the specific Python version
if ! python3 -c "import venv" > /dev/null 2>&1; then
    echo "${VENV_PACKAGE} is not installed. Installing it..."
    echo "You may be prompted for your password (sudo)."
    if command -v apt > /dev/null 2>&1; then
        sudo apt update && sudo apt install -y ${VENV_PACKAGE} python3-pip
    elif command -v dnf > /dev/null 2>&1; then
        sudo dnf install -y ${VENV_PACKAGE} python3-pip
    elif command -v yum > /dev/null 2>&1; then
        sudo yum install -y ${VENV_PACKAGE} python3-pip
    else
        echo "Package manager not found. Please install ${VENV_PACKAGE} manually."
        exit 1
    fi
    if [ $? -ne 0 ]; then
        echo "Failed to install ${VENV_PACKAGE}. Please install it manually and try again."
        echo "Run: sudo apt install ${VENV_PACKAGE}"
        exit 1
    fi
    echo "${VENV_PACKAGE} installed successfully."
fi

# Remove existing virtual environment if it's broken or has permission issues
if [ -d "$VENV_DIR" ]; then
    if [ ! -f "$VENV_DIR/bin/activate" ] || [ ! -w "$VENV_DIR" ]; then
        echo "Found broken or permission-denied virtual environment. Removing it..."
        # Try normal removal first
        rm -rf "$VENV_DIR" 2>/dev/null || sudo rm -rf "$VENV_DIR"
        
        # Make sure the directory is gone
        if [ -d "$VENV_DIR" ]; then
            echo "Failed to remove virtual environment. Please run manually:"
            echo "sudo rm -rf $VENV_DIR"
            exit 1
        fi
    fi
fi

# Check if the virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        echo "Please run: sudo apt install ${VENV_PACKAGE}"
        exit 1
    fi
    echo "Virtual environment created successfully."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

# Update pip
echo "Updating pip to the latest version..."
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Failed to update pip."
    exit 1
fi

# Install requirements if the file exists
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install some requirements. Please check the requirements.txt file."
    else
        echo "Requirements installed successfully."
    fi
elif [ -f "requierments.txt" ]; then
    echo "Found 'requierments.txt' (typo in filename). Installing requirements..."
    pip install -r requierments.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install some requirements. Please check the requierments.txt file."
    else
        echo "Requirements installed successfully."
    fi
else
    echo "No requirements.txt file found. Skipping requirements installation."
fi

# Install Jupyter and ipykernel
echo "Installing Jupyter and ipykernel for notebooks support..."
pip install ipykernel jupyter notebook
if [ $? -ne 0 ]; then
    echo "Failed to install Jupyter packages. Please install manually with:"
    echo "pip install ipykernel jupyter notebook"
else
    echo "Registering kernel for Jupyter..."
    python -m ipykernel install --user --name=self-awareness-env --display-name="Python (Self-Awareness)"
    if [ $? -ne 0 ]; then
        echo "Failed to register kernel. Please run manually:"
        echo "python -m ipykernel install --user --name=self-awareness-env"
    else
        echo "Jupyter setup completed successfully."
    fi
fi

# Virtual environment is now active
echo "Virtual environment is now active. You can run your project."
exec $SHELL
