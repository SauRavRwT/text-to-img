# Text to Image Converter

## Setup Guide

This guide will help you set up the Text to Image converter project. Please use Command Prompt (cmd) on Windows, or any terminal on Linux/Mac.

## Run on colab
The Google Colab notebook provides a convenient way to run the project without local setup:

1. Open `Text-to-img.ipynb` in Google Colab
2. Select 'Runtime' > 'Change runtime type' > Choose 'T4 GPU'
3. Run all cells in sequence

### Step 1: Get the Code
Clone the repository:
```bash
git clone https://github.com/SauRavRwT/text-to-img.git
cd text-to-img
```

### Step 2: Set Up Python Environment
Create and activate a virtual environment:

For Windows:
```bash
mkdir venv
python -m venv venv
venv\scripts\activate
```

For Linux/Mac:
```bash
mkdir venv
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
Install required packages:
```bash
# Install basic requirements
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install --upgrade torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install FastAI
pip install --upgrade fastai
```

### Step 4: Run the Application
Start the application:
```bash
python app.py
```

Now you're ready to convert text to images!
