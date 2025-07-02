# Battery Data Processing

This repository contains a Python script for processing battery test data. The script performs several key tasks:

- **Data Cleaning**: Handles missing values, outliers, and noise removal.
- **Capacity Calculation**: Calculates battery capacity (in mAh) based on charging/discharging curves.
- **Cycle Life Prediction**: Estimates battery cycle life based on voltage drop thresholds.
- **Feature Engineering**: Includes temperature and current rate as features for machine learning.
- **Machine Learning**: Uses a Random Forest model to predict battery capacity and other key metrics.
- **Batch Processing**: Processes multiple files and outputs cleaned and processed data.

### Features

- Data cleaning and formatting: Handles missing data, outliers, and noise.
- Capacity and cycle life calculation: Integrates electrochemical behavior to estimate key performance metrics.
- Machine learning integration: Predicts battery performance using Random Forest Regressor.
- Flexible input/output formats: Supports CSV, Excel, and text files.
- Easy batch processing via command-line interface.

### Installation

1. Clone this repository 
2. Install the required dependencies:
pip install -r requirements.txt

### Usage
- **Train a Machine Learning Model and Process Data:**
- To train a machine learning model and process your raw data, use the following command:

python process_data.py --input-dir ./raw_data --output-dir ./processed_data --columns-to-calibrate Voltage Current --train-model
- **Process Data Using Pre-trained Model:**

- If you already have a pre-trained model and just want to process new data using that model, use:

python process_data.py --input-dir ./raw_data --output-dir ./processed_data --columns-to-calibrate Voltage Current