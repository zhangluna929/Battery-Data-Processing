import os
import glob
import argparse
import pandas as pd
import yaml
import numpy as np

# Import refactored functions and the new config loader
from battery_analyzer.utils.config_loader import load_config
from battery_analyzer.processing.cleaning import fill_missing_data, handle_outliers, lowpass_filter
from battery_analyzer.processing.calibration import calibrate_device
from battery_analyzer.analysis.cycle_analyzer import calculate_capacity, calculate_cycle_lifetime, analyze_cycles
from battery_analyzer.analysis.feature_engineering import calculate_current_rate
from battery_analyzer.analysis.ica import calculate_ica # Import the new ICA function
from battery_analyzer.analysis.resistance import calculate_dcir # Import DCIR function
from battery_analyzer.models.regression import train_model
from battery_analyzer.models.soc_estimator import CoulombCountingSOEstimator, load_default_ocv_curve

def process_file(file_path, output_dir, config, model=None):
    """Processes a single data file based on the provided configuration."""
    print(f"Processing file: {file_path}")
    df = pd.read_csv(file_path)
    
    # Standardize column names based on config
    column_map = {v: k for k, v in config['columns'].items() if v in df.columns}
    df.rename(columns=column_map, inplace=True)

    # Data Cleaning
    df = fill_missing_data(df)
    df = handle_outliers(df, threshold=config['processing']['outlier_threshold_zscore'])

    # Device Calibration
    if 'columns_to_calibrate' in config['processing']:
        df = calibrate_device(df, 
                              columns_to_calibrate=config['processing']['columns_to_calibrate'], 
                              calibration_source_column=config['columns'].get('temperature', 'temperature'))

    # Data Smoothing
    filter_params = config['processing']['lowpass_filter']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(
        lambda x: lowpass_filter(
            x.values, 
            cutoff=filter_params['cutoff_frequency'],
            fs=filter_params['sampling_frequency'],
            order=filter_params['filter_order']
        )
    )
    
    # Calculate Capacity
    capacity = calculate_capacity(df, time_column=config['columns']['time'], current_column=config['columns']['current'])
    print(f"  - Calculated capacity: {capacity:.3f} Ah")
    df['Capacity_Ah'] = capacity

    # Calculate Cycle Lifetime
    cycle_lifetime = calculate_cycle_lifetime(df, voltage_column=config['columns']['voltage'], cycle_threshold=config['analysis']['cycle_voltage_threshold'])
    print(f"  - Calculated cycle lifetime: {cycle_lifetime} cycles")

    # --- New: Detailed Cycle Analysis (Coulombic Efficiency) ---
    if config.get('analysis', {}).get('cycle_analysis', {}).get('enabled', False):
        print("  - Performing detailed cycle analysis (incl. Coulombic Efficiency)...")
        cycle_summary_df = analyze_cycles(
            df,
            voltage_col=config['columns']['voltage'],
            current_col=config['columns']['current'],
            time_col=config['columns']['time'],
            cycle_index_col=config['columns'].get('cycle_index')
        )
        if not cycle_summary_df.empty:
            base_name, _ = os.path.splitext(os.path.basename(file_path))
            cycles_output_path = os.path.join(output_dir, f"{base_name}_cycles.csv")
            cycle_summary_df.to_csv(cycles_output_path, index=False)
            print(f"  - Cycle summary saved to {cycles_output_path}")
        else:
            print("  - Warning: Could not generate cycle summary.")

    # --- New: DC Internal Resistance (DCIR) Calculation ---
    if config.get('analysis', {}).get('dcir', {}).get('enabled', False):
        print("  - Calculating DC Internal Resistance (DCIR)...")
        dcir_params = config['analysis']['dcir']
        dcir_df = calculate_dcir(
            df,
            voltage_col=config['columns']['voltage'],
            current_col=config['columns']['current'],
            time_col=config['columns']['time'],
            current_change_threshold=dcir_params['current_change_threshold']
        )
        if not dcir_df.empty:
            base_name, _ = os.path.splitext(os.path.basename(file_path))
            dcir_output_path = os.path.join(output_dir, f"{base_name}_dcir.csv")
            dcir_df.to_csv(dcir_output_path, index=False)
            print(f"  - DCIR results saved to {dcir_output_path}")
        else:
            print("  - No significant current change events found for DCIR calculation.")

    # Feature Engineering
    df = calculate_current_rate(df, current_column=config['columns']['current'], time_column=config['columns']['time'])

    # --- SOC Estimation ---
    if config.get('soc_estimation', {}).get('enabled', False):
        soc_conf = config['soc_estimation']
        capacity_nominal = soc_conf['total_capacity_ah']
        ocv_path = soc_conf.get('ocv_curve_path')
        ocv_curve = load_default_ocv_curve() if ocv_path in (None, 'null') else pd.read_csv(ocv_path)
        soc_estimator = CoulombCountingSOEstimator(capacity_nominal, ocv_curve)
        df['SOC'] = soc_estimator.estimate_soc(df,
                                               current_col=config['columns']['current'],
                                               voltage_col=config['columns']['voltage'],
                                               time_col=config['columns']['time'])
    else:
        df['SOC'] = np.nan

    # Add aggregated DCIR as a feature if available
    if 'dcir_df' in locals() and not dcir_df.empty:
        df['mean_dcir_mohm'] = dcir_df['dc_internal_resistance_mohm'].mean()
    else:
        df['mean_dcir_mohm'] = 0

    # --- New: Incremental Capacity Analysis (ICA) ---
    if config.get('analysis', {}).get('ica', {}).get('enabled', False):
        print("  - Performing Incremental Capacity Analysis (ICA)...")
        # To perform ICA, we need to create a cumulative capacity column
        df['Cumulative_Capacity_Ah'] = (df[config['columns']['current']] * df[config['columns']['time']].diff().fillna(0) / 3600).cumsum()
        
        ica_params = config['analysis']['ica']
        ica_df = calculate_ica(
            df,
            voltage_col=config['columns']['voltage'],
            capacity_col='Cumulative_Capacity_Ah',
            voltage_step=ica_params['voltage_step'],
            savgol_window=ica_params['savgol_window'],
            savgol_polyorder=ica_params['savgol_polyorder']
        )
        
        # Save the ICA curve to a separate file
        base_name, _ = os.path.splitext(os.path.basename(file_path))
        ica_output_path = os.path.join(output_dir, f"{base_name}_ica.csv")
        ica_df.to_csv(ica_output_path, index=False)
        print(f"  - ICA curve saved to {ica_output_path}")

    # Prediction
    if model:
        feature_columns = config['model']['feature_columns']
        target_column = config['model']['target_column']
        
        # Ensure all required feature columns exist before attempting to predict
        if all(col in df.columns for col in feature_columns):
            X_pred = df[feature_columns].dropna() # Drop rows with NaN in features
            if not X_pred.empty:
                predicted_values = model.predict(X_pred)
                # Since prediction is row-wise, we might want to store the mean or full series
                df.loc[X_pred.index, f'Predicted_{target_column}'] = predicted_values
                print(f'  - Predicted {target_column} (mean): {predicted_values.mean():.3f}')
        else:
            missing_cols = [col for col in feature_columns if col not in df.columns]
            print(f"  - Warning: Not all feature columns available for prediction. Missing: {missing_cols}. Skipping.")

    # Save processed file
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    df.to_csv(output_file, index=False)
    print(f"  -> Processed file saved to {output_file}\n")


def process_batch(input_dir, output_dir, config, model=None):
    """Processes a batch of files in the input directory."""
    files = glob.glob(os.path.join(input_dir, '*.csv'))
    if not files:
        print(f"No CSV files found in '{input_dir}'.")
        return
        
    for file in files:
        try:
            process_file(file, output_dir, config, model)
        except Exception as e:
            print(f"Failed to process {os.path.basename(file)}: {e}")


def main():
    """Main function to run the data processing tool."""
    parser = argparse.ArgumentParser(description='Battery Data Processing and Analysis Tool')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory with CSV data files')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for processed files')
    parser.add_argument('--config', type=str, default='configs/default_config.yml', help='Path to the configuration YAML file')
    parser.add_argument('--train-model', action='store_true', help='Train a new regression model')

    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Configuration loaded from '{args.config}'")
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error: {e}")
        return

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    model = None
    if args.train_model:
        print("\n--- Training Model ---")
        all_files = glob.glob(os.path.join(args.input_dir, '*.csv'))
        if not all_files:
            print("No data available for training. Exiting.")
            return

        df_list = []
        for f in all_files:
            df = pd.read_csv(f)
            column_map = {v: k for k, v in config['columns'].items() if v in df.columns}
            df.rename(columns=column_map, inplace=True)
            df_list.append(df)
        
        training_df = pd.concat(df_list, ignore_index=True).dropna(
            subset=config['model']['feature_columns'] + [config['model']['target_column']]
        )

        feature_columns = config['model']['feature_columns']
        target_column = config['model']['target_column']
        
        if target_column not in training_df.columns:
            print(f"Target column '{target_column}' not found in the data, cannot train model.")
            return
            
        try:
            model, score = train_model(training_df, target_column, feature_columns)
            print(f"--- Model training completed with R^2 score: {score:.3f} ---")
        except ValueError as e:
            print(f"Error during model training: {e}")
            return

    print("\n--- Processing Batch ---")
    process_batch(args.input_dir, args.output_dir, config, model)
    print("--- Batch processing completed ---")


if __name__ == '__main__':
    main()
