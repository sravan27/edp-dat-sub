# Prepare features for October 1, 2024
print("Preparing features for October 1, 2024...")
# Generate date range for October 1, 2024 at 30-minute intervals
dates_oct1 = pd.date_range(start='2024-10-01 00:00:00', end='2024-10-01 23:30:00', freq='30T')
october_1_features = pd.DataFrame({'gmt_time': dates_oct1})

# Create the same time-based features as in training
print("Creating time-based features for October 1, 2024...")
october_1_features['hour'] = october_1_features['gmt_time'].dt.hour
october_1_features['day_of_week'] = october_1_features['gmt_time'].dt.dayofweek
october_1_features['month'] = october_1_features['gmt_time'].dt.month
october_1_features['is_weekend'] = (october_1_features['day_of_week'] >= 5).astype(int)

# Assign historical means to lag and rolling features
print("Assigning historical means to lag and rolling features...")
for col in lag_feature_cols + rolling_feature_cols:
    if col in historical_means:
        october_1_features[col] = historical_means[col]
    else:
        # Assign a default value or handle appropriately
        october_1_features[col] = 0
        print(f"Warning: {col} not found in historical means. Assigned default value 0.")

# Drop 'gmt_time' as it's not a feature used in the model
october_1_features = october_1_features.drop(columns=['gmt_time'])

# Ensure the order of columns matches the training data
print("Ensuring feature order matches the training data...")
october_1_features = october_1_features[training_feature_columns]

# Make predictions
print("Making predictions...")
october_1_predictions_price = best_model_price_loaded.predict(october_1_features)
october_1_predictions_niv = best_model_niv_loaded.predict(october_1_features)

# Create output DataFrame
output = pd.DataFrame({
    'GTM_TIME': dates_oct1,
    'SYSTEM_PRICE': october_1_predictions_price,
    'NIV_OUTTURN': october_1_predictions_niv
})

# Save predictions to CSV
print("Saving predictions...")
output.to_csv('predictions_october_1_2024.csv', index=False)
print("Predictions saved to 'predictions_october_1_2024.csv'.")
