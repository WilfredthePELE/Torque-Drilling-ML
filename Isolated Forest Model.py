import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the dataset with the correct delimiter
file_path = r'C:\Users\INTERNAL AUDIT\Desktop\data\cleaned_drilling_data.txt'  # Replace with the actual dataset path
data = pd.read_csv(file_path, delimiter='\t')  # Specify tab as the delimiter

# Debugging step: Print column names to verify
print("Column names in the dataset:")
print(data.columns)

# Clean column names (strip, replace spaces with underscores, make lowercase)
data.columns = data.columns.str.strip().str.replace(" ", "_").str.lower()

# Debugging step: Print cleaned column names
print("Cleaned column names:")
print(data.columns)

# Ensure relevant features are selected
features = [
    'weight_on_bit', 'rop_depth/hour', 'top_drive_rpm', 'top_drive_torque_(ft-lbs)',
    'flow_in', 'pump_pressure', 'spm_total', 'differential_pressure',
    'downhole_torque', 'mud_temp'
]

# Verify selected features exist in the DataFrame
missing_features = [feature for feature in features if feature not in data.columns]
if missing_features:
    raise KeyError(f"The following features are missing in the dataset: {missing_features}")

data_features = data[features]

# Handle missing values
data_features = data_features.fillna(method='ffill').fillna(method='bfill')

# Split the data into training (80%) and testing (20%) sets
train_data, test_data = train_test_split(data_features, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Initialize the Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(train_scaled)

# Predict anomalies for training and testing sets
train_predictions = model.predict(train_scaled)  # -1 = anomaly, 1 = normal
test_predictions = model.predict(test_scaled)

# Convert anomaly labels for easier understanding
train_predictions = np.where(train_predictions == 1, 0, 1)  # 0 = normal, 1 = anomaly
test_predictions = np.where(test_predictions == 1, 0, 1)

# Add predictions back to the original data
train_data['anomaly'] = train_predictions
test_data['anomaly'] = test_predictions

# Calculate error metrics using test data
mae = mean_absolute_error(test_predictions, test_data['anomaly'])
mse = mean_squared_error(test_predictions, test_data['anomaly'])
rmse = np.sqrt(mse)
r2 = r2_score(test_predictions, test_data['anomaly'])

print(f"Error Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Save the labeled dataset
output_path = 'labeled_data_with_anomalies.csv'
data_features['anomaly'] = model.predict(scaler.transform(data_features))
data_features['anomaly'] = data_features['anomaly'].replace({1: 0, -1: 1})
data_features.to_csv(output_path, index=False)
print(f"Labeled dataset saved to: {output_path}")

# Assuming ground truth is available in a column called "true_label"
if 'true_label' in data.columns:
    true_labels = data['true_label']
    predicted_labels = data_features['anomaly']

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print(f"Confusion Matrix:\n{cm}")

    # Classification Report
    report = classification_report(true_labels, predicted_labels, target_names=['Normal', 'Anomaly'])
    print(f"Classification Report:\n{report}")

    # Metrics Visualization
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Visualization of Anomalies
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data_features, x=data.index, y='top_drive_torque_(ft-lbs)', hue='anomaly', palette={0: 'blue', 1: 'red'})
plt.title('Anomaly Detection in Drilling Operations')
plt.xlabel('Index')
plt.ylabel('Top Drive Torque (ft-lbs)')
plt.legend(title='Anomaly')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
