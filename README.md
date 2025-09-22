Drilling Data Anomaly Detection

This project implements an anomaly detection system for drilling operations using the Isolation Forest algorithm. It processes drilling data to identify unusual patterns, such as equipment malfunctions or operational inefficiencies, by analyzing key parameters like weight on bit, top drive torque, and pump pressure. The project leverages Python, Pandas, Scikit-learn, and visualization libraries to deliver actionable insights for optimizing drilling performance.
Features

Data Preprocessing: Handles missing values, standardizes features, and cleans column names for robust analysis.
Anomaly Detection: Uses the Isolation Forest algorithm to detect anomalies in drilling data with a contamination rate of 5%.
Evaluation Metrics: Computes Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² scores.
Visualizations: Generates scatter plots and confusion matrices to visualize anomalies and model performance.
Output: Saves labeled data with anomaly flags to a CSV file for further analysis.
Modular Code: Well-documented Python script with clear sections for data loading, preprocessing, modeling, and evaluation.

Prerequisites

Python 3.8 or higher
Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

Installation

Clone the repository:git clone https://github.com/WilfredthePELE/Torque-Drilling-ML.git
cd Torque-Drilling-ML


Create a virtual environment (optional but recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt



Usage

Prepare the Dataset:

Place your drilling data file (e.g., cleaned_drilling_data.txt) in the appropriate directory (default: C:\Users\INTERNAL AUDIT\Desktop\data\).
Ensure the file is tab-delimited and contains the required features (e.g., weight_on_bit, top_drive_torque_(ft-lbs), etc.).


Run the Script:
python anomaly_detection.py

The script will:

Load and preprocess the dataset.
Train an Isolation Forest model.
Detect anomalies in training and testing sets.
Output error metrics (MAE, MSE, RMSE, R²).
Save the labeled dataset as labeled_data_with_anomalies.csv.
Generate visualizations (scatter plot and confusion matrix, if ground truth is available).


Example Output:
Error Metrics:
Mean Absolute Error (MAE): 0.0500
Mean Squared Error (MSE): 0.0500
Root Mean Squared Error (RMSE): 0.2236
R-squared (R²): 0.9000
Labeled dataset saved to: labeled_data_with_anomalies.csv



Dataset
The project expects a tab-delimited text file with the following columns:

weight_on_bit
rop_depth/hour
top_drive_rpm
top_drive_torque_(ft-lbs)
flow_in
pump_pressure
spm_total
differential_pressure
downhole_torque
mud_temp
(Optional) true_label for ground truth evaluation

Visualizations
Anomaly Scatter Plot
The script generates a scatter plot to visualize anomalies in top_drive_torque_(ft-lbs) over the dataset index:

Blue points: Normal data
Red points: Detected anomalies


Confusion Matrix
If a true_label column is provided, a confusion matrix is plotted to evaluate model performance:
Code Example
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = pd.read_csv('cleaned_drilling_data.txt', delimiter='\t')
features = ['weight_on_bit', 'rop_depth/hour', 'top_drive_torque_(ft-lbs)']
data_features = data[features].fillna(method='ffill').fillna(method='bfill')

# Standardize features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_features)

# Train Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
data_features['anomaly'] = model.fit_predict(data_scaled)
data_features['anomaly'] = data_features['anomaly'].replace({1: 0, -1: 1})

# Save results
data_features.to_csv('labeled_data_with_anomalies.csv', index=False)

Evaluation Metrics
The model is evaluated using the following metrics on the test set:

Mean Absolute Error (MAE): Measures average prediction error.
Mean Squared Error (MSE): Quantifies the average squared error.
Root Mean Squared Error (RMSE): Provides error in the same units as the data.
R-squared (R²): Indicates the proportion of variance explained by the model.

If ground truth labels (true_label) are available, a confusion matrix and classification report are generated to assess precision, recall, and F1-score.
Repository Structure
your-repo/
│
├── anomaly_detection.py        # Main script for anomaly detection
├── requirements.txt           # Project dependencies
├── labeled_data_with_anomalies.csv  # Output file with anomaly labels
├── README.md                  # Project documentation
└── data/                      # Directory for input dataset
    └── cleaned_drilling_data.txt

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.

See CONTRIBUTING.md for more details.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact
For questions or suggestions, feel free to open an issue or contact [your email or GitHub handle].
