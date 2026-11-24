import pandas as pd
import numpy as np
from src.preprocessing import prepare_data, preprocessor

# Load dataset
df = pd.read_csv("data/raw/synthetic_data_247.csv")

# Check dataset
print(df.head())
print(df.shape)

# Prepare features and split dataset
X_train, X_test, y_train, y_test = prepare_data(df)

# Check dataset dimensions
print("\nDataset split:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Check target distribution in training and testing sets
print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))

# Fit preprocessing pipeline on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Apply fitted preprocessing pipeline to testing data
X_test_processed = preprocessor.transform(X_test)

# Check processed dataset dimensions
print("\nProcessed dataset dimensions:")
print("X_train_processed:", X_train_processed.shape)
print("X_test_processed:", X_test_processed.shape)

# Check for missing values after preprocessing
print("\nMissing values in processed training data:")
print(np.isnan(X_train_processed).sum())

print("\nMissing values in processed testing data:")
print(np.isnan(X_test_processed).sum())

# Check numerical feature scaling
# [:, :5] is used to display the first 5 numerical features after scaling. The other 3 columns are the one-hot encoded categorical features.
print("\nNumerical feature means after scaling:")
print(X_train_processed[:, :5].mean(axis=0))

print("\nNumerical feature standard deviations after scaling:")
print(X_train_processed[:, :5].std(axis=0))

# Check encoded categorical feature names
print("\nProcessed feature names:")
print(preprocessor.get_feature_names_out())

# Check One-Hot Encoding output
print("\nOne-Hot Encoded membership tiers:")
print(X_train_processed[:5, 5:])

# Validate One-Hot Encoding across training and testing data
print("\nOne-Hot Encoding validation:")

print(
    "Training - invalid encoded rows:",
    (X_train_processed[:, 5:].sum(axis=1) != 1).sum()
)

print(
    "Testing - invalid encoded rows:",
    (X_test_processed[:, 5:].sum(axis=1) != 1).sum()
)