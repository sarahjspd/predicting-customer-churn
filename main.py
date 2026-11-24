import pandas as pd
import numpy as np
from src.preprocessing import (
    prepare_data,
    preprocessor,
    validate_input,
    build_preprocessor,
    TARGET
)

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


# ==========================================
# Additional Robustness Tests
# ==========================================

print("\nAdditional Robustness Tests:")

# 1. Test missing feature value imputation
df_missing = df.copy()
df_missing.loc[0, "monthly_spend"] = np.nan
df_missing.loc[1, "membership_tier"] = np.nan

X_missing = df_missing.drop(columns=[TARGET])
test_preprocessor = build_preprocessor()

X_missing_processed = test_preprocessor.fit_transform(X_missing)

assert not np.isnan(X_missing_processed).any()
print("PASSED: Missing feature values are imputed.")

# 2. Test rejection of an entirely empty feature column
df_empty = df.copy()
df_empty["monthly_spend"] = np.nan

try:
    validate_input(df_empty)
    raise AssertionError("Expected ValueError was not raised.")
except ValueError as error:
    if "only missing values" in str(error):
        print("PASSED: Entirely empty feature column rejected.")
    else:
        raise

# 3. Test rejection of missing target values
df_missing_target = df.copy()
df_missing_target.loc[0, TARGET] = np.nan

try:
    validate_input(df_missing_target)
    raise AssertionError("Expected ValueError was not raised.")
except ValueError as error:
    if "Target variable contains missing values" in str(error):
        print("PASSED: Missing target values rejected.")
    else:
        raise

# 4. Test rejection of unexpected target labels
df_invalid_target = df.copy()
df_invalid_target.loc[0, TARGET] = "Unknown"

try:
    validate_input(df_invalid_target)
    raise AssertionError("Expected ValueError was not raised.")
except ValueError as error:
    if "unexpected labels" in str(error):
        print("PASSED: Unexpected target labels rejected.")
    else:
        raise

# 5. Test rejection of missing required columns
df_missing_column = df.drop(columns=["membership_tier"])

try:
    validate_input(df_missing_column)
    raise AssertionError("Expected ValueError was not raised.")
except ValueError as error:
    if "missing required columns" in str(error):
        print("PASSED: Missing required columns rejected.")
    else:
        raise

# 6. Test handling of unseen membership categories
X_unseen = X_test.copy()
X_unseen.loc[X_unseen.index[0], "membership_tier"] = "Platinum"

X_unseen_processed = preprocessor.transform(X_unseen)

assert X_unseen_processed.shape[1] == 8
assert np.all(X_unseen_processed[0, 5:] == 0)

print("PASSED: Unseen membership categories handled.")

# 7. Test rejection of incorrect numerical data types
df_invalid_type = df.copy()
df_invalid_type["monthly_spend"] = "invalid"

try:
    validate_input(df_invalid_type)
    raise AssertionError("Expected TypeError was not raised.")
except TypeError:
    print("PASSED: Incorrect numerical data types rejected.")

# 8. Test rejection of non-DataFrame inputs
try:
    validate_input([])
    raise AssertionError("Expected TypeError was not raised.")
except TypeError:
    print("PASSED: Non-DataFrame inputs rejected.")



print("\nAll additional robustness tests completed.")
