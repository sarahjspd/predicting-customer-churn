from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


# Target variable
TARGET = "customer_churn"

# Numerical features
NUMERIC_FEATURES = [
    "monthly_spend",
    "number_of_store_visits_per_month",
    "customer_satisfaction_score",
    "number_of_complaints",
    "discount_usage_rate"
]

# Categorical features
CATEGORICAL_FEATURES = [
    "membership_tier"
]

# Target encoding
TARGET_MAPPING = {
    "Not Churned": 0,
    "Churned": 1
}


# Outlier handling
# Potential outliers were identified during EDA in monthly_spend,
# number_of_store_visits_per_month, number_of_complaints,
# and discount_usage_rate.
#
# Original values are retained because the legitimacy of these
# observations cannot be verified with the available dataset.
# No capping or removal is applied at this stage.
# The impact of potential outliers can be evaluated during
# subsequent model experimentation.

# Median imputation is included for pipeline robustness.
# EDA confirmed that the current dataset has no missing values,
# so this step does not alter the present dataset.



class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handle numerical outliers using configurable treatment.

    Supports:
    - None: Retain original values without outlier treatment.
    - clip: Cap values outside the IQR-based boundaries.

    The IQR boundaries are learned during fitting and subsequently
    applied to new data using the same learned boundaries.

    Parameters
    ----------
    method : str or None, default=None
        Outlier treatment method.
        Supported values are None and "clip".

    multiplier : float, default=1.5
        Multiplier applied to the interquartile range (IQR)
        when calculating the lower and upper outlier boundaries.

    Raises
    ------
    ValueError
        If an unsupported outlier treatment method is specified
        or if the multiplier is not positive.
    """

    def __init__(self, method=None, multiplier=1.5):
        self.method = method
        self.multiplier = multiplier

    def fit(self, X, y=None):
        """Learn outlier boundaries from the training data."""

        # Validate outlier treatment settings
        if self.method not in [None, "clip"]:
            raise ValueError(
                f"Unsupported outlier treatment method: {self.method}"
            )

        if self.multiplier <= 0:
            raise ValueError("Multiplier must be greater than zero.")

        # Calculate outlier boundaries if clipping is enabled
        if self.method == "clip":
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)

            iqr = q3 - q1

            self.lower_bounds_ = q1 - self.multiplier * iqr
            self.upper_bounds_ = q3 + self.multiplier * iqr

        return self

    def transform(self, X):
        """Apply the learned outlier treatment to the input data."""

        # Retain original values if outlier treatment is disabled
        if self.method is None:
            return X
        
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])

        # Apply the learned lower and upper boundaries
        return np.clip(
            X,
            self.lower_bounds_,
            self.upper_bounds_
        )
    
    def get_feature_names_out(self, input_features=None):
        """Return the input feature names unchanged."""

        if input_features is None:
            return None

        return input_features




def build_preprocessor(
    numeric_imputation="median",
    categorical_imputation="most_frequent",
    scaling="standard",
    outlier_method=None,
    outlier_multiplier=1.5
):
    """
    Build a configurable preprocessing pipeline.

    Parameters
    ----------
    numeric_imputation : str, default="median"
        Imputation strategy for numerical features.

    categorical_imputation : str, default="most_frequent"
        Imputation strategy for categorical features.

    scaling : str, default="standard"
        Scaling method. Supported values are "standard",
        "minmax", and "none".

    outlier_method : str or None, default=None
        Outlier treatment method: None or "clip".

    outlier_multiplier : float, default=1.5
        IQR multiplier used to calculate outlier boundaries.

    Raises
    ------
    ValueError
        If an unsupported scaling method is specified.
    """

    # Select the scaling method
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    elif scaling == "none":
        scaler = "passthrough"
    else:
        raise ValueError(
            f"Unsupported scaling method: {scaling}"
        )

    # Numerical preprocessing pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=numeric_imputation)),
            ("outlier_handler", OutlierHandler(
                method=outlier_method,
                multiplier=outlier_multiplier
            )),
            ("scaler", scaler)
        ]
    )

    # Categorical preprocessing pipeline
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=categorical_imputation)),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Combine numerical and categorical preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


# Create the default preprocessing pipeline
preprocessor = build_preprocessor()



def validate_input(df):
    """
    Validate the input dataset before preprocessing.

    Checks that:
    - The input is a pandas DataFrame.
    - All required feature and target columns are present.
    - Numerical features contain numeric data types.
    - Categorical features contain text or categorical data types.
    - Target labels belong to the expected categories.
    - Target values do not contain missing values.
    - Feature Columns are not entirely empty.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the customer features and target variable.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame, or if numerical
        or categorical features contain invalid data types.

    ValueError
        If required columns are missing, target labels are
        unexpected, target values contain missing values, or
        feature columns are entirely empty.
    """

    # 1. Check that the input is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # 2. Check that all required columns are present
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # 3. Check that numerical features contain numeric data types
    invalid_numeric_columns = [
        column for column in NUMERIC_FEATURES
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if invalid_numeric_columns:
        raise TypeError(
            f"Numerical features contain invalid data types: "
            f"{invalid_numeric_columns}"
        )

    # 4. Check that categorical features contain valid data types
    invalid_categorical_columns = [
        column for column in CATEGORICAL_FEATURES
        if not (
            pd.api.types.is_string_dtype(df[column].dtype)
            or isinstance(df[column].dtype, pd.CategoricalDtype)
        )
    ]

    if invalid_categorical_columns:
        raise TypeError(
            f"Categorical features contain invalid data types: "
            f"{invalid_categorical_columns}"
        )

    # 5. Check that feature columns are not entirely empty
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    empty_feature_columns = [
        column for column in feature_columns
        if df[column].isna().all()
    ]

    if empty_feature_columns:
        raise ValueError(
            f"Feature columns contain only missing values: "
            f"{empty_feature_columns}"
        )    

    # 6. Check that target values are not missing
    if df[TARGET].isna().any():
        raise ValueError("Target variable contains missing values.")

    # 7. Check that target labels belong to the expected categories
    invalid_target_labels = df.loc[
        ~df[TARGET].isin(TARGET_MAPPING.keys()), TARGET
    ].unique()

    if len(invalid_target_labels) > 0:
        raise ValueError(
            f"Target contains unexpected labels: {invalid_target_labels}"
        )



def prepare_data(df, test_size=0.2, random_state=42):
    # Validate input dataset
    validate_input(df)
    # Separate features and target
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].map(TARGET_MAPPING)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test