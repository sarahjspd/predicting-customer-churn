from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


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
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical preprocessing pipeline
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
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

def prepare_data(df):
    # Separate features and target
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].map(TARGET_MAPPING)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test