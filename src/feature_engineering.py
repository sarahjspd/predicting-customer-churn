"""
Feature Engineering Module
==========================

Purpose:
    Create meaningful features using domain knowledge and insights
    from exploratory data analysis (EDA).

Objectives:
    1. Develop new features that may improve churn classification.
    2. Apply feature selection techniques to identify relevant features.
    3. Ensure transformations are reusable and prevent data leakage.

Dataset:
    Retail customer churn dataset (1,000 records).
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def validate_feature_input(df):
    """
    Validate the input DataFrame before feature engineering.
    """
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    required_columns = [
        "monthly_spend",
        "discount_usage_rate",
        "membership_tier"
    ]
    
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]
    
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Input features contain missing values. "
            "please apply preprocessing first."
        )
        
    # Check that numerical features have numerical data types
    numeric_columns = [
        "monthly_spend",
        "discount_usage_rate"
    ]

    non_numeric_columns = [
        col for col in numeric_columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric_columns:
        raise TypeError(
            f"These columns must contain numerical values: {non_numeric_columns}"
        )
        
    # Check that numerical features contain finite values
    if not np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError(
            "Numerical input features cannot contain infinite values."
        )        
    
    # Check that membership tiers contain expected categories
    valid_membership_tiers = {"Basic", "Premium", "VIP"}

    unexpected_tiers = set(df["membership_tier"].unique()) - valid_membership_tiers

    if unexpected_tiers:
        raise ValueError(
            f"Unexpected membership tiers found: {unexpected_tiers}"
        )
    
    
    
    return True
    

def create_engineered_features(df):
    
    """
    Create additional features based on domain knowledge and EDA findings.

    Engineered features:
    1. spend_discount_interaction:
       Captures the potential interaction between monthly spending
       and discount usage.

    2. monthly_spend_squared:
       Captures potential non-linear relationships between
       monthly spending and customer churn.

    3. premium_spend_interaction:
       Captures the interaction between monthly spending and
       Premium membership status.

    4. vip_spend_interaction:
       Captures the interaction between monthly spending and
       VIP membership status.

    Feature engineering rationale:
    - EDA showed differences in spending patterns and churn rates
      across membership tiers:
        Basic: average spending 112.93, churn rate 19.4%
        Premium: average spending 76.41, churn rate 13.0%
        VIP: average spending 57.18, churn rate 7.0%

    - Membership tier and spending interactions are included to
      investigate whether the relationship between spending and
      churn differs across membership tiers.

    - Basic is used as the reference category. Its spending
      relationship is represented by the original monthly_spend
      feature, while the Premium and VIP interactions capture
      differences relative to Basic.

    - These engineered features are exploratory candidates.
      Their incremental predictive usefulness and redundancy
      should be evaluated before final feature selection.
      
    - The feature engineering calculations are deterministic and do
      not independently prevent data leakage.

    - Leakage is prevented through the workflow: split the data first,
      then fit data-dependent transformations and perform feature
      selection using training data only.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input customer dataset.

    Returns
    -------
    pd.DataFrame
        A copy of the input dataset with additional engineered
        features.
    """

    
    validate_feature_input(df)
    
    df = df.copy()
    
    # Interaction between monthly spending and discount usage
    df["spend_discount_interaction"] = (
        df["monthly_spend"] * df["discount_usage_rate"]
    )
    
    #Non-linear transformation of monthly spending
    df["monthly_spend_squared"] = df["monthly_spend"] ** 2
    
    # Membership tier and spending interactions
    # Basic is the reference category.
    # Premium and VIP interactions capture differences
    # in the spending relationship relative to Basic.
    
        
    df["premium_spend_interaction"] = (
        df["monthly_spend"] *
        (df["membership_tier"] == "Premium").astype(int)
    )
    
    df["vip_spend_interaction"] = (
        df["monthly_spend"] *
        (df["membership_tier"] == "VIP").astype(int)
    )
    
    return df


def rank_features_by_mutual_information(X_train, y_train, random_state=42):
    """
    Rank numerical features according to their Mutual Information
    with the customer churn target.

    Mutual Information measures the statistical dependence
    between each feature and the target.

    Parameters
    ----------
    X_train : pd.DataFrame
        Numerical training features, including engineered features
        and encoded categorical variables.

    y_train : pd.Series
        Corresponding training target labels.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Features ranked by Mutual Information score,
        in descending order.
    """

    # Validate input types
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    if not isinstance(y_train, (pd.Series, pd.DataFrame)):
        raise TypeError("y_train must be a pandas Series or DataFrame.")

    if isinstance(y_train, pd.DataFrame):
        if y_train.shape[1] != 1:
            raise ValueError("y_train must contain exactly one target column.")
        y_train = y_train.iloc[:, 0]

    # Check that features and target have matching records
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have matching lengths.")

    if len(X_train) == 0:
        raise ValueError("Training data cannot be empty.")

    # Check that features are numerical
    if not all(pd.api.types.is_numeric_dtype(dtype)
               for dtype in X_train.dtypes):
        raise TypeError("All training features must be numerical.")

    # Check for missing values
    if X_train.isnull().any().any() or y_train.isnull().any():
        raise ValueError("Training features and target cannot contain missing values.")

    # Check for infinite values in numerical features
    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError("Training features cannot contain infinite values.")        
    
    # Check that the target contains at least two classes
    if y_train.nunique() < 2:
        raise ValueError("Target must contain at least two classes.")
    
    # Check that target contains only valid binary labels
    if not y_train.isin([0, 1]).all():
        raise ValueError("Target must contain only binary labels: 0 and 1.")

    # Identify binary features as discrete variables.
    # Other numerical features are treated as continuous.
    discrete_features = X_train.nunique() <= 2

    # Calculate Mutual Information scores
    mi_scores = mutual_info_classif(
        X_train,
        y_train,
        discrete_features=discrete_features,
        random_state=random_state
    )

    # Create and rank the results
    feature_ranking = pd.DataFrame({
        "feature": X_train.columns,
        "mutual_information": mi_scores
    })

    feature_ranking = feature_ranking.sort_values(
        by="mutual_information",
        ascending=False
    ).reset_index(drop=True)

    return feature_ranking
