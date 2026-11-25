
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    OutlierHandler,
    build_preprocessor,
    validate_input,
    prepare_data,
)


@pytest.fixture
def sample_data():
    """Create a small, controlled customer dataset for testing."""

    df = pd.DataFrame({
        "monthly_spend": np.arange(50, 70),
        "number_of_store_visits_per_month": np.arange(1, 21),
        "customer_satisfaction_score": np.arange(1, 21),
        "number_of_complaints": np.arange(0, 20),
        "discount_usage_rate": np.linspace(0.1, 1.0, 20),
        "membership_tier": (
            ["Basic"] * 10 + ["Premium"] * 6 + ["VIP"] * 4
        ),
        "customer_churn": ["Not Churned"] * 16 + ["Churned"] * 4,
    })

    return df


# ------------------------------------------
# Input validation tests
# ------------------------------------------

def test_validate_input_accepts_valid_data(sample_data):
    """Valid input should pass without raising an error."""

    assert validate_input(sample_data) is None


def test_validate_input_rejects_missing_columns(sample_data):
    """Missing required columns should raise ValueError."""

    invalid_df = sample_data.drop(columns=["monthly_spend"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_input(invalid_df)


def test_validate_input_rejects_invalid_numeric_data(sample_data):
    """Numerical features should contain numerical data types."""

    invalid_df = sample_data.copy()
    invalid_df["monthly_spend"] = "invalid"

    with pytest.raises(TypeError, match="invalid data types"):
        validate_input(invalid_df)


def test_validate_input_rejects_missing_target(sample_data):
    """Missing target values should raise ValueError."""

    invalid_df = sample_data.copy()
    invalid_df.loc[0, TARGET] = None

    with pytest.raises(ValueError, match="Target variable contains missing"):
        validate_input(invalid_df)


def test_validate_input_rejects_unexpected_target_labels(sample_data):
    """Unexpected target labels should raise ValueError."""

    invalid_df = sample_data.copy()
    invalid_df.loc[0, TARGET] = "Unknown"

    with pytest.raises(ValueError, match="unexpected labels"):
        validate_input(invalid_df)


# ------------------------------------------
# Data preparation tests
# ------------------------------------------

def test_prepare_data_returns_correct_split_sizes(sample_data):
    """An 80/20 split should produce 16 training and 4 testing records."""

    X_train, X_test, y_train, y_test = prepare_data(
        sample_data,
        test_size=0.2,
        random_state=42
    )

    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4



def test_prepare_data_preserves_target_distribution(sample_data):
    """Stratified splitting should preserve the target class proportions."""

    X_train, X_test, y_train, y_test = prepare_data(sample_data)

    assert y_train.value_counts()[1] == 3
    assert y_test.value_counts()[1] == 1

    assert y_train.value_counts()[0] == 13
    assert y_test.value_counts()[0] == 3


def test_prepare_data_is_reproducible(sample_data):
    """The same random state should produce the same split."""

    first_split = prepare_data(sample_data, random_state=42)
    second_split = prepare_data(sample_data, random_state=42)

    pd.testing.assert_frame_equal(first_split[0], second_split[0])
    pd.testing.assert_frame_equal(first_split[1], second_split[1])
    pd.testing.assert_series_equal(first_split[2], second_split[2])
    pd.testing.assert_series_equal(first_split[3], second_split[3])


# ------------------------------------------
# Preprocessing pipeline tests
# ------------------------------------------

def test_preprocessor_output_shape_and_scaling(sample_data):
    """Pipeline should encode three tiers and standardise numerical features."""

    X_train, X_test, y_train, y_test = prepare_data(sample_data)

    preprocessor = build_preprocessor(scaling="standard")

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    assert X_train_transformed.shape == (16, 8)
    assert X_test_transformed.shape == (4, 8)

    # The five numerical features should have approximately zero mean.
    assert np.allclose(
        X_train_transformed[:, :5].mean(axis=0),
        0,
        atol=1e-10
    )

    # The transformed data should not contain missing values.
    assert not np.isnan(X_train_transformed).any()
    assert not np.isnan(X_test_transformed).any()


# ------------------------------------------
# Outlier handling tests
# ------------------------------------------

def test_outlier_handler_clips_extreme_values():
    """Clipping should cap values at the learned IQR boundary."""

    values = np.array([[1], [2], [3], [4], [5], [100]])

    handler = OutlierHandler(method="clip", multiplier=1.5)
    handler.fit(values)

    transformed = handler.transform(values)

    assert transformed[-1, 0] == pytest.approx(8.5)
    assert transformed[-1, 0] < 100


# ------------------------------------------
# Additional robustness tests
# ------------------------------------------

def test_validate_input_rejects_entirely_empty_feature_column(sample_data):
    """A feature column containing only missing values should be rejected."""

    invalid_df = sample_data.copy()
    invalid_df["monthly_spend"] = np.nan

    with pytest.raises(ValueError, match="only missing values"):
        validate_input(invalid_df)


def test_preprocessor_imputes_missing_feature_values(sample_data):
    """Missing numerical and categorical values should be imputed."""

    invalid_df = sample_data.copy()
    invalid_df.loc[0, "monthly_spend"] = np.nan
    invalid_df.loc[1, "membership_tier"] = np.nan

    X = invalid_df.drop(columns=[TARGET])

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    assert not np.isnan(transformed).any()


def test_preprocessor_handles_unseen_membership_category(sample_data):
    """Previously unseen membership categories should not cause errors."""

    X = sample_data.drop(columns=[TARGET]) 
    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    X_unseen = X.copy()
    X_unseen.loc[X_unseen.index[0], "membership_tier"] = "Platinum"

    transformed = preprocessor.transform(X_unseen)

    assert transformed.shape[1] == 8
    assert np.all(transformed[0, 5:] == 0)


def test_validate_input_rejects_non_dataframe():
    """Inputs that are not pandas DataFrames should be rejected."""

    with pytest.raises(TypeError, match="pandas DataFrame"):
        validate_input([])
