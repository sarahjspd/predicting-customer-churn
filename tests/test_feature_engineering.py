
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    validate_feature_input,
    create_engineered_features,
    rank_features_by_mutual_information,
)


@pytest.fixture
def sample_data():
    """Create a controlled dataset for feature engineering tests."""

    return pd.DataFrame({
        "monthly_spend": [100, 200, 150, 300, 250, 120],
        "discount_usage_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "membership_tier": [
            "Basic", "Premium", "VIP",
            "Basic", "Premium", "VIP"
        ],
    })


@pytest.fixture
def sample_training_data():
    """Create numerical training features and binary churn labels."""

    X_train = pd.DataFrame({
        "monthly_spend": [100, 120, 140, 160, 180, 200],
        "discount_usage_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "premium_member": [0, 1, 0, 1, 0, 1],
    })

    y_train = pd.Series([0, 0, 0, 1, 1, 1])

    return X_train, y_train


# ------------------------------------------
# Input validation tests
# ------------------------------------------

def test_validate_feature_input_accepts_valid_data(sample_data):
    """Valid feature input should pass validation."""

    assert validate_feature_input(sample_data) is True


def test_validate_feature_input_rejects_missing_columns(sample_data):
    """Missing required columns should raise ValueError."""

    invalid_df = sample_data.drop(columns=["monthly_spend"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_feature_input(invalid_df)


def test_validate_feature_input_rejects_missing_values(sample_data):
    """Missing feature values should raise ValueError."""

    invalid_df = sample_data.copy()
    invalid_df.loc[0, "monthly_spend"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        validate_feature_input(invalid_df)


def test_validate_feature_input_rejects_invalid_membership_tier(sample_data):
    """Unexpected membership tiers should raise ValueError."""

    invalid_df = sample_data.copy()
    invalid_df.loc[0, "membership_tier"] = "Gold"

    with pytest.raises(ValueError, match="Unexpected membership tiers"):
        validate_feature_input(invalid_df)


# ------------------------------------------
# Feature creation tests
# ------------------------------------------

def test_create_engineered_features_adds_expected_columns(sample_data):
    """Feature engineering should add all four candidate features."""

    result = create_engineered_features(sample_data)

    expected_features = [
        "spend_discount_interaction",
        "monthly_spend_squared",
        "premium_spend_interaction",
        "vip_spend_interaction",
    ]

    for feature in expected_features:
        assert feature in result.columns

    assert result.shape == (6, 7)


def test_create_engineered_features_calculates_correct_values(sample_data):
    """Engineered feature calculations should match expected values."""

    result = create_engineered_features(sample_data)

    # First record: Basic, spending 100, discount usage 0.1
    assert result.loc[0, "spend_discount_interaction"] == pytest.approx(10)
    assert result.loc[0, "monthly_spend_squared"] == pytest.approx(10000)
    assert result.loc[0, "premium_spend_interaction"] == 0
    assert result.loc[0, "vip_spend_interaction"] == 0

    # Second record: Premium, spending 200
    assert result.loc[1, "premium_spend_interaction"] == 200

    # Third record: VIP, spending 150
    assert result.loc[2, "vip_spend_interaction"] == 150


def test_create_engineered_features_does_not_modify_original(sample_data):
    """Feature engineering should preserve the original input DataFrame."""

    original = sample_data.copy()

    create_engineered_features(sample_data)

    pd.testing.assert_frame_equal(sample_data, original)


# ------------------------------------------
# Mutual Information feature selection tests
# ------------------------------------------

def test_mutual_information_returns_ranked_features(sample_training_data):
    """MI should return every feature ranked in descending order."""

    X_train, y_train = sample_training_data

    result = rank_features_by_mutual_information(X_train, y_train)

    assert list(result.columns) == ["feature", "mutual_information"]
    assert len(result) == X_train.shape[1]

    assert result["feature"].isin(X_train.columns).all()

    scores = result["mutual_information"].to_numpy()

    assert np.isfinite(scores).all()
    assert (scores >= 0).all()
    assert np.all(scores[:-1] >= scores[1:])


def test_mutual_information_is_reproducible(sample_training_data):
    """The same random state should produce identical MI rankings."""

    X_train, y_train = sample_training_data

    first_result = rank_features_by_mutual_information(
        X_train, y_train, random_state=42
    )

    second_result = rank_features_by_mutual_information(
        X_train, y_train, random_state=42
    )

    pd.testing.assert_frame_equal(first_result, second_result)


def test_mutual_information_rejects_invalid_target(sample_training_data):
    """Target labels outside 0 and 1 should raise ValueError."""

    X_train, y_train = sample_training_data
    invalid_target = y_train.copy()
    invalid_target.iloc[0] = 2

    with pytest.raises(ValueError, match="binary labels"):
        rank_features_by_mutual_information(X_train, invalid_target)
