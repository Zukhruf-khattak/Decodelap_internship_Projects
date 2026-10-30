"""Tests for the Content-Based Recommender."""

import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.recommender import TechStackRecommender

@pytest.fixture
def sample_df():
    """Create a tiny in-memory DataFrame for testing."""
    return pd.DataFrame({
        "Job Role": ["Data Scientist", "DevOps Engineer", "Frontend Developer"],
        "Skills": [
            "python machine learning sql",
            "python cloud docker kubernetes",
            "javascript react html css"
        ]
    })

def test_recommender_initialization(sample_df):
    engine = TechStackRecommender(sample_df)
    assert engine.vectorizer is not None
    assert engine.tfidf_matrix.shape[0] == 3

def test_recommendation_output(sample_df):
    engine = TechStackRecommender(sample_df)
    results = engine.recommend(["python", "cloud"], top_n=2)
    
    # Check dimensions
    assert len(results) <= 2
    # Check if DevOps is top (it matches Python + Cloud)
    assert results.iloc[0]["Job Role"] == "DevOps Engineer"
    # Score should be > 0
    assert results.iloc[0]["Match Score"] > 0.0

def test_empty_input_raises_error(sample_df):
    engine = TechStackRecommender(sample_df)
    with pytest.raises(ValueError):
        engine.recommend([])

def test_real_dataset_loads():
    """Requires scripts/create_dataset.py to have been run."""
    try:
        df = load_dataset()
        assert not df.empty
        assert "Job Role" in df.columns
        assert "Skills" in df.columns
    except FileNotFoundError:
        pytest.fail("raw_skills.csv not found. Run the dataset creation script.")