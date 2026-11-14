import pytest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_preprocessor import preprocess_molecular_data


class TestDataPreprocessor:
    """Test data preprocessing functions"""

    def test_preprocess_with_valid_data(self):
        """Test preprocessing with valid data"""
        # Create sample DataFrame
        df = pd.DataFrame({"formula": ["H2O", "NaCl", "CO2"], "Tc": [0, 0, 0]})

        X, y = preprocess_molecular_data(df)

        assert len(X) == 3
        assert len(y) == 3
