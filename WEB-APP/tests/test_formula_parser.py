import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.formula_parser import (
    parse_formula,
    standardize_formula,
    generate_random_formula,
)


class TestParseFormula:
    """Test parse_formula function"""

    def test_simple_formula(self):
        """Test simple binary compounds"""
        assert parse_formula("H2O") == {"H": 2.0, "O": 1.0}
        assert parse_formula("NaCl") == {"Na": 1.0, "Cl": 1.0}
        assert parse_formula("CO2") == {"C": 1.0, "O": 2.0}

    def test_complex_formula(self):
        """Test complex formulas like superconductors"""
        assert parse_formula("YBa2Cu3O7") == {"Y": 1.0, "Ba": 2.0, "Cu": 3.0, "O": 7.0}
        assert parse_formula("Bi2Sr2CaCu2O8") == {
            "Bi": 2.0,
            "Sr": 2.0,
            "Ca": 1.0,
            "Cu": 2.0,
            "O": 8.0,
        }

    def test_decimal_subscripts(self):
        """Test decimal subscripts"""
        assert parse_formula("Ba0.5In0.5") == {"Ba": 0.5, "In": 0.5}
        assert parse_formula("Ga0.3In0.7") == {"Ga": 0.3, "In": 0.7}

    def test_repeated_elements(self):
        """Test that repeated elements are handled correctly (BUG FIX VERIFICATION)"""
        # This was the bug: Fe2Fe2O3 would give Fe=4 instead of Fe=2
        result = parse_formula("Fe2Fe2O3")
        assert result["Fe"] == 2.0, f"Expected Fe=2.0, got Fe={result.get('Fe')}"
        assert result["O"] == 3.0

    def test_empty_formula(self):
        """Test empty input"""
        assert parse_formula("") == {}
        assert parse_formula(None) == {}

    def test_numeric_only(self):
        """Test formulas that are just numbers"""
        assert parse_formula("123") == {}


class TestStandardizeFormula:
    """Test standardize_formula function"""

    def test_simple(self):
        """Test simple standardization"""
        assert standardize_formula("H2O") == "H2O"

    def test_decimal_to_integer(self):
        """Test that .0 decimals are removed"""
        result = standardize_formula("Ba1.0In1.0")
        assert "1.0" not in result


class TestGenerateRandomFormula:
    """Test generate_random_formula function"""

    def test_generates_correct_count(self):
        """Test that correct number of formulas is generated"""
        formulas = generate_random_formula(["Ga", "In", "Sn"], 10)
        assert len(formulas) == 10

    def test_elements_present(self):
        """Test that specified elements are in generated formulas"""
        formulas = generate_random_formula(["Ga", "In"], 5)
        for formula in formulas:
            assert "Ga" in formula or "In" in formula
