"""
tests/test_doe.py
==================
Unit tests for the DoE (Design of Experiments) core module.
Tests verify that the generate_doe() function returns valid tables
with the correct structure and design quality metrics.
"""

import pytest
from src.core.doe import generate_doe


# ==============================================================================
# Fixtures — reusable feature/tweaks configurations
# ==============================================================================

@pytest.fixture
def three_cont_features():
    """3 continuous features — minimum for Box-Behnken design."""
    return [
        {'name': 'A', 'type': 'continuous', 'range': '[0, 10]'},
        {'name': 'B', 'type': 'continuous', 'range': '[5, 20]'},
        {'name': 'C', 'type': 'continuous', 'range': '[100, 200]'},
    ]


@pytest.fixture
def two_cont_features():
    """2 continuous features — valid for CCD, LHS but not Box-Behnken."""
    return [
        {'name': 'X', 'type': 'continuous', 'range': '[0, 1]'},
        {'name': 'Y', 'type': 'continuous', 'range': '[0, 1]'},
    ]


# ==============================================================================
# Test: Box-Behnken Design (bbdesign)
# ==============================================================================

class TestBoxBehnkenDesign:
    """Tests for the Box-Behnken design algorithm."""

    def test_returns_non_empty_table(self, three_cont_features):
        """generate_doe should return a non-empty table for 3 continuous features."""
        table, metrics = generate_doe(three_cont_features, {'model': 'bbdesign'})
        assert len(table) > 0

    def test_table_contains_all_features(self, three_cont_features):
        """Every row in the suggestions table should have all feature keys."""
        table, _ = generate_doe(three_cont_features, {'model': 'bbdesign'})
        for row in table:
            assert 'A' in row
            assert 'B' in row
            assert 'C' in row

    def test_metrics_structure(self, three_cont_features):
        """Metrics dictionary should have all required keys."""
        _, metrics = generate_doe(three_cont_features, {'model': 'bbdesign'})
        assert 'orthogonality' in metrics
        assert 'efficiency' in metrics
        assert 'resolution' in metrics
        assert 'curvature' in metrics
        assert 'runs' in metrics
        assert 'model' in metrics

    def test_values_within_bounds(self, three_cont_features):
        """All numerical values should be within the declared feature bounds."""
        table, _ = generate_doe(three_cont_features, {'model': 'bbdesign'})
        for row in table:
            assert 0.0 <= float(row['A']) <= 10.0
            assert 5.0 <= float(row['B']) <= 20.0
            assert 100.0 <= float(row['C']) <= 200.0

    def test_requires_at_least_3_features(self, two_cont_features):
        """Box-Behnken requires ≥3 features; should raise ValueError with 2."""
        with pytest.raises(ValueError, match="Box-Behnken"):
            generate_doe(two_cont_features, {'model': 'bbdesign'})


# ==============================================================================
# Test: Latin Hypercube Sampling (LHS)
# ==============================================================================

class TestLatinHypercubeSampling:
    """Tests for the Latin Hypercube Sampling design."""

    def test_lhs_returns_correct_run_count(self, two_cont_features):
        """LHS should return exactly max_runs rows."""
        table, _ = generate_doe(two_cont_features, {'model': 'lhs', 'max_runs': 15})
        assert len(table) == 15

    def test_lhs_default_run_count(self, two_cont_features):
        """LHS should default to 50 runs when max_runs not specified."""
        table, _ = generate_doe(two_cont_features, {'model': 'lhs'})
        assert len(table) == 50

    def test_lhs_values_within_bounds(self, two_cont_features):
        """All LHS sample values should be within declared feature bounds."""
        table, _ = generate_doe(two_cont_features, {'model': 'lhs', 'max_runs': 10})
        for row in table:
            assert 0.0 <= float(row['X']) <= 1.0
            assert 0.0 <= float(row['Y']) <= 1.0


# ==============================================================================
# Test: Categorical Features
# ==============================================================================

class TestCategoricalFeatures:
    """Tests for designs that include categorical features."""

    def test_categorical_values_are_valid_choices(self):
        """Categorical feature values in the output must only be from the declared choices."""
        features = [
            {'name': 'Solvent', 'type': 'categorical', 'range': 'Water, Ethanol, DMSO'},
            {'name': 'Temp',    'type': 'continuous',  'range': '[20, 80]'},
            {'name': 'pH',      'type': 'continuous',  'range': '[4, 9]'},
        ]
        table, _ = generate_doe(features, {'model': 'bbdesign'})
        valid_choices = {'Water', 'Ethanol', 'DMSO'}
        for row in table:
            assert row['Solvent'] in valid_choices


# ==============================================================================
# Test: Constraint Filtering
# ==============================================================================

class TestConstraints:
    """Tests for numerical and disallowed-combination constraint filtering."""

    def test_numerical_constraint_filters_violations(self):
        """Rows that violate a numerical constraint should not appear in the output."""
        features = [
            {'name': 'A', 'type': 'continuous', 'range': '[0, 10]'},
            {'name': 'B', 'type': 'continuous', 'range': '[0, 10]'},
            {'name': 'C', 'type': 'continuous', 'range': '[0, 10]'},
        ]
        tweaks = {
            'model': 'lhs',
            'max_runs': 50,
            'constraints': {'numerical': ['A + B <= 10'], 'disallowed': []}
        }
        table, _ = generate_doe(features, tweaks)
        for row in table:
            assert float(row['A']) + float(row['B']) <= 10.0 + 1e-6  # Small tolerance for float rounding

    def test_disallowed_combination_is_excluded(self):
        """Rows that match a disallowed categorical combination should be removed."""
        features = [
            {'name': 'Solvent', 'type': 'categorical', 'range': 'Water, Ethanol'},
            {'name': 'Catalyst','type': 'categorical', 'range': 'A, B'},
            {'name': 'Temp',    'type': 'continuous',  'range': '[20, 80]'},
        ]
        tweaks = {
            'model': 'lhs',
            'max_runs': 50,
            'constraints': {
                'numerical': [],
                'disallowed': [[
                    {'feature': 'Solvent',  'value': 'Water'},
                    {'feature': 'Catalyst', 'value': 'A'}
                ]]
            }
        }
        table, _ = generate_doe(features, tweaks)
        for row in table:
            assert not (row['Solvent'] == 'Water' and row['Catalyst'] == 'A')


# ==============================================================================
# Test: Error Cases
# ==============================================================================

class TestErrorCases:
    """Tests for invalid input configurations."""

    def test_single_feature_raises_error(self):
        """DoE requires at least 2 features; 1 should raise ValueError."""
        features = [{'name': 'A', 'type': 'continuous', 'range': '[0, 10]'}]
        with pytest.raises(ValueError, match="at least 2 features"):
            generate_doe(features, {'model': 'bbdesign'})

    def test_empty_features_raises_error(self):
        """An empty features list should raise ValueError."""
        with pytest.raises(ValueError):
            generate_doe([], {'model': 'bbdesign'})
