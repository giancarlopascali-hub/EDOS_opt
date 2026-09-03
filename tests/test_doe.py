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


# ==============================================================================
# Test: Feature Precision Rules
# ==============================================================================

class TestFeaturePrecision:
    """Tests for decimal precision of generated feature values."""

    def test_whole_number_range_defaults_to_one_decimal(self):
        """Range with whole numbers (0 decimals) should format values to 1 decimal place."""
        features = [
            {'name': 'Temp', 'type': 'continuous', 'range': '[10, 50]'},
            {'name': 'Press', 'type': 'continuous', 'range': '[1, 5]'},
        ]
        table, _ = generate_doe(features, {'model': 'lhs', 'max_runs': 10})
        for row in table:
            for feat_name in ('Temp', 'Press'):
                val_str = str(row[feat_name])
                parts = val_str.split('.')
                assert len(parts) == 2
                assert len(parts[1]) == 1

    def test_explicit_two_decimals_range(self):
        """Range with 2 decimal places (e.g. 0.00 to 1.00) should produce values with <=2 decimal places."""
        features = [
            {'name': 'Conc', 'type': 'continuous', 'range': '[0.00, 1.00]'},
            {'name': 'Flow', 'type': 'continuous', 'range': '[0.10, 0.50]'},
        ]
        table, _ = generate_doe(features, {'model': 'lhs', 'max_runs': 10})
        for row in table:
            for feat_name in ('Conc', 'Flow'):
                val_str = str(row[feat_name])
                parts = val_str.split('.')
                if len(parts) == 2:
                    assert len(parts[1]) <= 2


# ==============================================================================
# Test: Central Composite Design (ccdesign)
# ==============================================================================

class TestCentralCompositeDesign:
    """Tests for the Central Composite design algorithm."""

    def test_returns_non_empty_table(self, three_cont_features):
        """generate_doe should return a non-empty table for 3 continuous features."""
        table, metrics = generate_doe(three_cont_features, {'model': 'ccdesign'})
        assert len(table) > 0
        assert metrics['model'] == 'ccdesign'
        assert metrics['center_replicates'] > 0

    def test_ccdesign_no_unexpected_duplicates(self, three_cont_features):
        """The only duplicate rows in CCD should be the center-point replicates."""
        table, metrics = generate_doe(three_cont_features, {'model': 'ccdesign'})
        rows = [tuple(r.items()) for r in table]
        unique_rows = set(rows)
        # Total minus unique is the number of excess duplicate rows
        n_dups = len(rows) - len(unique_rows)
        # All duplicates should belong to the single center-point condition
        assert n_dups == metrics['center_replicates'] - 1


# ==============================================================================
# Test: Center-Point Ordering (RSM models)
# ==============================================================================

class TestCenterPointOrdering:
    """Tests verifying that replicate center points appear at the end of the table."""

    def test_bbdesign_center_points_are_last(self, three_cont_features):
        """For Box-Behnken, all center points (midpoint of all features) must be at the end."""
        table, metrics = generate_doe(three_cont_features, {'model': 'bbdesign'})
        n_centers = metrics['center_replicates']
        assert n_centers > 0

        # Midpoint for each feature: A: 5.0, B: 12.5, C: 150.0
        center_row = {'A': 5.0, 'B': 12.5, 'C': 150.0}

        # The last n_centers rows must all be the center point
        for row in table[-n_centers:]:
            for k, v in center_row.items():
                assert abs(row[k] - v) < 1e-3

        # None of the earlier rows should be the center point
        for row in table[:-n_centers]:
            is_center = all(abs(row[k] - center_row[k]) < 1e-3 for k in center_row)
            assert not is_center, "Center point found before the end of the table!"

    def test_ccdesign_center_points_are_last(self, three_cont_features):
        """For CCD, all center points must be at the end of the table (not split mid-design)."""
        table, metrics = generate_doe(three_cont_features, {'model': 'ccdesign'})
        n_centers = metrics['center_replicates']
        assert n_centers > 0

        center_row = {'A': 5.0, 'B': 12.5, 'C': 150.0}

        # The last n_centers rows must all be the center point
        for row in table[-n_centers:]:
            for k, v in center_row.items():
                assert abs(row[k] - v) < 1e-3

        # None of the earlier rows should be the center point
        for row in table[:-n_centers]:
            is_center = all(abs(row[k] - center_row[k]) < 1e-3 for k in center_row)
            assert not is_center, "Center point found before the end of the table!"

    def test_metrics_reports_center_replicates(self, three_cont_features):
        """center_replicates metric should be >0 for RSM models and 0 for others."""
        _, bb_metrics = generate_doe(three_cont_features, {'model': 'bbdesign'})
        assert bb_metrics['center_replicates'] == 3

        _, cc_metrics = generate_doe(three_cont_features, {'model': 'ccdesign'})
        assert cc_metrics['center_replicates'] == 8

        _, lhs_metrics = generate_doe(three_cont_features, {'model': 'lhs', 'max_runs': 10})
        assert lhs_metrics['center_replicates'] == 0

        _, dopt_metrics = generate_doe(three_cont_features, {'model': 'doptimal', 'max_runs': 10})
        assert dopt_metrics['center_replicates'] == 0


# ==============================================================================
# Test: D-Optimal Design Uniqueness
# ==============================================================================

class TestDOptimalUniqueness:
    """Tests verifying that D-Optimal design generates 100% unique experimental runs."""

    def test_doptimal_no_duplicate_runs_linear(self, three_cont_features):
        """Linear model D-optimal must contain zero duplicate rows."""
        table, metrics = generate_doe(three_cont_features, {
            'model': 'doptimal', 'dopt_model_order': 'linear', 'max_runs': 20
        })
        rows = [tuple(r.items()) for r in table]
        assert len(rows) == len(set(rows)), "D-optimal linear produced duplicate rows!"
        assert metrics['center_replicates'] == 0

    def test_doptimal_no_duplicate_runs_interaction(self, three_cont_features):
        """Interaction model D-optimal must contain zero duplicate rows."""
        table, metrics = generate_doe(three_cont_features, {
            'model': 'doptimal', 'dopt_model_order': 'interaction', 'max_runs': 20
        })
        rows = [tuple(r.items()) for r in table]
        assert len(rows) == len(set(rows)), "D-optimal interaction produced duplicate rows!"

    def test_doptimal_no_duplicate_runs_quadratic(self, three_cont_features):
        """Quadratic model D-optimal must contain zero duplicate rows."""
        table, metrics = generate_doe(three_cont_features, {
            'model': 'doptimal', 'dopt_model_order': 'quadratic', 'max_runs': 20
        })
        rows = [tuple(r.items()) for r in table]
        assert len(rows) == len(set(rows)), "D-optimal quadratic produced duplicate rows!"

    def test_doptimal_no_duplicate_runs_large(self, three_cont_features):
        """50-run D-optimal must contain zero duplicate rows."""
        table, _ = generate_doe(three_cont_features, {
            'model': 'doptimal', 'dopt_model_order': 'quadratic', 'max_runs': 50
        })
        rows = [tuple(r.items()) for r in table]
        assert len(rows) == len(set(rows)), f"Expected 50 unique rows, got {len(set(rows))}"


# ==============================================================================
# Test: Oversampling Uniqueness (Space-Filling Augmentation)
# ==============================================================================

class TestOversamplingUniqueness:
    """Tests verifying that exact-N oversampling produces unique points, not duplicate rows."""

    def test_fracfact_oversampling_no_duplicates(self, three_cont_features):
        """Oversampling fracfact from 8 to 25 runs must not duplicate rows."""
        table, metrics = generate_doe(three_cont_features, {
            'model': 'fracfact', 'use_exact_runs': True, 'max_runs': 25
        })
        assert len(table) == 25
        assert metrics['oversampling_warning'] is True
        rows = [tuple(r.items()) for r in table]
        assert len(rows) == len(set(rows)), f"Expected 25 unique rows, got {len(set(rows))}"

    def test_def_screening_oversampling_no_duplicates(self, three_cont_features):
        """Oversampling def_screening from 4 to 15 runs must not duplicate rows."""
        table, metrics = generate_doe(three_cont_features, {
            'model': 'def_screening', 'use_exact_runs': True, 'max_runs': 15
        })
        assert len(table) == 15
        assert metrics['oversampling_warning'] is True
        rows = [tuple(r.items()) for r in table]
        assert len(rows) == len(set(rows)), f"Expected 15 unique rows, got {len(set(rows))}"

    def test_bbdesign_oversampling_no_extra_duplicates(self, three_cont_features):
        """Oversampling Box-Behnken: only the 3 center points should be repeated."""
        table, metrics = generate_doe(three_cont_features, {
            'model': 'bbdesign', 'use_exact_runs': True, 'max_runs': 25
        })
        assert len(table) == 25
        rows = [tuple(r.items()) for r in table]
        unique_rows = set(rows)
        # Expected duplicates: exactly metrics['center_replicates'] - 1 (which is 2)
        n_dups = len(rows) - len(unique_rows)
        assert n_dups == metrics['center_replicates'] - 1, (
            f"Expected only {metrics['center_replicates'] - 1} duplicates from center points, but found {n_dups}!"
        )


