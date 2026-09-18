import unittest

import torch

from src.core.feasibility import constraint_probability


class ProbabilityOfFeasibilityTests(unittest.TestCase):
    def test_probability_at_boundary_is_half(self):
        probability = constraint_probability(torch.tensor(10.0), torch.tensor(2.0), '>=', 10.0)
        self.assertAlmostEqual(float(probability), 0.5, places=6)

    def test_probability_moves_in_expected_direction(self):
        lower_good = constraint_probability(torch.tensor(12.0), torch.tensor(1.0), '>=', 10.0)
        lower_bad = constraint_probability(torch.tensor(8.0), torch.tensor(1.0), '>=', 10.0)
        upper_good = constraint_probability(torch.tensor(8.0), torch.tensor(1.0), '<=', 10.0)
        self.assertGreater(float(lower_good), float(lower_bad))
        self.assertAlmostEqual(float(lower_good), float(upper_good), places=6)

    def test_zero_standard_deviation_is_numerically_safe(self):
        probability = constraint_probability(torch.tensor(11.0), torch.tensor(0.0), '>=', 10.0)
        self.assertTrue(torch.isfinite(probability))
        self.assertGreater(float(probability), 0.999)


if __name__ == '__main__':
    unittest.main()
