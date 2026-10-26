"""The exploratory guard must reject quality loss and excess empty masks."""
import unittest

from scripts.select_guarded import choose_guarded


class GuardedFollowupTests(unittest.TestCase):
    def test_guard_rejects_low_source_and_empty_shortcuts(self):
        rows = [
            {"checkpoint": "source.pt", "source_val_dice": 0.82,
             "target_instability": 0.03, "empty_prediction_fraction": 0.10},
            {"checkpoint": "low_source.pt", "source_val_dice": 0.79,
             "target_instability": 0.001, "empty_prediction_fraction": 0.08},
            {"checkpoint": "many_empty.pt", "source_val_dice": 0.81,
             "target_instability": 0.002, "empty_prediction_fraction": 0.15},
            {"checkpoint": "guarded.pt", "source_val_dice": 0.81,
             "target_instability": 0.02, "empty_prediction_fraction": 0.11},
        ]
        source, guarded, eligible = choose_guarded(rows, 0.02, 0.02)
        self.assertEqual(source["checkpoint"], "source.pt")
        self.assertEqual(guarded["checkpoint"], "guarded.pt")
        self.assertEqual({row["checkpoint"] for row in eligible},
                         {"source.pt", "guarded.pt"})

    def test_source_always_remains_eligible(self):
        rows = [{"checkpoint": "source.pt", "source_val_dice": 0.8,
                 "target_instability": 0.5, "empty_prediction_fraction": 0.0}]
        source, guarded, eligible = choose_guarded(rows, 0.0, 0.0)
        self.assertIs(source, guarded)
        self.assertEqual(eligible, rows)


if __name__ == "__main__":
    unittest.main()
