"""Focused behavioral checks for the two publication-time entry-point guards."""
import tempfile
import unittest
from pathlib import Path

from SecondPass.AngularContrastiveMotion.train import evaluate
from SecondPass.PredictiveMotionChange.train import main as train_ffn


class PublicationGuards(unittest.TestCase):
    def test_existing_classifier_attempt_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'budget.json'
            original = b'{"cap_started": 123, "hard_deadline": 456}\n'
            path.write_bytes(original)
            with self.assertRaisesRegex(RuntimeError, 'Existing attempt'):
                train_ffn(Path(folder))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual([p.name for p in Path(folder).iterdir()], ['budget.json'])

    def test_test_threshold_cannot_be_calibrated_on_test(self):
        # Rejection must precede model use, device work and data generation.
        with self.assertRaisesRegex(ValueError, 'validation-selected threshold'):
            evaluate(None, 'test', 384, 3072, deadline=0, threshold=None)


if __name__ == '__main__':
    unittest.main()
