import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solidity_source_v2_dataset import _select_windows, _windows


class SoliditySourceV2WindowTest(unittest.TestCase):
    def test_long_windows_cover_the_final_token(self):
        windows = _windows(list(range(1000)), width=384, stride=192)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][-1], 999)
        self.assertEqual(len(windows[-1]), 384)

    def test_budget_preserves_all_mandatory_windows(self):
        windows = [[index] for index in range(20)]
        selected = _select_windows(windows, mandatory_count=12, budget=8)
        self.assertEqual(selected, windows[:12])

    def test_budget_uniformly_selects_ordinary_windows(self):
        windows = [[index] for index in range(20)]
        selected = _select_windows(windows, mandatory_count=2, budget=8)
        self.assertEqual(selected[:2], windows[:2])
        self.assertEqual(selected[-1], windows[-1])
        self.assertEqual(len(selected), 8)


if __name__ == "__main__":
    unittest.main()
