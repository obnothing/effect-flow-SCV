from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
from audit_p11_error_candidates import audit_label


class Tests(unittest.TestCase):
    def test_time_and_call_signals(self):
        source="pragma solidity 0.6.0; contract X { function f() public { uint x=block.timestamp; msg.sender.call(\"\"); } }"
        self.assertIn("timestamp_or_block_number_present",audit_label("Time manipulation",source)["signals"])
        self.assertIn("potential_external_call_surface",audit_label("Reentrancy",source)["signals"])
        self.assertIn("low_level_call_review",audit_label("Unchecked Return Values",source)["signals"])


if __name__=="__main__": unittest.main()
