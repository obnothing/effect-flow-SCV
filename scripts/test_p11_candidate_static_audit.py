from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
from run_p11_candidate_static_audit import compiler_for,relevant_checks


class Tests(unittest.TestCase):
    def test_compiler_selection_and_detector_mapping(self):
        compiler,constraint=compiler_for("pragma solidity ^0.4.21; contract A{}")
        self.assertEqual(constraint,"^0.4.21"); self.assertTrue(str(compiler).endswith("solc-0.4.25"))
        found=relevant_checks("DoS",["calls-loop","reentrancy-eth","timestamp"])
        self.assertEqual(found,["calls-loop"])


if __name__=="__main__": unittest.main()
