"""Unit tests for P11 error-diagnosis statistics."""

from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
from diagnose_p11_errors import confusion,decision_confidence,ece_score,selective_risk


class Tests(unittest.TestCase):
    def test_confusion_and_confidence(self):
        labels=np.zeros((4,6),dtype=int); labels[:2,0]=1; labels[2:,1]=1
        probs=np.full((4,6),.1); probs[:,0]=[.9,.2,.8,.1]; probs[:,1]=[.1,.4,.7,.3]
        thresholds=np.full(6,.5); pred=(probs>=thresholds).astype(int)
        rows=confusion(labels,pred)
        self.assertEqual((rows[0]["tn"],rows[0]["fp"],rows[0]["fn"],rows[0]["tp"]),(1,1,1,1))
        confidence=decision_confidence(probs,pred,thresholds)
        self.assertTrue(((confidence>=0)&(confidence<=1)).all())
        risk=selective_risk(labels,pred,confidence); self.assertEqual(risk[0]["decisions"],24)

    def test_calibration(self):
        labels=np.array([0,0,1,1]); probs=np.array([.1,.2,.8,.9])
        ece,bins=ece_score(labels,probs,2,False)
        self.assertAlmostEqual(ece,.15); self.assertEqual(sum(row["count"] for row in bins),4)


if __name__=="__main__": unittest.main()
