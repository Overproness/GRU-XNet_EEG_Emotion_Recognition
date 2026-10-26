"""Manufactured execution guards; never open original files or cached outcomes."""
import unittest
from unittest.mock import patch
import numpy as np
from scripts.run_faced_representation import Study, source_sets
from gruxnet.faced_representation import fit_readout


def poisoned(*args,**kwargs):raise AssertionError('Protected operation invoked')


class RunnerTests(unittest.TestCase):
    def test_wrong_role_and_set_hold_before_original_authentication(self):
        x=Study.__new__(Study);x.source=[('source',2)];x.validation=[('validation',2)]
        x.gate=None;x.authenticate=poisoned;x.budget=poisoned
        for requested,role in (([('confirmation',2)],'source'),(x.validation,'validation'),(x.source,'source_bad')):
            with self.assertRaises(PermissionError):x.prepare(requested,role)

    def test_cached_targets_and_fits_require_source_gate(self):
        x=Study.__new__(Study);x.gate=None
        with self.assertRaises(PermissionError):x.open_cached_targets()
        with self.assertRaises(PermissionError):x.fit_all()

    def fixture(self):
        x=Study.__new__(Study);rng=np.random.default_rng(771)
        source=[('source-%02d'%i,2) for i in range(10)];bad=('validation',2)
        x.features={'context':{p:rng.normal(size=13) for p in source+[bad]},
                    'spectral':{p:rng.normal(size=150) for p in source}}
        x.targets={p:float(rng.uniform(0,7)) for p in source+[bad]}
        x.metadata={bad:{'family':'film','acquisition_group':'legacy_250'}}
        x.decisions={bad:{'policy_id':'faced-quality-context-fallback-v1','eeg_available':False,
                          'reasons':[{'channel':'F8','reason':'adc_rail'}]}}
        return x,source,bad

    def test_joint_unavailable_never_requests_eeg_features(self):
        x,source,bad=self.fixture()
        fitted=fit_readout(source,source,x.features,x.targets,'rbf',('context','spectral'),.1)
        rows=x.evaluated_rows(0,'context_spectral_kernel',None,'validation_unseen',[bad],fitted,{bad:2.5},3.)
        self.assertEqual(rows[0]['prediction'],2.5);self.assertTrue(rows[0]['fallback'])
        self.assertFalse(rows[0]['eeg_used'])

    def test_eeg_only_unavailable_uses_exact_source_median(self):
        x,source,bad=self.fixture()
        fitted=fit_readout(source,source,x.features,x.targets,'rbf',('spectral',),.1)
        median=float(np.median([x.targets[p] for p in source]))
        rows=x.evaluated_rows(0,'spectral_kernel',None,'validation_unseen',[bad],fitted,{},median)
        self.assertEqual(rows[0]['prediction'],median)
        self.assertEqual(rows[0]['branch'],'source_global_median_fallback')

    def test_context_predicts_even_when_eeg_is_unavailable(self):
        x,source,bad=self.fixture()
        fitted=fit_readout(source,source,x.features,x.targets,'rbf',('context',),.1)
        rows=x.evaluated_rows(0,'context_kernel',None,'validation_unseen',[bad],fitted,{},3.)
        self.assertFalse(rows[0]['fallback']);self.assertFalse(rows[0]['eeg_used'])


if __name__=='__main__':unittest.main()
