"""Scientific checks for training-label exclusion and exact matched controls."""
import unittest
import numpy as np
import pandas as pd
import torch
from gruxnet.full_context_controls import prior, contexts, normalize, batches
from gruxnet.full_context_models import FullControl, spectrogram
from gruxnet.material_controls import state_digest
from gruxnet.train import seed_everything


class FullContextTests(unittest.TestCase):
    def setUp(self):
        self.table = pd.DataFrame({'subject_id': ['a','a','b','b','c','c','d','d'],
            'material_key': ['v1','v2']*4, 'label': [0,1,1,0,1,1,0,0],
            'training_class': [0,1,1,0,1,1,0,0], 'trial_id': list('abcdefgh')})

    def test_own_participant_labels_cannot_change_own_training_prior(self):
        before = prior(self.table, np.arange(6), [0,1], 2, crossfit=True)
        changed = self.table.copy(); changed.loc[:1,'label'] = [1,0]
        np.testing.assert_array_equal(before, prior(changed, np.arange(6), [0,1], 2, crossfit=True))

    def test_heldout_labels_never_change_context(self):
        idx = {'train': np.arange(4), 'validation': np.array([4,5]), 'test': np.array([6,7])}
        before = contexts(self.table, idx, 2)
        changed = self.table.copy(); changed.loc[4:,'label'] = 1-changed.loc[4:,'label']
        np.testing.assert_array_equal(before, contexts(changed, idx, 2))

    def test_unseen_global_fallback_also_excludes_own_subject(self):
        table = self.table.copy(); table.loc[0,'material_key'] = 'unique'
        q = prior(table, np.arange(6), [0], 2, crossfit=True)
        expected = (np.bincount(table.iloc[2:6].label, minlength=2)+1)/6
        np.testing.assert_array_equal(q[0], expected)

    def test_normalizer_uses_source_only(self):
        rng = np.random.default_rng(1); x = rng.normal(size=(5,14,37,79)).astype(np.float32)
        _, mean, scale = normalize(x, [0,1,2]); x[3:] += 100
        _, other_mean, other_scale = normalize(x, [0,1,2])
        np.testing.assert_array_equal(mean, other_mean); np.testing.assert_array_equal(scale, other_scale)

    def test_residual_identical_backbone_and_prior_offset(self):
        seed_everything(42); a = FullControl('gru',2).eval()
        seed_everything(42); b = FullControl('gru_context',2).eval()
        self.assertEqual(state_digest(a.state_dict()), state_digest(b.state_dict()))
        x = torch.zeros(2,14,37,79); q = torch.log(torch.tensor([[.2,.8],[.4,.6]]))
        with torch.no_grad(): torch.testing.assert_close(b(x,q), a(x)+q, atol=1e-6, rtol=1e-6)

    def test_draws_balanced_and_stft_has_real_time_axis(self):
        sampled, _ = batches(self.table, np.arange(8), 42, 2)
        self.assertEqual(sampled.shape,(200,12))
        for row in sampled: np.testing.assert_array_equal(np.bincount(self.table.iloc[row].label,minlength=2),[6,6])
        x = torch.zeros(1,14,5120); x[:,:,2560:] = torch.sin(torch.arange(2560)*2*torch.pi*10/128)
        z = spectrogram(x); self.assertEqual(tuple(z.shape),(1,14,37,79))
        self.assertGreater(float(z[:,:,:,45:].sum()), float(z[:,:,:,:30].sum()))

    def test_corrected_local_reference_dropout_follows_global_pooling(self):
        from gruxnet.full_context_models_v2 import FullControl as Corrected
        model = Corrected('cbsatt_local',2).train(); captured=[]
        handle=model.cnn.drop.register_forward_pre_hook(lambda _,args: captured.append(tuple(args[0].shape)))
        with torch.no_grad(): model(torch.randn(2,14,37,79))
        handle.remove(); self.assertEqual(captured,[(2,14*128,1,1)])


if __name__ == '__main__': unittest.main()
