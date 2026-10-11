import unittest
import numpy as np
from scipy.linalg import expm, logm
from scipy.signal import welch
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from gruxnet.faced_classical_contract import BANDS
from gruxnet.faced_representation import (band_covariances, covariance_features,
    exact_pairs, fit_readout, predict_readout, SHRINKAGE, FEATURE_WIDTH)


class Poison(dict):
    def __getitem__(self, key):
        raise AssertionError('Forbidden mapping access')


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(441)
        self.windows = self.rng.normal(size=(7, 30, 512))
        self.pairs = [('source-%02d'%i, 2) for i in range(16)]
        self.features = {'context': {p:self.rng.normal(size=13) for p in self.pairs},
                         'spectral': {p:self.rng.normal(size=150) for p in self.pairs}}
        self.targets = {p:float(self.rng.uniform(0, 7)) for p in self.pairs}

    def test_diagonal_matches_independent_welch(self):
        f, power = welch(self.windows, fs=128, window='hann', nperseg=512,
                         noverlap=0, nfft=512, detrend='constant', axis=-1)
        expected = np.stack([power[..., (f>=lo)&(f<hi)].sum(axis=-1).mean(axis=0)*.25 for _,lo,hi in BANDS])
        np.testing.assert_allclose(np.diagonal(band_covariances(self.windows),axis1=1,axis2=2), expected, rtol=2e-14, atol=1e-14)

    def test_log_features_reconstruct_shrunken_matrices(self):
        vector, info = covariance_features(self.windows)
        self.assertEqual(vector.shape, (FEATURE_WIDTH,))
        upper=np.triu_indices(30)
        weights=np.where(upper[0]==upper[1],1,np.sqrt(2))
        for b, covariance in enumerate(band_covariances(self.windows)):
            matrix=np.zeros((30,30));matrix[upper]=vector[b*465:(b+1)*465]/weights
            matrix=matrix+matrix.T-np.diag(np.diag(matrix))
            expected=(1-SHRINKAGE)*covariance/np.trace(covariance)+SHRINKAGE*np.eye(30)/30
            np.testing.assert_allclose(matrix,logm(expected).real,atol=1e-13)
            np.testing.assert_allclose(expm(matrix),expected,atol=1e-13)
        self.assertGreaterEqual(min(info['minimum_shrunken_eigenvalues']),SHRINKAGE/30-1e-13)

    def test_scaling_only_changes_trace_features(self):
        a,_=covariance_features(self.windows);b,_=covariance_features(self.windows*3)
        np.testing.assert_allclose(a[:-5],b[:-5],atol=1e-12)
        np.testing.assert_allclose(b[-5:]-a[-5:],np.log10(9),atol=1e-12)

    def test_covariance_retains_cross_channel_sign(self):
        first=self.windows.copy();second=first.copy();second[:,0]*=-1
        a=band_covariances(first);b=band_covariances(second)
        np.testing.assert_allclose(np.diagonal(a,axis1=1,axis2=2),np.diagonal(b,axis1=1,axis2=2))
        np.testing.assert_allclose(a[:,0,1:],-b[:,0,1:])
        self.assertGreater(np.max(np.abs(covariance_features(first)[0]-covariance_features(second)[0])),1e-3)

    def test_car_singularity_has_fixed_positive_definite_shrinkage(self):
        x=self.windows-self.windows.mean(axis=1,keepdims=True)
        vector,info=covariance_features(x)
        self.assertTrue(np.isfinite(vector).all())
        np.testing.assert_allclose(info['minimum_shrunken_eigenvalues'],np.full(5,SHRINKAGE/30),atol=1e-13)

    def test_malformed_and_zero_energy_hold(self):
        for x in (np.zeros((7,30,512)),self.windows[:6],self.windows*np.nan):
            with self.assertRaises(ValueError): covariance_features(x)

    def test_identity_guard_precedes_features_and_targets(self):
        for requested in (self.pairs[:-1],self.pairs+[self.pairs[0]],[('confirmation',2)]):
            with self.assertRaises(PermissionError):
                fit_readout(requested,self.pairs,Poison(),Poison(),'rbf',('context',),.1)
        with self.assertRaises(PermissionError): exact_pairs(self.pairs,self.pairs+[self.pairs[0]])

    def test_kernel_matches_independent_sklearn_estimator(self):
        fitted=fit_readout(self.pairs,self.pairs,self.features,self.targets,'rbf',('context','spectral'),.1)
        z=fitted['source_z'];y=np.asarray([self.targets[p] for p in self.pairs])
        independent=KernelRidge(alpha=.1,kernel='rbf',gamma=fitted['gamma']).fit(z,y-y.mean())
        np.testing.assert_allclose(fitted['dual'],independent.dual_coef_,atol=1e-12)
        np.testing.assert_allclose(predict_readout(fitted,self.pairs,self.pairs,self.features),np.clip(y.mean()+independent.predict(z),0,7),atol=1e-12)
        np.testing.assert_allclose(rbf_kernel(z,gamma=fitted['gamma'])@fitted['dual']+.1*fitted['dual'],y-y.mean(),atol=1e-12)

    def test_linear_matches_independent_ridge(self):
        fitted=fit_readout(self.pairs,self.pairs,self.features,self.targets,'linear',('spectral',),1000.)
        x=np.stack([self.features['spectral'][p] for p in self.pairs]);z=fitted['scaler'].transform(x)
        y=np.asarray([self.targets[p] for p in self.pairs])
        independent=Ridge(alpha=1000.,solver='svd').fit(z,y)
        np.testing.assert_allclose(predict_readout(fitted,self.pairs,self.pairs,self.features),np.clip(independent.predict(z),0,7),atol=1e-12)

    def test_excluded_features_and_targets_cannot_change_fit(self):
        first=fit_readout(self.pairs,self.pairs,self.features,self.targets,'rbf',('spectral',),.1)
        self.features['spectral']['excluded']=np.full(150,np.nan);self.targets['excluded']=np.nan
        second=fit_readout(self.pairs,self.pairs,self.features,self.targets,'rbf',('spectral',),.1)
        np.testing.assert_array_equal(first['dual'],second['dual'])
        self.assertEqual(first['gamma'],second['gamma'])

    def test_degenerate_source_inputs_hold(self):
        feature={'context':{p:np.ones(13) for p in self.pairs}}
        with self.assertRaises(ValueError): fit_readout(self.pairs,self.pairs,feature,self.targets,'rbf',('context',),.1)

    def test_near_interpolation_is_memorization_even_for_shuffled_targets(self):
        y=np.asarray([self.targets[p] for p in self.pairs]);self.rng.shuffle(y)
        permuted=dict(zip(self.pairs,y))
        fitted=fit_readout(self.pairs,self.pairs,self.features,permuted,'rbf',('spectral',),1e-6)
        self.assertLess(fitted['training_raw_mae'],1e-3)

    def test_prediction_guard_precedes_feature_read(self):
        fitted=fit_readout(self.pairs,self.pairs,self.features,self.targets,'rbf',('spectral',),.1)
        with self.assertRaises(PermissionError): predict_readout(fitted,[('protected',2)],self.pairs,Poison())


if __name__=='__main__': unittest.main()
