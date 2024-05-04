from statsmodels.tsa.vector_ar.vecm import VECM

from ..utils.decorators import require_fitted

class VectorErrorCorrectionModel:
    """A wrapper for the statsmodels VECM class
    
    This model is used to estimate the long-run equilibrium relationship between a set of time series.
    A linear combination of the time series is a stationary process
    """
    def __init__(self, coint_rank=1, k_ar_diff=1, deterministic="co"):
        self.coint_rank = coint_rank
        self.k_ar_diff = k_ar_diff
        self.deterministic = deterministic
        self.model = None
        self.results = None
        self.fitted = False

    def fit(self, endog):
        self.model = VECM(endog, coint_rank=self.coint_rank, k_ar_diff=self.k_ar_diff, deterministic=self.deterministic)
        self.results = self.model.fit()
        self.fitted = True

    def predict(self, steps):
        return self.results.predict(steps)

    @property
    @require_fitted
    def Result(self):
        return self.results
    
    @property
    @require_fitted
    def ConintegrationVector(self):
        return self.results.beta
    
    @property
    @require_fitted
    def Beta(self):
        return self.results.beta
    
    @property
    @require_fitted
    def LoadingMatrix(self):
        return self.results.det_coef
    
    @require_fitted
    def getIntegratedSeries(self, endog):
        return (self.Beta.T @ endog.T).T