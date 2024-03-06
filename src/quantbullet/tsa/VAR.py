from statsmodels.tsa.api import VAR

class VectorAutoRegressionResults:
    def __init__(self, results):
        self.results = results

    @property
    def features(self):
        return self.results.names

    @property
    def coefficients(self):
        return self.results.coefs.reshape(-1, len(self.features))
    
    @property
    def sigma(self):
        return self.results.sigma_u.values
        

class VectorAutoRegression:
    def __init__(self):
        pass

    def fit(self, data, lags):
        self.model = VAR(data)
        self.results_ = self.model.fit(lags)
        self.results = VectorAutoRegressionResults(self.results_)
