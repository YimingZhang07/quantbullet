import pytest
import numpy as np
import pandas as pd
from quantbullet.model import ModelMetricsConsts

class TestModelSelection( object ):
    def setup_class(self):
        pass

    def test_model_objective(self):
        assert isinstance(ModelMetricsConsts.xgboost_objectives, list)