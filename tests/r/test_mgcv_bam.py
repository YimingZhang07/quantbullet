import unittest
import shutil
import time
from pathlib import Path
from quantbullet.r.mgcv_bam import MgcvBamWrapper

DEV_MODE = True

def print_environment_paths():
    import os
    print( "=====" * 10 )
    for key in ["PYTHONPATH", "Path", "R_HOME"]:
        value = os.environ.get(key, "Not Set")
        if value != "Not Set" and os.pathsep in value:
            print(f"{key}:")
            for path in value.split(os.pathsep):
                print(f"  {path}")
        else:
            print(f"{key}: {value}")

    # print the interpreter being used
    import sys
    print(f"Python Interpreter: {sys.executable}")
    import rpy2.robjects as ro
    print(f"R Library Paths: {list(ro.r('.libPaths()'))}")
    print( "=====" * 10 )

class TestMgcvBam(unittest.TestCase):
    def setUp(self):
        # print_environment_paths()
        self.cache_dir = "./tests/_cache_dir"
        # just remove all files in the cache dir, but not the dir itself
        if not DEV_MODE:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        # only clear cache dir in non-dev mode
        # we want to keep files for inspection in dev mode
        if not DEV_MODE:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def test_fit_and_predict(self):
        import pandas as pd
        import numpy as np

        # Create a simple test DataFrame
        n = 1_000_000
        df = pd.DataFrame({
            'x1': np.random.rand(n),
            'x2': np.random.rand(n),
            'y': np.random.rand(n)
        })

        formula = 'y ~ s(x1) + s(x2)'

        time_begin = time.perf_counter()
        mgcv_wrapper = MgcvBamWrapper()

        time_init = time.perf_counter() - time_begin
        print(f"**********MgcvBamWrapper init time**********: {time_init:.2f}s")

        time_begin = time.perf_counter()
        mgcv_wrapper.fit(df, formula, family='gaussian', num_cores=8, discrete=True, nthreads=8)
        time_fit = time.perf_counter() - time_begin
        print(f"**********MgcvBamWrapper fit time**********: {time_fit:.2f}s")
        
        time_begin = time.perf_counter()
        predictions = mgcv_wrapper.predict(df, type='response', chunk_size=500000)
        time_predict = time.perf_counter() - time_begin
        print(f"**********MgcvBamWrapper predict time**********: {time_predict:.2f}s")

        # Optionally, test plotting
        # fpath = Path(self.cache_dir) / "mgcv_plot.pdf"
        # mgcv_wrapper.plot_to_file( out_fpath=fpath, width=2400, height=800, dpi=300, pages=1 )
        mgcv_wrapper.stop_cluster()
        self.assertEqual(len(predictions), n)
        self.assertTrue(isinstance(predictions, (list, np.ndarray)))

    def test_fit_and_predict_pinned_data(self):
        import pandas as pd
        import numpy as np

        # Create a simple test DataFrame
        n = 1_000_000
        df = pd.DataFrame({
            'x1': np.random.rand(n),
            'x2': np.random.rand(n),
            'y': np.random.rand(n)
        })

        formula = 'y ~ s(x1) + s(x2)'

        time_begin = time.perf_counter()
        mgcv_wrapper = MgcvBamWrapper()

        time_init = time.perf_counter() - time_begin
        print(f"**********MgcvBamWrapper init time**********: {time_init:.2f}s")

        time_begin = time.perf_counter()
        mgcv_wrapper.pin_put( name = "test_data", df = df, as_datatable = True, lock = True )
        time_pin = time.perf_counter() - time_begin
        print(f"**********MgcvBamWrapper pin_put time**********: {time_pin:.2f}s")

        time_begin = time.perf_counter()
        mgcv_wrapper.fit_pinned_data( data_name = "test_data", 
                                      formula = formula, 
                                      family = 'gaussian', 
                                      num_cores = 8, 
                                      discrete = True )
        time_fit = time.perf_counter() - time_begin
        print(f"**********MgcvBamWrapper fit_pinned_data time**********: {time_fit:.2f}s")

        
        time_begin = time.perf_counter()
        predictions = mgcv_wrapper.predict_pinned_data( data_name = "test_data", type='response', chunk_size=500000)
        time_predict = time.perf_counter() - time_begin
        print(f"**********MgcvBamWrapper predict_pinned_data time**********: {time_predict:.2f}s")

        # Optionally, test plotting
        # fpath = Path(self.cache_dir) / "mgcv_plot.pdf"
        # mgcv_wrapper.plot_to_file( out_fpath=fpath, width=2400, height=800, dpi=300, pages=1 )
        mgcv_wrapper.stop_cluster()
        self.assertEqual(len(predictions), n)
        self.assertTrue(isinstance(predictions, (list, np.ndarray)))