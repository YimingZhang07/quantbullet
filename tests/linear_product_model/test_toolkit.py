"""LinearProductModelToolkit plots that draw through binned means."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("loss",["poisson","mse"])
def test_implied_actual_formula_and_cache_unchanged(loss):
    from quantbullet.linear_product_model import LinearProductModelToolkit
    from quantbullet.model.feature import DataType,Feature,FeatureRole,FeatureSpec
    from types import SimpleNamespace
    from unittest.mock import patch
    toolkit=LinearProductModelToolkit(FeatureSpec([
        Feature("x",DataType.FLOAT,FeatureRole.MODEL_INPUT),Feature("y",DataType.FLOAT,FeatureRole.TARGET)]))
    raw={"x":pd.DataFrame({"feature_value":[1,1,2],"y":[0.,1.,0.],
                           "m":[.2,.4,.3],"model_pred":[1.,2.,1.5],"w":[1.,3.,1.]})}
    reference=toolkit._aggregate_implied_data(raw,{"x":"discrete"},100,loss=loss)["x"]
    with patch.object(toolkit,"compute_implied_actual_data",return_value=raw):
        fig,axes=toolkit.plot_implied_actuals(SimpleNamespace(loss_=loss),None,bin_config={"x":"discrete"},min_count=2,n_cols=1)
    actual=toolkit.implied_actual_data_caches["x"].agg_df
    pd.testing.assert_frame_equal(actual,reference)
    expected=reference["implied_actual"].to_numpy().copy();expected[reference["count"].to_numpy()<2]=np.nan
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(),expected,equal_nan=True)
    assert [bar.get_height() for bar in toolkit.implied_actual_count_axes[0].patches]==[2,1]
    assert axes[0].get_legend() is not None


def test_generate_plots_pdf_titles_grid_pages_and_footer(tmp_path):
    import matplotlib.pyplot as plt
    from pypdf import PdfReader
    from quantbullet.linear_product_model import LinearProductModelToolkit
    from quantbullet.model.feature import DataType,Feature,FeatureRole,FeatureSpec
    toolkit=LinearProductModelToolkit(FeatureSpec([
        Feature("x",DataType.FLOAT,FeatureRole.MODEL_INPUT),Feature("y",DataType.FLOAT,FeatureRole.TARGET)]))
    fig,axes=plt.subplots(1,2)
    for ax in axes:
        ax.plot([0,1],[0,1])
    toolkit.implied_actual_plot_axes=axes
    extra_fig,extra_ax=plt.subplots()
    extra_ax.plot([0,1],[1,0])
    toolkit.add_additional_plots(extra_ax,layout=(1,1),title="Extra")
    path=toolkit.generate_plots_pdf(report_name=str(tmp_path/"Model"))
    plt.close(fig);plt.close(extra_fig)
    reader=PdfReader(path)
    assert [item.title for item in reader.outline]==["Numerical - Implied Errors","Extra"]
    assert "Numerical - Implied Errors" in reader.pages[0].extract_text()
    assert "Model" in reader.pages[0].extract_text()


def _tent_toolkit(raw=False):
    """Tent age, FlatRamp x and one-hot purpose; ``raw`` adds a feature z without a transformer."""
    from sklearn.preprocessing import OneHotEncoder
    from quantbullet.linear_product_model import LinearProductModelToolkit
    from quantbullet.model.feature import DataType,Feature,FeatureRole,FeatureSpec
    from quantbullet.preprocessing import FlatRampTransformer,TentTransformer
    rng=np.random.default_rng(0)
    df=pd.DataFrame({"age":rng.uniform(1.,120.,500),"x":rng.uniform(0.,10.,500),"z":rng.uniform(0.,1.,500),
                     "purpose":pd.Categorical(rng.choice(["C","N","P"],500))})
    df["y"]=rng.poisson(.3,500).astype(float)
    spec=FeatureSpec([Feature(name,DataType.FLOAT,FeatureRole.MODEL_INPUT) for name in ("age","x")+(("z",) if raw else ())]
                     +[Feature("purpose",DataType.CATEGORY,FeatureRole.MODEL_INPUT),Feature("y",DataType.FLOAT,FeatureRole.TARGET)])
    config={"age":TentTransformer((1.,12.,24.,60.,120.)),"x":FlatRampTransformer(knots=[2,5,8],include_bias=True),
            "purpose":OneHotEncoder(drop=None)}
    return df,LinearProductModelToolkit(spec,preprocess_config=config).fit(df)


def test_reference_rows_put_each_value_through_its_transformer():
    df,toolkit=_tent_toolkit(raw=True)
    rows=toolkit.reference_rows({"age":30.,"x":4.,"purpose":"P","z":.5})
    np.testing.assert_allclose(rows["age"],[0,0,5/6,1/6,0])
    np.testing.assert_allclose(rows["x"],toolkit.preprocess_config["x"].transform(pd.DataFrame({"x":[4.]}))[0])
    np.testing.assert_array_equal(rows["purpose"],[0,0,1])
    np.testing.assert_array_equal(rows["z"],[.5])
    with pytest.raises(ValueError,match="Feature income not found"):
        toolkit.reference_rows({"income":1.})


def test_fitting_summary_lists_tent_knots(tmp_path):
    from pypdf import PdfReader
    from quantbullet.linear_product_model import LinearProductRegressor
    from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer
    df,toolkit=_tent_toolkit()
    data=ProductModelDataContainer(df,toolkit.get_expanded_df(df),response=df["y"].to_numpy(),
                                   feature_groups=toolkit.feature_groups)
    model=LinearProductRegressor().fit(data,toolkit.feature_groups,loss="poisson",block_link="log",verbose=0)
    path=toolkit.generate_fitting_summary_pdf(model,report_name=str(tmp_path/"Model"))
    text=" ".join(page.extract_text() for page in PdfReader(path).pages)
    assert "TentTransformer" in text and "knots=[1, 12, 24, 60, 120]" in text
