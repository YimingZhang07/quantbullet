"""LinearProductModelToolkit plots that draw through binned means."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("loss",["poisson","mse"])
def test_implied_actual_formula_and_cache_unchanged(loss):
    from quantbullet.linear_product_model import LinearProductModelToolkit
    from quantbullet.linear_product_model import ModelSpec
    from quantbullet.linear_product_model.terms import TentTerm
    from types import SimpleNamespace
    from unittest.mock import patch
    toolkit=LinearProductModelToolkit(ModelSpec([TentTerm("x",(0.,1.))]))
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
    from quantbullet.linear_product_model import ModelSpec
    from quantbullet.linear_product_model.terms import TentTerm
    toolkit=LinearProductModelToolkit(ModelSpec([TentTerm("x",(0.,1.))]))
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


def _fitted_spec_model():
    """Tent age, FlatRamp x, one-hot purpose and the age curve by purpose, fitted through a ModelSpec."""
    from quantbullet.linear_product_model import LinearProductRegressor,ModelSpec
    from quantbullet.linear_product_model.terms import BasisTerm,ByTerm,CategoricalTerm,TentTerm
    from quantbullet.preprocessing import FlatRampTransformer
    rng=np.random.default_rng(0)
    df=pd.DataFrame({"age":rng.uniform(1.,120.,500),"x":rng.uniform(0.,10.,500),
                     "purpose":pd.Categorical(rng.choice(["C","N","P"],500))})
    df["y"]=rng.poisson(.3,500).astype(float)
    purpose=CategoricalTerm("purpose",reference="P")
    spec=ModelSpec([ByTerm(TentTerm("age",(1.,12.,24.,60.,120.)),by=purpose,drop=("age",)),purpose,
                    BasisTerm("x",FlatRampTransformer(knots=[2,5,8],include_bias=True))])
    data=spec.prepare(df,response="y")
    return df,data,LinearProductRegressor().fit(data,loss="poisson",block_link="log",verbose=0)


def test_toolkit_takes_a_model_spec_and_reads_terms_from_it():
    from quantbullet.linear_product_model import LinearProductModelToolkit
    from quantbullet.model.feature import DataType,Feature,FeatureRole,FeatureSpec
    with pytest.raises(TypeError,match="takes a ModelSpec"):
        LinearProductModelToolkit(FeatureSpec([Feature("x",DataType.FLOAT,FeatureRole.MODEL_INPUT),
                                               Feature("y",DataType.FLOAT,FeatureRole.TARGET)]))
    df,data,model=_fitted_spec_model()
    toolkit=LinearProductModelToolkit(model.spec_)
    assert toolkit.categorical_feature_group_names==["purpose"]
    assert toolkit.numerical_feature_group_names==["age","x"]
    assert toolkit.feature_groups==data.feature_groups
    np.testing.assert_array_equal(toolkit.get_single_feature_pred_given_values("x",[1.,4.,9.],model),
                                  model.term("x").curve([1.,4.,9.]))
    raw=toolkit.compute_implied_actual_data(model,data)
    assert sorted(raw)==["age|purpose=C","age|purpose=N","age|purpose=P","x"]


def test_fitting_summary_lists_each_term(tmp_path):
    from pypdf import PdfReader
    from quantbullet.linear_product_model import LinearProductModelToolkit
    df,data,model=_fitted_spec_model()
    path=LinearProductModelToolkit(model.spec_).generate_fitting_summary_pdf(model,report_name=str(tmp_path/"Model"))
    text=" ".join(page.extract_text() for page in PdfReader(path).pages)
    for line in ("ByTerm","knots=[1, 12, 24, 60, 120]","CategoricalTerm","3 levels","reference='P'",
                 "BasisTerm","FlatRampTransformer knots=[2, 5, 8]"):
        assert line in text
