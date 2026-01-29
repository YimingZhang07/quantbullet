import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quantbullet.model.gam import (
    load_partial_dependence_json,
    dump_partial_dependence_json,
    export_partial_dependence_payload,
    SplineTermData,
    SplineByGroupTermData,
    FactorTermData,
    TensorTermData,
)
from quantbullet.model.smooth_fit import make_monotone_predictor_pchip

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit(
        "This app requires streamlit. Install with: pip install streamlit"
    ) from exc

DEFAULT_INPUT = Path("./tests/_cache_dir/test_gam_pdep.json")
DEFAULT_OUTPUT = Path("./tests/_cache_dir/test_gam_pdep.edited.json")


def _term_id(key):
    if isinstance(key, tuple):
        return f"{key[0]}::{key[1]}"
    return str(key)


def _term_label(key, term):
    if isinstance(term, SplineTermData):
        return f"spline: {term.feature}"
    if isinstance(term, SplineByGroupTermData):
        return f"spline_by_category: {term.feature} by {term.by_feature}"
    if isinstance(term, FactorTermData):
        return f"factor: {term.feature}"
    if isinstance(term, TensorTermData):
        return f"tensor (ignored): {term.feature_x} x {term.feature_y}"
    return str(key)


def _clean_anchor_df(anchors_df):
    if anchors_df is None or anchors_df.empty:
        return pd.DataFrame(columns=["x", "y"])
    df = anchors_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["x", "y"])
    if df.empty:
        return pd.DataFrame(columns=["x", "y"])
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df = df.sort_values("x", kind="mergesort")
    df = df.drop_duplicates(subset=["x"], keep="last")
    df = df.reset_index(drop=True)
    return df


def _build_pchip_predictor(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return None
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    df = pd.DataFrame({"x": x, "y": y}).drop_duplicates("x", keep="last")
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    if len(x) == 1:
        return lambda x_new: np.full_like(np.asarray(x_new, dtype=float), y[0], dtype=float)
    return make_monotone_predictor_pchip(x, y, extrapolate="flat")


def _interp_on_curve(x, y, x_query):
    x_query = np.asarray(x_query, dtype=float)
    predictor = _build_pchip_predictor(x, y)
    if predictor is None:
        return np.zeros_like(x_query, dtype=float)
    return predictor(x_query)


def _default_anchors(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        return pd.DataFrame(columns=["x", "y"])
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_vals = _interp_on_curve(x, y, [x_min, x_max])
    return pd.DataFrame({"x": [x_min, x_max], "y": y_vals})


def _generate_anchors(x, y, n_anchors):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        return pd.DataFrame(columns=["x", "y"])
    n = max(2, int(n_anchors))
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_grid = np.linspace(x_min, x_max, n)
    y_grid = _interp_on_curve(x, y, x_grid)
    return pd.DataFrame({"x": x_grid, "y": y_grid})


def _interp_curve(x_full, anchors_df):
    df = _clean_anchor_df(anchors_df)
    xa = df["x"].to_numpy()
    ya = df["y"].to_numpy()
    if len(xa) == 0:
        return np.zeros_like(x_full, dtype=float)
    if len(xa) == 1:
        return np.full_like(x_full, ya[0], dtype=float)
    predictor = _build_pchip_predictor(xa, ya)
    if predictor is None:
        return np.zeros_like(x_full, dtype=float)
    return predictor(x_full)


def _plot_curve(x, y_orig, y_new, title, anchors_df=None):
    fig, ax = plt.subplots()
    ax.plot(x, y_orig, label="original", alpha=0.7)
    ax.plot(x, y_new, label="edited", alpha=0.9)
    if anchors_df is not None:
        df = _clean_anchor_df(anchors_df)
        if not df.empty:
            for xv in df["x"].to_numpy():
                ax.axvline(xv, color="tab:orange", alpha=0.6, linestyle="--", linewidth=1.2)
            ax.scatter(df["x"], df["y"], color="tab:orange", s=22, alpha=0.9, label="anchors")
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.minorticks_on()
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def _plot_bar(categories, values, title):
    fig, ax = plt.subplots()
    ax.bar(categories, values, alpha=0.8)
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    st.pyplot(fig, clear_figure=True)


def _build_modified_terms(term_data, modified_terms, term_key_map):
    updated = copy.deepcopy(term_data)
    for term_id, mod in modified_terms.items():
        key = term_key_map[term_id]
        term = updated[key]
        if mod["type"] == "spline":
            term.y = np.asarray(mod["y"])
        elif mod["type"] == "spline_by_category":
            for label, y in mod["group_curves"].items():
                term.group_curves[label]["y"] = np.asarray(y)
        elif mod["type"] == "factor":
            term.values = np.asarray(mod["values"])
    return updated


def _add_anchor_row(anchors_df, x_val, y_val):
    new_row = pd.DataFrame({"x": [float(x_val)], "y": [float(y_val)]})
    if anchors_df is None or anchors_df.empty:
        return _clean_anchor_df(new_row)
    combined = pd.concat([anchors_df, new_row], ignore_index=True)
    return _clean_anchor_df(combined)


def _snap_y_to_original(x_orig, y_orig, anchors_df):
    df = _clean_anchor_df(anchors_df)
    if df.empty:
        return df
    y_new = _interp_on_curve(x_orig, y_orig, df["x"].to_numpy())
    df["y"] = y_new
    return df


def main():
    st.set_page_config(page_title="GAM Partial Dependence Editor", layout="wide")
    st.title("GAM Partial Dependence Editor")
    st.caption("Edit spline curves and categorical coefficients, then export JSON.")

    with st.sidebar:
        st.header("Load JSON")
        input_path = st.text_input("JSON path", value=str(DEFAULT_INPUT))
        load_clicked = st.button("Load")
        st.divider()
        st.header("Export")
        output_path = st.text_input("Output path", value=str(DEFAULT_OUTPUT))
        st.caption("Output JSON will keep intercept/tensor terms unchanged.")

    if "term_data" not in st.session_state:
        st.session_state.term_data = None
        st.session_state.intercept = 0.0
        st.session_state.metadata = None
        st.session_state.term_key_map = {}
        st.session_state.modified_terms = {}

    if load_clicked:
        term_data, intercept, metadata = load_partial_dependence_json(input_path)
        st.session_state.term_data = term_data
        st.session_state.intercept = intercept
        st.session_state.metadata = metadata
        st.session_state.term_key_map = {_term_id(k): k for k in term_data.keys()}
        st.session_state.modified_terms = {}

    if st.session_state.term_data is None:
        st.info("Load a JSON file to start editing.")
        return

    term_data = st.session_state.term_data
    term_key_map = st.session_state.term_key_map
    modified_terms = st.session_state.modified_terms

    editable_terms = []
    ignored_terms = []
    for key, term in term_data.items():
        if isinstance(term, (SplineTermData, SplineByGroupTermData, FactorTermData)):
            editable_terms.append((key, term))
        else:
            ignored_terms.append((key, term))

    if ignored_terms:
        with st.expander("Ignored terms (kept unchanged)"):
            for key, term in ignored_terms:
                st.write(_term_label(key, term))

    labels = [_term_label(k, t) for k, t in editable_terms]
    selected_label = st.selectbox("Select a term to edit", labels)
    selected_idx = labels.index(selected_label)
    selected_key, selected_term = editable_terms[selected_idx]
    selected_id = _term_id(selected_key)

    if isinstance(selected_term, SplineTermData):
        x = selected_term.x
        y = selected_term.y
        anchor_state_key = f"anchors::{selected_id}"

        if anchor_state_key not in st.session_state:
            st.session_state[anchor_state_key] = _default_anchors(x, y)

        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])
        n_anchors = ctrl1.slider(
            "Auto anchors",
            min_value=2,
            max_value=40,
            value=12,
            step=1,
            key=f"n_anchors::{selected_id}",
        )
        if ctrl2.button("Generate evenly spaced", key=f"gen::{selected_id}"):
            st.session_state[anchor_state_key] = _generate_anchors(x, y, n_anchors)
            st.rerun()
        if ctrl3.button("Reset to min/max", key=f"reset::{selected_id}"):
            st.session_state[anchor_state_key] = _default_anchors(x, y)
            st.rerun()
        if ctrl4.button("Snap Y to original", key=f"snap::{selected_id}"):
            st.session_state[anchor_state_key] = _snap_y_to_original(
                x, y, st.session_state[anchor_state_key]
            )
            st.rerun()

        add_col1, add_col2 = st.columns([3, 1])
        x_min = float(np.min(x)) if len(x) else 0.0
        x_max = float(np.max(x)) if len(x) else 1.0
        step = (x_max - x_min) / 100 if x_max > x_min else 1.0
        add_x = add_col1.number_input(
            "Add anchor at x",
            min_value=x_min,
            max_value=x_max,
            value=x_min,
            step=step,
            key=f"add_x::{selected_id}",
        )
        if add_col2.button("Add anchor", key=f"add_btn::{selected_id}"):
            y_add = float(_interp_on_curve(x, y, [add_x])[0])
            st.session_state[anchor_state_key] = _add_anchor_row(
                st.session_state[anchor_state_key], add_x, y_add
            )
            st.rerun()

        table_col, plot_col = st.columns([1, 1])
        with table_col:
            with st.form(key=f"form::{selected_id}"):
                anchors_df = st.data_editor(
                    st.session_state[anchor_state_key],
                    width="stretch",
                    num_rows="dynamic",
                )
                if st.form_submit_button("Apply edits"):
                    anchors_df = _clean_anchor_df(anchors_df)
                    st.session_state[anchor_state_key] = anchors_df
                    st.rerun()

        anchors_df = st.session_state[anchor_state_key]
        y_new = _interp_curve(x, anchors_df)
        modified_terms[selected_id] = {"type": "spline", "y": y_new}
        with plot_col:
            _plot_curve(x, y, y_new, selected_label, anchors_df=anchors_df)

    elif isinstance(selected_term, SplineByGroupTermData):
        group_labels = list(selected_term.group_curves.keys())
        group_label = st.selectbox("Select group", group_labels)

        curve = selected_term.group_curves[group_label]
        x = curve["x"]
        y = curve["y"]
        anchor_state_key = f"anchors::{selected_id}::{group_label}"

        if anchor_state_key not in st.session_state:
            st.session_state[anchor_state_key] = _default_anchors(x, y)

        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])
        n_anchors = ctrl1.slider(
            "Auto anchors",
            min_value=2,
            max_value=40,
            value=12,
            step=1,
            key=f"n_anchors::{selected_id}::{group_label}",
        )
        if ctrl2.button("Generate evenly spaced", key=f"gen::{selected_id}::{group_label}"):
            st.session_state[anchor_state_key] = _generate_anchors(x, y, n_anchors)
            st.rerun()
        if ctrl3.button("Reset to min/max", key=f"reset::{selected_id}::{group_label}"):
            st.session_state[anchor_state_key] = _default_anchors(x, y)
            st.rerun()
        if ctrl4.button("Snap Y to original", key=f"snap::{selected_id}::{group_label}"):
            st.session_state[anchor_state_key] = _snap_y_to_original(
                x, y, st.session_state[anchor_state_key]
            )
            st.rerun()

        add_col1, add_col2 = st.columns([3, 1])
        x_min = float(np.min(x)) if len(x) else 0.0
        x_max = float(np.max(x)) if len(x) else 1.0
        step = (x_max - x_min) / 100 if x_max > x_min else 1.0
        add_x = add_col1.number_input(
            "Add anchor at x",
            min_value=x_min,
            max_value=x_max,
            value=x_min,
            step=step,
            key=f"add_x::{selected_id}::{group_label}",
        )
        if add_col2.button("Add anchor", key=f"add_btn::{selected_id}::{group_label}"):
            y_add = float(_interp_on_curve(x, y, [add_x])[0])
            st.session_state[anchor_state_key] = _add_anchor_row(
                st.session_state[anchor_state_key], add_x, y_add
            )
            st.rerun()

        table_col, plot_col = st.columns([1, 1])
        with table_col:
            with st.form(key=f"form::{selected_id}::{group_label}"):
                anchors_df = st.data_editor(
                    st.session_state[anchor_state_key],
                    width="stretch",
                    num_rows="dynamic",
                )
                if st.form_submit_button("Apply edits"):
                    anchors_df = _clean_anchor_df(anchors_df)
                    st.session_state[anchor_state_key] = anchors_df
                    st.rerun()

        anchors_df = st.session_state[anchor_state_key]
        y_new = _interp_curve(x, anchors_df)
        if selected_id not in modified_terms:
            modified_terms[selected_id] = {"type": "spline_by_category", "group_curves": {}}
        modified_terms[selected_id]["group_curves"][group_label] = y_new
        with plot_col:
            _plot_curve(x, y, y_new, f"{selected_label} - {group_label}", anchors_df=anchors_df)

    elif isinstance(selected_term, FactorTermData):
        categories = selected_term.categories
        values = selected_term.values
        factor_state_key = f"factor::{selected_id}"

        if factor_state_key not in st.session_state:
            st.session_state[factor_state_key] = pd.DataFrame({"category": categories, "value": values})

        table_col, plot_col = st.columns([1, 1])
        with table_col:
            with st.form(key=f"form::{selected_id}"):
                df_edit = st.data_editor(
                    st.session_state[factor_state_key],
                    width="stretch",
                    num_rows="fixed",
                    disabled=["category"],
                )
                if st.form_submit_button("Apply edits"):
                    st.session_state[factor_state_key] = df_edit
                    st.rerun()

        df_edit = st.session_state[factor_state_key]
        modified_terms[selected_id] = {"type": "factor", "values": df_edit["value"].to_numpy()}
        with plot_col:
            _plot_bar(df_edit["category"].tolist(), df_edit["value"].to_numpy(), selected_label)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save JSON to path"):
            updated_terms = _build_modified_terms(term_data, modified_terms, term_key_map)
            dump_partial_dependence_json(
                term_data=updated_terms,
                path=output_path,
                intercept=st.session_state.intercept,
                metadata=st.session_state.metadata,
            )
            st.success(f"Saved to {output_path}")

    with col2:
        updated_terms = _build_modified_terms(term_data, modified_terms, term_key_map)
        payload = export_partial_dependence_payload(
            term_data=updated_terms,
            intercept=st.session_state.intercept,
            metadata=st.session_state.metadata,
        )
        st.download_button(
            "Download JSON",
            data=json.dumps(payload, indent=2),
            file_name=Path(output_path).name,
            mime="application/json",
        )


if __name__ == "__main__":
    main()
