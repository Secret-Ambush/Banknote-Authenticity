from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.evaluation.shared_comparison import comparison_metrics, comparison_predictions


SCRIPT_DIR = Path(__file__).resolve().parent
COURSEWORK_ROOT = SCRIPT_DIR
RIDDHI_ROOT = COURSEWORK_ROOT / "Riddhi"
WORKSPACE_ROOT = COURSEWORK_ROOT.parent
RESULTS_DIR = RIDDHI_ROOT / "ann_results"
FEATURES = ["variance", "skewness", "curtosis", "entropy"]
CLASS_NAMES = {0: "Class 0", 1: "Class 1"}
COLORS = {0: "#54d6a0", 1: "#ff7d91"}


def find_file(candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate {description}. Searched: {searched}")


@st.cache_data(show_spinner=False)
def load_dataset(path: str, modified_ns: int) -> tuple[pd.DataFrame, pd.DataFrame, str, int]:
    source = Path(path)
    raw_rows: list[list[float]] = []
    repaired_labels = 0

    with source.open(newline="") as stream:
        for line_number, fields in enumerate(csv.reader(stream), start=1):
            if not fields or not any(field.strip() for field in fields):
                continue
            if len(fields) != 5:
                raise ValueError(f"Expected five comma-separated values on line {line_number}.")

            target = fields[-1].strip()
            if target.endswith("§") and target[:-1].strip() in {"0", "1"}:
                fields[-1] = target[:-1].strip()
                repaired_labels += 1
            try:
                raw_rows.append([float(field.strip()) for field in fields])
            except ValueError as error:
                raise ValueError(f"Non-numeric value on banknote data line {line_number}.") from error

    values = np.asarray(raw_rows, dtype=float)
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError("Expected finite rows with four features and one target.")
    if not np.isin(values[:, -1], [0, 1]).all():
        raise ValueError("Expected binary targets 0 and 1.")

    raw = pd.DataFrame(values, columns=FEATURES + ["class"])
    raw["class"] = raw["class"].astype(int)
    raw["source_row_index"] = np.arange(len(raw), dtype=int)
    conflicting = raw.groupby(FEATURES)["class"].nunique().gt(1)
    if conflicting.any():
        raise ValueError("Identical features have conflicting labels; investigate before splitting.")

    retained = raw.drop_duplicates(subset=FEATURES + ["class"], keep="first").copy()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return raw, retained, digest, repaired_labels


@st.cache_data(show_spinner=False)
def load_experiment(path: str, modified_ns: int) -> dict:
    with Path(path).open() as stream:
        return json.load(stream)


@st.cache_data(show_spinner=False)
def load_predictions(path: str, modified_ns: int) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_model(path: str, modified_ns: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def architecture_label(config: dict) -> str:
    hidden = "×".join(str(width) for width in config["hidden"])
    names = {"logistic": "sigmoid", "relu": "ReLU", "tanh": "tanh", "linear": "linear"}
    activations = " → ".join(names.get(name, name) for name in config["activations"])
    return f"[{hidden}] · {activations}"


def chart_layout(fig: go.Figure, height: int = 370) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=48, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce8f5", family="Inter, sans-serif"),
        title_font=dict(size=16),
        legend_title_text="",
        hoverlabel=dict(bgcolor="#162131", font_color="#f4f7fb"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#314155")
    fig.update_yaxes(gridcolor="#273548", zeroline=False, linecolor="#314155")
    return fig


def show_data_tab(
    raw: pd.DataFrame,
    data: pd.DataFrame,
    experiment: dict,
    data_digest: str,
    repaired_labels: int,
) -> None:
    st.subheader("Dataset explorer")
    st.caption(
        "Banknote Authentication · four continuous image-derived features · binary target. "
        "Exact duplicate rows are excluded to match the experiment split."
    )
    counts = data["class"].value_counts().sort_index()
    first, second, third, fourth = st.columns(4)
    first.metric("Unique examples", f"{len(data):,}", f"{len(raw) - len(data)} exact duplicates removed")
    second.metric("Features", str(len(FEATURES)), "continuous")
    third.metric("Class 0", f"{counts.get(0, 0):,}", f"{counts.get(0, 0) / len(data):.1%} of rows")
    fourth.metric("Class 1", f"{counts.get(1, 0):,}", f"{counts.get(1, 0) / len(data):.1%} of rows")

    left, right = st.columns([1, 1.6], gap="large")
    with left:
        st.markdown("#### Class balance")
        balance = counts.rename(index=CLASS_NAMES).rename_axis("class").reset_index(name="rows")
        fig = px.bar(
            balance,
            x="class",
            y="rows",
            color="class",
            color_discrete_map={"Class 0": COLORS[0], "Class 1": COLORS[1]},
            text="rows",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        chart_layout(fig, 330)
        fig.update_layout(showlegend=False, yaxis_title="Examples", xaxis_title="")
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    with right:
        st.markdown("#### Feature correlation")
        correlation = data[FEATURES].corr()
        fig = go.Figure(
            go.Heatmap(
                z=correlation.to_numpy(),
                x=FEATURES,
                y=FEATURES,
                zmin=-1,
                zmax=1,
                colorscale=[[0, "#6789f5"], [0.5, "#17263a"], [1, "#54d6a0"]],
                text=np.round(correlation.to_numpy(), 2),
                texttemplate="%{text}",
                hovertemplate="%{y} × %{x}<br>Correlation: %{z:.3f}<extra></extra>",
                colorbar=dict(title="r"),
            )
        )
        chart_layout(fig, 330)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    st.markdown("#### Feature distributions")
    selected_feature = st.selectbox("Feature", FEATURES, key="eda_feature")
    fig = px.histogram(
        data,
        x=selected_feature,
        color="class",
        color_discrete_map=COLORS,
        marginal="box",
        opacity=0.76,
        barmode="overlay",
        nbins=36,
        labels={"class": "Class", selected_feature: selected_feature.title()},
        hover_data=["source_row_index"],
    )
    chart_layout(fig, 390)
    fig.update_layout(yaxis_title="Examples")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    st.dataframe(data.drop(columns="source_row_index").describe().T, width="stretch")


def make_model_frame(results: list[dict]) -> pd.DataFrame:
    rows = []
    for index, result in enumerate(results, start=1):
        config = result["config"]
        rows.append(
            {
                "Model": f"M{index:02d} · {architecture_label(config)}",
                "Architecture": architecture_label(config),
                "Activation": " → ".join(config["activations"][:-1]),
                "Hidden layers": len(config["hidden"]),
                "Parameters": result["parameters"],
                "Informants": config.get("informant_count") if config.get("c_informant", 0) > 0 else 0,
                "Informant coefficient": config.get("c_informant", 0.0),
                "CV accuracy": result["mean_accuracy"],
                "CV std": result["std_accuracy"],
                "CV BCE": result["mean_bce"],
                "Seconds": result["seconds"],
                "Scores": result["scores"],
            }
        )
    return pd.DataFrame(rows)


def show_models_tab(experiment: dict) -> None:
    st.subheader("Riddhi's ANN architecture comparison")
    st.caption(
        "Results are from the notebook's repeated stratified 3-fold cross-validation "
        "(two seeds per fold). Error bars show standard deviation across the six fold/seed runs."
    )
    models = make_model_frame(experiment["architecture_results"])
    best_label = architecture_label(experiment["best_config"])
    perfect_count = int((models["CV accuracy"] == 1.0).sum())
    st.info(
        f"{perfect_count} of {len(models)} architectures achieved exactly 100% mean CV accuracy. "
        f"The full range is {models['CV accuracy'].min():.4%} to "
        f"{models['CV accuracy'].max():.4%}. Near-perfect accuracy can hide differences; "
        "compare validation BCE (lower is better) and individual fold scores as well. "
        "These are validation results, not separate test scores for every architecture."
    )

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown("#### Accuracy across architectures")
        chart_data = models.sort_values("CV accuracy", ascending=True)
        fig = go.Figure()
        for _, row in chart_data.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[row["CV accuracy"] * 100],
                    y=[row["Model"]],
                    orientation="h",
                    name=row["Activation"] or "default",
                    marker_color=px.colors.qualitative.Safe[
                        int(row["Hidden layers"]) % len(px.colors.qualitative.Safe)
                    ],
                    error_x=dict(type="data", array=[row["CV std"] * 100], visible=True, color="#e7eef8"),
                    hovertemplate=(
                        "%{y}<br>Mean accuracy: %{x:.3f}%<br>"
                        f"CV BCE: {row['CV BCE']:.5g}<br>Parameters: {row['Parameters']}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
        chart_layout(fig, max(430, len(models) * 34))
        fig.update_layout(
            xaxis=dict(title="Mean CV accuracy (%)", range=[97, 100.15]),
            yaxis_title="",
            bargap=0.28,
        )
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    with right:
        st.markdown("#### Accuracy vs. model size")
        fig = px.scatter(
            models,
            x="Parameters",
            y="CV accuracy",
            color="Activation",
            hover_name="Model",
            hover_data={"CV BCE": ":.5g", "CV std": ":.4g", "Parameters": True},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_traces(marker=dict(size=10))
        fig.update_yaxes(tickformat=".2%", title="Mean CV accuracy")
        fig.update_xaxes(title="Trainable parameters")
        chart_layout(fig, 430)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption("All models use visible, equal-sized markers, including those with zero CV standard deviation.")

    st.markdown("#### Candidate leaderboard")
    display = models.drop(columns=["Scores"]).sort_values(
        ["CV accuracy", "CV BCE", "Parameters"], ascending=[False, True, True]
    )
    st.dataframe(
        display.style.format(
            {
                "CV accuracy": "{:.3%}",
                "CV std": "{:.3%}",
                "CV BCE": "{:.5g}",
                "Seconds": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    chosen = st.selectbox("Inspect fold/seed scores", models["Model"].tolist(), key="cv_model")
    selected = models.loc[models["Model"] == chosen].iloc[0]
    scores = pd.DataFrame(selected["Scores"])
    scores["run"] = scores.apply(lambda row: f"Fold {int(row['fold'])} · seed {int(row['seed'])}", axis=1)
    fig = px.bar(
        scores,
        x="run",
        y="accuracy",
        color="fold",
        hover_data=["seed", "bce", "evaluations"],
        labels={"run": "CV run", "accuracy": "Accuracy", "fold": "Fold"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_yaxes(tickformat=".2%", range=[0.97, 1.002])
    chart_layout(fig, 330)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    if best_label in selected["Architecture"]:
        st.success(f"Selected final architecture: {best_label}. It is the only architecture evaluated on the locked test set.")
    else:
        st.info(f"The saved final model is {best_label}; this candidate has cross-validation results only.")

    with st.expander("PSO refinement runs"):
        refinements = make_model_frame(experiment["refinement_results"]).drop(columns=["Scores"])
        st.dataframe(
            refinements.sort_values(["CV accuracy", "CV BCE"], ascending=[False, True]).style.format(
                {"CV accuracy": "{:.3%}", "CV std": "{:.3%}", "CV BCE": "{:.5g}", "Seconds": "{:.2f}"}
            ),
            width="stretch",
            hide_index=True,
        )


def describe_layers(layers: list[int]) -> str:
    return " → ".join(str(layer) for layer in layers)


def build_people_comparison(
    riddhi: dict,
    maazuddin: dict,
    selected_group: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    experiment = riddhi["experiment"]
    if selected_group == "Architecture":
        for result in experiment["architecture_results"]:
            config = result["config"]
            layers = [4, *config["hidden"], 1]
            rows.append(
                {
                    "Person": "Riddhi",
                    "Configuration": f"ANN · {describe_layers(layers)} · {'/'.join(config['activations'][:-1])}",
                    "Architecture": describe_layers(layers),
                    "Activation": " → ".join(config["activations"][:-1]),
                    "Mean accuracy": result["mean_accuracy"],
                    "Std accuracy": result["std_accuracy"],
                    "Runs": len(result["scores"]),
                    "Metric": "CV validation accuracy",
                    "F1": np.nan,
                    "Execution seconds": np.nan,
                }
            )
        source_group = "architecture"
    else:
        source_group = selected_group

    maaz_group = maazuddin.get("groups", {}).get(source_group, {})
    for result in maaz_group.get("summary", []):
        layers = result["architecture"]
        rows.append(
            {
                "Person": "Maazuddin",
                "Configuration": (
                    f"{result['configuration_name']} · {describe_layers(layers)} · {result['activation']}"
                ),
                "Architecture": describe_layers(layers),
                "Activation": result["activation"],
                "Mean accuracy": result["mean_test_accuracy"],
                "Std accuracy": result["std_test_accuracy"],
                "Runs": sum(
                    1
                    for run in maaz_group.get("individual", [])
                    if run["configuration_name"] == result["configuration_name"]
                ),
                "Metric": "Fixed-holdout test accuracy",
                "F1": result.get("mean_test_f1_score", np.nan),
                "Execution seconds": result.get("mean_execution_time_seconds", np.nan),
            }
        )
    return pd.DataFrame(rows)


def show_people_tab(riddhi: dict, maazuddin: dict) -> None:
    st.subheader("Compare results by person and architecture")
    st.caption(
        "Notebook results are labeled Riddhi; the other saved experiment groups are labeled Maazuddin. "
        "Each row preserves its source configuration and evaluation protocol."
    )
    first, second, third = st.columns(3)
    first.metric("Riddhi · notebook architectures", len(riddhi["experiment"]["architecture_results"]))
    second.metric(
        "Maazuddin · experiment groups",
        len(maazuddin["groups"]),
    )
    third.metric(
        "Maazuddin · architecture candidates",
        len(maazuddin["groups"].get("architecture", {}).get("summary", [])),
    )

    group_names = {
        "Architecture": "Architecture",
        "Activation functions": "activation",
        "PSO budgets": "budget",
        "PSO coefficients": "coefficient",
        "Pilot architectures": "pilot",
    }
    choice = st.selectbox("Experiment group", list(group_names), key="people_group")
    rows = build_people_comparison(riddhi, maazuddin, group_names[choice])
    if rows.empty:
        st.warning(f"No saved results are available for the {choice.lower()} group.")
        return

    if choice == "Architecture":
        st.warning(
            "Protocol caveat: Riddhi's bars are repeated 3-fold validation accuracy; "
            "Maazuddin's are mean accuracy on a fixed 70/30 holdout across independent seeds. "
            "The dataset splits and metrics are different, so treat this as contextual comparison, "
            "not a controlled head-to-head ranking."
        )
    elif choice == "Pilot architectures":
        st.info("Pilot results contain one run per configuration; their zero standard deviations are not uncertainty estimates.")
    else:
        st.info("This group contains Maazuddin's saved experiments only; it has no matching notebook experiment group.")

    st.markdown("#### Accuracy by configuration")
    chart_data = rows.sort_values("Mean accuracy", ascending=True)
    fig = go.Figure()
    person_colors = {"Riddhi": "#54d6a0", "Maazuddin": "#87a9ff"}
    for person in ("Riddhi", "Maazuddin"):
        subset = chart_data[chart_data["Person"] == person]
        if subset.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=subset["Mean accuracy"] * 100,
                y=subset["Configuration"],
                orientation="h",
                name=person,
                marker_color=person_colors[person],
                error_x=dict(
                    type="data",
                    array=subset["Std accuracy"].fillna(0).to_numpy() * 100,
                    visible=True,
                ),
                customdata=np.column_stack(
                    [
                        subset["Metric"],
                        subset["Runs"].astype(str),
                        subset["Architecture"],
                        subset["Activation"],
                    ]
                ),
                hovertemplate=(
                    "%{y}<br>Accuracy: %{x:.3f}%<br>Metric: %{customdata[0]}"
                    "<br>Runs: %{customdata[1]}<br>Architecture: %{customdata[2]}"
                    "<br>Hidden activation(s): %{customdata[3]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="group",
        xaxis=dict(title="Mean accuracy (%) · zoomed for readability", range=[95, 100.5]),
        yaxis_title="",
        bargap=0.25,
    )
    chart_layout(fig, max(390, len(rows) * 35))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    table = rows.drop(columns=["Configuration"]).sort_values(
        ["Mean accuracy", "Person"], ascending=[False, True]
    )
    st.markdown("#### Results table")
    st.dataframe(
        table.style.format(
            {
                "Mean accuracy": "{:.3%}",
                "Std accuracy": "{:.3%}",
                "F1": "{:.3%}",
                "Execution seconds": "{:.3f}",
            },
            na_rep="—",
        ),
        width="stretch",
        hide_index=True,
    )
    riddhi_protocol = riddhi.get("protocol", {})
    if isinstance(riddhi_protocol, dict):
        cv_description = riddhi_protocol.get(
            "architecture_metric",
            "Repeated stratified cross-validation",
        )
        folds = riddhi_protocol.get("folds", riddhi["experiment"].get("fold_count", "—"))
        repeats = riddhi_protocol.get(
            "repeats",
            len(riddhi["experiment"].get("search_seeds", [])),
        )
    else:
        cv_description = str(riddhi_protocol)
        folds = riddhi["experiment"].get("fold_count", "—")
        repeats = len(riddhi["experiment"].get("search_seeds", []))
    st.caption(
        f"Riddhi protocol: {cv_description} ({folds} folds × {repeats} seeds). "
        f"Maazuddin protocol: {maazuddin['protocol']['split']} with "
        f"{maazuddin['protocol']['runs']} independent runs."
    )


def confusion_counts(actual: np.ndarray, predicted: np.ndarray) -> tuple[int, int, int, int]:
    true_negative = int(np.sum((actual == 0) & (predicted == 0)))
    false_positive = int(np.sum((actual == 0) & (predicted == 1)))
    false_negative = int(np.sum((actual == 1) & (predicted == 0)))
    true_positive = int(np.sum((actual == 1) & (predicted == 1)))
    return true_negative, false_positive, false_negative, true_positive


def architecture_figure(model: dict[str, np.ndarray], owner: str) -> go.Figure:
    layers = [int(width) for width in model["layer_sizes"]]
    activations = ["inputs", *[str(name) for name in model["activations"]]]
    positions = [np.arange(width) - (width - 1) / 2 for width in layers]
    edge_x, edge_y = [], []
    for layer in range(len(layers) - 1):
        for start in positions[layer]:
            for end in positions[layer + 1]:
                edge_x.extend([layer, layer + 1, None])
                edge_y.extend([start, end, None])
    fig = go.Figure(go.Scatter(
        x=edge_x, y=edge_y, mode="lines", line=dict(color="#314155", width=1),
        hoverinfo="skip", showlegend=False,
    ))
    for layer, (width, y, activation) in enumerate(zip(layers, positions, activations)):
        name = "Input" if layer == 0 else "Output" if layer == len(layers) - 1 else f"Hidden {layer}"
        fig.add_trace(go.Scatter(
            x=[layer] * width, y=y, mode="markers",
            marker=dict(size=17, color="#54d6a0" if layer == 0 else "#87a9ff"),
            text=[f"{name} · neuron {index + 1} · {activation}" for index in range(width)],
            hovertemplate="%{text}<extra></extra>", showlegend=False,
        ))
        fig.add_annotation(
            x=layer, y=max(layers) / 2 + 0.6,
            text=f"{name}<br>{width} · {activation}", showarrow=False,
        )
    chart_layout(fig, 340)
    fig.update_layout(title=f"{owner} · {describe_layers(layers)}")
    fig.update_xaxes(visible=False, range=[-0.4, len(layers) - 0.6])
    fig.update_yaxes(visible=False, range=[-max(layers) / 2 - 0.5, max(layers) / 2 + 2])
    return fig


def roc_points(actual: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-probabilities, kind="stable")
    truth, scores = actual[order], probabilities[order]
    endpoints = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    return (
        np.r_[0, np.cumsum(truth == 0)[endpoints] / np.sum(actual == 0)],
        np.r_[0, np.cumsum(truth == 1)[endpoints] / np.sum(actual == 1)],
    )


def show_shared_comparison(experiment: dict, raw: pd.DataFrame, data_digest: str) -> None:
    st.subheader("Same samples · two selected models")
    try:
        metadata, riddhi, maazuddin, rows = comparison_predictions(
            RESULTS_DIR, raw[FEATURES + ["class"]].to_numpy(), data_digest,
        )
        current_experiment = json.loads((RESULTS_DIR / "experiment.json").read_text())
        if current_experiment != experiment:
            raise ValueError("Riddhi's named bundle is stale; refresh it before comparing models.")
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Shared comparison is unavailable: {error}")
        st.caption("From the project root, refresh the named results and generate the shared-split model:")
        st.code("python Riddhi/Scripts/save_person_results.py\npython -m src.evaluation.shared_comparison")
        return

    actual = rows["actual_class"]
    models = {"Riddhi": riddhi, "Maazuddin": maazuddin}
    st.caption(
        f"Both models are evaluated on the exact same {len(actual)} deduplicated rows, excluded "
        f"from both training and CV. Shared development set: {len(riddhi['development_indices'])} rows. "
        "Decision threshold is fixed at 0.50 for both; neither model is selected by these test scores."
    )
    st.warning(
        "Maazuddin's model is newly selected by development CV and trained on the shared split, "
        "not taken from his historical 70/30 experiment. The test set has already been inspected. "
        "Training objectives, budgets and search spaces differ, so this is not an architecture-only "
        "or contributor-only ranking. Details are under Selection protocol below."
    )
    for column, (owner, model) in zip(st.columns(2, gap="large"), models.items()):
        with column:
            st.plotly_chart(architecture_figure(model, owner), width="stretch", config={"displaylogo": False})
            st.caption(
                f"{len(model['parameters'])} weights and biases · "
                f"Activations: {' → '.join(str(value) for value in model['activations'])}. "
                "Lines show connectivity, not weight magnitude; biases are included in the count."
            )

    summary, predictions = [], {}
    for owner in models:
        probabilities = rows[owner]
        predicted = (probabilities >= 0.5).astype(int)
        predictions[owner] = predicted
        scores = comparison_metrics(actual, probabilities)
        summary.append({
            "Model": owner, "Architecture": describe_layers(models[owner]["layer_sizes"]),
            "Parameters": len(models[owner]["parameters"]),
            "Accuracy": scores["accuracy"], "Balanced accuracy": scores["balanced_accuracy"],
            "Precision (class 1)": scores["precision"], "Recall (class 1)": scores["recall"],
            "F1 (class 1)": scores["f1_score"], "ROC AUC": scores["roc_auc"],
            "Brier score": scores["brier_score"],
            "Correct": int(np.sum(predicted == actual)), "Errors": int(np.sum(predicted != actual)),
        })
    summary_frame = pd.DataFrame(summary)
    st.markdown("#### Shared-test results")
    st.dataframe(
        summary_frame.style.format({
            **{name: "{:.2%}" for name in (
                "Accuracy", "Balanced accuracy", "Precision (class 1)", "Recall (class 1)", "F1 (class 1)",
            )},
            "ROC AUC": "{:.4f}", "Brier score": "{:.6g}",
        }), width="stretch", hide_index=True,
    )
    st.caption(
        "Balanced accuracy gives both classes equal weight. Precision/recall/F1 describe Class 1; "
        "undefined precision is shown as 0. Brier score measures probability error (lower is better). "
        "Perfect accuracy on this sample does not establish perfect generalisation or calibration."
    )
    bars = summary_frame.melt(
        id_vars="Model", value_vars=["Accuracy", "Balanced accuracy", "F1 (class 1)", "ROC AUC"],
        var_name="Metric", value_name="Score",
    )
    fig = px.bar(bars, x="Metric", y="Score", color="Model", barmode="group",
                 color_discrete_sequence=["#54d6a0", "#87a9ff"])
    chart_layout(fig)
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    probability_frame = pd.concat([
        pd.DataFrame({
            "Model": owner, "P(class 1)": rows[owner],
            "Actual class": [CLASS_NAMES[int(label)] for label in actual],
        })
        for owner in models
    ], ignore_index=True)
    fig = px.histogram(
        probability_frame, x="P(class 1)", color="Actual class", facet_col="Model",
        nbins=40, barmode="overlay", opacity=0.7,
        color_discrete_map={"Class 0": COLORS[0], "Class 1": COLORS[1]},
    )
    chart_layout(fig)
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(title="Confidence by model · same samples")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption(
        "Two models can classify the same rows correctly but give different probabilities. "
        "Class 0 should lie below 0.5 and Class 1 above it. Distance from 0.5 indicates confidence, "
        "not class imbalance; these histograms do not establish calibration on future data."
    )

    for column, owner in zip(st.columns(2, gap="large"), models):
        with column:
            tn, fp, fn, tp = confusion_counts(actual, predictions[owner])
            matrix = np.asarray([[tn, fp], [fn, tp]])
            fig = go.Figure(go.Heatmap(
                z=matrix, x=["Predicted 0", "Predicted 1"], y=["Actual 0", "Actual 1"],
                text=matrix, texttemplate="%{text}", colorscale="Blues", showscale=False,
            ))
            chart_layout(fig, 310)
            fig.update_layout(title=f"{owner} · confusion matrix")
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    fig = go.Figure()
    for owner, color in zip(models, ["#54d6a0", "#87a9ff"]):
        fpr, tpr = roc_points(actual, rows[owner])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=owner, line=dict(color=color, width=3),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="#8393a8"),
    ))
    chart_layout(fig)
    fig.update_layout(title="ROC curves · same test rows")
    fig.update_xaxes(title="False-positive rate", range=[0, 1])
    fig.update_yaxes(title="True-positive rate", range=[0, 1])
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    st.markdown("#### Compare individual test samples")
    table = pd.DataFrame({
        "Source row (0-based)": rows["original_row_index"], "Actual class": actual,
        "Riddhi P(class 1)": rows["Riddhi"], "Riddhi prediction": predictions["Riddhi"],
        "Maazuddin P(class 1)": rows["Maazuddin"], "Maazuddin prediction": predictions["Maazuddin"],
        "Models agree": predictions["Riddhi"] == predictions["Maazuddin"],
    }).sort_values("Source row (0-based)")
    selected_row = st.selectbox("Inspect a shared test sample (source row, 0-based)", table["Source row (0-based)"].tolist())
    st.dataframe(
        raw.loc[raw["source_row_index"] == selected_row, FEATURES + ["class"]],
        width="stretch", hide_index=True,
    )
    st.dataframe(table.loc[table["Source row (0-based)"] == selected_row], width="stretch", hide_index=True)
    disagreements = int(np.sum(predictions["Riddhi"] != predictions["Maazuddin"]))
    st.caption(f"Models disagree on {disagreements} of {len(actual)} samples. Each row below is evaluated by both models.")
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("Download paired predictions", table.to_csv(index=False),
                       file_name="shared_test_predictions.csv", mime="text/csv")
    with st.expander("Selection protocol and training settings"):
        config = experiment["best_config"]
        selected = metadata["selected"]
        st.write(
            f"Maazuddin: {len(metadata['candidates'])} existing profiles, "
            f"{metadata['fold_count']} stratified development folds, "
            f"optimizer seeds {metadata['search_seeds']} in each fold. "
            f"Selection: {metadata['selection']} Final restart chosen by training accuracy only; "
            f"seed {metadata['selected_final_seed']}. Fold scalers use fold-training rows only; "
            "final scalers use development rows only. Both use Riddhi's saved split and fold seeds."
        )
        st.dataframe(pd.DataFrame([
            {
                "Model": "Riddhi", "Training objective": "BCE + weight-only L2",
                "Particles": config["particles"], "Evaluations per fit": config["particles"] * (config["iterations"] + 1),
                "CV-selected accuracy": experiment["selected_validation_accuracy"],
            },
            {
                "Model": "Maazuddin", "Training objective": "Accuracy",
                "Particles": selected["config"]["swarm_size"],
                "Evaluations per fit": selected["config"]["max_evaluations"],
                "CV-selected accuracy": selected["mean_accuracy"],
            },
        ]), width="stretch", hide_index=True)
        st.json(selected["config"])
        st.caption("Maazuddin's development-only candidate scores (not test rankings):")
        st.dataframe(pd.DataFrame([
            {"Group": row["config"]["group"], "Configuration": row["config"]["name"],
             "Architecture": describe_layers(row["config"]["architecture"]),
             "CV accuracy": row["mean_accuracy"], "Parameters": row["parameters"]}
            for row in metadata["candidates"]
        ]), width="stretch", hide_index=True)


def show_test_tab(
    experiment: dict,
    model: dict[str, np.ndarray],
    predictions: pd.DataFrame,
    raw: pd.DataFrame,
) -> None:
    st.subheader("Locked test bench")
    st.caption(
        "One stratified holdout is reserved for the final selected configuration. "
        "Architecture selection uses cross-validation only; the test set is not used to rank candidates."
    )
    report = experiment["test_report"]
    expected_indices = np.asarray(model["retained_original_row_indices"])[model["test_indices"]]
    if not np.array_equal(
        np.sort(predictions["original_row_index"].to_numpy(dtype=int)),
        np.sort(expected_indices.astype(int)),
    ):
        raise ValueError("Saved prediction rows do not match the test indices in the model artifact.")
    ordered = predictions.sort_values("original_row_index").reset_index(drop=True)
    source_rows = raw.set_index("source_row_index")
    actual_from_data = source_rows.loc[ordered["original_row_index"], "class"].to_numpy(dtype=int)
    if not np.array_equal(actual_from_data, ordered["actual_class"].to_numpy(dtype=int)):
        raise ValueError("Saved test labels do not match the source data; check artifact compatibility.")

    probabilities = ordered["probability_class_1"].to_numpy(dtype=float)
    actual = ordered["actual_class"].to_numpy(dtype=int)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Saved test probabilities must be finite values in [0, 1].")
    config = experiment["best_config"]
    st.markdown(
        f"**Model:** Riddhi's {describe_layers([4, *config['hidden'], 1])} ANN "
        f"({' → '.join(config['activations'])}); "
        f"informant coefficient {config.get('c_informant', 0):g}, "
        f"informant count {config.get('informant_count', 0)}. "
        "The slider reclassifies saved probabilities; it does not retrain the model."
    )
    initial_threshold = float(model["threshold"])
    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=min(max(initial_threshold, 0.05), 0.95),
        step=0.01,
        help="Interactive what-if view only. The reported locked-test metrics were produced at threshold 0.50.",
    )
    predicted = (probabilities >= threshold).astype(int)
    if np.array_equal(probabilities >= 0.05, probabilities >= 0.95):
        st.info(
            "The slider cannot change any prediction in its 0.05–0.95 range: "
            "none of the saved probabilities falls where these thresholds would change its class. "
            "Unchanged metrics here are expected, not evidence that the slider is broken."
        )
    tn, fp, fn, tp = confusion_counts(actual, predicted)
    accuracy = float(np.mean(actual == predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    cards = st.columns(5)
    cards[0].metric("Test examples", f"{len(actual):,}")
    cards[1].metric("Accuracy", f"{accuracy:.2%}", f"{accuracy - report['accuracy']:+.2%} vs. 0.50")
    cards[2].metric("Precision · class 1", f"{precision:.2%}")
    cards[3].metric("Recall · class 1", f"{recall:.2%}")
    cards[4].metric("F1 · class 1", f"{f1:.2%}")
    st.caption(
        f"Correct: {int(np.sum(actual == predicted))}/{len(actual)} · "
        f"Errors: {int(np.sum(actual != predicted))} at threshold {threshold:.2f}. "
        "The four percentage metrics describe this one model on the same test examples."
    )
    with st.expander("Why can all four metrics be 100%? Inspect the probabilities"):
        st.markdown(
            "Accuracy, precision, recall and F1 all equal 100% when there are no classification "
            "errors and both classes are present. That does not mean every architecture has "
            "the same score, that the probabilities are exactly 0 or 1, or that future data "
            "will also be classified perfectly. Scientific notation below preserves tiny probabilities."
        )
        summaries = []
        for label in (0, 1):
            values = probabilities[actual == label]
            if len(values):
                summaries.append(
                    {
                        "Actual class": label,
                        "Examples": len(values),
                        "Minimum P(class 1)": f"{values.min():.12g}",
                        "Maximum P(class 1)": f"{values.max():.12g}",
                    }
                )
        st.dataframe(pd.DataFrame(summaries), width="stretch", hide_index=True)
        if np.any(actual == 0) and np.any(actual == 1):
            max_negative = float(probabilities[actual == 0].max())
            min_positive = float(probabilities[actual == 1].min())
            if max_negative < min_positive:
                st.caption(
                    "Every example is classified correctly for any threshold t satisfying "
                    f"{max_negative:.12g} < t ≤ {min_positive:.12g}."
                )

    left, right = st.columns([0.85, 1.15], gap="large")
    with left:
        st.markdown("#### Confusion matrix")
        matrix = np.array([[tn, fp], [fn, tp]])
        fig = go.Figure(
            go.Heatmap(
                z=matrix,
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                text=matrix,
                texttemplate="%{text}",
                colorscale=[[0, "#17263a"], [1, "#54d6a0"]],
                showscale=False,
                hovertemplate="%{y} · %{x}<br>Examples: %{z}<extra></extra>",
            )
        )
        chart_layout(fig, 360)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption(
            f"At the saved 0.50 threshold: ROC AUC {report['roc_auc']:.3f} · "
            f"BCE {report['binary_cross_entropy']:.3g} · "
            f"95% Wilson accuracy interval [{report['accuracy_wilson_95'][0]:.1%}, "
            f"{min(report['accuracy_wilson_95'][1], 1):.1%}]."
        )

    with right:
        st.markdown("#### ROC curve")
        order = np.argsort(-probabilities, kind="stable")
        ordered_actual = actual[order]
        true_positives = np.r_[0, np.cumsum(ordered_actual == 1)]
        false_positives = np.r_[0, np.cumsum(ordered_actual == 0)]
        positives = max(int(np.sum(actual == 1)), 1)
        negatives = max(int(np.sum(actual == 0)), 1)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=false_positives / negatives,
                y=true_positives / positives,
                mode="lines",
                line=dict(color="#54d6a0", width=3),
                name=f"Selected ANN · AUC {report['roc_auc']:.3f}",
                hovertemplate="False-positive rate: %{x:.3f}<br>True-positive rate: %{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#8393a8", dash="dash"),
                name="Chance",
                hoverinfo="skip",
            )
        )
        fig.update_xaxes(title="False-positive rate", range=[0, 1])
        fig.update_yaxes(title="True-positive rate", range=[0, 1])
        chart_layout(fig, 360)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    st.markdown("#### Test-set predictions")
    show = ordered.copy()
    show["threshold_prediction"] = predicted
    st.dataframe(show, width="stretch", hide_index=True)


def show_pso_equations(experiment: dict, maazuddin: dict) -> None:
    config = experiment["best_config"]
    informant_coefficient = config.get("c_informant", 0.0)
    st.markdown("#### How each person's swarm moves")
    st.markdown(
        "Each **particle** is a candidate ANN: its position **x** holds every weight and bias, "
        "and its velocity **v** is the proposed change. **p** is its own best position, "
        "**n** is its informants' best position, and **g** is the whole swarm's best position. "
        "The equations below apply separately to every coordinate **d** of every particle **i**."
    )
    profiles = [
        (f"{group.title()} · {row['configuration_name']}", row)
        for group, results in maazuddin["groups"].items()
        for row in results["summary"]
    ]
    labels = [label for label, _ in profiles]
    chosen = st.selectbox(
        "Maazuddin's saved PSO configuration",
        labels,
        index=labels.index("Coefficient · Balanced"),
        key="pso_configuration",
    )
    profile = profiles[labels.index(chosen)][1]

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Riddhi · personal / informant / global PSO")
        st.latex(
            r"v_{i,d}^{t+1}=w_t v_{i,d}^{t}"
            r"+c_1 r_{1,i,d}^{t}(p_{i,d}^{t}-x_{i,d}^{t})"
            r"+c_{\mathrm{informant}} r_{3,i,d}^{t}(n_{i,d}^{t}-x_{i,d}^{t})"
            r"+c_2 r_{2,i,d}^{t}(g_d^{t}-x_{i,d}^{t})"
        )
        st.latex(r"x_{i,d}^{t+1}=x_{i,d}^{t}+v_{i,d}^{t+1}")
        st.latex(
            r"w_t=w_{\mathrm{start}}+\frac{t}{\max(T-1,1)}"
            r"(w_{\mathrm{end}}-w_{\mathrm{start}}),\quad t=0,\ldots,T-1"
        )
        start, end = config["inertia"]
        st.markdown(
            f"- **Inertia:** decreases linearly from **{start:g}** to **{end:g}** "
            f"over **{config['iterations']}** velocity updates.\n"
            f"- **Personal/cognitive:** c₁ = **{config['c1']:g}**.\n"
            f"- **Global/social:** c₂ = **{config['c2']:g}**.\n"
            f"- **Informant coefficient:** **{informant_coefficient:g}**.\n"
            "- Each active random multiplier is independently sampled from Uniform(0, 1) "
            "per particle, coordinate and update.\n"
            "- These are the saved final settings, selected by CV accuracy, then BCE, then "
            "parameter count—not by test performance."
        )
        if informant_coefficient > 0:
            st.markdown(
                f"- **Informants:** **{config['informant_count']}**, including self. "
                "The others are resampled without replacement per particle and update; "
                "the target is their lowest-loss personal best."
            )
        else:
            st.warning(
                "This saved run has no informant attraction. The notebook source now supports "
                "c_informant and informant_count, but these historical results have not been "
                "recomputed with the new settings. Rerun the notebook and refresh the named "
                "result bundles to compare newly measured results."
            )
        velocity_limit = 2 * config["bound"] * config["velocity_fraction"]
        st.caption(
            f"The implemented velocity is then clipped to ±{velocity_limit:g}; positions "
            f"are clipped to ±{config['bound']:g}, with velocity zeroed at clipped coordinates. "
            "The objective is training binary cross-entropy plus an L2 weight penalty (lower is better)."
        )
    with right:
        st.markdown("##### Maazuddin · informant-based PSO")
        st.latex(
            r"v_{i,d}^{t+1}=\alpha v_{i,d}^{t}"
            r"+b_{i,d}^{t}(p_{i,d}^{t}-x_{i,d}^{t})"
            r"+c_{i,d}^{t}(n_{i,d}^{t}-x_{i,d}^{t})"
            r"+d_{i,d}^{t}(g_d^{t}-x_{i,d}^{t})"
        )
        st.latex(
            r"b\sim U(0,\beta),\quad c\sim U(0,\gamma),\quad d\sim U(0,\delta)"
        )
        st.latex(r"x_{i,d}^{t+1}=x_{i,d}^{t}+\epsilon v_{i,d}^{t+1}")
        informants = profile["informant_count"]
        if informants is None:
            informants = min(3, profile["swarm_size"])
        st.markdown(
            f"- **Inertia:** α = **{profile['alpha']:g}**, fixed throughout the run.\n"
            f"- **Personal/cognitive limit:** β = **{profile['beta']:g}**.\n"
            f"- **Informant/social limit:** γ = **{profile['gamma']:g}**.\n"
            f"- **Whole-swarm/global limit:** δ = **{profile['delta']:g}**.\n"
            f"- **Movement scale:** ε = **{profile['epsilon']:g}**.\n"
            f"- **Informants:** **{informants}**, including the particle itself; the others "
            "are randomly resampled without replacement on each update.\n"
            "- These values are fixed experiment profiles, not adaptively learned coefficients."
        )
        st.caption(
            "b, c and d are independent draws per coordinate and update. The informant target "
            "is the best personal-best position among the selected informants. "
            "Bounds initialise the particles; this implementation does not clamp their later "
            "positions or velocities. It maximises training classification accuracy."
        )

    st.markdown("#### Which knowledge receives more weight?")
    expected = pd.DataFrame(
        [
            {"Person": "Riddhi", "Knowledge": "Personal", "Expected coefficient": config["c1"] / 2},
            {"Person": "Riddhi", "Knowledge": "Informants", "Expected coefficient": informant_coefficient / 2},
            {"Person": "Riddhi", "Knowledge": "Global", "Expected coefficient": config["c2"] / 2},
            {"Person": "Maazuddin", "Knowledge": "Personal", "Expected coefficient": profile["beta"] / 2},
            {"Person": "Maazuddin", "Knowledge": "Informants", "Expected coefficient": profile["gamma"] / 2},
            {"Person": "Maazuddin", "Knowledge": "Global", "Expected coefficient": profile["delta"] / 2},
        ]
    )
    fig = px.bar(
        expected,
        x="Knowledge",
        y="Expected coefficient",
        color="Person",
        barmode="group",
        color_discrete_map={"Riddhi": "#54d6a0", "Maazuddin": "#87a9ff"},
    )
    chart_layout(fig, 340)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption(
        "Expected means average random multiplier: the mean of Uniform(0, a) is a/2. "
        "These bars are coefficient comparisons, not measured particle movements or percentages. "
        "Inertia is separate: it carries existing velocity, not a particular knowledge source."
    )
    st.markdown(
        f"**Riddhi (saved run):** personal coefficient {config['c1']:g} versus combined "
        f"informant/global coefficient {informant_coefficient + config['c2']:g}. "
        + (
            "The configured attraction favours individual experience."
            if config["c1"] > config["c2"] + informant_coefficient
            else "The combined configured attraction favours shared knowledge."
            if config["c2"] + informant_coefficient > config["c1"]
            else "The personal and combined shared coefficient budgets are equal."
        )
    )
    st.markdown(
        "**Maazuddin's saved coefficient profiles:**\n"
        "- **Balanced:** β = γ = δ = 1. Each source has the same expected multiplier, "
        "but informant + global coefficients together are twice the personal coefficient.\n"
        "- **Cognitive emphasis:** β = 1.5, γ = δ = 0.75. Personal is twice either shared "
        "source individually, but equals their combined coefficient budget.\n"
        "- **Social emphasis:** β = 0.75, γ = δ = 1.125. The combined shared-knowledge "
        "coefficient budget is three times the personal one.\n"
        "- The architecture, activation, budget and pilot groups use the balanced coefficients."
    )
    st.info(
        "Larger coefficients do not guarantee larger actual movements: each term also depends "
        "on a fresh random draw and the distance/direction to its target. Terms can reinforce or "
        "cancel one another; the informant and global bests may coincide. "
        "The saved artifacts do not contain per-particle term histories, so this is an equation "
        "and settings comparison, not a claim about observed influence."
    )


def show_pso_tab(experiment: dict, model: dict[str, np.ndarray], maazuddin: dict) -> None:
    st.subheader("Inside the swarm")
    config = experiment["best_config"]
    first, second, third, fourth = st.columns(4)
    first.metric("Particles", str(config["particles"]))
    second.metric("Iterations", str(config["iterations"]))
    third.metric("Cognitive c₁", f"{config['c1']:.2f}")
    fourth.metric("Social c₂", f"{config['c2']:.2f}")
    st.caption(
        f"Selected architecture [{', '.join(map(str, config['hidden']))}] with "
        f"{' → '.join(config['activations'])}; {model['parameters'].size} trainable parameters. "
        f"The final restart was selected from seeds {', '.join(map(str, experiment['final_seeds']))} "
        "using development-set objective only."
    )

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("#### Best objective during optimization")
        history = model["final_training_history"]
        fig = go.Figure(
            go.Scatter(
                x=np.arange(len(history)),
                y=history,
                mode="lines",
                line=dict(color="#54d6a0", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(84, 214, 160, 0.10)",
                name="Best-so-far objective",
                hovertemplate="Iteration %{x}<br>Objective %{y:.6g}<extra></extra>",
            )
        )
        fig.update_xaxes(title="Iteration")
        fig.update_yaxes(title="Training objective", type="log")
        chart_layout(fig, 385)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    with right:
        st.markdown("#### Inertia schedules · both people")
        start, end = config["inertia"]
        iterations = np.arange(config["iterations"])
        progress = iterations / max(config["iterations"] - 1, 1)
        inertia = start + progress * (end - start)
        fig = go.Figure(
            go.Scatter(
                x=iterations,
                y=inertia,
                mode="lines",
                line=dict(color="#87a9ff", width=3),
                name="Riddhi · decreasing inertia",
                hovertemplate="Iteration %{x}<br>w = %{y:.3f}<extra></extra>",
            )
        )
        maazuddin_alpha = maazuddin["groups"]["coefficient"]["summary"][0]["alpha"]
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=np.full(len(iterations), maazuddin_alpha),
                mode="lines",
                line=dict(color="#ffbc70", width=2, dash="dash"),
                name="Maazuddin · fixed inertia",
                hovertemplate="Constant α = %{y:.3f}<extra></extra>",
            )
        )
        fig.update_xaxes(title="Riddhi velocity-update index (0-based)")
        fig.update_yaxes(title="Inertia weight")
        chart_layout(fig, 385)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption(
            "Riddhi's exact update schedule is shown. Maazuddin's dashed line is a constant "
            "reference extended across this axis, not evidence that both runs used the same "
            "number of iterations or evaluation budget. All his saved profiles use α = 0.5."
        )

    show_pso_equations(experiment, maazuddin)

    st.info(
        "Going further: the decreasing inertia schedule transitions from broader exploration "
        "toward finer search. The experiment also compares activation functions and network depth "
        "under repeated cross-validation, then uses a held-out test split once for the selected model."
    )


def main() -> None:
    st.set_page_config(
        page_title="Banknote · PSO Lab",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(ellipse at 12% 0%, #142238 0%, #0b111b 42%, #0b111b 100%); }
        [data-testid="stSidebar"] { background: #101a28; border-right: 1px solid #263449; }
        h1, h2, h3 { letter-spacing: -0.035em; }
        .hero { padding: 1.15rem 0 0.45rem 0; }
        .eyebrow { color: #54d6a0; text-transform: uppercase; letter-spacing: .17em;
                   font-size: .76rem; font-weight: 700; }
        .hero-title { font-size: clamp(2.2rem, 5vw, 3.8rem); line-height: 1.0;
                      font-weight: 760; margin: .45rem 0; }
        .hero-copy { color: #9eafc3; font-size: 1.02rem; }
        [data-testid="stMetric"] { background: #111c2a; border: 1px solid #29384b;
                                   padding: .8rem 1rem; border-radius: 12px; }
        [data-testid="stMetricValue"] { color: #f0f6fc; }
        div[data-baseweb="tab-list"] { gap: .35rem; }
        div[data-baseweb="tab"] { background: #111c2a; border-radius: 9px 9px 0 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Biologically inspired computation · interactive lab</div>
          <div class="hero-title">BANKNOTE <span style="color:#54d6a0">/</span> PSO LAB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        data_path = find_file(
            [
                COURSEWORK_ROOT / "outline/data_banknote_authentication.txt",
                COURSEWORK_ROOT / "data/raw/data_banknote_authentication.txt",
            ],
            "Banknote Authentication data file",
        )
        raw, data, data_digest, repaired_labels = load_dataset(
            str(data_path), data_path.stat().st_mtime_ns
        )
        riddhi_path = RESULTS_DIR / "riddhi_results.json"
        maazuddin_path = find_file(
            [
                RESULTS_DIR / "maazuddin_results.json",
                WORKSPACE_ROOT / "ann_results/maazuddin_results.json",
            ],
            "Maazuddin's saved experiment results",
        )
        riddhi_results = load_experiment(str(riddhi_path), riddhi_path.stat().st_mtime_ns)
        maazuddin_results = load_experiment(str(maazuddin_path), maazuddin_path.stat().st_mtime_ns)
        experiment = riddhi_results["experiment"]
        model_path = RESULTS_DIR / "best_model.npz"
        model = load_model(str(model_path), model_path.stat().st_mtime_ns)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, KeyError) as error:
        st.error(f"Could not load the saved person-specific results: {error}")
        st.code(f'python "{RIDDHI_ROOT / "Scripts" / "save_person_results.py"}"')
        st.stop()

    data_tab, comparison_tab, pso_tab = st.tabs(
        [
            "01  Data explorer",
            "02  Shared-test model comparison",
            "03  PSO deep dive",
        ]
    )
    with data_tab:
        show_data_tab(raw, data, experiment, data_digest, repaired_labels)
    with comparison_tab:
        show_shared_comparison(experiment, raw, data_digest)
    with pso_tab:
        show_pso_tab(experiment, model, maazuddin_results)


if __name__ == "__main__":
    main()
