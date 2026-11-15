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


ROOT = Path(__file__).resolve().parent
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
    if data_digest != experiment.get("data_sha256"):
        st.warning("The current data file does not match the SHA-256 recorded in the experiment.")
    if repaired_labels:
        st.info(f"Repaired {repaired_labels} stray section-sign label(s) in memory; the source file is unchanged.")

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

    with st.expander("Explore pairwise feature relationships"):
        st.caption("Hover points to inspect feature values and class. Use the Plotly toolbar to zoom or select.")
        fig = px.scatter_matrix(
            data,
            dimensions=FEATURES,
            color="class",
            color_discrete_map=COLORS,
            opacity=0.64,
            labels={"class": "Class"},
            hover_data=["source_row_index"],
        )
        fig.update_traces(diagonal_visible=False, marker=dict(size=4))
        fig.update_layout(
            height=820,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dce8f5"),
        )
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
                "CV accuracy": result["mean_accuracy"],
                "CV std": result["std_accuracy"],
                "CV BCE": result["mean_bce"],
                "Seconds": result["seconds"],
                "Scores": result["scores"],
            }
        )
    return pd.DataFrame(rows)


def show_models_tab(experiment: dict) -> None:
    st.subheader("Architecture comparison")
    st.caption(
        "Results are from the notebook's repeated stratified 3-fold cross-validation "
        "(two seeds per fold). Error bars show standard deviation across the six fold/seed runs."
    )
    models = make_model_frame(experiment["architecture_results"])
    best_label = architecture_label(experiment["best_config"])

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
            size="CV std",
            size_max=18,
            hover_name="Model",
            hover_data={"CV BCE": ":.5g", "CV std": ":.4g", "Parameters": True},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_yaxes(tickformat=".2%", title="Mean CV accuracy")
        fig.update_xaxes(title="Trainable parameters")
        chart_layout(fig, 430)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

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
    st.caption(
        f"Riddhi protocol: {riddhi['protocol']['architecture_metric']} "
        f"({riddhi['protocol']['folds']} folds × {riddhi['protocol']['repeats']} seeds). "
        f"Maazuddin protocol: {maazuddin['protocol']['split']} with "
        f"{maazuddin['protocol']['runs']} independent runs."
    )


def confusion_counts(actual: np.ndarray, predicted: np.ndarray) -> tuple[int, int, int, int]:
    true_negative = int(np.sum((actual == 0) & (predicted == 0)))
    false_positive = int(np.sum((actual == 0) & (predicted == 1)))
    false_negative = int(np.sum((actual == 1) & (predicted == 0)))
    true_positive = int(np.sum((actual == 1) & (predicted == 1)))
    return true_negative, false_positive, false_negative, true_positive


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


def show_pso_tab(experiment: dict, model: dict[str, np.ndarray]) -> None:
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
        st.markdown("#### Linear inertia schedule")
        start, end = config["inertia"]
        iterations = np.arange(config["iterations"] + 1)
        inertia = np.linspace(start, end, len(iterations))
        fig = go.Figure(
            go.Scatter(
                x=iterations,
                y=inertia,
                mode="lines",
                line=dict(color="#87a9ff", width=3),
                name="Inertia weight",
                hovertemplate="Iteration %{x}<br>w = %{y:.3f}<extra></extra>",
            )
        )
        fig.update_xaxes(title="Iteration")
        fig.update_yaxes(title="Inertia weight")
        chart_layout(fig, 385)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

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
          <div class="hero-copy">Explore the data. Compare neural architectures. Inspect the swarm.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Experiment")
        st.markdown("**Optimizer**  Particle Swarm Optimization")
        st.markdown("**Classifier**  NumPy feedforward ANN")
        st.markdown("**Validation**  Repeated stratified 3-fold CV")
        st.markdown("**Test policy**  One locked holdout")
        st.divider()
        st.caption("Features: variance · skewness · curtosis · entropy")
        st.caption("Class labels are kept as numeric 0 / 1.")

    data_path = find_file(
        [
            ROOT / "Biologically-Inspired-Computation-Coursework/outline/data_banknote_authentication.txt",
            ROOT / "outline/data_banknote_authentication.txt",
        ],
        "Banknote Authentication data file",
    )
    results_dir = ROOT / "ann_results"
    try:
        raw, data, data_digest, repaired_labels = load_dataset(
            str(data_path), data_path.stat().st_mtime_ns
        )
        riddhi_path = results_dir / "riddhi_results.json"
        maazuddin_path = results_dir / "maazuddin_results.json"
        riddhi_results = load_experiment(str(riddhi_path), riddhi_path.stat().st_mtime_ns)
        maazuddin_results = load_experiment(str(maazuddin_path), maazuddin_path.stat().st_mtime_ns)
        experiment = riddhi_results["experiment"]
        notebook_artifacts = (ROOT / riddhi_results["source_artifact"]).parent
        predictions_path = notebook_artifacts / "test_predictions.csv"
        model_path = notebook_artifacts / "best_model.npz"
        predictions = load_predictions(str(predictions_path), predictions_path.stat().st_mtime_ns)
        model = load_model(str(model_path), model_path.stat().st_mtime_ns)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, KeyError) as error:
        st.error(f"Could not load the saved person-specific results: {error}")
        st.code(".venv/bin/python save_person_results.py")
        st.stop()

    st.markdown(
        f"<span class='hero-copy'>Data fingerprint <code>{data_digest[:16]}…</code>"
        f" &nbsp;·&nbsp; {len(data):,} unique rows &nbsp;·&nbsp; "
        f"{len(experiment['architecture_results'])} candidate architectures</span>",
        unsafe_allow_html=True,
    )
    data_tab, model_tab, people_tab, test_tab, pso_tab = st.tabs(
        [
            "01  Data explorer",
            "02  Notebook architectures",
            "03  People comparison",
            "04  Test bench",
            "05  PSO deep dive",
        ]
    )
    with data_tab:
        show_data_tab(raw, data, experiment, data_digest, repaired_labels)
    with model_tab:
        show_models_tab(experiment)
    with people_tab:
        show_people_tab(riddhi_results, maazuddin_results)
    with test_tab:
        show_test_tab(experiment, model, predictions, raw)
    with pso_tab:
        show_pso_tab(experiment, model)


if __name__ == "__main__":
    main()
