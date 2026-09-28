"""Streamlit dashboard for the fraud model monitoring system.

Launch from the project folder:
    streamlit run dashboard/app.py

Use a different database file:
    streamlit run dashboard/app.py -- --db path\\to\\monitoring.db

The dashboard only READS the monitoring database (read-only connection).
Run the monitoring commands to add new results; viewing never creates any.
"""

import argparse
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from monitoring import config
from monitoring import dashboard_data as dd
from monitoring.alerts import METRIC_LABELS, load_rules
from monitoring.drift import load_reference_data, summarize_drift
from monitoring.simulate import load_production_data
from monitoring.streaming import SIGNALS

st.set_page_config(page_title="Fraud Model Monitoring", page_icon=":material/monitoring:", layout="wide",
                   initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Settings, colors and formatting
# ---------------------------------------------------------------------------
def get_paths() -> dd.DashboardPaths:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return dd.DashboardPaths.from_env(args.db)


PATHS = get_paths()

# Validated categorical slots (light / dark) and reserved status colors.
SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"]}
INK = {"light": {"muted": "#52514e", "grid": "#e6e5e1", "surface": "#ffffff"},
       "dark": {"muted": "#c3c2b7", "grid": "#33332f", "surface": "#0e1117"}}
STATUS_COLORS = {"no_drift": "#0ca30c", "warning": "#fab219", "drift": "#d03b3b", "insufficient_data": "#9a9993"}

STATUS_LABELS = {
    "no_drift": "🟢 No drift", "warning": "🟠 Warning", "drift": "🔴 Drift",
    "insufficient_data": "⚪ Insufficient data", "stable": "🟢 Stable", "no_data": "⚪ No data",
    "ok": "🟢 OK",
}
SEVERITY_LABELS = {"critical": "🔴 Critical", "warning": "🟠 Warning", "info": "🔵 Info"}
DETECTOR_LABELS = {"adwin": "ADWIN", "page_hinkley": "Page-Hinkley"}
PERFORMANCE_METRICS = ["precision", "recall", "f1", "pr_auc", "false_positive_rate"]


def mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # theme information unavailable (e.g. in tests)
        return "light"


def series_color(i: int) -> str:
    return SERIES[mode()][i]


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def fmt(value, digits: int = 4) -> str:
    return "n/a" if is_missing(value) else f"{value:.{digits}f}"


def fmt_pct(value, digits: int = 2) -> str:
    return "n/a" if is_missing(value) else f"{value:.{digits}%}"


def fmt_int(value) -> str:
    return "n/a" if is_missing(value) else f"{int(value):,}"


def style(fig: go.Figure, height: int = 320, x_title: str | None = None, y_title: str | None = None,
          title: str | None = None) -> go.Figure:
    ink = INK[mode()]
    # The legend sits between the title and the plot, so the top margin
    # leaves room for both and they never overlap on narrow columns.
    has_legend = len(fig.data) > 1 and fig.layout.showlegend is not False
    top = (40 if title else 8) + (32 if has_legend else 0) + 8
    fig.update_layout(
        title=dict(text=title, x=0, y=1, yanchor="top", yref="container", pad=dict(t=8),
                   font=dict(size=15)) if title else None,
        height=height + (32 if has_legend else 0), margin=dict(l=8, r=8, t=top, b=8),
        hovermode="x unified", font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left"),
    )
    fig.update_xaxes(title=x_title, showgrid=False, linecolor=ink["grid"])
    fig.update_yaxes(title=y_title, gridcolor=ink["grid"], zeroline=False)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def reference_line(fig: go.Figure, y: float, text: str, dash: str = "dash") -> None:
    fig.add_hline(y=y, line=dict(color=INK[mode()]["muted"], width=1.5, dash=dash),
                  annotation=dict(text=text, font=dict(size=11, color=INK[mode()]["muted"])),
                  annotation_position="top left")


# ---------------------------------------------------------------------------
# Cached data loading (refreshes automatically when the database file changes)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load(loader: str, db_path: str, version: float):
    return getattr(dd, loader)(db_path)


def load(loader: str):
    return _load(loader, str(PATHS.db), dd.database_version(PATHS.db))


@st.cache_data(show_spinner="Loading reference data…")
def reference_frame(path: str) -> pd.DataFrame:
    return load_reference_data(path)


@st.cache_data(show_spinner="Loading production data…")
def production_frame(path: str) -> pd.DataFrame:
    return load_production_data(path)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def page_overview() -> None:
    st.title("Overview")
    st.caption("Headline numbers from every monitoring step, read from the monitoring database.")
    ov = load("load_overview")
    batches = load("load_batches")

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions monitored", fmt_int(ov["predictions"]), border=True,
              help=f"{ov['labeled']:,} true labels received, {ov['pending_labels']:,} still pending.")
    share = ov["flagged"] / ov["predictions"] if ov["predictions"] else None
    c2.metric("Fraud predictions", fmt_int(ov["flagged"]), border=True,
              help=f"Transactions the model flagged as fraud ({fmt_pct(share)} of all monitored).")
    c3.metric("Monitoring batches", fmt_int(ov["simulator_batches"]), border=True,
              help=(f"Completed production-replay batches. Failed batches: {ov['failed_batches']}. "
                    f"API requests logged separately: {ov['api_predictions']:,} predictions."))

    c1, c2, c3 = st.columns(3)
    c1.metric("Data drift events", fmt_int(ov["drift_events"]), border=True,
              help=(f"Feature checks marked 'drift' across {ov['drift_batches']} batches: "
                    f"KS {ov['ks_drift']:,}, PSI {ov['psi_drift']:,}. PSI warnings: {ov['psi_warning']:,}."))
    c2.metric("Streaming drift events", fmt_int(ov["stream_detections"]), border=True,
              help="Change points reported by ADWIN and Page-Hinkley across all signals.")
    c3.metric("Performance alerts", fmt_int(ov["alerts_total"]), border=True,
              help=(f"Critical {ov['alerts_critical']}, warning {ov['alerts_warning']}, "
                    f"info {ov['alerts_info']}."))

    st.subheader("Monitoring steps")
    steps = [
        ("Predictions logged", ov["predictions"] > 0, "python -m monitoring.simulate run"),
        ("Data drift checked (KS, PSI)", ov["drift_batches"] > 0, "python -m monitoring.drift run"),
        ("Streaming detectors run (ADWIN, Page-Hinkley)", ov["stream_batches"] > 0,
         "python -m monitoring.streaming run"),
        ("Performance evaluated with true labels", ov["performance_windows"] > 0,
         "python -m monitoring.performance run"),
    ]
    st.dataframe(
        pd.DataFrame([{"Step": name, "Status": "✅ Has results" if done else "⬜ No results yet",
                       "Command": cmd} for name, done, cmd in steps]),
        hide_index=True,
    )

    completed = batches[(batches["status"] == "completed") & (batches["source"] == "simulator")]
    if completed.empty:
        st.info("No completed batches yet. Run `python -m monitoring.simulate run`.")
        return

    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Scatter(
            x=completed["batch_id"], y=completed["flag_rate"], mode="lines+markers",
            line=dict(color=series_color(0), width=2), marker=dict(size=8),
            hovertemplate="Batch %{x}<br>Flagged: %{y:.2%}<extra></extra>"))
        fig.update_yaxes(tickformat=".1%")
        show(style(fig, title="Share of transactions flagged as fraud", x_title="Batch"))
    with right:
        fig = go.Figure(go.Scatter(
            x=completed["batch_id"], y=completed["mean_fraud_probability"], mode="lines+markers",
            line=dict(color=series_color(0), width=2), marker=dict(size=8),
            hovertemplate="Batch %{x}<br>Mean score: %{y:.4f}<extra></extra>"))
        show(style(fig, title="Average fraud probability per batch", x_title="Batch"))

    alerts = load("load_alerts")
    st.subheader("Latest alerts")
    if alerts.empty:
        st.caption("No alerts stored.")
    else:
        for _, a in alerts.sort_values("alert_id", ascending=False).head(5).iterrows():
            st.markdown(f"**{SEVERITY_LABELS.get(a['severity'], a['severity'])}** · {a['description']}")


# ---------------------------------------------------------------------------
# Data drift
# ---------------------------------------------------------------------------
def page_data_drift() -> None:
    st.title("Data drift")
    st.caption("Each production batch compared with the reference (training-period) data, feature by feature.")
    results = load("load_drift_results")
    if results.empty:
        st.info("No data drift results yet. Run `python -m monitoring.drift run`.")
        return

    with st.expander("How to read KS and PSI"):
        st.markdown(
            f"""
- **KS statistic** (0–1): the biggest gap between the reference and batch distributions.
  The **p-value** is the chance of a gap this big if nothing changed; below **{config.KS_ALPHA}** → *drift*.
  With thousands of rows, KS flags even tiny differences, so read it together with PSI.
- **PSI**: how much the share of values in each of {config.PSI_N_BINS} reference-decile bins moved.
  Below **{config.PSI_WARNING_THRESHOLD}** no drift · up to **{config.PSI_DRIFT_THRESHOLD}** warning · above → drift.
- **Insufficient data**: fewer than {config.MIN_DRIFT_SAMPLE_SIZE} usable values, so no decision is made.
"""
        )

    summary = summarize_drift(results)
    latest = summary.index.max()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Batches checked", fmt_int(len(summary)), border=True)
    c2.metric(f"Batch {latest}: KS drift", f"{summary.loc[latest, 'ks_drift']} / {summary.loc[latest, 'features']}",
              border=True, help="Features with KS p-value below the significance level.")
    c3.metric(f"Batch {latest}: PSI drift", f"{summary.loc[latest, 'psi_drift']} / {summary.loc[latest, 'features']}",
              border=True, help=f"Features with PSI ≥ {config.PSI_DRIFT_THRESHOLD}.")
    c4.metric(f"Batch {latest}: PSI warning", f"{summary.loc[latest, 'psi_warning']} / {summary.loc[latest, 'features']}",
              border=True, help=f"Features with PSI between {config.PSI_WARNING_THRESHOLD} and {config.PSI_DRIFT_THRESHOLD}.")

    tab_status, tab_trends, tab_batch, tab_dist = st.tabs(
        ["Feature status", "Trends", "Batch details", "Distribution comparison"])

    with tab_status:
        test = st.radio("Test", ["psi", "ks"], horizontal=True,
                        format_func=lambda t: "PSI" if t == "psi" else "KS test", key="drift_test")
        drift_heatmap(results, test)
        st.markdown("**Summary per batch**")
        st.dataframe(summary.reset_index(), hide_index=True, column_config={
            "batch_id": "Batch", "features": "Features", "ks_drift": "KS drift", "psi_warning": "PSI warning",
            "psi_drift": "PSI drift", "max_psi": st.column_config.NumberColumn("Max PSI", format="%.4f"),
            "top_psi_feature": "Top PSI feature"})

    with tab_trends:
        psi = results[results["test_name"] == "psi"]
        ranked = psi.groupby("feature_name")["score"].mean().sort_values(ascending=False).index.tolist()
        features = st.multiselect("Features (up to 4)", config.DRIFT_FEATURES, default=ranked[:3],
                                  max_selections=4, key="drift_features")
        if not features:
            st.info("Pick at least one feature.")
        else:
            left, right = st.columns(2)
            with left:
                fig = feature_trend(results, features, "psi")
                reference_line(fig, config.PSI_DRIFT_THRESHOLD, f"drift ≥ {config.PSI_DRIFT_THRESHOLD}")
                reference_line(fig, config.PSI_WARNING_THRESHOLD, f"warning ≥ {config.PSI_WARNING_THRESHOLD}", "dot")
                show(style(fig, title="PSI per batch", x_title="Batch", y_title="PSI"))
            with right:
                fig = feature_trend(results, features, "ks")
                show(style(fig, title="KS statistic per batch", x_title="Batch", y_title="KS statistic"))
            st.caption("Gaps in a line mean the batch had insufficient data for that feature.")

    with tab_batch:
        batch_ids = sorted(int(b) for b in results["batch_id"].unique())
        batch = st.selectbox("Batch", batch_ids, index=len(batch_ids) - 1, key="drift_batch")
        st.dataframe(batch_detail_table(results, batch), hide_index=True, column_config={
            "KS statistic": st.column_config.NumberColumn(format="%.4f"),
            "KS p-value": st.column_config.NumberColumn(format="%.3g"),
            "PSI": st.column_config.NumberColumn(format="%.4f")})

    with tab_dist:
        distribution_comparison(results)


def drift_heatmap(results: pd.DataFrame, test: str) -> None:
    rows = results[results["test_name"] == test]
    order = {s: i for i, s in enumerate(["insufficient_data", "no_drift", "warning", "drift"])}
    status = rows.pivot(index="feature_name", columns="batch_id", values="status")
    status = status.reindex([f for f in config.DRIFT_FEATURES if f in status.index])
    scores = rows.pivot(index="feature_name", columns="batch_id", values="score").reindex(status.index)

    z = status.apply(lambda col: col.map(order)).to_numpy(dtype=float)
    hover = [[f"{feat} · batch {b}<br>{STATUS_LABELS.get(status.loc[feat, b], '')}<br>"
              f"{'PSI' if test == 'psi' else 'KS statistic'}: {fmt(scores.loc[feat, b])}"
              for b in status.columns] for feat in status.index]
    # Status codes 0-3 with zmin=-0.5 / zmax=3.5 sit at 0.125, 0.375, 0.625, 0.875
    # on the color scale, so each status owns one quarter of it.
    c = STATUS_COLORS
    colorscale = [[0.0, c["insufficient_data"]], [0.25, c["insufficient_data"]],
                  [0.25, c["no_drift"]], [0.5, c["no_drift"]],
                  [0.5, c["warning"]], [0.75, c["warning"]],
                  [0.75, c["drift"]], [1.0, c["drift"]]]
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(b) for b in status.columns], y=status.index.tolist(), zmin=-0.5, zmax=3.5,
        colorscale=colorscale, showscale=False, xgap=2, ygap=2,
        text=hover, hovertemplate="%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    show(style(fig, height=max(320, 22 * len(status.index) + 60), x_title="Batch"))
    st.caption("🟢 No drift · 🟠 Warning (PSI only) · 🔴 Drift · ⚪ Insufficient data — hover a cell for its score.")


def feature_trend(results: pd.DataFrame, features: list[str], test: str) -> go.Figure:
    fig = go.Figure()
    rows = results[results["test_name"] == test]
    for i, feature in enumerate(features):
        data = rows[rows["feature_name"] == feature].sort_values("batch_id")
        fig.add_trace(go.Scatter(
            x=data["batch_id"], y=data["score"], name=feature, mode="lines+markers",
            line=dict(color=series_color(i), width=2), marker=dict(size=8),
            customdata=data["status"].map(STATUS_LABELS),
            hovertemplate=f"{feature}: %{{y:.4f}} (%{{customdata}})<extra></extra>"))
    return fig


def batch_detail_table(results: pd.DataFrame, batch_id: int) -> pd.DataFrame:
    rows = results[results["batch_id"] == batch_id]
    ks = rows[rows["test_name"] == "ks"].set_index("feature_name")
    psi = rows[rows["test_name"] == "psi"].set_index("feature_name")
    table = pd.DataFrame({
        "Feature": ks.index,
        "KS statistic": ks["score"].to_numpy(),
        "KS p-value": ks["p_value"].to_numpy(),
        "KS status": ks["status"].map(STATUS_LABELS).to_numpy(),
        "PSI": psi["score"].reindex(ks.index).to_numpy(),
        "PSI status": psi["status"].reindex(ks.index).map(STATUS_LABELS).to_numpy(),
        "Values": ks["n_current"].to_numpy(),
        "Missing": ks["n_missing"].to_numpy(),
        "Note": ks["detail"].fillna("").to_numpy(),
    })
    return table.sort_values("PSI", ascending=False, na_position="last")


def distribution_comparison(results: pd.DataFrame) -> None:
    st.caption("Reference data vs. one batch for a single feature. Scores shown are the stored results.")
    missing = [p for p in (PATHS.reference_data, PATHS.production_data) if not p.is_file()]
    if missing:
        st.info("Distribution charts need the reference and production files: "
                + ", ".join(f"`{p}`" for p in missing) + " not found. Run `python -m monitoring.train`.")
        return

    left, right = st.columns(2)
    feature = left.selectbox("Feature", config.DRIFT_FEATURES, key="dist_feature")
    batch_ids = sorted(int(b) for b in results["batch_id"].unique())
    batch = right.selectbox("Batch", batch_ids, index=len(batch_ids) - 1, key="dist_batch")

    try:
        reference = reference_frame(str(PATHS.reference_data))[feature].dropna().to_numpy(dtype=float)
        current = dd.batch_feature_values(PATHS.db, production_frame(str(PATHS.production_data)), batch, feature)
    except (FileNotFoundError, ValueError) as exc:
        st.warning(f"Could not load feature values: {exc}")
        return
    if current.size == 0:
        st.info("This batch has no values for the selected feature.")
        return

    low, high = np.percentile(np.concatenate([reference, current]), [0.5, 99.5])
    bins = dict(start=low, end=high, size=(high - low) / 50 if high > low else 1)
    fig = go.Figure()
    for i, (name, values) in enumerate([("Reference", reference), (f"Batch {batch}", current)]):
        fig.add_trace(go.Histogram(x=values, name=name, xbins=bins, histnorm="probability", opacity=0.6,
                                   marker=dict(color=series_color(i), line=dict(width=0)),
                                   hovertemplate=f"{name}: %{{y:.2%}}<extra></extra>"))
    fig.update_layout(barmode="overlay", bargap=0.04)
    fig.update_yaxes(tickformat=".0%")
    show(style(fig, height=360, x_title=feature, y_title="Share of values",
               title=f"{feature}: reference vs. batch {batch} (middle 99% of values)"))

    stored = results[(results["batch_id"] == batch) & (results["feature_name"] == feature)].set_index("test_name")
    if not stored.empty:
        ks, psi = stored.loc["ks"], stored.loc["psi"]
        st.markdown(f"KS statistic **{fmt(ks['score'])}** (p = {fmt(ks['p_value'], 3)}) "
                    f"→ {STATUS_LABELS[ks['status']]} · PSI **{fmt(psi['score'])}** → {STATUS_LABELS[psi['status']]}")


# ---------------------------------------------------------------------------
# Streaming monitoring
# ---------------------------------------------------------------------------
def page_streaming() -> None:
    st.title("Streaming monitoring")
    st.caption("ADWIN and Page-Hinkley watch per-transaction signals in time order and report change points.")
    results = load("load_stream_results")
    labels = load("load_label_status")
    if results.empty:
        st.info("No streaming results yet. Run `python -m monitoring.streaming run`.")
        return

    with st.expander("What are these signals and detectors?"):
        st.markdown("\n".join(f"- **{name}** — {spec['description']}" for name, spec in SIGNALS.items()))
        st.markdown(
            "- **ADWIN** keeps a window of recent values and reports a change when an older part of the window "
            "has a clearly different average from the newer part.\n"
            "- **Page-Hinkley** adds up how far each value lands from the running average and reports a change "
            "once that running total passes a threshold.\n\n"
            "Changes in *fraud_probability* and *flag_rate* mean the model's **outputs** changed. Only "
            "*error_rate* (which needs true labels) says whether the model makes **more mistakes**.")

    events = results[results["status"] == "drift"].copy()
    st.subheader(f"Detected changes ({int(results['n_detections'].sum())})")
    if events.empty:
        st.caption("No detector has reported a change.")
    else:
        events["Detector"] = events["detector"].map(DETECTOR_LABELS)
        st.dataframe(
            events[["batch_id", "signal", "Detector", "n_detections", "first_detection_transaction_id",
                    "first_detection_event_time", "signal_mean"]].sort_values(["batch_id", "signal"]),
            hide_index=True, column_config={
                "batch_id": "Batch", "signal": "Signal", "n_detections": "Detections",
                "first_detection_transaction_id": "First detection at",
                "first_detection_event_time": st.column_config.NumberColumn("Event time (s)", format="%.0f"),
                "signal_mean": st.column_config.NumberColumn("Batch mean", format="%.4f")})

    left, right = st.columns(2)
    with left:
        show(signal_chart(results, "fraud_probability", "Average fraud probability per batch", ".3f"))
    with right:
        show(signal_chart(results, "flag_rate", "Flag rate per batch", ".1%"))

    st.subheader("Error rate (needs true labels)")
    error_rows = results[results["signal"] == "error_rate"]
    ready, waiting = labels["ready"], labels["waiting"]
    if error_rows.empty:
        total = len(ready) + len(waiting)
        st.info(
            "The error rate cannot be monitored yet: no batch has received **all** of its true labels "
            f"(fully labeled batches: {len(ready)} of {total}). Labels arrive later than predictions; "
            "run `python -m monitoring.simulate labels` and then `python -m monitoring.streaming run`.")
    else:
        show(signal_chart(error_rows, "error_rate", "Error rate per batch (labeled batches only)", ".2%"))
        covered = sorted(error_rows["batch_id"].unique())
        if waiting:
            st.caption(f"Covers batches {covered[0]}–{covered[-1]}. Batches {waiting[0]}–{waiting[-1]} are waiting "
                       "for true labels, so their errors are not counted yet.")
        else:
            st.caption(f"Covers batches {covered[0]}–{covered[-1]}; every logged batch is fully labeled.")
        stored_until = max(covered)
        newly_ready = [b for b in ready if b > stored_until]
        if newly_ready:
            st.caption(f"Labels have since arrived for batches {newly_ready[0]}–{newly_ready[-1]}; "
                       "run `python -m monitoring.streaming run` to include them.")

    label_progress(labels)


def signal_chart(results: pd.DataFrame, signal: str, title: str, tick_format: str) -> go.Figure:
    rows = results[results["signal"] == signal].sort_values("batch_id")
    means = rows.drop_duplicates("batch_id")[["batch_id", "signal_mean"]]
    ink = INK[mode()]
    fig = go.Figure(go.Scatter(
        x=means["batch_id"], y=means["signal_mean"], name="Batch mean", mode="lines",
        line=dict(color=ink["muted"], width=2),
        hovertemplate=f"Batch %{{x}}: %{{y:{tick_format}}}<extra></extra>"))
    for i, (detector, symbol) in enumerate([("adwin", "triangle-up"), ("page_hinkley", "diamond")]):
        hits = rows[(rows["detector"] == detector) & (rows["status"] == "drift")]
        fig.add_trace(go.Scatter(
            x=hits["batch_id"], y=hits["signal_mean"], name=f"{DETECTOR_LABELS[detector]} change",
            mode="markers", marker=dict(symbol=symbol, size=13, color=series_color(i),
                                        line=dict(width=2, color=ink["surface"])),
            hovertemplate=f"{DETECTOR_LABELS[detector]}: change in batch %{{x}}<extra></extra>"))
    fig.update_yaxes(tickformat=tick_format)
    return style(fig, title=title, x_title="Batch")


def label_progress(labels: dict) -> None:
    per_batch = labels["per_batch"]
    if per_batch.empty:
        return
    with st.expander("Label arrival per batch"):
        share = per_batch["n_labeled"] / per_batch["n_predictions"]
        fig = go.Figure(go.Bar(
            x=per_batch["batch_id"], y=share, marker=dict(color=series_color(0), cornerradius=4),
            hovertemplate="Batch %{x}: %{y:.0%} labeled<extra></extra>"))
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        show(style(fig, height=260, title="Share of predictions with a true label", x_title="Batch"))


# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
def page_performance() -> None:
    st.title("Model performance")
    st.caption("Measured only with true labels that have arrived, over windows of consecutive labeled batches.")
    windows = load("load_performance_windows")
    labels = load("load_label_status")
    baseline = dd.load_training_baseline(PATHS.baseline)

    with st.expander("What do these metrics mean?"):
        st.markdown(
            """
- **Precision** — of the transactions flagged as fraud, the share that really were fraud.
- **Recall** — of the real fraud, the share the model caught.
- **F1-score** — one number that is high only when precision *and* recall are both good.
- **PR-AUC** — how well the fraud scores rank fraud above normal transactions, at every threshold (1.0 = perfect).
- **False positive rate** — of the normal transactions, the share wrongly flagged.
"""
            f"\nA window is evaluated once it has at least **{config.PERFORMANCE_MIN_POSITIVES} fraud** and "
            f"**{config.PERFORMANCE_MIN_NEGATIVES:,} non-fraud** labeled cases; with fewer, the numbers are too noisy.")

    if windows.empty:
        st.info("No performance windows yet. Run `python -m monitoring.performance run` once true labels "
                "have arrived (`python -m monitoring.simulate labels`).")
        open_window_status(labels)
        return

    latest = windows.iloc[-1]
    st.subheader(f"Latest window: batches {latest['window_label']}")
    cols = st.columns(5)
    for col, metric in zip(cols, PERFORMANCE_METRICS):
        value = latest[metric]
        base = (baseline or {}).get(metric)
        delta = None if is_missing(value) or base is None else f"{value - base:+.4f} vs baseline"
        col.metric(METRIC_LABELS[metric], fmt(value), delta=delta, border=True,
                   delta_color="inverse" if metric == "false_positive_rate" else "normal",
                   help=f"Training baseline: {fmt(base)}" if base is not None else "No baseline available.")
    if latest["status"] != "ok" or latest["notes"]:
        st.warning("Some metrics in this window cannot be calculated reliably:\n\n"
                   + "\n".join(f"- {note}" for note in latest["notes"]))
    if baseline is None:
        st.caption(f"No training baseline found at `{PATHS.baseline}`.")
    else:
        st.caption("Baseline = validation-set results saved at training time (data the model never trained on).")

    st.subheader("Trends across windows")
    rules = load_rules()
    grid = st.columns(3) + st.columns(3)
    for col, metric in zip(grid, PERFORMANCE_METRICS):
        with col:
            show(metric_trend(windows, metric, (baseline or {}).get(metric), rules))
    with grid[5]:
        st.markdown("**Reading the charts**")
        st.caption("Dashed line: training baseline. Dotted line: alert limit from the configured rules. "
                   "Hollow markers: window marked as insufficient data.")

    st.subheader("Windows")
    table = windows[["window_label", "n_labeled", "n_positive", "true_positives", "false_positives",
                     "true_negatives", "false_negatives", *PERFORMANCE_METRICS, "status"]].copy()
    table["status"] = table["status"].map(STATUS_LABELS)
    table["notes"] = windows["notes"].map(lambda notes: "; ".join(notes))
    st.dataframe(table, hide_index=True, column_config={
        "window_label": "Batches", "n_labeled": "Labeled", "n_positive": "Fraud",
        "true_positives": "TP", "false_positives": "FP", "true_negatives": "TN", "false_negatives": "FN",
        **{m: st.column_config.NumberColumn(METRIC_LABELS[m], format="%.4f") for m in PERFORMANCE_METRICS},
        "status": "Status", "notes": "Notes"})

    st.subheader("Confusion matrix")
    labels_by_id = dict(zip(windows["window_id"], windows["window_label"]))
    window_id = st.selectbox("Window (batches)", list(labels_by_id), index=len(labels_by_id) - 1,
                             format_func=lambda w: labels_by_id[w], key="cm_window")
    confusion_matrix(windows.set_index("window_id").loc[window_id])

    open_window_status(labels)


def metric_trend(windows: pd.DataFrame, metric: str, baseline: float | None, rules) -> go.Figure:
    ink = INK[mode()]
    ok = windows["status"] == "ok"
    fig = go.Figure(go.Scatter(
        x=windows["window_label"], y=windows[metric], mode="lines+markers", name=METRIC_LABELS[metric],
        line=dict(color=series_color(0), width=2),
        marker=dict(size=9, color=[series_color(0) if o else ink["surface"] for o in ok],
                    line=dict(width=2, color=series_color(0))),
        hovertemplate=f"Batches %{{x}}<br>{METRIC_LABELS[metric]}: %{{y:.4f}}<extra></extra>"))
    if baseline is not None:
        reference_line(fig, baseline, f"baseline {baseline:.3f}")
    for rule in rules:
        if rule.metric != metric:
            continue
        if rule.condition in ("below", "above"):
            reference_line(fig, rule.threshold, f"alert {'<' if rule.condition == 'below' else '>'} {rule.threshold}", "dot")
        elif baseline is not None:
            limit = baseline - rule.threshold
            reference_line(fig, limit, f"alert < {limit:.3f}", "dot")
    fig.update_layout(showlegend=False)
    return style(fig, height=260, title=METRIC_LABELS[metric], x_title="Window (batches)")


def confusion_matrix(window: pd.Series) -> None:
    if window["n_labeled"] == 0:
        st.info("This window has no labeled transactions.")
        return
    counts = np.array([[window["true_negatives"], window["false_positives"]],
                       [window["false_negatives"], window["true_positives"]]], dtype=float)
    row_share = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    text = [[f"{int(counts[r, c]):,}<br>{row_share[r, c]:.1%} of row" for c in range(2)] for r in range(2)]
    fig = go.Figure(go.Heatmap(
        z=row_share, x=["Predicted: normal", "Predicted: fraud"], y=["Actual: normal", "Actual: fraud"],
        text=text, texttemplate="%{text}", textfont=dict(size=15), zmin=0, zmax=1, showscale=False,
        colorscale=[[0, "#f0f5fc"], [1, "#1c5cab"]], xgap=2, ygap=2, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    left, right = st.columns([3, 2])
    with left:
        show(style(fig, height=300))
    with right:
        st.markdown(
            f"- **{int(window['true_positives']):,}** fraud caught (TP)\n"
            f"- **{int(window['false_negatives']):,}** fraud missed (FN)\n"
            f"- **{int(window['false_positives']):,}** normal transactions flagged (FP)\n"
            f"- **{int(window['true_negatives']):,}** normal transactions passed (TN)")
        if window["n_positive"] < config.PERFORMANCE_MIN_POSITIVES:
            st.warning(f"Only {int(window['n_positive'])} fraud cases: rates in the bottom row are unreliable.")


def open_window_status(labels: dict) -> None:
    if labels["waiting"]:
        st.caption(f"Batches {labels['waiting'][0]}–{labels['waiting'][-1]} are still waiting for true labels "
                   "and are not included.")
    if labels["open_batches"]:
        b = labels["open_batches"]
        st.info(
            f"Batches {b[0]}–{b[-1]} are fully labeled but not evaluated yet: a window needs at least "
            f"{config.PERFORMANCE_MIN_POSITIVES} fraud and {config.PERFORMANCE_MIN_NEGATIVES:,} non-fraud cases, "
            f"and these batches have {labels['open_positives']} fraud and {labels['open_negatives']:,} "
            "non-fraud cases so far.")


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
def page_alerts() -> None:
    st.title("Alerts")
    st.caption("Alerts stored by `python -m monitoring.performance run`. Viewing this page never creates alerts.")
    alerts = load("load_alerts")
    if alerts.empty:
        st.success("No alerts stored.")
        return

    c1, c2, c3 = st.columns(3)
    for col, severity in zip((c1, c2, c3), ("critical", "warning", "info")):
        col.metric(SEVERITY_LABELS[severity], fmt_int((alerts["severity"] == severity).sum()), border=True)

    left, right = st.columns(2)
    severities = left.multiselect("Severity", list(SEVERITY_LABELS), default=list(SEVERITY_LABELS),
                                  format_func=SEVERITY_LABELS.get, key="alert_severity")
    metrics = sorted(alerts["metric_name"].unique())
    chosen = right.multiselect("Metric", metrics, default=metrics, format_func=lambda m: METRIC_LABELS.get(m, m),
                               key="alert_metric")
    shown = alerts[alerts["severity"].isin(severities) & alerts["metric_name"].isin(chosen)].copy()
    if shown.empty:
        st.info("No alerts match the filters.")
        return

    shown["Severity"] = shown["severity"].map(SEVERITY_LABELS)
    shown["Metric"] = shown["metric_name"].map(lambda m: METRIC_LABELS.get(m, m))
    shown["Batches"] = [f"{s}–{e}" for s, e in zip(shown["start_batch_id"], shown["end_batch_id"])]
    shown["Created"] = pd.to_datetime(shown["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    st.dataframe(
        shown.sort_values("alert_id", ascending=False)[
            ["Created", "Severity", "rule_name", "Metric", "observed_value", "threshold", "baseline_value",
             "Batches", "description"]],
        hide_index=True, column_config={
            "rule_name": "Rule",
            "observed_value": st.column_config.NumberColumn("Observed", format="%.4f"),
            "threshold": st.column_config.NumberColumn("Threshold", format="%.4f"),
            "baseline_value": st.column_config.NumberColumn("Baseline", format="%.4f"),
            "description": st.column_config.TextColumn("Description", width="large")})
    st.caption("For baseline rules, *Threshold* is the allowed drop below the baseline.")

    st.markdown("**Full descriptions**")
    for _, alert in shown.sort_values("alert_id", ascending=False).iterrows():
        st.markdown(f"- **#{alert['alert_id']}** {alert['Severity']} · {alert['description']}")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
PAGES = {
    "Overview": page_overview,
    "Data drift": page_data_drift,
    "Streaming monitoring": page_streaming,
    "Model performance": page_performance,
    "Alerts": page_alerts,
}


def sidebar() -> str:
    with st.sidebar:
        st.markdown("### :material/monitoring: Fraud model monitoring")
        page = st.radio("Page", list(PAGES), key="page", label_visibility="collapsed")
        st.divider()
        st.caption(f"Database: `{PATHS.db.name}`", help=f"Full path: {PATHS.db}")
        if PATHS.db.is_file():
            updated = datetime.fromtimestamp(dd.database_version(PATHS.db))
            st.caption(f"Last updated: {updated:%Y-%m-%d %H:%M:%S}")
        if st.button("Refresh data", icon=":material/refresh:", width="stretch"):
            st.cache_data.clear()
            st.rerun()
    return page


def missing_database() -> None:
    st.title("Fraud model monitoring")
    st.warning(f"No monitoring database found at `{PATHS.db}`.")
    st.markdown("Create it by running the monitoring steps from the project folder (PowerShell):")
    st.code(
        ".\\.venv\\Scripts\\Activate.ps1\n"
        "python -m monitoring.simulate run\n"
        "python -m monitoring.drift run\n"
        "python -m monitoring.streaming run\n"
        "python -m monitoring.simulate labels\n"
        "python -m monitoring.performance run",
        language="powershell")


def main() -> None:
    page = sidebar()
    if not PATHS.db.is_file():
        missing_database()
        return
    try:
        PAGES[page]()
    except sqlite3.Error as exc:
        st.error(f"Could not read the monitoring database: {exc}")
    except KeyError as exc:
        # Streamlit re-reads this file on every rerun but keeps the imported
        # `monitoring` modules from when it started, so after a code update the
        # two can disagree until the dashboard is restarted.
        st.error(
            f"The page expected a value named {exc} that the loaded monitoring code does not provide. "
            "This usually means the project code was updated while the dashboard was running. "
            "Restart the dashboard (Ctrl+C in its window, then "
            "`streamlit run dashboard\\app.py --server.address localhost`). "
            "If it still appears after a restart, it is a bug."
        )


main()
