import glob
import os
from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterator, List, Optional

import mlflow
import numpy as np
import streamlit as st
from mlflow.entities import RunInfo
from mlflow.tracking import MlflowClient
from scipy.stats import ttest_ind

st.set_page_config(layout="wide")
st.title("A/B Testing with MLflow")


@dataclass
class MLflowData:
    run_id: str
    metrics: Dict[str, float]
    info: RunInfo
    tags: Dict[str, str]

    @property
    def artifact_path(self):
        return f"/tmp/dl/{self.run_id}"

    @property
    def images(self) -> List[str]:
        _images = glob.glob(f"{self.artifact_path}/*.png")
        return _images

    def download_artifacts(self):
        client = MlflowClient()
        local_path = self.artifact_path
        os.system(f"mkdir -p {local_path}")
        client.download_artifacts(run_id=self.run_id, path=".", dst_path=local_path)


def print_graphs(run_list: List[MLflowData]):
    for d in run_list:
        for im in d.images:
            st.image(im)


def get_data(D: Dict[str, bool]) -> Iterator[MLflowData]:
    client = MlflowClient()
    for run_id in D:
        if D[run_id]:
            # st.text(run_id)
            run = client.get_run(run_id)
            # st.write(run)
            _metrics = run.data.metrics
            # for metric_name, metric_val in M[run_id].items():
            #     st.text(metric_name)

            mlflow_data = MLflowData(
                run_id=run_id,
                metrics=run.data.metrics,
                info=run.info,
                tags=run.data.tags,
            )
            mlflow_data.download_artifacts()
            yield mlflow_data


query = "tags.compare LIKE '%'"
runs = mlflow.search_runs(experiment_names=["test-experiment"], filter_string=query)
with st.expander("Run data"):
    st.write(runs)
unique_tags = sorted(runs["tags.compare"].unique().tolist())
assert (
    len(unique_tags) > 1
), "must have at least two unique `compare` tags for comparison."

with st.sidebar:
    # c1, c2 = st.columns(2)
    # with c1:
    left_tag = st.selectbox("A", unique_tags, index=0)
    # with c2:
    right_tag = st.selectbox("B", unique_tags, index=1)

    metric_columns = runs.columns[
        list(map(lambda x: x.startswith("metrics"), list(runs.columns)))
    ]

    metric = st.selectbox(
        "Metrics", [m.replace("metrics.", "") for m in metric_columns]
    )
    metric = f"metrics.{metric}"
    alternative = st.selectbox("Alternative", ["two-sided", "less", "greater"])

    show_imgs = st.checkbox("Show Images", value=False)
    num_rounding = st.slider(
        "Rounding Decimal Places", min_value=2, max_value=12, value=4
    )
left_runs = runs[runs["tags.compare"] == left_tag]
right_runs = runs[runs["tags.compare"] == right_tag]

run_id_list_left = left_runs["run_id"].tolist()
run_id_list_right = right_runs["run_id"].tolist()


# run = client.get_run(run_id)
# metrics = run.data.metrics

# st.markdown(run.data.metrics)


def letter_filter(mlflow_run: MLflowData, letter="a") -> bool:
    compare = mlflow_run.tags.get("compare", "")
    if compare == letter:
        return True
    return False


with st.expander("Select Runs"):
    lcol, rcol = st.columns(2)
    D = {}
    with lcol:
        for run_id in run_id_list_left:
            D[run_id] = st.checkbox(f"{run_id}", value=True)

    with rcol:
        for run_id in run_id_list_right:
            D[run_id] = st.checkbox(f"{run_id}", value=True)

# st.write(D)
left_runs_compare = left_runs[
    left_runs["run_id"].isin(list(filter(lambda x: D[x] is True, D.keys())))
]
right_runs_compare = right_runs[
    right_runs["run_id"].isin(list(filter(lambda x: D[x] is True, D.keys())))
]

data = list(get_data(D))
# st.write(data)
left_runs = list(filter(lambda x: letter_filter(x, letter=left_tag), data))
right_runs = list(filter(lambda x: letter_filter(x, letter=right_tag), data))

# st.write(left_runs_compare)
# st.write(right_runs_compare)
# st.write(data)
# available_metrics = [d.metrics for d in data]
# import pandas as pd
# df = pd.DataFrame(available_metrics)
# st.write(df)


# st.write(runs[metric_columns])

test_samples = lambda a, b: ttest_ind(a, b, equal_var=False, alternative=alternative)

l_vals = left_runs_compare[metric].values
r_vals = right_runs_compare[metric].values

_metric = metric.replace("metrics.", "")


stat, pval = test_samples(l_vals, r_vals)

display_stat_cols = st.columns([2, 1])
display_metric_cols = st.columns([1, 1, 1])
with display_stat_cols[1]:
    st.metric(
        f"T-Statistic for {_metric}",
        np.round(stat, num_rounding),
        help="Test Statistic for Null Hypothesis of equal averages.",
    )
with display_stat_cols[0]:
    st.metric("p-value", pval, help="p-value associated with hypothesis test.")


with display_metric_cols[0]:
    r_avg = r_vals.mean()
    l_avg = l_vals.mean()
    avg, delta_avg = r_avg, (r_avg - l_avg)
    st.metric(
        f"Average of {_metric}(B)",
        np.round(avg, num_rounding),
        delta=delta_avg,
        help="average of B-column for metric, with delta showing difference against A",
    )
with display_metric_cols[2]:
    r_var = r_vals.var()
    l_var = l_vals.var()
    var, delta_var = r_var, (r_var - l_var)
    st.metric(
        f"Variance of {_metric}(B)",
        np.round(var, num_rounding),
        delta=delta_var,
        help="variance of B-column for metric, with delta showing difference against A",
    )

with display_metric_cols[1]:
    r_med = np.median(r_vals)
    l_med = np.median(l_vals)
    med, delta_med = r_med, (r_med - l_med)
    st.metric(
        f"Median of {_metric}(B)",
        np.round(med, num_rounding),
        delta=delta_med,
        help="median of B-column for metric, with delta showing difference against A",
    )

left_summary_col, right_summary_col = st.columns(2)


with left_summary_col:
    # st.write(left_runs_compare[metric])
    if show_imgs:
        print_graphs(left_runs)
with right_summary_col:
    # st.write(right_runs_compare[metric])
    if show_imgs:
        print_graphs(right_runs)
