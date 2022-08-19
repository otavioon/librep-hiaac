from typing import List

import plotly.express as px
import plotly.graph_objects as go


def visualize_sample(ys, xs=None, label: str = "", mode: str = "lines+text", title: str = None):
    if xs is None:
        xs = list(range(len(ys)))

    fig = go.Figure()
    fig.add_scatter(x=xs, y=ys, name=label, mode=mode)
    fig.update_layout(title=title)
    fig.show()


def visualize_sample_windowed(
    window_size: int, ys, xs=None, labels: List[str] = None, mode: str = "lines+text",
    sample_size: int = None, title: str = None
):
    if xs is None:
        xs = list(range(window_size))
    if sample_size is None:
        sample_size = window_size

    fig = go.Figure()
    for i, start in enumerate(range(0, len(ys), window_size)):
        if labels is not None:
            label = labels[i]
        else:
            label = None
        fig.add_scatter(x=xs, y=ys[start : start + sample_size], name=label, mode=mode)

    fig.update_layout(title=title)
    fig.show()