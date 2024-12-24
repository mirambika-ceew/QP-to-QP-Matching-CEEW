import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st


def get_percentile_rank(number: float, number_list: list) -> float:
    # Returns what percentage of numbers are below this number
    return (np.searchsorted(np.sort(number_list), number) / len(number_list)) * 100


def highlight_rows(row):
    value = row["similarity_score"]
    color = "background-color: "
    if value >= 0.90:
        return [color + "green"] * len(row)
    elif value >= 0.80:
        return [color + "yellow"] * len(row)
    else:
        return [color + "grey"] * len(row)


def add_colorscale(data):
    percentiles = [5, 95]
    percentile_values = np.percentile(data, percentiles)
    min_value = np.min(data)
    max_value = np.max(data)

    # Create color map
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)

    # Create the figure and axis for the colorbar
    fig, ax = plt.subplots(figsize=(8, 0.2))

    # Create colorbar based on the data
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    cb.set_label("Percentile scores for cosine similarities across all job pairs", rotation=0, labelpad=2)

    # # Mark percentiles and min/max values on the horizontal colorbar
    # for value, label in zip([min_value] + percentile_values.tolist() + [max_value],
    #                         ["Min"] + [f"{p}th" for p in percentiles] + ["Max"]):
    #     ax.plot([norm(value), norm(value)], [0, 1], color='black', lw=1)
    #     ax.text(norm(value), 1.05, f"{label}: {value:.2f}", va="bottom", ha="center", fontsize=4)
    return fig

