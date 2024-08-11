import os
import json
import pickle
import shutil
import sys

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy
import pandas as pd
import pint

from config import DATA_DIR


plt.rc("font", family="Arial")


def get_report_path(filename, refresh=True):
    base, ext = os.path.splitext(filename)
    report_path = os.path.join(base, "results")
    if refresh and os.path.exists(filename):
        shutil.rmtree(report_path, ignore_errors=True)
        shutil.unpack_archive(filename, DATA_DIR)
    return report_path


def load_results(exp_name, trials=True, refresh=True):
    if trials:
        filename = os.path.join(DATA_DIR, "{}.trials.pickle".format(exp_name))
        with open(filename, "rb") as f:
            trials = pickle.load(f)
        df = pd.DataFrame(trials.results)
        df = df.loc[df["status"] == "ok"]
        idx = df.index.to_numpy()
        for key, val in trials.vals.items():
            df = df.assign(**{key: np.asarray(val)[idx]})
    else:
        filename = os.path.join(DATA_DIR, "{}.tgz".format(exp_name))
        report_path = get_report_path(filename, refresh)
        results = []
        for file in os.listdir(report_path):
            if os.path.isfile(os.path.join(report_path, file)):
                with open(os.path.join(report_path, file), "r") as f:
                    result = json.load(f)
                    results.append({**result["returned_dict"], **result["current_params"]})
        df = pd.DataFrame(results)
        
    def convert(x):
        val, unit = x[:-3], x[-3:]
        if unit == "MHz":
            val = float(val) * 1e6
        elif unit == "kHz":
            val = float(val) * 1e3
        else:
            raise ValueError("Invalid frequency unit {}".format(unit))
        return val
    
    df["fir_rate"] = df["fir_rate"].apply(convert)
    if "delay" not in df:
        df["delay"] = df["fir_length"] / df["fir_rate"] * df["logging_rate"] / df["nodes"]
    keys = ["Vpi", "Pmax", "M", "ext_ratio"]
    vals = [np.nan, np.nan, np.nan, np.nan]
    for key, val in zip(keys, vals):
        if key in df:
            df.loc[np.isnan(df[key]), key] = val
        else:
            df[key] = val
        # df.loc[np.isnan(df["phi"]), "phi"] = 1.56
    df["G"] = df["pmax"] * 10 / df["Vpi"]
    df["rho"] = df["input_scaling"] * 0.25 * np.pi
    df["nrmse"] = df["nmse"].apply(np.sqrt)
    df["Pmin"] = df["Pmax"] * (1 - df["M"]) * (1 + df["M"])
    t0 = df.start_time.max()
    df["t"] = (df["start_time"] - t0) / 3600
    return df
  

def plot_stats(df, metric, param1, param2=None, ax=None, **scatter_kwargs):
    x = df[param1].to_numpy()
    data = df[metric].to_numpy()
    if param2:
        y = df[param2].to_numpy()

    if ax is None:
        fig, ax = plt.subplots(dpi=300)
        do_cb = True
    else:
        fig = ax.get_figure()
        do_cb = False
    ax.set_xlabel(param1)
    idx_sort = np.argsort(data)[::-1]
    if param2:
        if isinstance(ax, Axes3D):
            z = df.nmse.to_numpy()[idx_sort]
            im = ax.scatter(x[idx_sort], y[idx_sort], z[idx_sort],
                            c=data[idx_sort], s=1 / data[idx_sort],
                            **scatter_kwargs)
        else:
            im = ax.scatter(x[idx_sort], y[idx_sort],
                            c=data[idx_sort], s=1 / data[idx_sort],
                            **scatter_kwargs)
        if do_cb:
            cb = fig.colorbar(im, ax=[ax,])
            cb.set_label(metric.upper())
        ax.set_ylabel(param2)
        # ax.set_yscale("log")
    else:
        ax.scatter(x[idx_sort], data[idx_sort],
                   c=data[idx_sort], s=1 / data[idx_sort],
                   **scatter_kwargs)
        ax.set_ylabel(metric)
    if exp_name:
        ax.set_title(exp_name)
    return fig, ax


def plot_history(df, metric, x_is_time=False, **kwargs):
    fig, ax = plt.subplots(dpi=300)
    idx = np.argsort(df["start_time"].to_numpy())
    data = df[metric].to_numpy()[idx]
    accumulate = kwargs.pop("accumulate", None)
    if accumulate:
        data = accumulate.accumulate(data)
    if x_is_time:
        x = df["start_time"].to_numpy()[idx]
        x -= x[0]
        x /= 3600
    else:
        x = np.arange(len(data))
    ax.plot(x, data, label=metric, **kwargs)
    ax.legend()
    if exp_name:
        ax.set_title(exp_name)
    if x_is_time:
        ax.set_xlabel("Time (h)")
    else:
        ax.set_xlabel("Iteration")
    # ax.set_ylim(0.55, 1)
    # ax.set_yscale("log")
    return fig, ax


# %%
if __name__ == "__main__":
    if len(sys.argv) == 1:
        exp_name = "hyperopt_sin_square_random_search"
    elif len(sys.argv) == 2:
        exp_name = sys.argv[1]
    else:
        raise TypeError("Too many arguments")
    df = load_results(exp_name, False, False)
    df_plot = df.loc[df.nmse < 0.2].loc[df.nodes <= 50]
    #df_plot = df.loc[df.nmse < 0.5]
    # %% Plot stats
    fig = plt.figure(figsize=(12, 6), dpi=150,
        
    )
    gs = plt.GridSpec(3, 3,
                      width_ratios=[2, 1, 1],
                      hspace=0.25, wspace=0.3)
    ax00 = fig.add_subplot(gs[:, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])
    #ax22 = fig.add_subplot(gs[2, 2])
    plot_stats(df_plot, "nmse", "delay", ax=ax00, marker='.')
    plot_stats(df_plot, "nmse", "delay", "rho", ax=ax01, marker=".")
    plot_stats(df_plot, "nmse", "delay", "phi0", ax=ax02, marker=".")
    plot_stats(df_plot, "nmse", "delay", "G", ax=ax11, marker=".")
    plot_stats(df_plot, "nmse", "delay", "ridge", ax=ax12, marker=".")
    #plot_stats(df_plot, "nmse", "nodes", ax=ax22, marker='.')

    
    ax00.set_ylabel("NMSE")
    ax01.set_ylabel("$\\rho$")
    ax11.set_ylabel("$G$")
    ax02.set_ylabel("$\Phi_0$")
    #ax22.set_ylabel('NMSE')
    #ax22.set_xlabel('NODES')

    ticks = list(range(5))
    denominator = max(ticks)
    tlocs = [t * np.pi / denominator for t in ticks]
    ticklbl = [t * sympy.pi / denominator for t in ticks]
    
    ax02.set_yticks(tlocs, ["${}$".format(sympy.latex(l)) for l in ticklbl])
    
    ax12.set_yscale("log")
    ax12.set_ylabel("$\lambda$")
    
    for _, label in zip(fig.axes, "abcde"):
        _.set_xlabel("$\\tau/T$")
        _.set_title(label, position=(-0.12, 0))
    cb = fig.colorbar(
        ax01.collections[0], ax=ax00, location="bottom",
        pad=0.15, fraction=0.04
    )
    cb.set_label("NMSE in {} Task".format(exp_name))
    plt.show()
    plt.figure()
    plot_stats(df_plot, "nmse", "nodes", marker='.')



