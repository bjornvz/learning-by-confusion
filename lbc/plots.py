import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# from sklearn.metrics import r2_score # NOTE: use pytorch r2_score instead
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

sns.set_theme()
sns.color_palette("Paired")

import numpy as np
import matplotlib.pyplot as plt

def save_fig(fig, folder_path, file_name):
    """
    Saves fig to path as a png file.
    """
    folder = Path(folder_path)
    save_path = folder.joinpath(file_name)
    fig.savefig(save_path)


def plot_history(metrics, cf, file_name):
    """"
    Plots losses and the bottleneck activations for each epoch, and phase indicator if applicable.
    """
    metric1, metric2 = f"train_{cf.loss_str}", "train_l1"
    metric3, metric4 = f"val_{cf.loss_str}", "val_l1"
    metric5 = cf.goodness_str

    temp = metrics[f"val_"+metric5]
    if cf.model == "tetriscnn":
        pass
        # z = np.array([ metrics[f'z_{k}'] for k in range(len(cf.kernels)) ] ) # shape ( no_kernels, epochs )
        # z = np.array([ np.mean(metrics[f'z_{k}'], axis=1)  for k in range(len(cf.kernels)) ]) # mean over the batch
    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])  # 4 rows, 2 columns

    # subplot 1: all the losses
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(metrics[metric1], label='train', color="red")
    ax1.plot(metrics[metric3], label='val', color="blue")
    if cf.model != "tetriscnn":
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel(f"{cf.loss_str}")
    ax1.set_title(fr"$\mathbf{{{cf.loss_str}}}$, train= {metrics[metric1][-1]:.2e}, val={metrics[metric3][-1]:.2e}", loc='left')

    if cf.model == "tetriscnn":
        pass
        # # subplot 2: l1
        # ax2 = fig.add_subplot(gs[0, 1])
        # ax2.plot(metrics[metric2], label='train', color="red")
        # ax2.plot(metrics[metric4], label='val', color="blue")
        # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # ax2.set_yscale("log")
        # ax2.set_xlabel("epoch")
        # ax2.set_ylabel("l1")
        # ax2.set_title(
        #     fr"$\mathbf{{l1}}$, train= {metrics[metric2][-1]:.2e}, val={metrics[metric4][-1]:.2e}",
        #     loc='left'
        # )
    # subplot 3: acc/r2
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(metrics[f"val_"+metric5], label=metric5, color="blue")
    ax3.plot(metrics[f"train_"+metric5], label=f"train_{metric5}", color="red")
    if cf.task == "regression":
        ax3.plot(metrics["train_r2agg"], label="train_r2agg", color="red", linestyle='--')
        ax3.plot(metrics["val_r2agg"], label="val_r2agg", color="blue", linestyle='--')

    ax3.set_xlabel("epoch")
    ax3.set_ylabel(cf.goodness_str)
    ax3.set_title(fr"$\mathbf{{{cf.goodness_str}}}$, train= {metrics[f'train_{metric5}'][-1]:.2f}, val={metrics[f'val_{metric5}'][-1]:.2f}", loc='left')
    lower_limit = 0 if cf.task == "regression" else 0.49
    ax3.set_ylim(lower_limit, 1.01)

    if cf.model == "tetriscnn":
        pass
        # # subplot 4: the bottleneck
        # ax1b = fig.add_subplot(gs[1, 1])
        # for i in range(len(cf.kernels)):
        #     ax1b.plot(np.abs(z[i]), label=kernel_to_latex_pattern(cf.kernels[i]))
        # legend = ax1b.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
        # for text, line in zip(legend.get_texts(), legend.get_lines()):
        #     text.set_color(line.get_color())
        # ax1b.set_xlabel("epoch")
        # ax1b.set_ylabel("abs(z)")
        # ax1b.set_title("bottleneck activations")
        # ax1b.set_yscale("log")


    # set grid to true for all axes so far
    # for ax in [ax1, ax2, ax3, ax1b]:
    for ax in [ax1, ax3]:
        ax.grid(True)


    # subplot 5: phase indicator spanning two rows
    ax5 = fig.add_subplot(gs[2:, 0])  # Span rows 2 and 3, both columns
    ax5b = fig.add_subplot(gs[2:, 1])  # Span rows 2 and 3, both columns
    sns.set_style("dark")


    plt.tight_layout()
    save_fig(fig, cf.logdir, file_name)
    plt.close()





def plot_lbc(all_metrics, cf, file_name, enabled_metrics=None, show=False):
    """
    Plot Learning by Confusion (LBC) results aggregated over seeds.
    """

    # if cf.remake_lambda_plot:
    #     unique_labels = np.fromstring(cf.unique_labels.strip("[]"), sep=" ")
    # else:
    #     unique_labels = np.array(cf.val_dataset.processor.unique_labels)
        
    unique_labels = np.array(cf.unique_labels)
    partitions = (unique_labels[:-1] + unique_labels[1:]) / 2

    metrics_to_plot = [
        # dict(name="z", ylabel="abs(z)", title="final bottleneck values", per_kernel=True),
        dict(name="acc", ylabel="accuracy", title="acc"),
        dict(name="loss", ylabel="loss", title="loss", log=True),
    ]
    if enabled_metrics:
        active_metrics = [m for m in metrics_to_plot if m["name"] in enabled_metrics]
    else:
        active_metrics = metrics_to_plot

    fig, ax = plt.subplots(
        len(active_metrics), 1,
        figsize=(9, 3 + (len(active_metrics) - 1) * 2),
        sharex=True,
        gridspec_kw={'height_ratios': [5] + [2] * (len(active_metrics) - 1)}
    )
    if len(active_metrics) == 1:
        ax = [ax]
    # print(all_metrics.keys())
    # assert len(cf.lambdas) == 1
    # lam = str(cf.lambdas[0])
    seeds = list(all_metrics.keys())

    for i, m in enumerate(active_metrics):
        metric_name = m["name"]

        # collect data across seeds
        def stack(metric_key):
            arrs = []
            for seed in seeds:
                arrs.append(np.array(all_metrics[seed][metric_key]))
            return np.stack(arrs)  # shape: (n_seeds, n_partitions, ...)
        
        if metric_name == "z":
            data = stack("z")  # (seeds, partitions, kernels) or (seeds, partitions)
            if data.ndim == 3:  # multiple kernels
                for k in range(data.shape[2]):
                    mean = np.mean(np.abs(data[:, :, k]), axis=0)
                    std = np.std(np.abs(data[:, :, k]), axis=0)
                    # ax[i].plot(partitions, mean, marker="o", linestyle="-", markersize=4,
                    #            label=kernel_to_latex_pattern(cf.kernels[k]))
                    # ax[i].fill_between(partitions, mean - std, mean + std, alpha=0.2)
            else:  # single kernel
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                ax[i].plot(partitions, mean, marker="o", linestyle="-", markersize=4)
                ax[i].fill_between(partitions, mean - std, mean + std, alpha=0.2)
            legend = ax[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2, fontsize=8)
            for text, line in zip(legend.get_texts(), legend.get_lines()):
                text.set_color(line.get_color())
            ax[i].set_yscale("log")

        elif metric_name == "acc":
            for split, color in [("train_acc", "red"), ("val_acc", "blue")]:
                data = stack(split)  # (seeds, partitions)
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                ax[i].plot(partitions, mean, marker="o", linestyle="-", markersize=4, label=split.split("_")[0], color=color)
                ax[i].fill_between(partitions, mean - std, mean + std, color=color, alpha=0.2)
            ax[i].legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")

        elif metric_name == "loss":
            for split, color in [("train_CEL", "red"), ("val_CEL", "blue")]:
                data = stack(split)
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                ax[i].plot(partitions, mean, marker="o", linestyle="-", markersize=4, label=split.split("_")[0], color=color)
                ax[i].fill_between(partitions, mean - std, mean + std, color=color, alpha=0.2)
            if m.get("log"):
                ax[i].set_yscale("log")

        ax[i].set_ylabel(m["ylabel"])
        ax[i].set_title(m["title"], loc="left")
        ax[i].grid(True)

    ax[-1].set_xlabel(cf.label_param)
    ax[-1].tick_params(axis='x', rotation=45)
    ax[-1].set_xticks(partitions)

    fig.suptitle(rf"learning by confusion: {cf.dataset}, {len(cf.seeds)} seeds")
    plt.tight_layout()
    save_fig(fig, cf.logdir, file_name)
    if show:
        plt.show()
    plt.close()








