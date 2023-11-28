import matplotlib.pyplot as plt


def plot_shared_x(
    x,
    y1,
    y2,
    color1="tab:red",
    color2="tab:blue",
):
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, color=color1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y1", color=color1)
    ax1.tick_params("y", colors=color1)

    ax2 = ax1.twinx()
    ax2.plot(x, y2, color=color2)
    ax2.set_ylabel("y2", color=color2)
    ax2.tick_params("y", colors=color2)

    fig.tight_layout()
    return fig, ax1, ax2
