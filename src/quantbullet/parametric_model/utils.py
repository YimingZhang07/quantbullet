import matplotlib.pyplot as plt
import numpy as np
from quantbullet.plot.cycles import use_economist_cycle

def compare_models(models, x, y = None):
    fig, ax = plt.subplots()
    
    with use_economist_cycle():
        for model in models:
            x_min = np.min(x)
            x_max = np.max(x)
            x_grid = np.linspace(x_min, x_max, 100)
            ax.plot(x_grid, model.predict(x_grid), 
                label=model.model_name,
                linewidth=3
            )
        if y is not None:
            ax.scatter(x, y, label="Data", color="gray", alpha=0.4)
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Model Comparison")
    return fig, ax