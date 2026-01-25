from matplotlib import pyplot as plt
import numpy as np


def display_roc(
    x: np.ndarray, 
    y: np.ndarray,
    title: str = "ROC Curve",
) -> None:
    plt.plot(x, y)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$P_{fa}$")
    plt.ylabel(r"$P_d$")
    plt.title(title)
    plt.show()