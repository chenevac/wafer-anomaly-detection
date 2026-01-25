from matplotlib import pyplot as plt
import numpy as np


def display_roc(
    x: np.ndarray, 
    y: np.ndarray,
    title: str = "ROC Curve",
) -> None:
    """Display ROC curve."""
    plt.plot(x, y)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$P_{fa}$")
    plt.ylabel(r"$P_d$")
    plt.title(title)
    plt.show()
    
    
def display_learning_curve(
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    title: str = "Learning Curve",
) -> None:
    """Display learning curve."""
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlim(1, len(train_losses))
    plt.ylim(0, max(max(train_losses), max(val_losses)) * 1.1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()