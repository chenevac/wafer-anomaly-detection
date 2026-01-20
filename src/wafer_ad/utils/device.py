"""Device utility functions."""

import logging
import torch


def get_device(verbose: bool = True) -> str:
    """Get cpu, gpu or mps device."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if verbose:
        logging.info("Computation device %s", device)
    return device