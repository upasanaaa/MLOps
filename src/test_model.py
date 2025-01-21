import torch
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_solution import MyAwesomeModel
import pytest


def test_error_on_wrong_shape():
    model = MyAwesomeModel()

    # Test for incorrect input dimensions (not 4D)
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(torch.randn(1, 2, 3))  # Incorrect shape

    # Test for incorrect shape of each sample
    with pytest.raises(ValueError, match="Expected each sample to have shape \\[1, 28, 28\\]"):
        model(torch.randn(1, 1, 28, 29))  # Incorrect shape
