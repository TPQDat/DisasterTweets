"""
Utility Functions
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import CFG

def seed_everything(seed: int):
    """seed everything
    Args:
        seed (int): hash seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True