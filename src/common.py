"""Shared imports and constants."""
import math
from dataclasses import dataclass, field, replace
import numpy as np
import pandas as pd
import pulp

CLASSES = ["low", "medium", "high", "severe"]
K = len(CLASSES)
INF = np.inf
