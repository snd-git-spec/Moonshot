from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import zlib
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("moonshot")



# Core majors (stable sample) + hot movers (ZEC-style explosions). Hunt/paper share the set.
# Hot names can thin samples and spike DD — kept because dollar upside is the goal.
DEFAULT_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
    "DOGE/USDT:USDT", "ZEC/USDT:USDT", "SUI/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT",
]
HOT_SYMBOLS = [
    "ZEC/USDT:USDT", "SUI/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT",
    "SOL/USDT:USDT", "DOGE/USDT:USDT",
]
PAPER_SYMBOLS = list(DEFAULT_SYMBOLS)
HUNT_SYMBOLS = list(DEFAULT_SYMBOLS)
# Extra perps available in the dashboard; not default paper universe.
OPTIONAL_SYMBOLS = [
    "ADA/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT", "LTC/USDT:USDT",
    "DOT/USDT:USDT", "ATOM/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT",
    "ARB/USDT:USDT", "OP/USDT:USDT", "INJ/USDT:USDT", "UNI/USDT:USDT", "AAVE/USDT:USDT",
    "TRX/USDT:USDT", "BCH/USDT:USDT", "ETC/USDT:USDT",
    "RENDER/USDT:USDT", "FET/USDT:USDT", "SEI/USDT:USDT", "TIA/USDT:USDT", "WLD/USDT:USDT",
    "POL/USDT:USDT",
]


