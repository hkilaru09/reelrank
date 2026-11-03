"""Print the ids of users the served model knows, as JSON, for loadtest/recommend.js."""

import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/ml-1m/users.npz"
u = np.load(path)
print(json.dumps(u["raw_ids"][u["in_train"]].tolist()))
