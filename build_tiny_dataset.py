import pickle
import numpy as np
import sys
from pathlib import Path
if not hasattr(np, '_core') and hasattr(np, 'core'):
    sys.modules.setdefault('numpy._core', np.core)
    np._core = np.core
base = Path('data/woodscape_mock')
with open(base / 'woodscape_infos_train.pkl', 'rb') as f:
    train = pickle.load(f)
with open(base / 'woodscape_infos_val.pkl', 'rb') as f:
    val = pickle.load(f)
mini_train = dict(metadata=train['metadata'], infos=train['infos'][:8])
mini_val = dict(metadata=val['metadata'], infos=val['infos'][:4])
with open(base / 'woodscape_infos_train_tiny.pkl', 'wb') as f:
    pickle.dump(mini_train, f)
with open(base / 'woodscape_infos_val_tiny.pkl', 'wb') as f:
    pickle.dump(mini_val, f)
print('train_tiny', len(mini_train['infos']))
print('val_tiny', len(mini_val['infos']))
