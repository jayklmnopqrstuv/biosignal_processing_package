import os
from joblib import Memory

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
memory = Memory(cachedir=CACHE_DIR, verbose=0)
