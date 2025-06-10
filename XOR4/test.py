import gzip
import pickle
from typing import Tuple

# --- 1.  load the DB --------------------------------------------------
def load_db(n: int, path_template: str = "DB/Binary/logic_class_db_n{}.pkl.gz"):
    with gzip.open(path_template.format(n), "rb") as f:
        return pickle.load(f)
    
db=load_db(4)

x = len(db["classes"])
y = sum(db["separability"])


print(x)
print(y)
print(y/x)
    
