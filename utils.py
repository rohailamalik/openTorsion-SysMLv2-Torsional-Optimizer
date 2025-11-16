import numpy as np

def to_python(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic): 
            return obj.item()
        elif isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_python(x) for x in obj]
        else:
            return obj