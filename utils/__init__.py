from .data_module import (
    Evaluator,
    MgsmEval,
)
from .skip_layer import sl_generate


NAME_TO_DATASET = {
    "mgsm" : MgsmEval
}

def get_eval(name: str) -> Evaluator:
    return NAME_TO_DATASET[name]


__all__ = ["Evaluator", "MgsmEval", "sl_generate", "get_eval"]