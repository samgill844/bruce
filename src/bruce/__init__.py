"""Tools for modelling and analysing exoplanets and binary stars."""
from importlib import import_module

__all__ = [
    "binarystar", "data", "sampler", "template_match", "ambiguous_period",
    "visual", "empirical_relations",
]


def __getattr__(name):
    # Keep archive, GUI and plotting dependencies out of unrelated workflows.
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
