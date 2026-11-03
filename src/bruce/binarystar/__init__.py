from .binarystar import lc, rv1, rv2, transit_width, stellar_density

_TTV_EXPORTS = ['scan_companion_period', 'chopping_model', 'nearest_resonances', 'synodic_period', 'synodic_to_periods', 'plot_single_epoch', 'stroboscopic_ttv_plot', 'plot_period_scan', 'plot_ttv_model']


def __getattr__(name):
    if name in _TTV_EXPORTS:
        from importlib import import_module
        value = getattr(import_module(".ttv", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["lc", "rv1", "rv2", "transit_width", "stellar_density"] + _TTV_EXPORTS
