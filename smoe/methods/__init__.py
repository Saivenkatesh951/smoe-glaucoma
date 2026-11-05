from . import balanced, baseline, cggm, smoe

METHODS = {
    "baseline": baseline,
    "normal": balanced,
    "opm": balanced,
    "ogm": balanced,
    "opm_ogm": balanced,
    "cggm": cggm,
    "smoe": smoe,
}
