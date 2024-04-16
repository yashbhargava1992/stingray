import os
import warnings
from .rxte import rxte_calibration_func, rxte_pca_event_file_interpretation


def rough_calibration(pis, mission):
    """Make a rough conversion between PI channel and energy.

    Only works for NICER, NuSTAR, IXPE, and XMM.

    Parameters
    ----------
    pis: float or array of floats
        PI channels in data
    mission: str
        Mission name

    Returns
    -------
    energies : float or array of floats
        Energy values

    Examples
    --------
    >>> rough_calibration(0, 'nustar')
    1.62
    >>> rough_calibration(0.0, 'ixpe')
    0.0
    >>> # It's case-insensitive
    >>> rough_calibration(1200, 'XMm')
    1.2
    >>> rough_calibration(10, 'asDf')
    Traceback (most recent call last):
        ...
    ValueError: Mission asdf not recognized
    >>> rough_calibration(100, 'nicer')
    1.0
    """
    if mission.lower() == "nustar":
        return pis * 0.04 + 1.62
    elif mission.lower() == "xmm":
        return pis * 0.001
    elif mission.lower() == "nicer":
        return pis * 0.01
    elif mission.lower() == "ixpe":
        return pis / 375 * 15
    raise ValueError(f"Mission {mission.lower()} not recognized")


def _patch_mission_info(info, mission=None):
    """Add some information that is surely missing in xselect.mdb.

    Examples
    --------
    >>> info = {'gti': 'STDGTI', 'ecol': 'PHA'}
    >>> new_info = _patch_mission_info(info, mission=None)
    >>> assert new_info['gti'] == info['gti']
    >>> new_info = _patch_mission_info(info, mission="xmm")
    >>> new_info['gti']
    'STDGTI,GTI0'
    >>> new_info = _patch_mission_info(info, mission="xte")
    >>> new_info['ecol']
    'PI'
    """
    if mission is None:
        return info
    if mission.lower() == "xmm" and "gti" in info:
        info["gti"] += ",GTI0"
    if mission.lower() == "xte" and "ecol" in info:
        info["ecol"] = "PI"
        info["ccol"] = "PCUID"
    return info


def read_mission_info(mission=None):
    """Search the relevant information about a mission in xselect.mdb."""
    curdir = os.path.abspath(os.path.dirname(__file__))
    fname = os.path.join(curdir, "..", "datasets", "xselect.mdb")

    # If HEADAS is defined, search for the most up-to-date version of the
    # mission database
    if os.getenv("HEADAS"):
        hea_fname = os.path.join(os.getenv("HEADAS"), "bin", "xselect.mdb")
        if os.path.exists(hea_fname):
            fname = hea_fname
    if mission is not None:
        mission = mission.lower()

    db = {}
    with open(fname) as fobj:
        for line in fobj.readlines():
            line = line.strip()
            if mission is not None and not line.lower().startswith(mission):
                continue
            if line.startswith("!") or line == "":
                continue
            allvals = line.split()
            string = allvals[0]
            value = allvals[1:]
            if len(value) == 1:
                value = value[0]

            data = string.split(":")[:]
            if mission is None:
                if data[0] not in db:
                    db[data[0]] = {}
                previous_db_step = db[data[0]]
            else:
                previous_db_step = db
            data = data[1:]
            for key in data[:-1]:
                if key not in previous_db_step:
                    previous_db_step[key] = {}
                previous_db_step = previous_db_step[key]
            previous_db_step[data[-1]] = value
    return _patch_mission_info(db, mission)


def get_rough_conversion_function(mission, instrument=None, epoch=None):
    """Get a rough PI-Energy conversion function for a mission.

    The function should accept a PI channel and return the corresponding energy.
    Additional keyword arguments (e.g. epoch, detector) can be passed to the function.

    Parameters
    ----------
    mission : str
        Mission name
    instrument : str
        Instrument onboard the mission
    epoch : float
        Epoch of the observation in MJD (important for missions updating their calibration).
    Returns
    -------
    function
        Conversion function
    """

    if mission.lower() == "nustar":
        return lambda pi: pi * 0.04 + 1.62
    if mission.lower() == "xmm":
        return lambda pi: pi * 0.001
    if mission.lower() == "nicer":
        return lambda pi: pi * 0.01
    if mission.lower() == "ixpe":
        return lambda pi: pi / 375 * 15
    if mission.lower() == "xte":
        return rxte_calibration_func(instrument, epoch)
    raise ValueError(f"Mission {mission.lower()} not recognized")


def mission_specific_event_interpretation(mission):
    """Get the mission-specific FITS interpretation function."""

    if mission.lower() == "xte":
        return rxte_pca_event_file_interpretation

    def _empty(x):
        return x

    return _empty
