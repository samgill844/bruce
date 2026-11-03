import numpy as np


R_EARTH_IN_RJUP = 0.0892147
M_EARTH_IN_MJUP = 317.828


def mass_radius(mass, mass_unit="earth", radius_unit="earth"):
    """
    Dransfield et al. (2026) empirical mass-radius relation.

    Parameters
    ----------
    mass : float or array-like
        Planet mass.
    mass_unit : {"earth", "jupiter"}
        Unit of input mass.
    radius_unit : {"earth", "jupiter"}
        Unit of returned radius.

    Returns
    -------
    radius : float or ndarray
        Predicted planet radius.
    """

    mass = np.asarray(mass, dtype=float)

    # Convert input mass to Earth masses
    if mass_unit.lower() in ["jupiter", "mj", "mjup"]:
        mass = mass * M_EARTH_IN_MJUP
    elif mass_unit.lower() not in ["earth", "me", "mearth"]:
        raise ValueError("mass_unit must be 'earth' or 'jupiter'")

    radius = np.empty_like(mass)

    mask = mass < 5.1
    radius[mask] = 1.02 * mass[mask]**0.289

    mask = (mass >= 5.1) & (mass < 54.3)
    radius[mask] = 0.554 * mass[mask]**0.668

    mask = (mass >= 54.3) & (mass < 257.8)
    radius[mask] = 2.72 * mass[mask]**0.270

    mask = mass >= 257.8
    radius[mask] = 12.6 * mass[mask]**(-0.007)

    # Convert radius to Jupiter radii if requested
    if radius_unit.lower() in ["jupiter", "rj", "rjup"]:
        radius *= R_EARTH_IN_RJUP
    elif radius_unit.lower() not in ["earth", "re", "rearth"]:
        raise ValueError("radius_unit must be 'earth' or 'jupiter'")

    return radius

def radius_to_mass(radius):
    """
    Inverse of the Dransfield et al. (2026) mass-radius relation.

    Parameters
    ----------
    radius : float or array-like
        Planet radius in Earth radii.

    Returns
    -------
    mass : float or ndarray
        Planet mass in Earth masses.
    """

    radius = np.asarray(radius, dtype=float)

    mass = np.empty_like(radius)

    # Terrestrial: M < 5.1 M_earth
    mask = radius < 1.63
    mass[mask] = (radius[mask] / 1.02)**(1 / 0.289)

    # Neptunian: 5.1 < M < 54.3 M_earth
    mask = (radius >= 1.63) & (radius < 8.0)
    mass[mask] = (radius[mask] / 0.554)**(1 / 0.668)

    # Saturnian: 54.3 < M < 257.8 M_earth
    mask = (radius >= 8.0) & (radius < 12.2)
    mass[mask] = (radius[mask] / 2.72)**(1 / 0.270)

    # Jovian: M > 257.8 M_earth
    mask = radius >= 12.2
    mass[mask] = (radius[mask] / 12.6)**(1 / -0.007)

    return mass