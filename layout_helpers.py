"""
Helper functions for Aerosense layout setup.
This file contains all the necessary functions to run Aerosense_layout_notebook.ipynb
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from typing import Tuple, Optional
import plotly.graph_objects as go


# ==================== BUILD FEATURES FUNCTIONS ====================

def position2distance(
    wing: np.ndarray,
    i0: Optional[int] = None,
    xsens: Optional[np.ndarray] = None,
    isensor_le: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the distance between the points given as input.

    The leading edge (0,0) must be present, as well as the trailing edge (1,0). chord=1.
    Wing must start at the trailing edge, pressure side, goes at leading edge and finishes at
    trailing edge via upper side.

    Parameters:
    wing (complex): The complex coordinates of the wing.
    i0 (int, optional): If provided, distance won't be calculated from leading edge.
    xsens (array, optional): If provided, it represents the x coordinate of the sensors (along the chord)
                              and the function will return their distance from i0.
    isensor_le (int, optional): If provided, it represents the index of the leading edge in the sensors array which is the first on the suction side.

    Returns:
    Lwing (array): Curvilinear length of the points from the trailing edge.
    twing (array): Tangent of each point given in wing.
    nwing (array): Normal of each point given in wing.
    zsens (array): It could be the position of the points defining the wing (same as input) or the sensors if xsens is provided.
    If sensors are placed, wing is replaced by sensors!
    """

    # find leading edge
    if i0 is None:
        ile = np.argmin(wing.real)
    else:
        ile = i0

    # first side (should be lower side)
    dwing_l = np.diff(wing[ile::-1])
    lwing_l = np.abs(dwing_l)
    Lwing_l = np.flip(np.cumsum(lwing_l))

    # second side (should be upper side)
    dwing_u = np.diff(wing[ile:])
    lwing_u = np.abs(dwing_u)
    Lwing_u = np.cumsum(np.concatenate(([0], lwing_u)))

    Lwing = np.concatenate((-Lwing_l, Lwing_u))

    Ww = np.concatenate((wing, [wing[-1]]))

    with np.errstate(divide="ignore", invalid="ignore"):
        twing = np.mean(
            [
                np.diff(Ww) / np.abs(np.diff(Ww)),
                np.roll(np.diff(Ww) / np.abs(np.diff(Ww)), 1),
            ],
            axis=0,
        )
    twing[np.isnan(twing.real)] = 0
    nwing = twing * np.exp(1j * np.pi / 2)

    zsens = wing

    if xsens is not None:
        if np.iscomplex(xsens).any():
            xsens = xsens.real
        if isensor_le is None:
            isen_le = cKDTree(xsens.reshape(-1, 1)).query([0])[1]
        else:
            isen_le = isensor_le
        si_sen = np.ones(xsens.shape)
        si_sen[:isen_le] = -si_sen[:isen_le]
        Lsens = np.interp(si_sen * xsens, np.sign(Lwing) * wing.real, Lwing)

        zsens, tsens, nsens = distance2position(Lsens, wing[ile], wing=wing)

        Lwing = Lsens
        twing = tsens
        nwing = nsens

    return Lwing, twing, nwing, zsens


def distance2position(
    Lsens: np.ndarray, zsens0: np.ndarray, wing: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gives the positions of the sensors on the given wing.
    The leading edge (0,0) must be present, as well as the trailing edge (1,0). chord=1.
    Wing must start at the trailing edge goes at leading edge and finishes at trailing edge.

    Parameters:
    Lsens: Distance from sensor isen0 positioned at zsens0. We should have Lsens[isen0-1]=0.
    zsens0: Position of the sensor where we know the position.
    wing: Points defining the blade in complex coordinate.

    Returns:
    zsensor: Position of the sensor on the wing.
    tsens: Normal of each sensor.
    nsens: Tangent of each sensor.
    """

    # Check if wing is complex
    if np.isreal(wing).all():
        wing = wing[:, 0] + 1j * wing[:, 1]

    tree = cKDTree(c2m(wing))
    # Query the nearest point
    _, isens0 = tree.query(c2m(zsens0))

    Lwing, twing, _, _ = position2distance(wing, i0=isens0[0])

    zsens = np.interp(Lsens, Lwing, wing)
    tsens = np.interp(Lsens, Lwing, twing)
    tsens /= np.abs(tsens)

    nsens = tsens * np.exp(1j * np.pi / 2)
    nsens /= np.abs(nsens)

    return zsens, tsens, nsens


def c2m(complex_array):
    """Convert complex array to matrix."""
    return np.column_stack((complex_array.real, complex_array.imag))


def find_leading_edge(wing: np.ndarray) -> Tuple[int, complex]:
    """Find the leading edge of a wing shape."""
    tree = cKDTree(c2m(wing))
    _, indices = tree.query([[0, 0]])
    i0 = indices[0]
    return i0, wing[i0]


# ==================== MAKE DATASET FUNCTIONS ====================

def ynaca(x: np.ndarray, th: float) -> np.ndarray:
    """
    Calculate the y-coordinates of a NACA 00XX airfoil.

    Args:
        x (np.ndarray): x-coordinates from 0 to 1.
        th (float): Thickness of the airfoil as a fraction of the chord length (e.g., 0.18 for NACA 0018).

    Returns:
        np.ndarray: y-coordinates of the airfoil.
    """
    return (th / 0.2) * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4
    )


def create_wing(wing_shape_dir: str = None) -> np.ndarray:
    """
    Function to create a wing shape based on the provided directory or NACA code.

    Returns:
        np.ndarray: Complex array representing wing coordinates.
    """
    # Extract thickness from 'NACA00XX'
    if wing_shape_dir and str(wing_shape_dir).startswith("NACA00"):
        thickness_str = str(wing_shape_dir)[-2:]
        th = int(thickness_str) / 100  # e.g., 18 -> 0.18 for NACA0018
        xwing = np.linspace(0, 1, 1000)
        ywing = ynaca(xwing, th)
        # Create both sides of the wing
        x_upper = xwing
        y_upper = ywing
        x_lower = xwing[1:]  # Exclude the leading edge point to avoid duplication
        y_lower = -ywing[1:]  # Mirror the upper surface for the lower surface
        # Concatenate upper and lower surfaces
        x_full = np.concatenate([x_lower[::-1], x_upper])
        y_full = np.concatenate([y_lower[::-1], y_upper])
        # Create complex wing shape coordinates
        wing = x_full + 1j * y_full
    else:
        # Extract from any file wing_shape_dir which is set correctly
        datwing = np.genfromtxt(wing_shape_dir, delimiter=" ")
        wing = np.flipud(datwing[:, 0] + 1j * datwing[:, 1])
        # index of leading edge
        i0, _ = find_leading_edge(wing)
        # add a leading edge 0,0 (will simplify our work afterwards)
        wing = np.insert(wing, i0, 0)
    return wing


def make_labbook(
    chord: float = 1.25, span: float = 5.0, wing_shape_dir: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Function to create a labbook for the project.

    Returns:
        Tuple of (zwing, lwing, twing, nwing, blade_properties)
    """
    blade_properties = {
        "chord": chord,  # chord length in m
        "span": span,  # span in m
    }

    # Create the wing
    wing = create_wing(wing_shape_dir)

    # Find the unique curvilinear length (lw), the tangent vector (tw) and the normal vector (nw)
    [lw, tw, nw, zw] = position2distance(wing)

    # make more discretised wing
    newLwing = np.linspace(lw[0], lw[-1], 1001)
    interp_func_real = interp1d(lw, np.real(wing), kind="cubic")
    interp_func_imag = interp1d(lw, np.imag(wing), kind="cubic")

    # Interpolate the wing values
    wing_i = interp_func_real(newLwing) + 1j * interp_func_imag(newLwing)
    [lw_i, tw_i, nw_i, zw_i] = position2distance(wing_i)
    blade_properties["wing_shape"] = zw_i
    blade_properties["Lwing"] = lw_i
    blade_properties["tan_wing"] = tw_i
    blade_properties["norm_wing"] = nw_i

    # return the points defining the wing, the curvilinear length, the tangent and the normal
    return zw_i, lw_i, tw_i, nw_i, blade_properties


# ==================== VISUALIZATION FUNCTIONS ====================

def make_go_wing(wing: np.ndarray) -> go.Scatter:
    """
    Create a plotly graph object for the wing.
    """
    wing_trace = go.Scatter(
        x=wing.real,
        y=wing.imag,
        mode="lines",
        line=dict(color="darkgrey", width=2),
        fill="toself",
        name="Wing",
        showlegend=False,
    )
    return wing_trace


def position_aerosense_sensors(
    zwing: np.ndarray,
    nwing: np.ndarray,
    chord: float,
    target_real_positions: list,
    target_sides: list,
    isens0_baros: list,
    nbaros: int = 24,
    sensor_spacing: float = 14.0,
    sensor_thickness: float = 4.0,
    limit_before: float = 35.0,
    limit_after: float = 41.0,
) -> Tuple[list, list, list, np.ndarray, float]:
    """
    Position Aerosense sensors on the wing surface.

    Parameters:
    -----------
    zwing : np.ndarray
        Complex array of wing coordinates
    nwing : np.ndarray
        Complex array of wing normal vectors
    chord : float
        Wing chord length in meters
    target_real_positions : list
        List of chordwise positions for each sensor block
    target_sides : list
        List of surface sides ('pressure' or 'suction') for each block
    isens0_baros : list
        Index of the reference sensor for each block (0-indexed)
    nbaros : int, optional
        Number of sensors per block (default: 24)
    sensor_spacing : float, optional
        Distance between sensors in mm (default: 14.0)
    sensor_thickness : float, optional
        Thickness of sensors in mm (default: 4.0)
    limit_before : float, optional
        Distance before first sensor for limit marker in mm (default: 35.0)
    limit_after : float, optional
        Distance after last sensor for limit marker in mm (default: 41.0)

    Returns:
    --------
    zBaros_list : list
        List of sensor positions for each block (complex arrays)
    limit_points : list
        List of limit marker positions (complex numbers)
    zsens0_baros : list
        List of reference sensor positions (complex numbers)
    zBaros_all : np.ndarray
        All sensor positions concatenated
    db : float
        Normalized sensor spacing
    """
    # Calculate sensor spacing normalized by chord
    db = sensor_spacing * 10**-3 / chord

    # Find positions on wing for sensor alignment
    zsens0_baros = []
    for target_real, side in zip(target_real_positions, target_sides):
        if side == 'pressure':
            # Filter for pressure side (negative imaginary part)
            pressure_mask = np.imag(zwing) <= 0
            zwing_side = zwing[pressure_mask]
        else:  # suction side
            # Filter for suction side (positive imaginary part)
            suction_mask = np.imag(zwing) >= 0
            zwing_side = zwing[suction_mask]

        # Find closest real coordinate
        real_diffs = np.abs(np.real(zwing_side) - target_real)
        closest_idx = np.argmin(real_diffs)
        zsens0_baros.append(zwing_side[closest_idx])

    # Create wing surface offset by sensor thickness
    th_sens = sensor_thickness * 10**-3 / chord  # normalized thickness
    zwing_sens = zwing + th_sens * nwing

    # Calculate sensor positions for each block
    zBaros_list = []
    limit_points = []  # To store the limit markers

    for i in range(len(isens0_baros)):
        lbaros = db * np.arange(-isens0_baros[i], nbaros - isens0_baros[i])
        zBaros, _, _ = distance2position(lbaros, zsens0_baros[i], zwing_sens)
        zBaros_list.append(zBaros)

        # Add limit markers using configuration parameters
        l_start = lbaros[0] - limit_before * 10**-3 / chord
        l_end = lbaros[-1] + limit_after * 10**-3 / chord

        z_start, _, _ = distance2position(np.array([l_start]), zsens0_baros[i], zwing_sens)
        z_end, _, _ = distance2position(np.array([l_end]), zsens0_baros[i], zwing_sens)

        limit_points.extend([z_start[0], z_end[0]])

    zBaros_all = np.concatenate(zBaros_list)

    return zBaros_list, limit_points, zsens0_baros, zBaros_all, db
