# -*- coding: utf-8 -*-
"""
static/visualization.py
Created on: 26/03/2019
Last edited: 17/07/2020
Author: William van Doorn

This file contains all static information about visualization such as colors.
"""

# imports
from typing import Tuple, List


def rgb_to_plt(rgb_values: Tuple[float]) -> Tuple[float]:
    """Main function which performs the algorithm
    comparison.

    Parameters
    ----------
    rgb_values: Tuple[float]
        float of RGB values in format (r, g, b)

    Returns
    -------
    Tuple[float]
        returns a float of RGB values corrected for the matplotlib API

    """

    return tuple(s / 255.0 for s in rgb_values)


# Static variable containing the JAMA colors
JAMA_COLORS: List[tuple] = [((247, 148, 132),  # orange
                            (253, 203, 139)),  # orange (shade)
                            ((94, 131, 143),  # blue
                            (188, 208, 217)),  # blue (shade)
                            ((138, 129, 114),  # grey
                            (219, 212, 195)),
                            ((178, 71, 69),  # purple ish
                            (225, 180, 179)),
                            ((106, 101, 153),  # blue ish
                            (184, 182, 207))]

# Contains the string for the outcome variables
OUT_STRINGS: List[str] = ["",  # empty so we can use actual models (0 is not a model!)
                          "Septic shock",
                          "In-house mortality",
                          "1-month mortality"]

# Contains the string for the model datasets
MODEL_STRINGS: List[str] = ["",
                            "Lab",
                            "Lab + drugs + history",
                            "Lab + clinical"]
