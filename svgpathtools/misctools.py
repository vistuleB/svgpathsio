"""This submodule contains miscellaneous tools that are used internally, but
aren't specific to SVGs or related mathematical objects."""

# External dependencies:
from __future__ import division, absolute_import, print_function
import os
import sys
import webbrowser


# stackoverflow.com/questions/214359/converting-hex-color-to-rgb-and-vice-versa
def hex2rgb(value):
    """Converts a hexadeximal color string to an RGB 3-tuple

    EXAMPLE
    -------
    >>> hex2rgb('#0000FF')
    (0, 0, 255)
    """
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i: i + lv // 3], 16) for i in range(0, lv, lv // 3))


def _amalgamate_args(*args):
    a = []
    for z in args:
        try:
            a.extend(z)
        except TypeError:
            a.append(z)
    return a


def _ints_array_to_hex(ints, upper):
    ans = '#' if len(ints) > 1 else ''
    for x in ints:
        ans += '%02x' % x
    return ans.upper() if upper else ans


# stackoverflow.com/questions/214359/converting-hex-color-to-rgb-and-vice-versa
def rgb2hex(*args, upper=True):
    """Converts an RGB 1-, 3- or 4-tuple to a hexadeximal color string;
    only prends the '#' if the tuple has length > 1

    EXAMPLES
    --------
    >>> rgb2hex((0,0,255))
    '#0000FF'
    >>> rgb2hex(0, 0, 255)
    '#0000FF'
    >>> rgb2hex((0, 0, 255, 255))
    '#0000FFFF'
    >>> rgb2hex((0, 0, 255), 255)
    '#0000FFFF'
    >>> rgb2hex(0)
    '00'
    """
    ints = _amalgamate_args(*args)

    if not all(isinstance(x, int)  and 0 <= x <= 255 for x in ints):
        raise ValueError("expecting integers in range 0-255")

    return _ints_array_to_hex(ints, upper)


# stackoverflow.com/questions/214359/converting-hex-color-to-rgb-and-vice-versa
def rgb012hex(*args, upper=True):
    """Like rgb2hex, but assumes 0-1 float values instead of 0-255 integers.

    EXAMPLE
    -------
    >>> rgb2hex(0, 0, 1)
    '#0000FF'
    """
    ints = [round(255 * x) for x in _amalgamate_args(*args)]

    if not all(0 <= x <= 255 for x in ints):
        raise ValueError("expecting floats in range 0-1")

    return _ints_array_to_hex(ints, upper)


def rgb_affine_combination(c, rgb1, rgb2):
    assert len(rgb1) == len(rgb2)
    return [max(0, min(255, int(c * rgb1[i] + (1 - c) * rgb2[i]))) for i in range(len(rgb1))]


def isclose(a, b, rtol=1e-5, atol=1e-8):
    """This is essentially np.isclose, but slightly faster."""
    return abs(a - b) < (atol + rtol * abs(b))


def open_in_browser(file_location):
    """Attempt to open file located at file_location in the default web
    browser."""

    #  For some reason webbrowser.get.().open() wants an absolute path:
    file_location = os.path.abspath(file_location)
    if not os.path.isfile(file_location):
        raise IOError("\n\nFile not found.")

    #  For some reason OSX requires this adjustment (tested on 10.10.4, 10.12.6)
    if sys.platform == "darwin":
        file_location = "file:///" + file_location

    new = 0  # open in a new tab, if possible
    try:
        webbrowser.get().open(file_location, new=new)
    except webbrowser.Error:
        print("got an error")


BugException = Exception("This code should never be reached.  You've found a "
                         "bug.  Please submit an issue to \n"
                         "https://github.com/mathandy/svgpathtools/issues"
                         "\nwith an easily reproducible example.")
