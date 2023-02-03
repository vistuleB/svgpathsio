"""This submodule contains miscellaneous tools that are used internally, but
aren't specific to SVGs or related mathematical objects."""

# External dependencies:
from __future__ import division, absolute_import, print_function
from numbers import Real, Complex
import numpy as np
import os
import sys
import webbrowser
import random


HtmlColors = {
    'AliceBlue': '#F0F8FF',
    'AntiqueWhite': '#FAEBD7',
    'Aqua': '#00FFFF',
    'Aquamarine': '#7FFFD4',
    'Azure': '#F0FFFF',
    'Beige': '#F5F5DC',
    'Bisque': '#FFE4C4',
    'Black': '#000000',
    'BlanchedAlmond': '#FFEBCD',
    'Blue': '#0000FF',
    'BlueViolet': '#8A2BE2',
    'Brown': '#A52A2A',
    'BurlyWood': '#DEB887',
    'CadetBlue': '#5F9EA0',
    'Chartreuse': '#7FFF00',
    'Chocolate': '#D2691E',
    'Coral': '#FF7F50',
    'CornflowerBlue': '#6495ED',
    'Cornsilk': '#FFF8DC',
    'Crimson': '#DC143C',
    'Cyan': '#00FFFF',
    'DarkBlue': '#00008B',
    'DarkCyan': '#008B8B',
    'DarkGoldenRod': '#B8860B',
    'DarkGray': '#A9A9A9',
    'DarkGrey': '#A9A9A9',
    'DarkGreen': '#006400',
    'DarkKhaki': '#BDB76B',
    'DarkMagenta': '#8B008B',
    'DarkOliveGreen': '#556B2F',
    'DarkOrange': '#FF8C00',
    'DarkOrchid': '#9932CC',
    'DarkRed': '#8B0000',
    'DarkSalmon': '#E9967A',
    'DarkSeaGreen': '#8FBC8F',
    'DarkSlateBlue': '#483D8B',
    'DarkSlateGray': '#2F4F4F',
    'DarkSlateGrey': '#2F4F4F',
    'DarkTurquoise': '#00CED1',
    'DarkViolet': '#9400D3',
    'DeepPink': '#FF1493',
    'DeepSkyBlue': '#00BFFF',
    'DimGray': '#696969',
    'DimGrey': '#696969',
    'DodgerBlue': '#1E90FF',
    'FireBrick': '#B22222',
    'FloralWhite': '#FFFAF0',
    'ForestGreen': '#228B22',
    'Fuchsia': '#FF00FF',
    'Gainsboro': '#DCDCDC',
    'GhostWhite': '#F8F8FF',
    'Gold': '#FFD700',
    'GoldenRod': '#DAA520',
    'Gray': '#808080',
    'Grey': '#808080',
    'Green': '#008000',
    'GreenYellow': '#ADFF2F',
    'HoneyDew': '#F0FFF0',
    'HotPink': '#FF69B4',
    'IndianRed': '#CD5C5C',
    'Indigo': '#4B0082',
    'Ivory': '#FFFFF0',
    'Khaki': '#F0E68C',
    'Lavender': '#E6E6FA',
    'LavenderBlush': '#FFF0F5',
    'LawnGreen': '#7CFC00',
    'LemonChiffon': '#FFFACD',
    'LightBlue': '#ADD8E6',
    'LightCoral': '#F08080',
    'LightCyan': '#E0FFFF',
    'LightGoldenRodYellow': '#FAFAD2',
    'LightGray': '#D3D3D3',
    'LightGrey': '#D3D3D3',
    'LightGreen': '#90EE90',
    'LightPink': '#FFB6C1',
    'LightSalmon': '#FFA07A',
    'LightSeaGreen': '#20B2AA',
    'LightSkyBlue': '#87CEFA',
    'LightSlateGray': '#778899',
    'LightSlateGrey': '#778899',
    'LightSteelBlue': '#B0C4DE',
    'LightYellow': '#FFFFE0',
    'Lime': '#00FF00',
    'LimeGreen': '#32CD32',
    'Linen': '#FAF0E6',
    'Magenta': '#FF00FF',
    'Maroon': '#800000',
    'MediumAquaMarine': '#66CDAA',
    'MediumBlue': '#0000CD',
    'MediumOrchid': '#BA55D3',
    'MediumPurple': '#9370DB',
    'MediumSeaGreen': '#3CB371',
    'MediumSlateBlue': '#7B68EE',
    'MediumSpringGreen': '#00FA9A',
    'MediumTurquoise': '#48D1CC',
    'MediumVioletRed': '#C71585',
    'MidnightBlue': '#191970',
    'MintCream': '#F5FFFA',
    'MistyRose': '#FFE4E1',
    'Moccasin': '#FFE4B5',
    'NavajoWhite': '#FFDEAD',
    'Navy': '#000080',
    'OldLace': '#FDF5E6',
    'Olive': '#808000',
    'OliveDrab': '#6B8E23',
    'Orange': '#FFA500',
    'OrangeRed': '#FF4500',
    'Orchid': '#DA70D6',
    'PaleGoldenRod': '#EEE8AA',
    'PaleGreen': '#98FB98',
    'PaleTurquoise': '#AFEEEE',
    'PaleVioletRed': '#DB7093',
    'PapayaWhip': '#FFEFD5',
    'PeachPuff': '#FFDAB9',
    'Peru': '#CD853F',
    'Pink': '#FFC0CB',
    'Plum': '#DDA0DD',
    'PowderBlue': '#B0E0E6',
    'Purple': '#800080',
    'RebeccaPurple': '#663399',
    'Red': '#FF0000',
    'RosyBrown': '#BC8F8F',
    'RoyalBlue': '#4169E1',
    'SaddleBrown': '#8B4513',
    'Salmon': '#FA8072',
    'SandyBrown': '#F4A460',
    'SeaGreen': '#2E8B57',
    'SeaShell': '#FFF5EE',
    'Sienna': '#A0522D',
    'Silver': '#C0C0C0',
    'SkyBlue': '#87CEEB',
    'SlateBlue': '#6A5ACD',
    'SlateGray': '#708090',
    'SlateGrey': '#708090',
    'Snow': '#FFFAFA',
    'SpringGreen': '#00FF7F',
    'SteelBlue': '#4682B4',
    'Tan': '#D2B48C',
    'Teal': '#008080',
    'Thistle': '#D8BFD8',
    'Tomato': '#FF6347',
    'Turquoise': '#40E0D0',
    'Violet': '#EE82EE',
    'Wheat': '#F5DEB3',
    'White': '#FFFFFF',
    'WhiteSmoke': '#F5F5F5',
    'Yellow': '#FFFF00',
    'YellowGreen': '#9ACD32'
}


HtmlColorsLowerCase = {}
for z, item in HtmlColors.items():
    HtmlColorsLowerCase[z.lower()] = item


# HtmlColorsLowerCaseNames = [x.lower() for x in HtmlColors]


class RgbaAncestor():
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], str):
            string = args[0].lower()

            if string in HtmlColorsLowerCase:
                string = HtmlColorsLowerCase[string]

            assert string.startswith('#')
            if len(string) in [4, 7]:
                r, g, b = hex2rgb(string)
                a = 255

            elif len(string) in [5, 9]:
                r, g, b, a = hex2rgb(string)

            else:
                assert False

        elif len(args) == 3 and all(isinstance(x, Real) for x in args):
            r, g, b = args
            a = 255

        elif len(args) == 4 and all(isinstance(x, Real) for x in args):
            r, g, b, a = args

        elif len(args) == 1 and isinstance(args[0], Rgba):
            o = args[0]
            r, g, b, a = o.r, o.g, o.b, o.a

        else:
            print("args:", args)
            assert False

        assert all(-255 <= x <= 255 for x in [r, g, b, a])

        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def __mul__(self, scalar):
        assert -1 <= scalar <= 1
        return RgbaDif(scalar * self.r,
                       scalar * self.g,
                       scalar * self.b,
                       scalar * self.a)

    def __rmul__(self, scalar):
        assert -1 <= scalar <= 1
        return RgbaDif(scalar * self.r,
                       scalar * self.g,
                       scalar * self.b,
                       scalar * self.a)


class RgbaDif(RgbaAncestor):
    def __init__(self, r, g, b, a):
        super().__init__(r, g, b, a)

    def __repr__(self):
        return f'RgbaDif({self.r}, {self.g}, {self.b}, {self.a})'


class Rgba(RgbaAncestor):
    def __init__(self, *args):
        super().__init__(*args)

        assert all(x >= 0 for x in self.fields())

    def fields(self):
        return [self.r, self.g, self.b, self.a]

    def __add__(self, other):
        assert isinstance(other, RgbaDif)

        return Rgba(self.r + other.r,
                    self.g + other.g,
                    self.b + other.b,
                    self.a + other.a)

    def __sub__(self, other):
        assert isinstance(other, Rgba)

        return RgbaDif(self.r - other.r,
                       self.g - other.g,
                       self.b - other.b,
                       self.a - other.a)

    def __repr__(self):
        return f'Rgba({self.r}, {self.g}, {self.b}, {self.a})'

    def __str__(self):
        string = rgb2hex(self.r, self.g, self.b, self.a).lower()
        if string.endswith('ff'):
            string = string[:-2]
            assert len(string) == 7
        if all(string[i] == string[i + 1] for i in range(1, len(string), 2)):
            string = '#' + ''.join(string[i] for i in range(1, len(string), 2))
        return string

    def rgb_str(self):
        string = rgb2hex(self.r, self.g, self.b).lower()
        if all(string[i] == string[i + 1] for i in range(1, len(string), 2)):
            string = '#' + ''.join(string[i] for i in range(1, len(string), 2))
        return string

    def change_a(self, new_value):
        assert 0 <= new_value <= 255
        self.a = new_value
        return self

    def changed_a(self, new_value):
        return Rgba(self.r, self.g, self.b, new_value)

    def tweaked(self, r=None, g=None, b=None, a=None):
        return Rgba(
            self.r if r is None else r,
            self.g if g is None else g,
            self.b if b is None else b,
            self.a if a is None else a
        )

    def with_opacity(self, amt):
        assert 0 <= amt <= 1
        return Rgba(self.r, self.g, self.b, amt * 255)

    def as_stop(self):
        return f'stop-color={self.rgb_str()}.stop-opacity={(self.a / 255):.2f}'


def is_css_color(string):
    assert isinstance(string, str)

    if string.lower() in HtmlColorsLowerCase:
        return True

    if string.startswith('#'):
        if not all(x in '0123456789abcdef' for x in string[1:].lower()):
            return False
        return len(string) in [4, 5, 7, 9]

    if string.startswith('rgb('):
        if not string.endswith(')'):
            return False
        pieces = string[4:-1].replace(',', ' ').split()
        if len(pieces) != 3:
            return False
        for p in pieces:
            if '.' in p or 'e' in p or 'E' in p:
                return False
            try:
                q = int(p)
                if not 0 <= q <= 255:
                    return False
            except ValueError:
                return False
        return True

    return False


def random_color():
    return random.choice(list(HtmlColors.keys()))


def to_decimals(value, decimals):
    # it seems that the threshold at which python
    # chooses to print large floats representing
    # integers in scientific notation is 10^16, so
    # we'll use the same here (i.e., avoiding converting
    # something like 10^16 to an int, otherwise it will
    # print as 10000000000000000 instead of as 10^16)
    if int(value) == value and abs(value) < 1e16:
        return str(int(value))

    if decimals is None:
        return str(value)

    string = f'{value:.{decimals}f}'
    assert '.' in string
    
    while string.endswith('0'):
        string = string[:-1]

    if string.endswith('.'):
        string = string[:-1]

    assert len(string) > 0
    return string


def int_else_float(z):
    assert isinstance(z, Real)
    return int(z) if int(z) == z else z


def real_numbers_in(thing):
    y = None

    def last_minute_changes():
        to_return_x = int_else_float(x)
        to_return_y = y if y is None else int_else_float(y)
        return to_return_x, to_return_y

    if isinstance(thing, str):
        try:
            x = float(thing)
            return last_minute_changes()

        except ValueError:
            pass

        try:
            z = complex(thing)
            x = z.real
            y = z.imag
            return last_minute_changes()

        except ValueError:
            pass

        if thing.endswith('deg'):
            try:
                x = float(thing[:-3])
                return last_minute_changes()

            except ValueError:
                raise

        if thing.endswith('rad'):
            try:
                x = float(thing[:-3]) * 180.0 / np.pi
                return last_minute_changes()

            except ValueError:
                raise

        raise ValueError

    if isinstance(thing, Real):
        x = float(thing)
        return last_minute_changes()

    if isinstance(thing, Complex):
        x = float(thing.real)
        y = float(thing.imag)
        return last_minute_changes()

    if isinstance(thing, bool):
        x = float(thing)
        return last_minute_changes()

    if isinstance(thing, np.bool_):
        x = float(thing)
        return last_minute_changes()

    try:
        x = float(thing.x)
        y = float(thing.y)
        return last_minute_changes()

    except AttributeError:
        pass

    if isinstance(thing, list):
        if len(thing) not in [1, 2]:
            raise ValueError

        try:
            x = float(thing[0])
            y = float(thing[1]) if len(thing) > 1 else None
            return last_minute_changes()

        except TypeError:
            raise ValueError

    if isinstance(thing, np.ndarray):
        if thing.shape != (2, 1) and \
           (thing.shape != (3, 1) or thing[2, 0] != 1):
            raise ValueError

        x = float(thing[0, 0])
        y = float(thing[1, 0])
        return last_minute_changes()

    try:
        try:
            x = float(thing['x'])
            y = float(thing['y'])
            return last_minute_changes()

        except TypeError:
            raise ValueError

    except KeyError:
        raise ValueError

    assert False


def real_numbers_in_iterator(*args):
    for a in args:
        x, y = real_numbers_in(a)
        yield x
        if y is not None:
            yield y


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
    chunk_size = lv // 3
    if chunk_size not in [1, 2]:
        raise ValueError("bad length in hex2rgb")
    addendum = 0 if chunk_size == 2 else 16
    values = tuple(int(value[i: i + chunk_size], 16) for i in range(0, lv, chunk_size))
    return tuple(x + addendum * x for x in values)


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
    ints = [round(x) for x in _amalgamate_args(*args)]

    if not all(isinstance(x, int) and 0 <= x <= 255 for x in ints):
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
        print("u were using this file_location:", file_location)
        raise IOError("\n\nFile not found.")

    #  For some reason OSX requires this adjustment (tested on 10.10.4, 10.12.6)
    if sys.platform == "darwin":
        file_location = "file:///" + file_location

    new = 0  # open in a new tab, if possible
    try:
        webbrowser.get().open(file_location, new=new)
    except webbrowser.Error:
        print("got an error")


BugException = Exception("This code should never be reached. You've found a bug.")

