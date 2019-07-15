"""
(Experimental) replacement for import/export functionality SAX
"""

# External dependencies
from __future__ import division, absolute_import, print_function
import os
import numpy as np
from xml.etree.ElementTree import iterparse, Element, ElementTree, SubElement
from copy import copy, deepcopy
from numbers import Number
from random import choice as randomchoice
from collections import MutableSequence

# Internal dependencies
from .parser import parse_path
from .svg_to_paths import (path2pathd, ellipse2pathd, line2pathd,
                           polyline2pathd, polygon2pathd, rect2pathd)
from .misctools import open_in_browser
from .path import Path


# To maintain forward/backward compatibility
try:
    str = basestring
except NameError:
    pass


NAME_SVG = "svg"
NAME_PATH = "path"
NAME_RECT = "rect"

ATTR_VERSION = "version"
VALUE_SVG_VERSION = "1.1"
ATTR_XMLNS = "xmlns"
VALUE_XMLNS = "http://www.w3.org/2000/svg"
ATTR_XMLNS_LINK = "xmlns:xlink"
VALUE_XLINK = "http://www.w3.org/1999/xlink"
ATTR_XMLNS_EV = "xmlns:ev"
VALUE_XMLNS_EV = "http://www.w3.org/2001/xml-events"
ATTR_WIDTH = "width"
ATTR_HEIGHT = "height"
ATTR_VIEWBOX = "viewBox"
ATTR_DATA = "d"
ATTR_FILL = "fill"
ATTR_STROKE = "stroke"
ATTR_STROKE_WIDTH = "stroke-width"
ATTR_TRANSFORM = "transform"
ATTR_CLASS = "class"
VALUE_NONE = "none"

ATTR_BACKGROUND_COLOR = "bg-color"


unique_object = Path()  # unique id object, don't use pls

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


HtmlColorsLowerCaseNames = [x.lower() for x in HtmlColors]


def random_color():
    return randomchoice(list(HtmlColors.keys()))


def new_style(fill='none', **kw):
    string = ''
    for key, value in kw.items():
        if key == 'width':
            key = 'stroke-width'
        string += key + ':' + str(value) + ';'
    return string


unique_id_counter = 1


_dot_and_attributes_keymap = {
    'classname': 'class',
    'width': 'stroke-width',
    'radius': 'r',
    'x': 'cx',
    'y': 'cy'
}

_text_and_attributes_keymap = {
    'classname': 'class',
    'width': 'stroke-width',
    'anything-that-ends-with-href': 'href'  # haha... this is just a mnemonic!
}

_path_and_attributes_keymap = {
    'classname': 'class',
    'width': 'stroke-width'
}

_use_and_attributes_keymap = {
    'classname': 'class',
    'width': 'stroke-width'
}


class GlorifiedDictionary():
    def __init__(self, **kw):
        for key, value in kw.items():
            self.__setitem__(key, value)

    def update(self, other):
        for key, value in other.items():
            self.__setitem__(key, value)

    def __contains__(self, key):
        if key == '_keymap':
            return '_keymap' in self.__dict__
        if key in self.__dict__['_keymap']:
            key = self.__dict__['_keymap'][key]
        return key in self.__dict__

    def __iter__(self):
        for key in self.__dict__:
            if key == '_keymap':
                continue
            yield key

    def items(self):
        for key, value in self.__dict__.items():
            if key == '_keymap':
                continue
            yield (key, value)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattr__(self, key, default_value=unique_object):
        if key == '__getstate__' or \
           key == '__setstate__':
            return super.__getattr__(key)
        return self.__getitem__(key, default_value)

    def __getitem__(self, key, default_value=unique_object):
        if key in self.__dict__['_keymap']:
            key = self.__dict__['_keymap'][key]
        if default_value is not unique_object:
            return self.__dict__.get(key, default_value)
        return self.__dict__.get(key)

    def __setitem__(self, key, value):
        if key in self._keymap:
            key = self._keymap[key]

        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass

        self.__dict__[key] = value

    def get(self, key, default_value=unique_object, **kw):
        return self.__getitem__(key, default_value=default_value, **kw)

    def __repr__(self):
        return self.__dict__.__repr__()

    def pop(self, key, default_value=unique_object):
        if key in self._keymap:
            key = self._keymap[key]
        if default_value is not unique_object:
            return self.__dict__.pop(key, default_value)
        return self.__dict__.pop(key)


class DotAndAttributes(GlorifiedDictionary):
    def __init__(self, **kw):
        self.__dict__['_keymap'] = _dot_and_attributes_keymap
        GlorifiedDictionary.__init__(self, **kw)

    def generate_Element(self):
        el = Element('circle')
        for key, value in self.items():
            if key in ['original_tag']:
                continue
            el.set(key, str(value))
        return el, None


class UseAndAttributes(GlorifiedDictionary):
    def __init__(self, **kw):
        self.__dict__['_keymap'] = _use_and_attributes_keymap
        GlorifiedDictionary.__init__(self, **kw)

    def flatten(self):
        raise NotImplementedError

    def instantiate(self, defs_list):
        transform = self.get('transform', '')
        x = self.get('x', 0)
        y = self.get('y', 0)
        if x or y:
            transform += f'translate({x}, {y})'

        ref = None
        for key in self:
            if key.endswith('href'):
                ref = self[key]
                if ref.startswith('#'):
                    ref = ref[1:]
                break
        if ref is None:
            raise ValueError("ref missing from 'use' element")

        for d in defs_list:
            if 'id' in d and d['id'] == ref:
                if 'transform' in d:
                    transform += d['transform']
                to_return = deepcopy(d)

                if transform != '':
                    to_return['transform'] = transform
                else:
                    to_return.pop('transform', None)

                return to_return, d

        raise ValueError(f"UseAndAttributes.instantiate could not find defs with id {ref}")

    def generate_Element(self):
        el = Element('use')
        for key, value in self.items():
            if key in ['original_tag']:
                continue
            el.set(key, str(value))
        return el, None


class TextAndAttributes(GlorifiedDictionary):
    def __init__(self, **kw):
        self.__dict__['_keymap'] = _text_and_attributes_keymap
        GlorifiedDictionary.__init__(self, **kw)

    def __setitem__(self, key, value):
        if key.endswith('href'):
            key = 'href'
        GlorifiedDictionary.__setitem__(self, key, value)

    def __getitem__(self, key, default_value=unique_object):
        if key.endswith('href'):
            key = 'href'
        return GlorifiedDictionary.__getitem__(self, key, default_value)

    def __str__(self):
        tmp = copy(self.__dict__)
        tmp.pop('_keymap')
        return tmp.__str__()

    def generate_Element(self, **kw):
        global unique_id_counter

        inner = el = Element('text')
        href = def_el = None

        for key in self:
            if key in ['href', 'text', 'path', 'original_tag']:
                continue
            el.set(key, str(self[key]))

        if 'href' in self:
            href = str(self['href'])
            if not href.startswith('#'):
                href = '#' + href

        if 'path' in self:
            if href is None:
                unique_id_counter += 1
                href = '#' + 'sax_auto_generated_id_' + str(unique_id_counter)

            def_el = Element('path')
            def_el.set('id', href[1:])

            if isinstance(self.path, str):
                if kw:
                    def_el.set('d', parse_path(self.path).d(**kw))
                else:
                    def_el.set('d', self.path)
            else:
                def_el.set('d', self.path.d(**kw))

        if href is not None:
            inner = SubElement(el, 'textPath')
            inner.set('xlink:href', href)

        if 'text' in self:
            inner.text = self['text']

        return el, def_el


class PathAndAttributes(GlorifiedDictionary):
    """
    Main element class for storing paths. Attributes: stroke, width
    (for stroke-width), fill, classname (for class) etc.

    It also has 'd' and 'object' fields. The former stores the string
    for the path, the latter the object representation. These two
    fields are kept consistent automatically. (Internally, the
    representation has no redundancy.)

    Has a 'flatten' method.

    Access the d-string in a particular format using the 'get' method
    and keyword options, as in:

        path_a_a.get('d', use_V_and_H=False, spacing_after_command='')

    Or like this:

        path_aa.get(
            'd',
            {
                'spacing_after_command': '',
                'spacing_within_coordinate': ' ',
                'use_V_and_H': False
            }
        )
    """

    def __init__(self, **kw):
        self.__dict__['_keymap'] = _path_and_attributes_keymap
        self._last_d_options = None
        GlorifiedDictionary.__init__(self, **kw)

    def __setitem__(self, key, value):
        # 'path' is resolved to either 'd' or 'object' (types checked
        # are checked below):
        if key == 'path':
            key = 'object'
            if isinstance(value, str):
                key = 'd'

        # pop object if given d-string:
        if key == 'd':
            if not isinstance(value, str):
                raise ValueError("d should be string")
            self.__dict__.pop('object', None)
            self._last_d_options = None

        # pop d-string if given object:
        if key == 'object':
            if not isinstance(value, Path):
                raise ValueError("object should be Path")
            self.__dict__.pop('d', None)

        GlorifiedDictionary.__setitem__(self, key, value)

    def __getitem__(self, key, default_value=unique_object, options={}, **kw):
        assert 'd' not in self or 'object' not in self

        if key == 'd':
            this_d_options = {}
            this_d_options.update(options)
            this_d_options.update(kw)

        if key == 'd' and 'd' in self:
            if this_d_options == self._last_d_options:
                return self.__dict__['d']
            self.__dict__['object'] = parse_path(self.__dict__['d'])
            self.__dict__.pop('d')
            return self.object.d(options=this_d_options)

        if key == 'd' and 'object' in self:
            return self.object.d(options=this_d_options)

        if key == 'object' and 'd' in self:
            self.__dict__['object'] = parse_path(self.__dict__['d'])
            self.__dict__.pop('d')

        return GlorifiedDictionary.__getitem__(self, key, default_value)

    def __str__(self):
        tmp = copy(self.__dict__)
        tmp.pop('_last_d_options')
        tmp.pop('_keymap')
        if 'object' in self:
            tmp_use_oneline = self.object._repr_use_oneline
            self.object._repr_use_oneline = True
        to_return = tmp.__repr__()
        if 'object' in self:
            self.object._repr_use_oneline = tmp_use_oneline
        return to_return

    def flatten(self):
        if 'transform' in self:
            self.object = self.object.transformed(self['transform'])
            self.__dict__.pop('transform')

    def flattened_path(self):
        if 'transform' in self:
            return self.object.transformed(self.transform)
        return self.object

    def generate_Element(self, **kw):
        el = Element('path')

        d = self.get('d', default_value=None, **kw)
        if d is not None:
            el.set('d', d)

        for key, item in self.items():
            if key in ['original_tag', 'd', 'object', '_last_d_options']:
                continue
            el.set(key, str(item))

        return el, None


class SaxDocument(MutableSequence):
    def __init__(self, filename=None):
        """
        A container for a SAX SVG light tree objects document.

        This class provides functions for extracting SVG data into Path objects.

        Args:
            filename(str): The filename of the SVG file
        """
        self.root_attrs = {}
        self.defs = []
        self.elements = []
        self.styles = {}

        # off we go
        if filename is not None:
            self.sax_parse(filename)

    def __len__(self):  # (MutableSequence abstract class)
        return len(self.elements)

    def __iter__(self):  # (MutableSequence abstract class)
        return self.elements.__iter__()

    def __getitem__(self, key_or_index, default_value=unique_object):  # (MutableSequence abstract class)
        if isinstance(key_or_index, str):
            if key_or_index == 'viewbox':
                key_or_index = 'viewBox'
            if default_value is not None:
                return self.root_attrs.get(key_or_index, default_value)
            return self.root_attrs[key_or_index]

        if isinstance(key_or_index, Number):
            return self.elements[key_or_index]

        raise ValueError("bad key_or_index")

    def __setitem__(self, key_or_index, value):  # (MutableSequence abstract class)
        if isinstance(key_or_index, str):
            if key_or_index == 'viewbox':
                key_or_index = 'viewBox'
            self.root_attrs[key_or_index] = value
            return

        if isinstance(key_or_index, Number):
            if not isinstance(value, PathAndAttributes):
                raise ValueError("value must be PathAndAttributes instance")
            self.elements[key_or_index] = value

        raise ValueError("bad key_or_index")

    def __contains__(self, key):
        if isinstance(key, str):
            if key == 'viewbox':
                assert 'viewbox' not in self.root_attrs
                key = 'viewBox'
            return key in self.root_attrs
        raise ValueError("expecting a string")

    def __delitem__(self, index):  # (MutableSequence abstract class)
        del self.elements[index]

    def insert(self, index, value):  # (MutableSequence abstract class)
        if not isinstance(value, PathAndAttributes) and \
           not isinstance(value, TextAndAttributes) and \
           not isinstance(value, DotAndAttributes):
            raise ValueError("unexpected type of element")
        self.elements.insert(index, value)

    @property
    def width(self):
        return self.__getitem__('width')

    @width.setter
    def width(self, value):
        self.__setitem__('width', value)

    @property
    def height(self):
        return self.__getitem__('height')

    @height.setter
    def height(self, value):
        self.__setitem__('height', value)

    @property
    def viewBox(self):
        return self.__getitem__('viewBox')

    @viewBox.setter
    def viewBox(self, value):
        self.__setitem__('viewBox', value)

    @property
    def viewbox(self):
        return self.__getitem__('viewBox')

    @viewbox.setter
    def viewbox(self, value):
        self.__setitem__('viewBox', value)

    def append_all(self, *things):
        self.extend(things)

    def sax_parse(self, filename):
        """
        This function reinitializes the SaxDocument to the contents
        of filename.
        """
        def pop_attributes_for(name):
            nonlocal attributes
            to_pop = {
                'circle': ['cx', 'cy', 'r'],
                'line': ['x1', 'y1', 'x2', 'y2'],
                'polyline': ['points'],
                'polygon': ['points'],
                'rect': ['x', 'y', 'width', 'height']
            }
            for z in to_pop[name]:
                try:
                    attributes.pop(z)
                except KeyError:
                    print("(warning:", name, "missing", z, "attribute)")

        self.root_attrs = {}
        self.defs = []
        self.elements = []
        self.styles = {}

        # remember location of original svg file (why?)
        self.location = filename
        if os.path.dirname(filename) == '':
            self.location = os.path.join(os.getcwd(), filename)

        # local bookkeeping variables
        stack = []
        attributes = {}  # Ordinary dictionary for the root_attrs
        cumulative_transform = ''  # This remains a string the whole time
        inside_defs = False

        for event, elem in iterparse(filename, events=('start', 'end')):
            if event == 'start':
                stack.append(
                    (attributes, cumulative_transform, inside_defs)
                )
                attributes = copy(attributes)

                assert 'd' not in attributes

                attrs = elem.attrib  # we won't ever see elem again...

                if 'style' in attrs:
                    for equate in attrs['style'].split(';'):
                        equal_item = equate.split(':')
                        attributes[
                            equal_item[0].strip()
                        ] = equal_item[1].strip()
                    attrs.pop('style')  # ...so we can modify its attrs

                if 'transform' in attrs:
                    if len(cumulative_transform) > 0:
                        cumulative_transform += ' '
                    cumulative_transform += attrs.pop('transform')

                assert 'transform' not in attrs

                if 'http://www.w3.org/2000/svg' in elem.tag:
                    name = elem.tag[28:]
                else:
                    name = elem.tag

                assert 'd' not in attributes

                if name == 'style':
                    # impromptu css styles parsing
                    def get_next_keys(text):
                        keys, rest = text.split('{', 1)
                        keys = [
                            k.strip(' \n')
                            for k in keys.split() if k.strip(' \n') != ''
                        ]
                        return keys, rest

                    def get_next_style(text):
                        text, rest = text.split('}', 1)
                        text = text.strip(' \n')
                        text = ''.join(text.split())  # not sure if well-advised
                        return text, rest.strip(' \n')

                    text = elem.text
                    while text != '':
                        keys, text = get_next_keys(text)
                        style, text = get_next_style(text)
                        for k in keys:
                            self.styles[k] = style

                assert 'd' not in attributes
                attributes.update(attrs)

                if cumulative_transform != '':
                    attributes['transform'] = cumulative_transform

                # we're not very high tech... we expect the svg element to come
                # first, and no nested svgs:
                assert ('svg' == name) == (len(stack) == 1)

                if 'svg' == name:
                    # We store self.root_attrs as an ordinary dictionary, while
                    # switching to PathAndAttributes() dictionaries for the rest
                    # of the document:
                    self.root_attrs = attributes
                    attributes = {}  # start afresh
                    continue

                if 'defs' == name:
                    inside_defs = True
                    continue

                elif 'g' == name:
                    continue

                elif 'path' == name:
                    attributes['d'] = path2pathd(attrs)

                elif 'circle' == name:
                    attributes['d'] = ellipse2pathd(attrs)
                    pop_attributes_for('circle')

                elif 'ellipse' == name:
                    attributes['d'] = ellipse2pathd(attrs)
                    pop_attributes_for('ellipse')

                elif 'line' == name:
                    attributes['d'] = line2pathd(attrs)
                    pop_attributes_for('line')

                elif 'polyline' == name:
                    attributes['d'] = polyline2pathd(attrs['points'])
                    pop_attributes_for('polyline')

                elif 'polygon' == name:
                    attributes['d'] = polygon2pathd(attrs['points'])
                    pop_attributes_for('polygon')

                elif 'rect' == name:
                    try:
                        attributes['d'] = rect2pathd(attrs)
                        pop_attributes('rect')

                    except ValueError:
                        print("(note: ignoring a <rect> element with non-float width or height)")  # who? what? when does this happen?
                        continue

                elif 'text' == name and len(elem) == 0:
                    assert 'd' not in attributes

                elif 'textPath' == name and len(elem) == 0:
                    assert 'd' not in attributes

                elif 'use' == name:
                    pass

                else:
                    continue

                attributes['original_tag'] = name

                if 'd' in attributes:
                    if inside_defs:
                        self.defs.append(PathAndAttributes(**attributes))

                    else:
                        self.elements.append(PathAndAttributes(**attributes))

                elif 'text' == name or 'textPath' == name:
                    self.elements.append(TextAndAttributes(**attributes))
                    self.elements[-1].text = elem.text

                elif 'use' == name:
                    self.elements.append(UseAndAttributes(**attributes))

                else:
                    assert False, name

            else:
                attributes, cumulative_transform, inside_defs = stack.pop()

    def flatten(self):
        for element in self.elements:
            if isinstance(element, PathAndAttributes):
                element.flatten()
        return self

    def style_value(self, stylename, propertyname):
        if stylename not in self.styles:
            return None
        stylestring = self.styles[stylename]
        pairs = stylestring.split(';')
        for p in pairs:
            equate = [z.strip() for z in p.split(':')]
            if equate[0] == propertyname:
                return equate[1]
        return None

    def style_aware_value(self, attributes4path, key):
        """
        if attributes4path has a class name(s), searches the .styles
        dictionary for matching style names that contain property 'key',
        and returns the first match

        """
        if 'class' in attributes4path:
            mapped_key = key

            if key == 'width':
                mapped_key = 'stroke-width'

            names = ['.' + n for n in attributes4path.classname.split() if n != '']

            for name in names:
                val = self.style_value(name, mapped_key)
                if val is not None:
                    if mapped_key == 'stroke-width':
                        return float(val)
                    return val

        if key in attributes4path:
            return attributes4path[key]

        return None

    def reset_viewbox(self, absolute_margins=[], percentage_margins=[], with_strokes=False):
        """
        Sets the self.root_attrs['viewBox'] following path data
        """
        xmin, ymin = np.inf, np.inf
        xmax, ymax = -np.inf, -np.inf

        assert all(abs(x) == np.inf for x in [xmin, xmax, ymin, ymax])

        if not self.elements:
            raise ValueError("cannot set viewBox on empty elements list")

        for elem in self.elements:
            if isinstance(elem, DotAndAttributes):
                elem = elem.convert_to_PathAndAttributes()

            if not isinstance(elem, PathAndAttributes):
                continue

            path = None

            if with_strokes:
                width = self.style_aware_value(elem, 'width')
                transform = self.style_aware_value(elem, 'transform')
                if width is not None:
                    path = elem.object.stroke(width)
                    if transform is not None:
                        path = path.transformed(transform)

            if path is None:
                path = elem.flattened_path()

            xi, xa, yi, ya = path.bbox()
            xmin = min(xi, xmin)
            xmax = max(xa, xmax)
            ymin = min(yi, ymin)
            ymax = max(ya, ymax)

        assert all(abs(x) != np.inf for x in [xmin, xmax, ymin, ymax])

        if isinstance(absolute_margins, Number):
            absolute_margins = [absolute_margins]

        if isinstance(percentage_margins, Number):
            percentage_margins = [percentage_margins]

        for a in [absolute_margins, percentage_margins]:
            if not isinstance(a, list):
                raise ValueError("please provide margins as lists")
            if len(a) not in [0, 1, 2, 4]:
                raise ValueError("margin should be 0, 1, 2 or 4 values")
            if any(x < 0 for x in a):
                raise ValueError("expecting nonnegative margin values")
            if len(a) == 0:
                a.append(0)
            if len(a) == 1:
                a.append(a[0])
            if len(a) == 2:
                a.append(a[0])
                a.append(a[1])
            assert len(a) == 4

        if any(x >= 1 for x in percentage_margins):
            print("\nwarning: large percentage_margin in set_or_reset_viewbox")
            print("expected value in the range 0-1; will truncate to 1\n")

        for i in range(4):
            if percentage_margins[i] >= 1:
                percentage_margins[i] = 1

        width, height = xmax - xmin, ymax - ymin
        base = min(width, height)

        final_margins = [0, 0, 0, 0]
        for i in range(4):
            final_margins[i] = \
                absolute_margins[i] + \
                base * percentage_margins[i]

        # margins go in css-order top/right/bottom/left
        TOP = 0
        RIGHT = 1
        BOTTOM = 2
        LEFT = 3

        final_x = xmin - final_margins[LEFT]
        final_y = ymin - final_margins[TOP]
        final_width = width + final_margins[LEFT] + final_margins[RIGHT]
        final_height = height + final_margins[TOP] + final_margins[BOTTOM]

        self.viewbox = \
            str(final_x) + " " + \
            str(final_y) + " " + \
            str(final_width) + " " + \
            str(final_height)

    def viewbox_accessor(self, field):
        if field.lower() not in ['x', 'y', 'width', 'height']:
            raise ValueError("invalid viewbox field")
        if 'viewbox' not in self:  # note: viewbox -> viewBox automatically
            raise ValueError("viewbox missing")
        [x, y, width, height] = self.viewbox.split(' ')
        return eval("float(" + field.lower() + ")")

    def parse_css_measure(self, quantity):
        try:
            return float(quantity), None
        except ValueError:
            pass
        assert isinstance(quantity, str)
        quantity = quantity.strip()
        for units in ['%', 'cm', 'em', 'ex', 'in', 'mm', 'pc', 'pt', 'px']:
            if quantity.endswith(units):
                try:
                    f = float(quantity[:-len(units)])
                    return f, units
                except ValueError:
                    raise ValueError("unable to parse css measure")

    def compose_css_measure(self, amount, units):
        if not isinstance(amount, Number):
            raise ValueError("expecting number")
        if units not in [None, '%', 'cm', 'em', 'ex', 'in', 'mm', 'pc', 'pt', 'px']:
            raise ValueError("unexpected units")
        if units is not None:
            return str(amount) + units
        return str(amount)

    def set_width_from_height(self):
        if 'height' not in self:
            raise ValueError("height missing")

        w = self.viewbox_accessor('width')
        h = self.viewbox_accessor('height')

        if h < 0.001:
            raise ValueError("small or negative height")

        H, units = self.parse_css_measure(self.height)

        if units == '%':
            print("warning: \% units in set_width_from_height")

        W = H * w / h
        self.width = self.compose_css_measure(W, units)

    def set_height_from_width(self):
        if 'width' not in self:
            raise ValueError("width missing")

        w = self.viewbox_accessor('width')
        h = self.viewbox_accessor('height')

        if h < 0.001:
            raise ValueError("small or negative width")

        W, units = self.parse_css_measure(self.width)

        if units == '%':
            print("warning: \% units in set_height_from_width")

        H = W * h / w
        self.height = self.compose_css_measure(H, units)

    def collect_classnames(self, prepend_dot=True):
        names = set()
        for p in self.elements:
            if 'class' in p:
                for name in p['class'].split():
                    if name != '':
                        if prepend_dot:
                            name = '.' + name
                        names.add(name)
        return list(names)

    def set_background_color(self, color):
        if not isinstance(color, str):
            raise ValueError("expecting color to be a string")
        self.root_attrs[ATTR_BACKGROUND_COLOR] = color

    def generate_dom(self, **kw):
        root = Element(NAME_SVG)
        root.set(ATTR_VERSION, VALUE_SVG_VERSION)
        root.set(ATTR_XMLNS, VALUE_XMLNS)
        root.set(ATTR_XMLNS_LINK, VALUE_XLINK)
        root.set(ATTR_XMLNS_EV, VALUE_XMLNS_EV)
        width = self.root_attrs.get(ATTR_WIDTH, None)
        height = self.root_attrs.get(ATTR_HEIGHT, None)

        if width is not None:
            root.set('width', str(width))

        if height is not None:
            root.set('height', str(height))

        if 'viewbox' in self:
            root.set('viewBox', self['viewbox'])

        if len(self.defs) > 0:
            defs = SubElement(root, 'defs')
            for z in self.defs:
                main, aux = z.generate_Element()
                assert aux is None
                defs.append(main)

        if len(self.styles) > 0:
            styles = SubElement(root, 'style')
            text = ''
            for style, val in self.styles.items():
                text += style + ' {' + val + '} '
            styles.text = text

        bgcolor = self.root_attrs.get(ATTR_BACKGROUND_COLOR, None)
        if bgcolor is not None:
            if 'viewbox' not in self:
                raise ValueError("viewbox missing; use reset_viewbox()?")
            bgcolor_rect = SubElement(root, 'rect')
            bgcolor_rect.set("fill", bgcolor)
            # bgcolor_rect.set("width", "100%")
            # bgcolor_rect.set("height", "100%")
            for z in ['x', 'y', 'width', 'height']:
                bgcolor_rect.set(z, str(self.viewbox_accessor(z)))

        for z in self.elements:
            main, aux = z.generate_Element()
            root.append(main)
            if aux is not None:
                defs.append(aux)

        return ElementTree(root)

    def save(self, filename='display_temp.svg', **kw):
        with open(filename, 'wb') as output_svg:
            dom_tree = self.generate_dom(**kw)
            dom_tree.write(output_svg)

    def display(self, filename='display_temp.svg'):
        """Displays/opens the doc using the OS's default application."""
        self.save(filename)
        open_in_browser(filename)
