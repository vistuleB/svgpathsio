"""This submodule contains the path_parse() function used to convert SVG path
element d-strings into svgpathtools Path objects.
Note: This file was taken (nearly) as is from the svg.path module (v 2.0)."""

# External dependencies
from __future__ import division, absolute_import, print_function
from .misctools import real_numbers_in, to_decimals
from numbers    import Real
import re

# Internal dependencies
from .path import Path, Subpath, Line, QuadraticBezier, CubicBezier, Arc


COMMANDS = set('MmZzLlHhVvCcSsQqTtAa')
UPPERCASE = set('MZLHVCSQTA')

COMMAND_RE = re.compile(r"([MmZzLlHhVvCcSsQqTtAa])")
FLOAT_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def _tokenize_path_string(pathdef):
    for x in COMMAND_RE.split(pathdef):
        if x in COMMANDS:
            yield x

        for token in FLOAT_RE.findall(x):
            yield float(token)


def _unpack_tokens(pathdef):
    for token in pathdef:
        if isinstance(token, str):
            if token not in COMMANDS:
                print("token", token)
                raise ValueError("unrecognized string token in svgpathtools.parser._unpack_tokens")
            yield token
            continue

        x, y = real_numbers_in(token)
        yield x
        if y is not None:
            yield y


def generate_path(*args, decimals=None):
    if len(args) == 0:
        raise ValueError("empty args in generate_path")

    if len(args) == 1 and isinstance(args[0], list):
        tokens = args[0]

    else:
        tokens = args

    def stringifier(thing):
        if isinstance(thing, str):
            return thing
        assert isinstance(thing, Real)
        return to_decimals(thing, decimals)

    return " ".join(stringifier(c) for c in _unpack_tokens(tokens))


# The following function returns a Subpath when it can, and otherwise, if
# accept_paths is True, a Path. Does not accept an empty token list / spec.
def parse_subpath(*args, accept_paths=False, auto_add_M=False):
    # In the SVG specs, initial movetos are absolute, even if
    # specified as 'm'. This is the default behavior here as well.
    if len(args) == 0:
        raise ValueError("empty args in parse_subpath")

    if len(args) == 1 and isinstance(args[0], str):
        elements = list(_tokenize_path_string(args[0]))

    elif len(args) == 1 and isinstance(args[0], list):
        elements = list(_unpack_tokens(args[0]))

    else:
        elements = list(_unpack_tokens(args))

    if any(not isinstance(x, str) and not isinstance(x, Real) for x in elements):
        print("args:", args)
        print("elements:", elements)
        assert False

    if len(elements) == 0:
        # raise ValueError("Empty token list in parse_subpath.")
        return Subpath()

    if isinstance(elements[0], Real):
        if not auto_add_M:
            raise ValueError("path not starting with 'M' or 'm' (use auto_add_M flag if desired)")
        elements.insert(0, 'M')

    # Reverse for easy use of .pop()
    elements.reverse()

    path = Path()
    subpath = Subpath()
    subpath_start = None
    command = None
    current_pos = 0  # if path starts with an 'm'...

    def append_to_path(subpath):
        if len(path) > 0 and not accept_paths:
            raise ValueError("parse_subpath given multi-subpath path")
        path.append(subpath)

    def pop_float():
        nonlocal elements
        el = elements.pop()
        if isinstance(el, str):
            print("el:", el)
            print("elements:", elements)
            raise ValueError("string found in tokens when float expected")
        assert isinstance(el, Real)
        return el

    while elements:
        if elements[-1] in COMMANDS:
            # New command.
            last_command = command  # Used by S and T
            command = elements.pop()
            absolute = command in UPPERCASE
            command = command.upper()

        else:
            # If this element starts with numbers, it is an implicit command
            # and we don't change the command. Check that it's allowed:
            if command is None:
                raise ValueError("Missing command after 'Z'")

        if command == 'M':
            # Moveto command.
            if len(subpath) > 0:
                append_to_path(subpath)
                subpath = Subpath()
            x = pop_float()
            y = pop_float()
            pos = x + y * 1j
            if absolute:
                current_pos = pos

            else:
                current_pos += pos

            # when M is called, reset subpath_start
            # This behavior of Z is defined in svg spec:
            # http://www.w3.org/TR/SVG/paths.html#PathDataClosePathCommand
            subpath_start = current_pos

            # Implicit moveto commands are treated as lineto commands.
            # So we set command to lineto here, in case there are
            # further implicit commands after this moveto.
            command = 'L'

        elif command == 'Z':
            # Close path
            if len(subpath) > 0:
                subpath.set_Z(forceful=True)
                assert subpath.Z
                append_to_path(subpath)
                subpath = Subpath()
            assert subpath_start is not None
            current_pos = subpath_start
            command = None

        elif command == 'L':
            x = pop_float()
            y = pop_float()
            pos = x + y * 1j
            if not absolute:
                pos += current_pos
            subpath.append(Line(current_pos, pos))
            current_pos = pos

        elif command == 'H':
            x = pop_float()
            pos = x + current_pos.imag * 1j
            if not absolute:
                pos += current_pos.real
            subpath.append(Line(current_pos, pos))
            current_pos = pos

        elif command == 'V':
            y = pop_float()
            pos = current_pos.real + y * 1j
            if not absolute:
                pos += current_pos.imag * 1j
            subpath.append(Line(current_pos, pos))
            current_pos = pos

        elif command == 'C':
            control1 = pop_float() + pop_float() * 1j
            control2 = pop_float() + pop_float() * 1j
            end = pop_float() + pop_float() * 1j

            if not absolute:
                control1 += current_pos
                control2 += current_pos
                end += current_pos

            subpath.append(CubicBezier(current_pos, control1, control2, end))
            current_pos = end

        elif command == 'S':
            # Smooth curve. First control point is the "reflection" of
            # the second control point in the previous path.

            if last_command not in 'CS':
                # If there is no previous command or if the previous command
                # was not an C, c, S or s, assume the first control point is
                # coincident with the current point.
                control1 = current_pos

            else:
                # The first control point is assumed to be the reflection of
                # the second control point on the previous command relative
                # to the current point.
                control1 = current_pos + current_pos - subpath[-1].control2

            control2 = pop_float() + pop_float() * 1j
            end = pop_float() + pop_float() * 1j

            if not absolute:
                control2 += current_pos
                end += current_pos

            subpath.append(CubicBezier(current_pos, control1, control2, end))
            current_pos = end

        elif command == 'Q':
            control = pop_float() + pop_float() * 1j
            end = pop_float() + pop_float() * 1j

            if not absolute:
                control += current_pos
                end += current_pos

            subpath.append(QuadraticBezier(current_pos, control, end))
            current_pos = end

        elif command == 'T':
            # Smooth curve. Control point is the "reflection" of
            # the second control point in the previous path.

            if last_command not in 'QT':
                # If there is no previous command or if the previous command
                # was not an Q, q, T or t, assume the first control point is
                # coincident with the current point.
                control = current_pos
            else:
                # The control point is assumed to be the reflection of
                # the control point on the previous command relative
                # to the current point.
                control = current_pos + current_pos - subpath[-1].control

            end = pop_float() + pop_float() * 1j

            if not absolute:
                end += current_pos

            subpath.append(QuadraticBezier(current_pos, control, end))
            current_pos = end

        elif command == 'A':
            radius = pop_float() + pop_float() * 1j
            rotation = pop_float()
            arc = pop_float()
            sweep = pop_float()
            end = pop_float() + pop_float() * 1j

            if not absolute:
                end += current_pos

            subpath.append(Arc(current_pos, radius, rotation, arc, sweep, end))
            current_pos = end

    if len(subpath) > 0:
        append_to_path(subpath)
        subpath = Subpath()

    if not accept_paths:
        assert len(path) <= 1

    if len(path) <= 1:
        if len(path) > 0:
            assert len(subpath) == 0
            return path[-1]
        assert len(subpath) == 0
        return subpath

    return path


def parse_path(*args, auto_add_M=False):
    """
    Parses a path from a single string or from a list of tokens. The 'accept_paths'
    option is accepted here for uniformity with 'parse_subpath', but it has no
    effect in this function.

    All of the following are valid usages, and will parse to the same path:

    parse_path("M 0 0 10 10")
    parse_path(['M', 0, 0, 10, 10])
    parse_path('M', 0, 0, 10, 10)
    parse_path("M 0+0j 10+10j")
    parse_path(['M', 0+0j, 10+10j])
    parse_path('M', 0+0j, 10+10j)
    parse_path('M', 0, 0, 10+10j)

    (Etc.)
    """
    s = parse_subpath(*args, accept_paths=True, auto_add_M=auto_add_M)

    if isinstance(s, Subpath):
        s = Path(s)

    return s
