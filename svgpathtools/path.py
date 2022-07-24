"""This submodule contains the class definitions of the the main five
classes svgpathtools is built around: Path, Line, QuadraticBezier,
CubicBezier, and Arc. This sentence here was added to make sure git was
workin."""

# External dependencies
from __future__ import division, absolute_import, print_function
from itertools import chain

from math import sqrt, cos, sin, degrees, radians, log, pi, tan, atan2, floor, ceil
from cmath import exp, phase
from warnings import warn
from collections import MutableSequence
from numbers import Number, Real
import numpy as np

try:
    from scipy.integrate import quad
    _quad_available = True

except ImportError:
    _quad_available = False


# Internal dependencies
from .bezier import \
    bezier_intersections, split_bezier, \
    bezier_x_value_intersections, bezier_y_value_intersections, \
    bezier_xbox, bezier_ybox, \
    bezier_by_line_intersections, \
    polynomial2bezier, bezier2polynomial

from .misctools import BugException, real_numbers_in

from .polytools import \
    rational_limit, polyroots, polyroots01, imag, real

from .transform_parser import parse_transform

# trig

tau = 2 * pi
eta = pi / 2


def cos_deg(a):
    z = a % 360
    if z == 180:
        return -1
    elif z == 90 or z == 270:
        return 0
    elif z == 0:
        return 1
    return cos(radians(a))


def sin_deg(a):
    z = a % 360
    if z == 180 or z == 0:
        return 0
    elif z == 90:
        return 1
    elif z == 270:
        return -1
    return sin(radians(a))


def cis_deg(a):
    return cos_deg(a) + 1j * sin_deg(a)


def list_reals_to_list_complex(list_reals):
    assert len(list_reals) % 2 == 0
    assert all(isinstance(x, Real) for x in list_reals)
    return [complex(x, y) for (x, y) in zip(list_reals[::2], list_reals[1::2])]


# Default Parameters  ########################################################

# path segment .length() parameters for arc length computation
LENGTH_MIN_DEPTH  = 5
LENGTH_ERROR = 1e-12
USE_SCIPY_QUAD = True  # for elliptic Arc segment arc length computation

# path segment .ilength() parameters for inverse arc length computation
ILENGTH_MIN_DEPTH = 5
ILENGTH_ERROR = 1e-12
ILENGTH_S_TOL = 1e-12
ILENGTH_MAXITS = 10000

# compatibility / implementation related warnings and parameters
CLOSED_WARNING_ON = True

# d-string printing defaults:
SUBPATH_TO_SUBPATH_SEPARATOR = ' '
COMMAND_TO_NUMBER_SEPARATOR = ' '
NUMBER_TO_NUMBER_SEPARATOR = ','
SEGMENT_TO_SEGMENT_SEPARATOR = '  '

svgpathtools_d_string_params = {
    'decimals': None,
    'use_V_and_H': True,
    'use_S_and_T': False,
    'use_relative_cors': False,
    'spacing_before_new_subpath': ' ',
    'spacing_after_command': ' ',
    'spacing_before_command': ' ',
    'spacing_within_coordinate': ',',
    'include_elidable_commands': True,
    'include_elidable_line_commands': False,
    'include_elidable_first_line': True,
    'elided_command_replacement': ' ',
    'elided_line_command_replacement': ' ',
    'elided_first_line_command_replacement': ' ',
}

_d_string_must_be_strings_params = {
    f for f in svgpathtools_d_string_params if isinstance(svgpathtools_d_string_params[f], str)
}

_d_string_must_be_pure_spaces_params = _d_string_must_be_strings_params
_d_string_must_be_pure_spaces_params.remove('spacing_within_coordinate')

_d_string_must_be_nonempty_params = _d_string_must_be_strings_params
_d_string_must_be_nonempty_params.remove('spacing_after_command')
_d_string_must_be_nonempty_params.remove('spacing_before_command')
_d_string_must_be_nonempty_params.remove('spacing_before_new_subpath')

_NotImplemented4ArcException = \
    Exception("This method has not yet been implemented for Arc objects.")


# Convenience Constructors  ##################################################


def points2lines(*points):
    if len(points) < 2:
        raise ValueError("please provide at least two points")

    lines = []
    for start, end in zip(points[:-1], points[1:]):
        lines.append(Line(start, end))

    return lines


def points2polyline(*points, return_subpath=False):
    if return_subpath:
        return Subpath(*points2lines(*points))

    else:
        return Path(Subpath(*points2lines(*points)))


def points2polygon(*points, return_subpath=False):
    if return_subpath:
        return Subpath(*points2lines(*points)).set_Z(forceful=True)

    else:
        return Path(Subpath(*points2lines(*points)).set_Z(forceful=True))


def bbox2subpath(xmin, xmax, ymin, ymax):
    """
    Returns a Subpath object containing a closed subpath that delimits
    the bounding box.
    """
    return points2polygon(
        xmin + 1j * ymin,
        xmax + 1j * ymin,
        xmax + 1j * ymax,
        xmin + 1j * ymax,
        return_subpath=True
    )


def bbox2path(xmin, xmax, ymin, ymax):
    """
    Returns a Path object containing a closed subpath that delimits
    the bounding box.
    """
    return Path(bbox2subpath(xmin, xmax, ymin, ymax))


def rounded_corner_constructor(a, b, c, r, invert_corner=False):
    assert r >= 0

    l1 = a - b
    l2 = c - b

    if abs(l1) < r or \
       abs(l2) < r:
        raise ValueError

    e1 = b + l1 * r / abs(l1)
    e2 = b + l2 * r / abs(l2)

    if r / abs(l1) == 0.5:
        e1 = a + (b - a) * 0.5

    elif r / abs(l1) == 1:
        e1 = a

    if r / abs(l2) == 1:
        e2 = c

    if a.imag == b.imag:
        imag = a.imag
        real = b.real + r * (1 if a.real > b.real else -1)
        if r / abs(l1) == 0.5:
            real = a.real - r * (1 if a.real > b.real else -1)
        elif r / abs(l1) == 1:
            real = a.real
        e1 = complex(real, imag)

    elif a.real == b.real:
        real = a.real
        imag = b.imag + r * (1 if a.imag > b.imag else -1)
        if r / abs(l1) == 0.5:
            imag = a.imag - r * (1 if a.imag > b.imag else -1)
        elif r / abs(l1) == 1:
            imag = a.imag
        e1 = complex(real, imag)

    if c.imag == b.imag:
        imag = c.imag
        real = b.real + r * (1 if c.real > b.real else -1)
        if r / abs(l2) == 0.5:
            real = c.real - r * (1 if c.real > b.real else -1)
        elif r / abs(l2) == 1:
            real = c.real
        e2 = complex(real, imag)

    elif c.real == b.real:
        real = c.real
        imag = b.imag + r * (1 if c.imag > b.imag else -1)
        if r / abs(l2) == 0.5:
            imag = c.imag - r * (1 if c.imag > b.imag else -1)
        elif r / abs(l2) == 1:
            imag = c.imag
        e2 = complex(real, imag)

    sweep = 1 if (l2 / l1).imag < 0 else 0
    if invert_corner:
        sweep = 1 - sweep

    return e1, e2, sweep


def rounded_polyline(*args, radius=None, invert_corners=False, split_arcs=False):
    corners = list_reals_to_list_complex(*args)

    data = []
    for a, b, c in zip(corners, corners[1:], corners[2:]):
        e1, e2, sweep = rounded_corner_constructor(a, b, c, radius, invert_corners)
        data.append((e1, e2, sweep))

    if len(data) == 0:
        return points2polyline(corners)

    R = radius + 1j * radius
    mister = Subpath()

    if corners[0] != data[0][0]:
        mister.append(Line(corners[0], data[0][0]))

    for e1, e2, sweep in data:
        if e1 != mister.end and mister.end is not None:
            mister.append(Line(mister.end, e1))

        a = Arc(start=e1,
                radius=R,
                rotation=0,
                large_arc=0,
                sweep=sweep,
                end=e2)

        if not split_arcs:
            mister.append(a)

        else:
            arcs = a.split(0.5)
            assert len(arcs) == 2
            assert all(isinstance(b, Arc) for b in arcs)
            assert arcs[0].start == a.start
            assert arcs[-1].end == a.end
            mister.extend(*arcs)

    if corners[-1] != data[-1][1]:
        mister.append(Line(mister.end, corners[-1]))

    return Path(mister)


def rounded_polygon(*args, radius=None):
    if radius is None:
        raise ValueError("radius argument missing")

    corners = list_reals_to_list_complex(*args)

    if corners[0] != corners[-1]:
        corners.append(corners[0])
    corners.append(corners[1])

    if len(corners) < 3 + 2:
        raise ValueError

    data = []
    for a, b, c in zip(corners, corners[1:], corners[2:]):
        e1, e2, sweep = rounded_corner_constructor(a, b, c, radius)
        data.append((e1, e2, sweep))

    segments = []
    R = radius + 1j * radius
    data.insert(0, data[-1])
    for (prev_e1, prev_e2, prev_sweep), (next_e1, next_e2, next_sweep) in zip(
        data, data[1:]
    ):
        if prev_e2 != next_e1:
            segments.append(Line(prev_e2, next_e1))

        segments.append(Arc(start=next_e1,
                            radius=R,
                            rotation=0,
                            large_arc=0,
                            sweep=next_sweep,
                            end=next_e2))

    return Path(Subpath(*segments).set_Z())


def reflect_complex_through(c, direction):
    return direction * (c / direction).conjugate()


def vanilla_cubic_interpolator(*args, z=0.37):
    points = list_reals_to_list_complex(*args)

    if len(points) <= 1:
        raise ValueError("not enough points in vanilla_cubic_interpolator")

    if len(points) == 2:
        return Subpath(Line(points[0], points[1]))

    lengths = [abs(points[i + 1] - points[i]) for i in range(len(points) - 1)]
    v_dirs = [None]
    for i in range(1, len(points) - 1):
        v_dirs.append(points[i + 1] - points[i - 1])
        v_dirs[-1] /= abs(v_dirs[-1])
    v_dirs[0] = -reflect_complex_through(v_dirs[1], (points[1] - points[0]) * 1j)
    v_dirs.append(-reflect_complex_through(v_dirs[-1], (points[-1] - points[-2]) * 1j))

    assert all(np.isclose(1, abs(v)) for v in v_dirs)
    assert len(v_dirs) == len(points)

    to_return = Subpath()
    for i in range(len(points) - 1):
        c = lengths[i] * z
        to_return.append(CubicBezier(points[i], points[i] + v_dirs[i] * c, points[i + 1] - v_dirs[i + 1] * c, points[i + 1]))

    assert len(to_return) == len(points) - 1

    return to_return


# Miscellaneous  #############################################################


def is_path_or_subpath(thing):
    return isinstance(thing, Path) or isinstance(thing, Subpath)


# Segment counting, iteration  ###############################################


def segment_iterator_of(thing, back_to_front=False):
    if thing is None:
        return [].__iter__()

    if isinstance(thing, Segment):
        return [thing]

    if isinstance(thing, Subpath):
        if back_to_front:
            return reversed(thing)
        return thing

    if isinstance(thing, Path):
        return thing.segment_iterator(back_to_front=back_to_front)

    raise ValueError("expecting Segment, Subpath, or Path")


def subpath_iterator_of(thing):
    if thing is None:
        return [].__iter__()

    if isinstance(thing, Segment):
        raise NotImplementedError("Not sure we needed this feature...?")

    if isinstance(thing, Subpath):
        return [thing]

    if isinstance(thing, Path):
        return thing

    raise ValueError("expecting Segment, Subpath, or Path")


def segments_and_partial_addresses_in(thing):
    if isinstance(thing, Segment):
        yield (thing, Address())

    elif isinstance(thing, Subpath):
        for index, seg in enumerate(thing):
            yield (seg, Address(segment_index=thing))

    elif isinstance(thing, Path):
        for index1, subpath in enumerate(thing):
            for index2, segment in enumerate(subpath):
                yield (segment,
                       Address(subpath_index=index1, segment_index=index2))

    else:
        raise ValueError("unknown thing in segments_and_partial_addresses_in")


def num_segments_in(thing):
    if isinstance(thing, Segment):
        return 1

    if isinstance(thing, Subpath):
        return len(thing)

    if isinstance(thing, Path):
        return sum(len(s) for s in thing)

    raise ValueError("unknown thing in num_segments_in")


# Could maybe have been in bezier.py, but is not:  ###########################


def complex_determinant(z, w):
    return np.linalg.det(np.array([[real(z), real(w)],
                                   [imag(z), imag(w)]]))


def complex_linear_congruence(a, z, b, w):
    """
    Solves a + l * z = b + m * w for real l, m where a, z, b, w are complex.
    """
    Q = np.array([[real(z), -real(w)],
                  [imag(z), -imag(w)]])
    try:
        l_and_m = np.linalg.inv(Q).dot(np.array([[real(b - a)], [imag(b - a)]]))

    except np.linalg.LinAlgError:
        raise ValueError

    return l_and_m.item(0, 0), l_and_m.item(1, 0)


def line_by_line_intersections(l1, l2):
    """returns values a list that is either empty or else contains a single
    pair (t1, t2) such that l1.point(t1) ~= l2.point(t2)"""
    assert len(l1) == 2
    assert len(l2) == 2
    a = l1[0]
    z = l1[1] - l1[0]
    b = l2[0]
    w = l2[1] - l2[0]
    try:
        t1, t2 = complex_linear_congruence(a, z, b, w)
        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            assert np.isclose(l1.point(t1), l2.point(t2))
            return [(t1, t2)]
        return []

    except ValueError:
        # try to certify an empty intersection
        pairs = [(l1, l2)]
        # ...by trying to rotate to get a 'flat' bounding box:
        if abs(z) > 1e-8:
            pairs.append((Line(0, 1), Line(b - a, b - a + (w / z))))
        if abs(w) > 1e-8:
            pairs.append((Line(a - b, a - b + (z / w)), Line(0, 1)))
        for (q1, q2) in pairs:
            xmax1, ymax1, xmin1, ymin1 = q1.bbox()
            xmax2, ymax2, xmin2, ymin2 = q2.bbox()
            if xmax1 < xmin2 or xmax2 < xmin1 or ymax1 < ymin2 or ymax2 < ymin1:
                return []
        raise


# Conversion  #################################################################


def bpoints2bezier(*bpoints):
    if len(bpoints) == 1:
        if not isinstance(bpoints[0], list) and not isinstance(bpoints[0], tuple):
            raise ValueError("bad bpoints passed to bpoints2bezier (1)")
        bpoints = bpoints[0]

    if len(bpoints) == 2:
        return Line(*bpoints)

    if len(bpoints) == 4:
        return CubicBezier(*bpoints)

    if len(bpoints) == 3:
        return QuadraticBezier(*bpoints)

    raise ValueError("bad points passed to bpoints2bezier (2)")


def poly2bez(poly, return_bpoints=False):
    """Converts a cubic or lower order Polynomial object (or a sequence of
    coefficients) to a CubicBezier, QuadraticBezier, or Line object as
    appropriate.  If return_bpoints=True then this will instead only return
    the control points of the corresponding Bezier curve.
    Note: The inverse operation is available as a method of CubicBezier,
    QuadraticBezier and Line objects."""
    bpoints = polynomial2bezier(poly)
    return bpoints if return_bpoints else bpoints2bezier(*bpoints)


def bez2poly(bez, numpy_ordering=True, return_poly1d=False):
    """Converts a Bezier object or tuple of Bezier control points to a tuple
    of coefficients of the expanded polynomial.
    return_poly1d : returns a numpy.poly1d object.  This makes computations
    of derivatives / anti - derivatives and many other operations quite quick.
    numpy_ordering : By default (to accommodate numpy) the coefficients will
    be output in reverse standard order.
    Note:  This function is redundant thanks to the .poly() method included
    with all bezier segment classes."""
    if isinstance(bez, BezierSegment):
        bez = bez.bpoints
    return bezier2polynomial(bez,
                             numpy_ordering=numpy_ordering,
                             return_poly1d=return_poly1d)


# Geometric  ###################################################################


def x_val_cut(curve, x_val, jiggle=True):
    assert isinstance(x_val, Real)

    if isinstance(curve, Path):
        to_return = Path()
        for subpath in curve:
            to_return.append(x_val_cut(subpath, x_val))
        return to_return

    elif isinstance(curve, Subpath):
        to_return = Subpath()
        for segment in curve:
            to_return.extend(*x_val_cut(segment, x_val))
        return to_return

    elif isinstance(curve, Segment):
        intersection_addresses = curve.x_val_intersect(x_val)
        pieces = multisplit(curve, intersection_addresses)
        to_return = []
        for p in pieces:
            tweaked = p
            if jiggle and np.isclose(p.start.real, x_val) and p.start.real != x_val:
                tweaked = tweaked.tweaked(start=(x_val + 1j * tweaked.start.imag))
            if jiggle and np.isclose(tweaked.end.real, x_val) and tweaked.end.real != x_val:
                tweaked = tweaked.tweaked(end=(x_val + 1j * p.end.imag))
            to_return.append(tweaked)
        return to_return

    else:
        assert False



def custom_x_y_to_x_y_transform(curve, f):
    def to_complex(t):
        return t[0] + 1j * t[1]

    if isinstance(curve, Path):
        return Path(*[custom_x_y_to_x_y_transform(subpath, f) for subpath in curve])

    elif isinstance(curve, Subpath):
        return Subpath(*[custom_x_y_to_x_y_transform(segment, f) for segment in curve]).set_Z(following=curve)

    elif isinstance(curve, BezierSegment):
        return bpoints2bezier([to_complex(f(p.real, p.imag)) for p in curve.bpoints])

    elif isinstance(curve, Arc):
        raise TypeError("Please use converted_to_bezier before using custom_x_y_to_x_y_transform")

    else:
        raise TypeError("Input `curve` should be a Path, Subpath, Line, "
                        "QuadraticBezier, CubicBezier, or Arc object.")


def transform(curve, tf):
    """Returns transformed copy of the curve by the homogeneous transformation matrix tf"""
    def to_point(p):
        return np.array([[p.real], [p.imag], [1.0]])

    def to_complex(v):
        return v.item(0) + 1j * v.item(1)

    tf = parse_transform(tf)

    if not isinstance(tf, np.ndarray):
        raise ValueError("Expecting numpy.ndarray instance.")

    if tf.shape != (3, 3):
        raise ValueError("Expecting 3x3 numpy.ndarray.")

    if isinstance(curve, Path):
        return Path(*[transform(subpath, tf) for subpath in curve])

    elif isinstance(curve, Subpath):
        return Subpath(*[transform(segment, tf) for segment in curve]).set_Z(following=curve)

    elif isinstance(curve, BezierSegment):
        return bpoints2bezier([to_complex(tf.dot(to_point(p)))
                               for p in curve.bpoints])

    elif isinstance(curve, Arc):
        new_start = to_complex(tf.dot(to_point(curve.start)))
        new_end = to_complex(tf.dot(to_point(curve.end)))

        t = radians(curve.rotation)
        rx = real(curve.radius)
        ry = imag(curve.radius)

        a = np.array([[cos(t)**2 / rx**2 + sin(t)**2 / ry**2,     cos(t) * sin(t) * (1 / rx**2 - 1 / ry**2)],
                      [cos(t) * sin(t) * (1 / rx**2 - 1 / ry**2), sin(t)**2 / rx**2 + cos(t)**2 / ry**2]])
        s_inverse = np.linalg.inv(tf[0:2, 0:2])
        b = s_inverse.transpose().dot(a.dot(s_inverse))
        d_squared, p = np.linalg.eigh(b)

        new_t = np.arctan2(p[1, 0], p[0, 0])
        new_rotation = degrees(new_t)
        new_rx = sqrt(1 / d_squared[0])
        new_ry = sqrt(1 / d_squared[1])

        if b[0, 0] == b[1, 1] and \
           b[1, 0] == b[0, 1] == 0:
            # the eigenvector basis was arbitrary; rever to old orientation
            # for 'least surprise'
            new_rotation = curve.rotation

        return Arc(new_start,
                   radius=(new_rx + 1j * new_ry),
                   rotation=new_rotation,
                   large_arc=curve.large_arc,
                   sweep=curve.sweep if np.linalg.det(a) > 0 else not curve.sweep,
                   end=new_end)

    else:
        raise TypeError("Input `curve` should be a Path, Subpath, Line, "
                        "QuadraticBezier, CubicBezier, or Arc object.")


def rotate(curve, degs, origin=0j):
    """
    Returns curve rotated by `degs` degrees (CCW) around the point `origin`
    (a complex number), which defaults to 0.
    """
    return \
        transform(
            curve,
            ['translate', origin, 'rotate', degs, 'translate', -origin]
        )


def scale(curve, x, y=None, origin=0j):
    """
    Scales all coordinates in the curve by x in the x direction and by y
    direction; y defaults to x if not supplied.
    """
    return \
        transform(
            curve,
            ['translate', origin, 'scale', x, x if y is None else y, 'translate', -origin]
        )


def translate(curve, x, y=None):
    """
    Returns a shifted copy of 'curve' by the complex quantity z = x + 1 j * y, such that
    translate(curve, x, y).point(t) = curve.point(t) + z. Note that x can be complex.
    Therefore, translate(curve, 10, 10) and translate(curve, 10 + 10j) are equivalent.
    """

    # Note: we could implement this via transform like 'scale' and 'rotate'
    # above, but since translations are common, and since general Arc transforms are
    # slow-ish, we do things from scratch
    if y is None: 
        if isinstance(x, Real):
            y = 0
            
        else:
            x, y = real_numbers_in(x)

    assert isinstance(x, Real) and isinstance(y, Real)
    z = x + 1j * y

    if isinstance(curve, Path):
        return Path(*[
            translate(subpath, z) for subpath in curve
        ])

    elif isinstance(curve, Subpath):
        return Subpath(*[
            translate(seg, z) for seg in curve
        ]).set_Z(following=curve)

    elif isinstance(curve, BezierSegment):
        return bpoints2bezier(*[
            bpt + z for bpt in curve.bpoints
        ])

    elif isinstance(curve, Arc):
        new_start = curve.start + z
        new_end = curve.end + z
        return Arc(start=new_start,
                   radius=curve.radius,
                   rotation=curve.rotation,
                   large_arc=curve.large_arc,
                   sweep=curve.sweep,
                   end=new_end)

    else:
        raise TypeError("Input `curve` should be a Path, Line, "
                        "QuadraticBezier, CubicBezier, or Arc object.")


def bezier_unit_tangent(seg, t):
    """
    Returns the unit tangent of the segment at t.

    Notes
    - - - - -
    If you receive a RuntimeWarning, try the following:
    >>> import numpy
    >>> old_numpy_error_settings = numpy.seterr(invalid='raise')
    This can be undone with:
    >>> numpy.seterr(**old_numpy_error_settings)
    """
    def compute_error_message(place):
        bef = seg.poly().deriv()(t - 1e-4)
        aft = seg.poly().deriv()(t + 1e-4)
        mes = (f"thrown at {place} in bezier_unit_tangent:" +
               f"unit tangent appears to not be well-defined at t = {t},\n" +
               f"seg.poly().deriv()(t - 1e-4) = {bef}\n" +
               f"seg.poly().deriv()(t + 1e-4) = {aft}")
        return mes

    def normalize(thing):
        return thing / abs(thing)

    assert 0 <= t <= 1
    dseg = seg.derivative(t)

    # Note: dseg might be numpy value, use np.seterr(invalid='raise')
    try:
        unit_tangent = normalize(dseg)
    except (ZeroDivisionError, FloatingPointError):
        # the following a previous solution based on csqrt, which I
        # (vistuleB) deemed iffy, because the csqrt chooses its root
        # arbitrarily among two choices
        pts = seg.bpoints

        if len(pts) > 2 and t == 0 and np.isclose(pts[0], pts[1]):
            try:
                unit_tangent = normalize(pts[2] - pts[0])
            except (ZeroDivisionError, FloatingPointError):
                if len(pts) > 3 and np.isclose(pts[0], pts[2]):
                    try:
                        unit_tangent = normalize(pts[3] - pts[0])
                    except (ZeroDivisionError, FloatingPointError):
                        raise ValueError(compute_error_message("@A"))
                raise ValueError(compute_error_message("@B"))

        elif len(pts) > 2 and t == 1 and np.isclose(pts[-1], pts[-2]):
            try:
                unit_tangent = normalize(pts[-1] - pts[-3])
            except (ZeroDivisionError, FloatingPointError):
                if len(pts) > 3 and np.isclose(pts[-1], pts[-3]):
                    try:
                        unit_tangent = normalize(pts[-1] - pts[-4])
                    except (ZeroDivisionError, FloatingPointError):
                        raise ValueError(compute_error_message("@C"))
                raise ValueError(compute_error_message("@D"))

        else:
            raise ValueError(compute_error_message("@E"))

    return unit_tangent


def bezier_radialrange(seg, origin, return_all_global_extrema=False):
    """returns the tuples (d_min, t_min) and (d_max, t_max) which minimize and
    maximize, respectively, the distance d = |self.point(t) - origin|.
    return_all_global_extrema:  Multiple such t_min or t_max values can exist.
    By default, this will only return one. Set return_all_global_extrema=True
    to return all such global extrema."""

    def _radius(t):
        return abs(seg.point(t) - origin)

    shifted_seg_poly = seg.poly() - origin
    r_squared = real(shifted_seg_poly)**2 + imag(shifted_seg_poly)**2
    extremizers = [0, 1] + polyroots01(r_squared.deriv())
    extrema = [(_radius(t), t) for t in extremizers]

    if return_all_global_extrema:
        raise NotImplementedError

    else:
        seg_global_min = min(extrema, key=(lambda x: x[0]))
        seg_global_max = max(extrema, key=(lambda x: x[0]))
        return seg_global_min, seg_global_max


def segment_length(curve, start, end, start_point, end_point,
                   error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH, depth=0):
    """
    Recursively approximates the length by straight lines.

    Designed for internal consumption by a knowledgeable caller.
    """
    mid = (start + end) / 2
    mid_point = curve.point(mid)
    length = abs(end_point - start_point)
    first_half = abs(mid_point - start_point)
    second_half = abs(end_point - mid_point)

    length2 = first_half + second_half
    if (length2 - length > error) or (depth < min_depth):
        # Calculate the length of each segment:
        depth += 1
        return (segment_length(curve, start, mid, start_point, mid_point,
                               error, min_depth, depth) +
                segment_length(curve, mid, end, mid_point, end_point,
                               error, min_depth, depth))
    # This is accurate enough.
    return length2


def inv_arclength_01(curve, s, s_tol=ILENGTH_S_TOL, maxits=ILENGTH_MAXITS,
                     error=ILENGTH_ERROR, min_depth=ILENGTH_MIN_DEPTH):
    """
    Takes a value s in [0, 1] and returns a value t such that curve.point(t)
    is a distance s * curve.length() from curve.point(0) along curve.
    """
    if not 0 <= s <= 1:
        raise ValueError("expecting value in [0, 1]")
    return inv_arclength(curve, s * curve.length(), s_tol, maxits,
                         error, min_depth)


def inv_arclength(curve, s, s_tol=ILENGTH_S_TOL, maxits=ILENGTH_MAXITS,
                  error=ILENGTH_ERROR, min_depth=ILENGTH_MIN_DEPTH):
    """
    INPUT: curve may be any type of Segment, a Subpath or a Path
    OUTPUT: Returns a float, t, such that the arc length of curve from 0 to
    t is approximately s.
    s_tol - exit when |s(t) - s| < s_tol where
        s(t) = seg.length(0, t, error, min_depth) and seg is either curve or,
        if curve is a Path object, then seg is a segment in curve.
    error - used to compute lengths of cubics and arcs
    min_depth - used to compute lengths of cubics and arcs
    Note:  This function is not designed to be efficient, but if it's slower
    than you need, make sure you have scipy installed.
    """
    curve_length = curve.length(error=error, min_depth=min_depth)
    assert curve_length > 0
    if not 0 <= s <= curve_length:
        raise ValueError("s is not in interval [0, curve.length()].")

    if s == 0:
        return param2address(curve, 0)
    if s == curve_length:
        return param2address(curve, 1)

    if isinstance(curve, Line):
        t = s / curve.length(error=error, min_depth=min_depth)
        return Address(t=t)

    elif isinstance(curve, Segment):
        t_upper = 1
        t_lower = 0
        iteration = 0
        while iteration < maxits:
            iteration += 1
            t = (t_lower + t_upper) / 2
            s_t = curve.length(t1=t, error=error, min_depth=min_depth)

            if abs(s_t - s) < s_tol:
                return Address(t=t)

            elif s_t < s:  # t too small
                t_lower = t

            else:  # s < s_t, t too big
                t_upper = t

            if t_upper == t_lower:
                warn(f"t is as close as a float can be to the correct value, but |s(t) - s| = {abs(s_t - s)} > s_tol")
                return Address(t=t)

        raise Exception(f"Maximum iterations reached with s(t) - s = {s_t - s}.")

    elif isinstance(curve, Subpath):
        seg_lengths = [
            seg.length(error=error, min_depth=min_depth) for seg in curve
        ]
        lsum = 0
        # Find which segment the point we search for is located on
        for k, len_k in enumerate(seg_lengths):
            if lsum <= s <= lsum + len_k:
                a = inv_arclength(curve[k], s - lsum, s_tol=s_tol,
                                  maxits=maxits, error=error,
                                  min_depth=min_depth)
                return curve.t2address(a, segment_index=k)
            lsum += len_k
        return curve.T2address(1)

    elif isinstance(curve, Path):
        subpath_lengths = [
            subpath.length(error=error,
                           min_depth=min_depth) for subpath in curve
        ]
        lsum = 0
        for k, len_k in enumerate(subpath_lengths):
            if lsum <= s <= lsum + len_k:
                a = inv_arclength(curve[k], s - lsum, s_tol=s_tol,
                                  maxits=maxits, error=error,
                                  min_depth=min_depth)
                return curve.T2address(a, subpath_index=k)
            lsum += len_k
        return curve.W2address(1)

    else:
        raise TypeError("First argument must be a Line, QuadraticBezier, "
                        "CubicBezier, Arc, Subpath, or Path object.")


def fancy_multisplit(subpath, addresses):
    def address_with_T(T):
        for a in addresses:
            if a.T == T:
                return a
        assert False

    assert isinstance(subpath, Subpath)
    assert len(addresses) > 0

    Ts = list({a.T for a in addresses})
    Ts.sort()
    assert len(Ts) == len(addresses)
    assert all(0 <= T <= 1 for T in Ts)

    pieces = []
    for T1, T2 in zip(Ts, chain(Ts[1:], [Ts[0]] if (Ts[-1] < 1 or Ts[0] > 0) else [])):
        assert T1 < T2 or (
            T2 == Ts[0] and
            T1 == Ts[-1] and
            T1 > T2 and
            subpath.Z
        )
        pieces.append((
            subpath.cropped(T1, T2),
            (address_with_T(T1), address_with_T(T2))
        ))

    assert len(pieces) == len(addresses)
    assert all(isinstance(p, Subpath) for p, _ in pieces)
    assert all(not p.Z for p, _ in pieces)
    assert all(a1a2 != b1b2 or p == q for p, a1a2 in pieces for q, b1b2 in pieces)

    assert np.isclose(
        sum(q[0].length() for q in pieces),
        subpath.length()
    )

    return pieces


def multisplit(curve, addresses_or_params, debug=False):
    assert all(isinstance(x, Number) or isinstance(x, Address) for x in addresses_or_params)
    addresses = [param2address(curve, a) for a in addresses_or_params]

    if debug:
        print("")
        print("in multisplit debug mode; full list of addresses before insertions of 0, 1:")
        for a in addresses:
            print(a.__repr__(decimals=3))

    assert all(a1 < a2 for a1, a2 in zip(addresses, addresses[1:]))

    if isinstance(curve, Segment) or isinstance(curve, Subpath):
        inserted_0 = False
        if all(address2param(curve, a) > 0 for a in addresses):
            addresses.insert(0, param2address(curve, 0))
            inserted_0 = True

        inserted_1 = False
        if all(address2param(curve, a) < 1 for a in addresses):
            addresses.append(param2address(curve, 1))
            inserted_1 = True
        
        assert 0 in [address2param(curve, a) for a in addresses]
        assert 1 in [address2param(curve, a) for a in addresses]

        if debug:
            print("")
            print("in multisplit debug mode; full list of addresses after insertions of 0, 1:")
            for a in addresses:
                print(a.__repr__(decimals=3))
            print("")

        assert all(a1 < a2 for a1, a2 in zip(addresses, addresses[1:]))

    pieces = []

    if isinstance(curve, Segment):
        for a1, a2 in zip(addresses, addresses[1:]):
            pieces.append(curve.cropped(a1, a2))

    elif isinstance(curve, Subpath):
        for a1, a2 in zip(addresses, addresses[1:]):
            pieces.append(curve.cropped(a1, a2))

        if curve.Z and inserted_0 and inserted_1:  # (recently changed from "not inserted_0 and not inserted_1")
            if len(pieces) == 1:
                assert len(pieces[0]) == len(curve)  # recently changed from full equality
                assert pieces[0].Z is False
                pieces[0].set_Z()

            else:
                assert len(pieces) > 1
                num_segments = sum(x.num_segments() for x in pieces)
                last_piece = pieces.pop()
                pieces[0].splice(0, 0, last_piece)
                assert num_segments == sum(x.num_segments() for x in pieces)

                if debug:
                    print("pieces after recomp:")
                    print(pieces)

        assert all(isinstance(piece, Subpath) for piece in pieces)
        assert len(pieces) > 0 or curve.is_empty()

    elif isinstance(curve, Path):
        for subpath_index, subpath in enumerate(curve):
            pieces.extend(multisplit(subpath, [a.erase_W_subpath_index_part() for a in addresses if a.subpath_index == subpath_index]))
        assert all(isinstance(piece, Subpath) for piece in pieces)

    return pieces


def crop_bezier(seg, t0, t1):
    """returns a cropped copy of this segment which starts at self.point(t0)
    and ends at self.point(t1)."""
    if t0 < t1:
        swap = False
    else:
        t1, t0 = t0, t1
        swap = True

    if t0 == 0:
        cropped_seg = seg if t1 == 1 else seg.split(t1)[0]  # trying to reuse segments here...

    elif t1 == 1:
        cropped_seg = seg.split(t0)[1]

    else:
        # trim off the 0 <= t < t0 part
        trimmed_seg = seg.split(t0)[1]
        t1_adj = 1 if t0 == 1 else (t1 - t0) / (1 - t0)
        cropped_seg = trimmed_seg.split(t1_adj)[0]

    assert isinstance(cropped_seg, BezierSegment)
    assert type(cropped_seg) == type(seg)

    return cropped_seg if not swap else cropped_seg.reversed()


# For external convenience geometric #####################################


def heuristic_has_point_outside(p, enclosure, tol=1e-8):
    xmin, xmax, ymin, ymax = p.bbox()
    Xmin, Xmax, Ymin, Ymax = enclosure.bbox()
    if xmax > Xmax + tol or \
       xmin < Xmin - tol or \
       ymax > Ymax + tol or \
       ymin < Ymin - tol:
        return True

    for t in np.arange(0.05, 0.96, 0.05):  # avoid endpoints as too touchy
        if not enclosure.even_odd_encloses(p.point(t)):
            return True

    return False


def crop(thing, enclosure, crop_to_inside=True, debug=False):
    if debug:
        print("")
        print("crop starting in DEBUG mode")
        print("")
        print("thing:")
        print(thing.__repr__(decimals=2))
        print("")
        print("enclosure:")
        print(enclosure.__repr__(decimals=2))

    intersections = thing.intersect(enclosure)

    if isinstance(thing, Segment):
        time_values = list({a1.t for (a1, _) in intersections})
    
    elif isinstance(thing, Subpath):
        time_values = list({a1.T for (a1, _) in intersections})

    elif isinstance(thing, Path):
        time_values = list({a1.W for (a1, _) in intersections})

    else:
        assert False

    time_values.sort()
    if debug:
        print("")
    pieces = multisplit(thing, time_values, debug=debug)
    if debug:
        print("")
        print("number of pieces obtained:", len(pieces))
        print("")

    kept_pieces = []
    for i, p in enumerate(pieces):
        if heuristic_has_point_outside(p, enclosure) != crop_to_inside:
            kept_pieces.append(p)
            if debug:
                print(f"piece (i={i}), kept:")
                print(p.__repr__(decimals=2))

        else:
            if debug:
                print(f"piece (i={i}), not kept:")
                print(p.__repr__(decimals=2))

    to_return = Path(*kept_pieces)

    if debug:
        print("the path that we are returning:")
        print(to_return.__repr__(decimals=2))

    assert all(heuristic_has_point_outside(x, enclosure) != crop_to_inside for x in to_return)

    return to_return


def intersect_subpaths(s1, s2):
    assert isinstance(s1, Subpath)
    assert isinstance(s2, Subpath)
    assert s1.Z
    assert s2.Z

    def find_matching_a2(a1):
        assert isinstance(a1, Address)
        for (b1, b2) in intersections:
            if b1 == a1:
                return b2
        print("a1:", a1)
        print("intersections:", intersections)
        assert False

    def find_matching_a1(a2):
        assert isinstance(a2, Address)
        for (b1, b2) in intersections:
            if b2 == a2:
                return b1
        assert False

    intersections = s1.intersect(s2, normalize=True)
    for index, (a1, a2) in enumerate(intersections):
        for (b1, b2) in intersections[index + 1:]:
            assert a1.T != b1.T
            assert a2.T != b2.T

    if len(intersections) <= 1:
        if not heuristic_has_point_outside(s1, s2):
            return Path(s1.cloned())
        if not heuristic_has_point_outside(s2, s1):
            return Path(s2.cloned())
        return Path()

    pieces_and_endpoints_1 = fancy_multisplit(s1, [a1 for (a1, a2) in intersections])
    pieces_and_endpoints_2 = fancy_multisplit(s2, [a2 for (a1, a2) in intersections])

    pieces_and_endpoints_1 = [
        (p, (s, e)) for p, (s, e)
        in pieces_and_endpoints_1 if not heuristic_has_point_outside(p, s2, tol=1e-6)
    ]

    pieces_and_endpoints_2 = [
        (p, (s, e)) for p, (s, e)
        in pieces_and_endpoints_2 if not heuristic_has_point_outside(p, s1, tol=1e-6)
    ]

    answer = Path()

    total_pieces_used_so_far = []

    while True:
        pieces_and_endpoints_1 = [
            (p, (s, e)) for p, (s, e) in pieces_and_endpoints_1 if p not in total_pieces_used_so_far
        ]
        pieces_and_endpoints_2 = [
            (p, (s, e)) for p, (s, e) in pieces_and_endpoints_2 if p not in total_pieces_used_so_far
        ]

        if len(pieces_and_endpoints_1) > 0:
            piece = pieces_and_endpoints_1[0][0]
            piece_subpath_owner = s1
            piece_index = 0
            piece_forward = True

        elif len(pieces_and_endpoints_2) > 0:
            piece = pieces_and_endpoints_2[0][0]
            piece_subpath_owner = s2
            piece_index = 0
            piece_forward = False  # what does it matter?

        else:
            break

        pieces_so_far = [(
            piece,
            piece_forward,
            piece_subpath_owner,
            piece_index
        )]

        assert piece not in total_pieces_used_so_far
        total_pieces_used_so_far.append(piece)

        while True:
            if piece_subpath_owner is s1:
                a1 = pieces_and_endpoints_1[
                    piece_index
                ][
                    1
                ][
                    1 if piece_forward else 0
                ]
                a2 = find_matching_a2(a1)

            elif piece_subpath_owner is s2:
                a2 = pieces_and_endpoints_2[
                    piece_index
                ][
                    1
                ][
                    1 if piece_forward else 0
                ]
                a1 = find_matching_a1(a2)

            else:
                assert False

            assert (a1, a2) in intersections

            candidates = []

            # enumerate all p1 candidates
            for index, (p1, (start1, end1)) in enumerate(pieces_and_endpoints_1):
                assert isinstance(p1, Subpath)
                assert isinstance(start1, Address)
                assert isinstance(end1, Address)

                if start1 != a1 and end1 != a1:
                    continue

                # if any(q[0] == p1 for q in pieces_so_far):
                #     continue
                if p1 in total_pieces_used_so_far:
                    continue

                if heuristic_has_point_outside(p1, s2):
                    continue

                assert p1 != piece

                if start1 == a1:
                    candidates.append((
                        p1,
                        True,
                        s1,
                        index
                    ))

                if end1 == a1:
                    candidates.append((
                        p1,
                        False,
                        s1,
                        index
                    ))

            # enumerate all p2 candidates
            for index, (p2, (start2, end2)) in enumerate(pieces_and_endpoints_2):
                assert isinstance(p2, Subpath)
                assert isinstance(start2, Address)
                assert isinstance(end2, Address)

                if any(q[0] == p2 for q in pieces_so_far):
                    continue

                # if any(q[0] == p2 for q in pieces_so_far):
                #     continue
                if p2 in total_pieces_used_so_far:
                    continue

                if heuristic_has_point_outside(p2, s1):
                    continue

                assert p2 != piece

                if start2 == a2:
                    candidates.append((
                        p2,
                        True,
                        s2,
                        index
                    ))

                if end2 == a2:
                    candidates.append((
                        p2,
                        False,
                        s2,
                        index
                    ))

            if len(candidates) == 0:
                break

            # let's be primitive:
            pieces_so_far.append(candidates[0])

            (
                piece,
                piece_forward,
                piece_subpath_owner,
                piece_index
            ) = pieces_so_far[-1]

            assert piece not in total_pieces_used_so_far
            total_pieces_used_so_far.append(piece)

        oriented_pieces = [
            piece if forward else piece.reversed()
            for piece, forward, owner, index in pieces_so_far
        ]

        assert len(oriented_pieces) > 0

        if not all(np.isclose(p.end, q.start) for p, q in zip(oriented_pieces, oriented_pieces[1:])):
            raise ValueError("You thought it would be that easy? Whu-ah-ah-ah-ah-ah!!!! (1)")

        if not np.isclose(
            oriented_pieces[0].start,
            oriented_pieces[-1].end
        ):
            raise ValueError("You thought it would be that easy? Whu-ah-ah-ah-ah-ah!!!! (2)")

        answer.append(Subpath(*oriented_pieces).set_Z())

    return answer


def intersect_paths(p1, p2):
    answer = Path()

    for s1 in subpath_iterator_of(p1):
        for s2 in subpath_iterator_of(p2):
            answer.extend(intersect_subpaths(s1, s2), even_if_empty=False)

    return answer


# Offset-related functions ###############################################


def offset_intersection(a, b, c, offset_amount):
    normal_a_b = (b - a) * (-1j) / abs(b - a)
    normab_b_c = (c - b) * (-1j) / abs(c - b)
    A = a + normal_a_b * offset_amount
    C = c + normab_b_c * offset_amount

    # first possible system:
    # A + l * (b - a) = C + m * (b - c)

    # second possible system:
    # A + l * (b - a) = b + m * (sum_of_normals)

    sum_of_normals = normal_a_b + normab_b_c

    first_system_determinant = imag((b - a) * (b - c).conjugate())
    secnd_system_determinant = imag((b - a) * sum_of_normals.conjugate())

    try:
        if (abs(first_system_determinant) > abs(secnd_system_determinant)):
            l, m = complex_linear_congruence(A, b - a, C, b - c)
            return A + l * (b - a)
        else:
            l, m = complex_linear_congruence(A, b - a, b, sum_of_normals)
            return A + l * (b - a)
    except ValueError:
        raise


def divergence_of_offset(p1, p2, putatitve_offset, safety=10,
                         early_return_threshold=None):
    safety = int(max(1, safety))
    t_min = 0
    max_distance = 0
    # t = 0, 1 are not tested (not useful):
    for t in [(x / (safety + 1)) for x in range(1, safety + 1)]:
        extruded = p1.point(t) + p1.normal(t) * putatitve_offset

        if t_min > 0:
            # keep sequence of points in monotone order
            __, p2 = p2.split(t_min)

        closest = p2.closest_point_to(extruded)
        assert closest.value == abs(p2.point(closest.address) - extruded)
        max_distance = max(max_distance, closest.value)

        if early_return_threshold is not None and \
           max_distance > early_return_threshold:
            return max_distance

    return max_distance


def compute_offset_joining_subpath(seg1, off1, seg2, off2,
                                   offset_amount,
                                   join='miter',
                                   miter_limit=4):
    """returns a triple of the form 'subpath, t1, t2' where 'subpath' is the
    connecting subpath, t1 and t2 are the new end and start points of off1,
    off2, respectively"""
    assert all(isinstance(x, Segment) for x in [seg1, off1, seg2, off2])
    assert seg1.end == seg2.start
    assert join in ['miter', 'round', 'bevel']

    if join == 'bevel':
        join = 'miter'
        miter_limit = 1

    assert join in ['miter', 'round']

    base_corner = seg1.end

    n1 = seg1.normal(1)
    n2 = seg2.normal(0)

    theoretical_start = seg1.end   + n1 * offset_amount
    theoretical_end   = seg2.start + n2 * offset_amount

    assert np.isclose(theoretical_start, off1.end)  # also checks offset_amount!
    assert np.isclose(theoretical_end, off2.start)

    tangent1_base = seg1.unit_tangent(1)
    tangent2_base = seg2.unit_tangent(0)

    # note: real(w * z.conjugate()) is the dot product of w, z as vectors

    if offset_amount * real(n1 * tangent2_base.conjugate()) > 0:
        # acute case
        intersections = off1.intersect(off2)
        if len(intersections) > 0:
            a1, a2 = intersections[-1]
            t1, t2 = a1.t, a2.t
            assert 0 <= t1 <= 1 and 0 <= t2 <= 1
            assert np.isclose(off1.point(t1), off2.point(t2))
            if t2 > 0 and t1 < 1:
                return None, t1, t2
        return Subpath(Line(off1.end, off2.start)), 1, 0

    if offset_amount * real(n1 * tangent2_base.conjugate()) < 0:
        # obtuse case
        if join == 'miter':
            a = base_corner - tangent1_base
            b = base_corner
            c = base_corner + tangent2_base

            apex = offset_intersection(a, b, c, offset_amount)
            z1 = apex - off1.end
            z2 = apex - off2.start
            assert real(z1 * tangent1_base.conjugate()) > 0
            assert real(z2 * tangent2_base.conjugate()) < 0
            assert np.isclose(imag(z1 * tangent1_base.conjugate()), 0)
            assert np.isclose(imag(z2 * tangent2_base.conjugate()), 0)

            miter = abs(apex - base_corner) / abs(offset_amount)
            if miter > miter_limit:
                return Subpath(Line(off1.end, off2.start)), 1, 0
            else:
                # the only case we actually need a path instead of a segment:
                return Subpath(Line(off1.end, apex),
                               Line(apex, off2.start)), 1, 0

        assert join == 'round'

        r1 = abs(off1.end - base_corner)
        r2 = abs(off2.start - base_corner)

        assert np.isclose(r1, r2)

        sweep = 1 if imag(tangent1_base * tangent2_base.conjugate()) < 0 else 0

        return \
            Subpath(Arc(off1.end, r1 + 1j * r1, 0, 0, sweep, off2.start)), \
            1, 0

    assert real(n1 * tangent2_base.conjugate()) == 0
    assert np.isclose(off1.end, off2.start)
    assert False


def join_offset_segments_into_subpath(skeleton, offsets, putative_amount,
                                      join, miter_limit):
    """for-internal-use function, assumes all the following:"""
    assert isinstance(skeleton, Subpath) and isinstance(offsets, Path)
    assert len(skeleton) == offsets.num_segments()
    assert skeleton.is_bezier_subpath()
    assert join in ['miter', 'round', 'bevel']

    if len(skeleton) == 0:
        return Subpath()

    assert all(isinstance(thing, Segment) for thing in skeleton)
    assert all(isinstance(thing, Subpath) for thing in offsets)

    segment_pairs = [(u, v) for u, v in zip(skeleton,
                                            offsets.segment_iterator())]

    if skeleton.Z:
        segment_pairs.append((skeleton[0], offsets[0][0]))

    successive_pairs = zip(segment_pairs, segment_pairs[1:])

    to_return = Subpath(offsets[0][0])

    for index, (u, v) in enumerate(successive_pairs):
        is_loop_around_iteration = False
        if skeleton.Z and index == len(segment_pairs) - 2:
            is_loop_around_iteration = True

        assert isinstance(u[0], Segment)
        assert isinstance(v[0], Segment)
        assert isinstance(v[1], Segment)
        assert isinstance(u[1], Segment)

        seg1 = u[0]
        seg2 = v[0]
        off1 = u[1] if index == 0 else to_return[-1]
        off2 = v[1] if not is_loop_around_iteration else to_return[0]

        assert all(isinstance(x, Segment) for x in [seg1, seg2, off1, off2])

        if np.isclose(off1.end, off2.start):
            if off1.end != off2.start:
                o2 = off2.tweaked(start=off1.end)
            else:
                o2 = off2

        else:
            p, t1, t2 = compute_offset_joining_subpath(
                seg1, off1, seg2, off2,
                putative_amount,
                join=join, miter_limit=miter_limit
            )

            assert t1 >= 0 and t2 <= 1
            o1 = off1 if t1 == 1 else off1.cropped(0, t1)
            o2 = off2 if t2 == 0 else off2.cropped(t2, 1)

            if p is None:
                if t1 == 1 and t2 == 0:
                    p = Subpath(Line(off1.end, off2.start))
                else:
                    assert np.isclose(o1.end, o2.start)
                    o2 = o2.tweaked(start=o1.end)

            to_return[-1] = o1  # (overwrite previous )

            if p is not None:
                assert p.start == o1.end
                assert p.end == o2.start
                to_return.extend(p)

        if not is_loop_around_iteration:
            to_return.append(o2)
        else:
            to_return[0] = o2
            to_return.set_Z(forceful=False)

    assert to_return.Z == skeleton.Z

    return to_return


# Stroke-related #############################################################


def endcap_for_curve(p, offset_amount, cap_style):
    n     = p.normal(1)
    start = p.end + n * offset_amount
    end   = p.end - n * offset_amount
    t     = p.unit_tangent(1) * abs(offset_amount)

    if cap_style == 'square':
        l1 = Line(start, start + t)
        l2 = Line(start + t, end + t)
        l3 = Line(end + t, end)
        return Subpath(l1, l2, l3)

    if cap_style == 'round':
        mid = p.end + t
        sweep = 1 if offset_amount > 0 else 0
        a1 = Arc(start, offset_amount + 1j * offset_amount, 0, 0, sweep, mid)
        a2 = Arc(mid,   offset_amount + 1j * offset_amount, 0, 0, sweep, end)
        return Subpath(a1, a2)

    if cap_style == 'butt':
        return Subpath(Line(start, end))

    assert False


# Main Classes ###############################################################

"""
(IMPROMPTU README) The main user-facing classes are:

Line
QuadraticBezier
CubicBezier
Arc
Subpath
Path

The first four types of objects ('Line', 'QuadraticBezier', 'CubicBezier'
and 'Arc') are commonly known as "segments".

A subpath is a list, possibly empty, of end-to-end contiguous segments. A
nonempty subpath whose first and last points coincide *may* be "closed". A
subpath prints with a 'Z' attached to the end if and only if it is closed.
Closure is controlled by the Subpath.set_Z(), Subpath.unset_Z() methods. See
the implementation of those methods for more details.

A path is an ordered list of subpaths.

There are also superclasses to help group and categorize the various types of
segments. These are 'Segment', from which all segments inherit, and
'BezierSegment', from which only 'Line', 'QuadraticBezier' and 'CubicBezier'
inherit. The class inheritance diagram for all segments is as follows:

- Segment:
  - Arc
  - BezierSegment:
    - Line
    - QuadraticBezier
    - CubicBezier

Note that only 'Arc', 'Line', 'QuadraticBezier' and 'CubicBezier' are meant to
be user-instantiated, but the Segment and BezierSegment superclasses can be
useful, e.g., for identifying the type of an object via 'isinstance()'.

Furthermore, Segment and Subpath both inherit from 'ContinuousCurve', while
Path and ContinuousCurve inherit from Curve. A full inheritance diagram is
thus as follows:

- Curve:
  - Path
  - ContinuousCurve
    - Subpath
    - Segment
      - Arc
      - BezierSegment
        - Line
        - QuadraticBezier
        - CubicBezier

...but only 'Path', 'Subpath', 'Arc', 'Line', 'QuadraticBezier' and
'CubicBezier' can be instantiated.

In particular, one should keep in mind that there only exist three main
categories of user-instantiated objects, these being paths, subpaths and
segments. Segments are the 'primitives', subpaths are containers for segments,
and paths are containers for subpaths. (!!! Important !!!)

PARAMETERIZATION AND ADDRESSES

Points on paths are parameterized by a value from 0 to 1, denoted 'W' in the
source code.
Points on subpaths are parameterized by a value from 0 to 1, denoted 'T' in
the source code.
Points on segments are parameterized by a value from 0 to 1, denoted 't' in
the source code.

To save the user the headache of keep track of the various parameters, a
single type of object called 'Address' is used to encode the position of
points on paths / subpaths / segments. An address object will have a different
number of fields set depending on which type of object returns (creates) it,
as an object may not have full knowledge of its own address within another
object (and indeed may be owned by several different objects) (i.e., a segment
may appear in more than one subpath, a subpath may appear in more than one
path). The fields of an address are as follows:

.t               ...the segment-level parameter of the addressed point
.T               ...the subpath-level parameter of the addressed point
.segment_index   ...the index of the segment containing the addressed point
                 within its subpath
.W               ...the path-level parameter of the addressed point
.subpath_index   ...the index of the subpath containing the addressed point
                 within its path

In general:

    --- instead of returning a t-value, a Segment method will return an
        address a such that a.t != None.
    --- instead of returning a T-value, a Subpath method will return an
        address a such that a.T != None, a.segment_index != None,
        a.t != None
    --- instead of returning a W-value, a Path method will return a full
        address

Also:

    --- user-facing methods of Segment will accept either t-values or
        addresses with non-None .t fields
    --- user-facing methods of Subpath will accept either T-values or
        addresses with non-None .T fields
    --- user-facing methods of Path will accept either W-values or
        addresses with non-None .W fields

The different classes offer various courtesy conversion and 'address filling
out' methods, such as:

- Path.W2address
- Path.T2address
- Path.t2address
- Subpath.T2address
- Subpath.t2address

Some of these functions require more parameters than others to succeed, but
all are polymorphic and will accept a mix of a partially filled out address
and standalone arguments. If supplied, the partially filled out address will
be filled out and returned. (I.e., the address is filled out as an intended
side effect.) For example, all of the following are valid calls:

    path.W2address(0.5)
    path.W2address(W=0.5)
    path.W2address(Address(W=0.5))

    path.T2address(0.1, 2)
    path.T2address(T=0.1, subpath_index=2)
    path.T2address(Address(T=0.1), subpath_index=2)
    path.T2address(Address(T=0.1, subpath_index=2))

    path.t2address(0.1, segment_index=1, subpath_index=2)
    path.t2address(t=0.1, segment_index=1, subpath_index=2)
    ...(etc)
    path.t2address(Address(t=0.1, segment_index=1, subpath_index=2)

And so on for Subpath.T2address and subpath.t2address (which leave the .W,
.subpath_index fields blank). See the docstrings of each function for more
information.

Note that previously existing methods T2t, t2T have been removed in order to
encourage the use of addresses as a more uniform (and less error-prone and
also more complete, see below) currency for describing positions.

Lastly it can be noted that addresses provide a more fine-grained control over
the position of a point than a global parameter such as 'T' or 'W' alone does.
E.g., anywhere a curve is discontinuous, the two endpoints on either side of
the discontinuity are associated to the same value of the global parameter,
but do not have the same address. (In such cases, methods such as
Path.point(W) return the 'first occurrence' of a point with parameter W. By
calling Path.point(address) for an appropriate value of address, by contrast,
one can retrieve points on either side of the discontinuity.) Likewise,
T-values may point to points on the boundary between two segments, and thus
may not uniquely specify an associated segment, whereas an address can be used
to unambiguously specify a segment on a Subpath, in addition to a point within
that segment.

(END OF IMPROMPTU README)
"""

# Address and related methods ##################################################


ADDRESS_FIELD_NAMES = ['t', 'T', 'W', 'segment_index', 'subpath_index']


class Address(object):
    # (this constructor is very permissive; I wonder what would crash if it implemented some more checks)
    def __init__(self, **kw):
        assert all(key in ADDRESS_FIELD_NAMES for key in kw)
        for key in ['t', 'T', 'W']:
            val = kw.get(key, None)
            if val is not None and (val < 0 or val > 1):
                raise ValueError("out-of-bounds Address parameter:", key, "val:", val)
        self._t = kw.get('t')
        self._T = kw.get('T', None)
        self._W = kw.get('W', None)
        self._segment_index = kw.get('segment_index', None)
        self._subpath_index = kw.get('subpath_index', None)

    def __repr__(self, decimals=None):
        if decimals is None:
            return \
                f'Address t={self.t} T={self.T} W={self.W} ' + \
                f'segment_index={self.segment_index} ' + \
                f'subpath_index={self.subpath_index}'

        else:
            t_str = f'{self.t:.{decimals}f}' if self.t is not None else 'None'
            T_str = f'{self.T:.{decimals}f}' if self.T is not None else 'None'
            W_str = f'{self.W:.{decimals}f}' if self.W is not None else 'None'
            return \
                f'Address t={t_str} T={T_str} W={W_str} ' + \
                f'segment_index={self.segment_index} ' + \
                f'subpath_index={self.subpath_index}'

    # this is useful because otherwise float '1.0' vs int '1' causes two
    # numerically identical addresses to be considered unequal:
    def __eq__(self, other):
        if not isinstance(other, Address):
            raise ValueError
        return \
            self._t == other._t and \
            self._T == other._T and \
            self._W == other._W and \
            self._segment_index == other._segment_index and \
            self._subpath_index == other._subpath_index

    def __lt__(self, other):
        if not isinstance(other, Address):
            raise ValueError("Address compared to non-Address")

        if (self.W is None) != (other.W is None):
            raise ValueError

        if self.W is not None and self.W > other.W:
            if self.subpath_index < other.subpath_index:
                raise ValueError("inconsistency between W, subpath_index fields in address __lt__ comparison")
            if self.subpath_index == other.subpath_index:
                assert self.erase_W_subpath_index_part() > other.erase_W_subpath_index_part()
            return False

        if self.W is not None and self.W < other.W:
            if self.subpath_index > other.subpath_index:
                raise ValueError("inconsistency between W, subpath_index fields in address __lt__ comparison")
            if self.subpath_index == other.subpath_index:
                assert self.erase_W_subpath_index_part() < other.erase_W_subpath_index_part()
            return True

        if (self.subpath_index is None) != (other.subpath_index is None):
            raise ValueError

        if self.subpath_index is not None and self.subpath_index > other.subpath_index:
            return False
        if self.subpath_index is not None and self.subpath_index < other.subpath_index:
            return True

        if (self.T is None) != (other.T is None):
            raise ValueError

        if self.T is not None and self.T > other.T:
            if self.segment_index < other.segment_index:
                raise ValueError("inconsistency between W, segment_index fields in address __lt__ comparison")
            if self.segment_index == other.segment_index:
                assert self.erase_T_segment_index_part() > other.erase_T_segment_index_part()
            return False
        if self.T is not None and self.T < other.T:
            if self.segment_index > other.segment_index:
                raise ValueError("inconsistency between W, segment_index fields in address __lt__ comparison")
            if self.segment_index == other.segment_index:
                assert self.erase_T_segment_index_part() < other.erase_T_segment_index_part()
            return True

        if (self.segment_index is None) != (other.segment_index is None):
            raise ValueError
        if self.segment_index is not None and self.segment_index > other.segment_index:
            return False
        if self.segment_index is not None and self.segment_index < other.segment_index:
            return True

        if (self.t is None) != (other.t is None):
            raise ValueError
        if self.t is not None and self.t > other.t:
            return False
        if self.t is not None and self.t < other.t:
            return True

        return False

    def __gt__(self, other):
        return other.__lt__(self)

    def is_complete(self, for_object=None):
        if for_object is None:
            for_object = Path()

        assert isinstance(for_object, Curve)

        if self._t is None:
            return False

        if not isinstance(for_object, Segment) and \
           (self._T is None or self._segment_index is None):
            return False

        if not isinstance(for_object, ContinuousCurve) and \
           (self._W is None or self._subpath_index is None):
            return False

        return True

    def __getitem__(self, name):
        assert name in ADDRESS_FIELD_NAMES
        return eval("self._" + name)

    def __setitem__(self, name, value):  # why am I allowing this?
        assert name in ADDRESS_FIELD_NAMES
        exec("self._" + name + " = value")

    def tweaked(self, **kw):
        assert all(key in ADDRESS_FIELD_NAMES for key in kw)
        # a = Address(**kw)
        # for name in ADDRESS_FIELD_NAMES:
        #     if name not in kw:
        #         a[name] = self[name]
        # return a
        new_kws = {}
        for name in ADDRESS_FIELD_NAMES:
            new_kws[name] = kw[name] if name in kw else self[name]
        return Address(**new_kws)

    def erase_W_subpath_index_part(self):
        return self.tweaked(W=None, subpath_index=None)

    def erase_T_segment_index_part(self):
        return self.tweaked(T=None, segment_index=None)

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, val):
        if val is not None:
            assert 0 <= val <= 1
            if self._t is None:
                self._t = val
            else:
                if self._t != val and not np.isclose(self._t, val):
                    raise ValueError(f"Attempt to overwrite Address.t = "
                                     f"{self._t} with {val}")

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, val):
        if val is not None:
            assert 0 <= val <= 1
            if self._T is None:
                self._T = val
            else:
                if self._T != val and not np.isclose(self._T, val):
                    raise ValueError(f"Attempt to overwrite Address.T = "
                                     f"{self._T} with {val}")

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, val):
        if val is not None:
            assert 0 <= val <= 1
            if self._W is None:
                self._W = val
            else:
                if self._W != val and not np.isclose(self._W, val):
                    raise ValueError(f"Attempt to overwrite Address.W = "
                                     f"{self._W} with {val}")

    @property
    def segment_index(self):
        return self._segment_index

    @segment_index.setter
    def segment_index(self, val):
        if val is not None:
            assert isinstance(val, int) and val >= 0
            if self._segment_index is None:
                self._segment_index = val
            else:
                if self._segment_index != val:
                    raise ValueError(f"Attempt to overwrite Address."
                                     f"segment_index = {self._segment_index}"
                                     f"with {val}")

    @property
    def subpath_index(self):
        return self._subpath_index

    @subpath_index.setter
    def subpath_index(self, val):
        if val is not None:
            assert isinstance(val, int) and val >= 0
            if self._subpath_index is None:
                self._subpath_index = val
            else:
                if self._subpath_index != val:
                    raise ValueError("Attempt to overwrite Address."
                                     f"subpath_index = {self._subpath_index}"
                                     f"with {val}")


class ValueAddressPair(object):
    def __init__(self, value=None, address=None):
        self.value = value
        self.address = address

    def __iter__(self):
        return [self.value, self.address].__iter__()


def address2param(owner, param_or_address):
    if isinstance(param_or_address, Number):
        return param_or_address

    elif isinstance(owner, Segment):
        return param_or_address.t

    elif isinstance(owner, Subpath):
        return param_or_address.T

    elif isinstance(owner, Path):
        return param_or_address.W

    raise ValueError("Unknown type of owner in address2param.")


def param2address(owner, param_or_address):
    assert isinstance(param_or_address, Number) or isinstance(param_or_address, Address)

    if isinstance(param_or_address, Address):
        return param_or_address

    if isinstance(owner, Segment):
        return owner.t2address(param_or_address)

    if isinstance(owner, Subpath):
        return owner.T2address(param_or_address)

    if isinstance(owner, Path):
        return owner.W2address(param_or_address)

    raise ValueError("recalcitrant owner in param2address")


def complete_address_from_t_and_indices(thing, address):
    if address.t is None:
        raise ValueError("missing t")

    if isinstance(thing, Segment):
        pass

    elif isinstance(thing, Subpath):
        if address.segment_index is None:
            raise ValueError("missing segment_index")
        thing.t2address(address)

    if isinstance(thing, Path):
        if address.segment_index is None:
            raise ValueError("missing segment_index")
        if address.subpath_index is None:
            raise ValueError("missing subpath_index")
        thing[address.subpath_index].t2address(address)
        thing.T2address(address)

    else:
        raise ValueError("unknown type of thing")

    assert address.is_complete(for_object=thing)


def address_pair_from_t1t2(t1, t2):
    return (Address(t=t1), Address(t=t2))


def address_pair_from_t1t2_tuple(p):
    return (Address(t=p[0]), Address(t=p[1]))


# __repr__ craziness... all the options & their processing ###################


_repr_option_names = {
    'use_keywords',
    'use_parens',
    'use_commas',
    'use_oneline',
    'use_fixed_indent',
    'constructor_ready',
    'pad_operators',
    'indent_size',
    'decimals'
}


def _load_repr_options_for(shortname):
    to_return = {
        'use_keywords': False,
        'use_parens': False,
        'use_commas': True,
        'use_oneline': False,
        'use_fixed_indent': True,
        'constructor_ready': False,
        'pad_operators': False,
        'indent_size': 4,
        'decimals': None
    }

    assert all(k in _repr_option_names for k in to_return)
    assert all(k in to_return for k in _repr_option_names)

    if shortname in ['path', 'subpath']:
        return to_return

    to_return['use_oneline'] = True

    return to_return


def _parse_repr_options(shortname, options, give_details=False):
    def print_lists(msg):
        print(msg + ":")
        for l in lists:
            print("  ", l)

    shortnames = {'path', 'subpath', 'line', 'quadratic', 'cubic', 'arc'}
    qualifiers = shortnames.union({'segment'})

    assert shortname in shortnames

    is_segment = shortname in ['line', 'quadratic', 'cubic', 'arc']

    if isinstance(options, dict):
        if any(o not in _repr_option_names for o in options):
            raise ValueError("bad repr_option")
        return options

    if not isinstance(options, str):
        raise ValueError("expecting options in form of string or dict")

    relevant_qualifiers = {shortname}.union({'segment'} if is_segment else set([]))
    extraneous_qualifiers = qualifiers.difference(relevant_qualifiers)

    # normalizing...
    options = options.replace('segments', 'segment')
    options = options.replace(',', ' ')

    # turn options into lists
    options_list = options.split()
    assert '' not in options_list
    lists = [o.split('.') for o in options_list]
    if give_details:
        print_lists("original lists")

    # remove options targeting other objects
    lists = [o for o in lists if any(x in o for x in relevant_qualifiers)]
    if give_details:
        print_lists(f"lists after keeping only {shortname}-relevant options")

    # removing extraneous qualifiers:
    lists = [[e for e in o if e not in extraneous_qualifiers] for o in lists]
    if give_details:
        print_lists("lists after purging extraneous qualifiers")

    # create list of used keys, and normalize missing values to 'true':
    keys = []
    for o in lists:
        if o[0] in _repr_option_names:
            val_index = 1

        else:
            if o[0] not in relevant_qualifiers or \
               o[1] not in _repr_option_names:
                raise ValueError("format error (1) for _repr_ option:", o)
            val_index = 2

        if not val_index <= len(o) <= val_index + 1:
            raise ValueError("format error (2) for _repr_ option:", o)

        if len(o) == val_index + 1 and \
           o[val_index] not in ['true', 'false'] and \
           o[val_index - 1] != 'indent_size':
            raise ValueError("format error (3) for _repr_ option:", o)

        if len(o) < val_index + 1 and \
           o[val_index - 1] != 'indent_size':
            o.append('true')

        if len(o) < val_index + 1:
            assert o[val_index] == 'indent_size'
            raise ValueError("format error (4) for _repr_ option:", o)

        assert len(o) == val_index + 1  # everybody has a value

        keys.append(o[:val_index])

    if give_details:
        print_lists(f"lists after adding default 'true' fields")

    # remove keys whose specializations already exist
    indices_to_delete = []
    assert len(keys) == len(lists)
    for index, key in enumerate(keys):
        if len(key) == 1:
            assert key[0] in _repr_option_names
            new_key = [shortname, key[0]]

        elif (len(key) == 2 and
              key[0] == 'segment'):
            assert is_segment
            assert key[1] in _repr_option_names
            new_key = [shortname, key[1]]

        else:
            continue

        if new_key in keys:
            indices_to_delete.append(index)

    last_deleted = len(lists)
    for index in reversed(indices_to_delete):
        assert index < last_deleted
        del lists[index]
        last_deleted = index

    if give_details:
        print_lists(f"lists after purging superceded keys")

    # now we can pop shortname or 'segment' from lists
    for o in lists:
        if o[0] == shortname:
            del o[0]

        elif o[0] == 'segment':
            assert is_segment
            del o[0]

        assert len(o) == 2
        assert o[0] in _repr_option_names

    if give_details:
        print_lists(f"lists after popping shortname, 'segment'")

    # parse the second (last) coordinate of each list
    for o in lists:
        assert o[0] in _repr_option_names

        if o[0] == 'indent_size':
            try:
                val = int(o.pop())
                if val < 0:
                    raise ValueError("")
                o.append(val)
            except ValueError:
                raise ValueError("couldn't convert indent_size to int")

        else:
            val = o.pop()
            if val == 'true':
                o.append(True)

            elif val == 'false':
                o.append(False)

            else:
                assert False

        assert len(o) == 2

    # check against duplicates
    for index1, o1 in enumerate(lists):
        for index2, o2 in enumerate(lists):
            if index1 == index2:
                continue
            if o1[0] == o2[0]:
                raise ValueError("redundant repr_option")

    parsed_options = {}
    for o in lists:
        assert len(o) == 2
        assert o[0] not in parsed_options
        assert o[0] in _repr_option_names
        assert \
            isinstance(o[1], bool) or \
            (isinstance(o[1], int) and o[0] == 'indent_size')
        parsed_options[o[0]] = o[1]

    if 'constructor_ready' in parsed_options and \
       'pad_operators' not in parsed_options and \
       parsed_options['constructor_ready'] is True:
        parsed_options['pad_operators'] = True

    return parsed_options


def _check_repr_options(options):
    for key, value in options.items():
        if key not in _repr_option_names:
            raise ValueError("unknown key:", key)
        if key == 'indent_size':
            if not isinstance(value, int):
                raise ValueError("bad indent_size")
        else:
            if not isinstance(value, bool):
                raise ValueError("expecting bool option for key", key)


# Curve and its descendants ##################################################


class Curve(object):
    @classmethod
    def shortname(cls):
        if cls is Path:
            return 'path'
        elif cls is Subpath:
            return 'subpath'
        elif cls is Line:
            return 'line'
        elif cls is QuadraticBezier:
            return 'quadratic'
        elif cls is CubicBezier:
            return 'cubic'
        elif cls is Arc:
            return 'arc'
        raise ValueError("unknown class in shortname")

    def _repr_options_init(self):
        self._own_repr_options = self._class_repr_options.copy()

    @classmethod
    def update_class_repr_options(cls, options):
        if cls is BezierSegment:
            Line.update_class_repr_options(options)
            QuadraticBezier.update_class_repr_options(options)
            CubicBezier.update_class_repr_options(options)

        elif cls is Segment:
            BezierSegment.update_class_repr_options(options)
            Arc.update_class_repr_options(options)

        else:
            if cls not in [Path, Subpath, Line, Arc,
                           QuadraticBezier, CubicBezier]:
                raise ValueError("unknown class")
            options = _parse_repr_options(cls.shortname(), options)
            _check_repr_options(options)
            cls._class_repr_options.update(options)

    def update_repr_options(self, options):
        options = _parse_repr_options(cls.shortname(), options)
        _check_repr_options(options)
        self._own_repr_options.update(options)

    def __repr__(self, tmp_options={}, forced_options={}, decimals=None):
        def repr_num(z):
            if options['decimals'] is None:
                string = str(z)

            else:
                string = f"{z:.{options['decimals']}f}"

            if not options['use_parens']:
                string = string.strip('()')

            if options['pad_operators']:
                string = ' + '.join(string.split('+'))
                if string.startswith('-'):
                    string = '-' + ' - '.join(string.split('-')[1:])
                else:
                    string = ' - '.join(string.split('-'))

            return string

        options = self._own_repr_options.copy()
        local_tmp_options = _parse_repr_options(self.shortname(), tmp_options)
        if decimals is not None:
            local_tmp_options.update({'decimals': decimals})
        options.update(local_tmp_options)
        options.update(forced_options)

        if options['constructor_ready']:
            options['use_commas'] = True

        if options['use_oneline']:
            options['use_fixed_indent'] = False

        # first join
        first_join = ''
        if options['use_fixed_indent']:
            first_join = '\n'  # I guess we'll leave this here even for paths; not much harm?
            first_join += ' ' * options['indent_size']

        # comma
        comma = ',' if options['use_commas'] else ''

        # join
        join = comma + ' '
        if not options['use_oneline']:
            if options['use_fixed_indent']:
                num_spaces = options['indent_size']
            else:
                num_spaces = len(self.__class__.__name__) + 1
            join = comma + '\n' + ' ' * num_spaces

        # newline_replacement
        newline_replacement = \
            ' ' if options['use_oneline'] \
            else '\n' + ' ' * num_spaces

        # last join
        last_join = ''
        if options['use_fixed_indent']:
            last_join = '\n'

        # ztring
        zstring = ''
        if isinstance(self, Subpath) and self._Z:
            zstring = \
                '.set_Z()' if options['constructor_ready'] \
                else '.Z'

        # and here we go...
        string = self.__class__.__name__ + f'({first_join}'

        if isinstance(self, Segment):
            for index, key in enumerate(self._field_names, 1):
                if options['use_keywords']:
                    string += key + '='
                string += repr_num(eval("self._" + key))
                if index < len(self._field_names):
                    string += join

        else:
            assert isinstance(self, Subpath) or isinstance(self, Path)
            # options that should be inherited (are there more?):
            forced_options = {}

            if options['use_oneline']:
                forced_options.update({'use_oneline': True})

            if options['constructor_ready']:
                forced_options.update({'constructor_ready': True})

            for index, thing in enumerate(self, 1):
                s = thing.__repr__(tmp_options=tmp_options,
                                   forced_options=forced_options)
                s = s.replace('\n', newline_replacement)
                string += s
                if index < len(self):
                    string += join

        string += f'{last_join}){zstring}'

        return string

    def rotated(self, degs, origin=0j):
        """Returns a copy of self rotated by `degs` degrees (CCW) around the
        point `origin` (a complex number).  By default `origin` is either
        `self.point(0.5)`, or in the case that self is an Arc object, `origin`
        defaults to `self.center`."""
        return rotate(self, degs, origin=origin)

    def translated(self, z0):
        """Returns a copy of self shifted by the complex quantity `z0` such
        that self.translated(z0).point(t) = self.point(t) + z0 for any t."""
        return translate(self, z0)

    def scaled(self, x, y=None, origin=0j):
        """Scales x coordinates by x, y coordinates by y; y defaults to x if
        y == None."""
        if x.imag != 0 or (y is not None and y.imag != 0):
            raise ValueError("scaled takes one or two real-valued inputs to avoid ambiguities")
        return scale(self, x, y, origin)

    def transformed(self, tf):
        return transform(self, tf)

    def cloned(self):
        return self.translated(0)

    def ilength_01(self, s, s_tol=ILENGTH_S_TOL, maxits=ILENGTH_MAXITS,
                   error=ILENGTH_ERROR, min_depth=ILENGTH_MIN_DEPTH):
        """Returns a float, u, such that self.length(0, u) is approximately s * self.length().
        See the inv_arclength_01() docstring for more details."""
        return inv_arclength_01(self, s, s_tol=s_tol, maxits=maxits, error=error, min_depth=min_depth)

    def ilength(self, s, s_tol=ILENGTH_S_TOL, maxits=ILENGTH_MAXITS,
                error=ILENGTH_ERROR, min_depth=ILENGTH_MIN_DEPTH):
        """Returns a float, u, such that self.length(0, u) is approximately s.
        See the inv_arclength() docstring for more details."""
        return inv_arclength(self, s, s_tol=s_tol, maxits=maxits, error=error, min_depth=min_depth)

    def arcpoint(self, s, s_tol=ILENGTH_S_TOL, maxits=ILENGTH_MAXITS,
                 error=ILENGTH_ERROR, min_depth=ILENGTH_MIN_DEPTH):
        """Returns point at arclength s from start of curve."""
        return self.point(self.ilength(s, s_tol=s_tol, maxits=maxits, error=error, min_depth=min_depth))

    def closest_point_to(self, pt):
        """returns a pair (d_min, address_min) where d_min minimizes
        d = |self.point(address) - pt|
        over all addresses for self, and where address_min is the first
        address (w.r.t. self) for which the minimum value is attained."""
        return self.radialrange(pt)[0]

    def farthest_point_from(self, pt):
        """returns a pair (d_max, address_max) where d_max maximizes
        d = |self.point(address) - pt|
        over all addresses for self, and where address_min is the first
        address (w.r.t. self) for which the maximum value is attained."""
        return self.radialrange(pt)[1]

    def radialrange(self, origin, return_all_global_extrema=False):
        """returns the tuples (d_min, address_min), (d_max, address_max)
        which minimize and maximize, respectively, the distance
        d = |self.point(address) - origin|."""
        """overwritten by BezierSegment and Arc:"""
        assert isinstance(self, Path) or isinstance(self, Subpath)

        if return_all_global_extrema:
            raise NotImplementedError

        global_min = ValueAddressPair(value=np.inf, address=None)
        global_max = ValueAddressPair(value=0, address=None)

        # in the following loop, a is an address:
        for segment, a in segments_and_partial_addresses_in(self):
            seg_min, seg_max = segment.radialrange(origin)
            if seg_min.value < global_min.value:
                global_min.value = seg_min.value
                global_min.address = a.tweaked(t=seg_min.address.t)
            if seg_max.value > global_max.value:
                global_max.value = seg_max.value
                global_max.address = a.tweaked(t=seg_max.address.t)

        if num_segments_in(self) > 0:
            assert isinstance(global_min.address, Address)
            assert isinstance(global_max.address, Address)
            self.t2address(global_min.address)
            self.t2address(global_max.address)

        return global_min, global_max

    def multisplit(self, time_values):
        """
        Takes a possibly empty list of values u1, ..., un such that
        0 <= u1 < ... < un <= 1 and returns a list of objects of the same type
        as curve whose union is curve and whose endpoints are curve.point(0),
        curve.point(s1), ..., curve.point(k), curve.point(1) where 0 < s1 < ...
        < k < 1 and

        {0, s1, ..., sk, 1} = {0, u1, ..., un, 1 }
        """
        assert isinstance(time_values, list)
        assert all(isinstance(x, Number) for x in time_values)
        return multisplit(self, time_values)

    def xbox(self, stroke_width=None):
        """
        Returns xmin, xmax: smallest and lowest x-coordinates of curve.
        Overwritten by BezierSegment, Line and Arc.
        """
        if stroke_width is not None:
            return self.stroke(stroke_width).xbox()

        boxes = [thing.xbox() for thing in self]
        return min(b[0] for b in boxes), max(b[1] for b in boxes)

    def ybox(self, stroke_width=None):
        """
        Returns ymin, ymax: smallest and lowest y-coordinates of curve.
        Overwritten by BezierSegment, Line and Arc.
        """
        if stroke_width is not None:
            return self.stroke(stroke_width).ybox()

        boxes = [thing.ybox() for thing in self]
        return \
            min(b[0] for b in boxes), \
            max(b[1] for b in boxes)

    def bbox(self, stroke_width=None):
        """
        Returns the bounding box for the curve in the form xmin, xmax, ymin, ymax.
        """
        if stroke_width is not None:
            return self.stroke(stroke_width).bbox()

        xmin, xmax = self.xbox(stroke_width)
        ymin, ymax = self.ybox(stroke_width)
        return xmin, xmax, ymin, ymax

    def bbox_width(self, stroke_width=None):
        xmin, xmax = self.xbox(stroke_width=stroke_width)
        return xmax - xmin

    def bbox_height(self, stroke_width=None):
        ymin, ymax = self.ybox(stroke_width=stroke_width)
        return ymax - ymin

    def scale_about_centroid(self, sx, sy):
        xmin, xmax, ymin, ymax = self.bbox()
        centroid = (xmin + xmax) / 2 + 1j * (ymin + ymax) / 2
        return self.transformed(parse_transform(
            'translate', centroid,
            'scale', sx, sy,
            'translate', -centroid
        ))

    @property
    def xmin(self):
        return self.xbox()[0]

    @property
    def xmax(self):
        return self.xbox()[1]

    @property
    def ymin(self):
        return self.ybox()[0]

    @property
    def ymax(self):
        return self.ybox()[1]

    def point_outside(self):
        """
        Returns an arbitrary point outside the curve's bounding box.
        """
        xmin, xmax, ymin, ymax = self.bbox()
        return xmin - 42 + (ymin - 43) * 1j

    def normal(self, param_or_address):
        """
        Returns the (right hand rule) unit normal vector to self at the
        global time parameter specified by 'param_or_address'. (Nb: 'global'
        means 'W' for paths, 'T' for subpaths, 't' for segments.)

        E.g., call as 'Path.normal(0.5)', 'Path.normal(Address(W=0.5))', etc.
        """
        return -1j * self.unit_tangent(param_or_address)

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, val):
        raise Exception("Sorry, segments are immutable!")

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, val):
        raise Exception("Sorry, segments are immutable!")

    def num_segments(self):
        return num_segments_in(self)


class ContinuousCurve(Curve):
    pass


class Segment(ContinuousCurve):
    def curvature(self, t_or_address, use_inf=False):
        """returns the curvature of the segment at t.

        (overwritten by Line, for which the value is 0)

        Notes   # [vistuleB: is this note still relevant, given code below??]
        - - - - -
        If you receive a RuntimeWarning, run command
        >>> old = np.seterr(invalid='raise')
        This can be undone with
        >>> np.seterr(**old)
        """
        t = param2address(self, t_or_address).t
        dz = self.derivative(t)
        ddz = self.derivative(t, n=2)
        dx, dy = dz.real, dz.imag
        ddx, ddy = ddz.real, ddz.imag
        old_np_seterr = np.seterr(invalid='raise')
        try:
            kappa = abs(dx * ddy - dy * ddx) / sqrt(dx * dx + dy * dy)**3
        except (ZeroDivisionError, FloatingPointError):
            # tangent vector is zero at t, use polytools to find limit
            p = self.poly()
            dp = p.deriv()
            ddp = dp.deriv()
            dx, dy = real(dp), imag(dp)
            ddx, ddy = real(ddp), imag(ddp)
            f2 = (dx * ddy - dy * ddx)**2
            g2 = (dx * dx + dy * dy)**3
            lim2 = rational_limit(f2, g2, t)
            if lim2 < 0:  # impossible, must be numerical error
                return 0
            kappa = sqrt(lim2)
        finally:
            np.seterr(**old_np_seterr)
        return kappa

    def segment_at_address(self, address):
        return self

    def t2address(self, t):
        a = t if isinstance(t, Address) else Address(t=t)
        if a.t is None:
            raise ValueError("no source of value for t in Segment.t2address")
        return a

    def is_or_has_arc(self):
        return isinstance(self, Arc)

    def is_vertical(self):
        # overwritten by Line
        return False

    def is_horizontal(self):
        # overwritten by Line
        return False

    def first_segment(self):
        return self

    def last_segment(self):
        return self

    def matched_end_with_start_of(self, other):
        other = other.first_segment()
        if other is None:
            raise ValueError("'None' segment in Segment.matched_end_with_start_of")

        if self.end == other.start:
            return self.cloned()

        if not np.isclose(self.end, other.start):
            raise ValueError("unexpected distance in matched_end_with_start_of")

        if self.is_vertical() and \
           other.is_vertical() and \
           self.end.real != other.start.real:
            raise ValueError("incompatibility between vertical segments in"
                             "Segment.matched_end_with_start_of")

        if self.is_horizontal() and \
           other.is_horizontal() and \
           self.end.imag != other.start.imag:
            raise ValueError("incompatibility between horizontal segments in"
                             "Segment.matched_end_with_start_of")

        if self.is_vertical():
            new_x = self.end.real

        elif other.is_vertical():
            new_x = other.start.real

        else:
            new_x = (self.end.real + other.start.real) * 0.5

        if self.is_horizontal():
            new_y = self.start.imag

        elif other.is_horizontal():
            new_y = other.start.imag

        else:
            new_y = (self.end.imag + other.start.imag) * 0.5

        return self.tweaked(end=new_x + 1j * new_y)

    def matched_start_with_end_of(self, other):
        other = other.last_segment()
        if other is None:
            raise ValueError("'None' segment in Segment.matched_start_with_end_of")

        if self.start == other.end:
            return self.cloned()

        if not np.isclose(self.start, other.end):
            raise ValueError("unexpected distance in matched_end_with_start_of")

        if self.is_vertical() and \
           other.is_vertical() and \
           self.start.real != other.end.real:
            raise ValueError("incompatibility between vertical segments in"
                             "Segment.matched_start_with_end_of")

        if self.is_horizontal() and \
           other.is_horizontal() and \
           self.start.imag != other.end.imag:
            raise ValueError("incompatibility between horizontal segments in"
                             "Segment.matched_start_with_end_of")

        if self.is_vertical():
            new_x = self.start.real

        elif other.is_vertical():
            new_x = other.end.real

        else:
            new_x = (self.start.real + other.end.real) * 0.5

        if self.is_horizontal():
            new_y = self.start.imag

        elif other.is_horizontal():
            new_y = other.end.imag

        else:
            new_y = (self.start.imag + other.end.imag) * 0.5

        return self.tweaked(start=new_x + 1j * new_y)

    def stroke(self, width, quality=0.01, safety=5,
               cap='butt', reversed=False):
        return Subpath(self).stroke(width, quality=quality, safety=safety,
                                    cap=cap, reversed=reversed)


class BezierSegment(Segment):
    def pro_offset(self, amount, quality=0.01, safety=5, two_sided=False):
        """all offset "hard work" ends up happening here and in
        BezierSegment.naive_offset"""
        assert quality > 0
        to_test = [self, self.reversed()] if two_sided else [self]
        naive = []
        max_divergence = 0

        for s in to_test:
            naive.append(s.naive_offset(amount))
            divergence = divergence_of_offset(s, naive[-1], amount,
                                              early_return_threshold=quality,
                                              safety=safety)
            max_divergence = max(divergence, max_divergence)
            if max_divergence > quality:
                break

        if max_divergence <= quality:
            way_out  = Subpath(naive[0])
            way_in   = Subpath(naive[1]) if two_sided else Subpath()
            skeleton = Subpath(self)

        else:
            s1, s2 = self.split(0.5)
            wo1, sk1, wi1 = s1.pro_offset(amount, quality, safety, two_sided)
            wo2, sk2, wi2 = s2.pro_offset(amount, quality, safety, two_sided)
            if wo1.end != wo2.start:
                assert np.isclose(wo1.end, wo2.start)
                wo2[0]._start = wo1.end
                wo2._reset()
            if wi2.end != wi1.start:
                assert np.isclose(wi2.end, wi1.start)
                wi1[0]._start = wi2.end
                wi1._reset()
            way_out  = wo1.extend(wo2)
            way_in   = wi2.extend(wi1)
            skeleton = sk1.extend(sk2)

        return way_out, skeleton, way_in

    def offset(self, amount, quality=0.01, safety=5):
        return self.pro_offset(amount, quality, safety)[0]

    def radialrange(self, origin, return_all_global_extrema=False):
        """
        Returns the tuples (d_min, address_min) and (d_max, address_max)
        which minimize and maximize, respectively, the distance
        d = |self.point(address.t) - origin|.
        """
        min_d_and_t, max_d_and_t = bezier_radialrange(
            self,
            origin,
            return_all_global_extrema=return_all_global_extrema
        )
        assert len(min_d_and_t) == 2
        assert all(isinstance(x, Number) for x in min_d_and_t)

        return \
            ValueAddressPair(value=min_d_and_t[0], address=Address(t=min_d_and_t[1])), \
            ValueAddressPair(value=max_d_and_t[0], address=Address(t=max_d_and_t[1]))

    @property
    def xmin(self):
        xmin, _ = bezier_xbox(self)
        return xmin

    @property
    def xmax(self):
        _, xmax = bezier_xbox(self)
        return xmax

    @property
    def ymin(self):
        ymin, _ = bezier_ybox(self)
        return ymin

    @property
    def ymax(self):
        _, ymax = bezier_ybox(self)
        return ymax

    def xbox(self, stroke_width=None):
        # Overwritten by Line
        if stroke_width is not None:
            return self.stroke(stroke_width).xbox()
        return bezier_xbox(self)

    def ybox(self, stroke_width=None):
        # Overwritten by Line
        if stroke_width is not None:
            return self.stroke(stroke_width).ybox()
        return bezier_ybox(self)

    def unit_tangent(self, t_or_address):
        """
        (overwritten by Line)

        returns the unit tangent vector of the segment at t (centered at the
        origin and expressed as a complex number).  If the tangent vector's
        magnitude is zero, this method will find the limit of
        self.derivative(u) / abs(self.derivative(u)) as u approaches t.
        """
        t = address2param(self, t_or_address)
        return bezier_unit_tangent(self, t)

    def __eq__(self, other, tol=0):
        if type(self) != type(other):
            return NotImplemented
        return all(abs(x - y) <= tol for x, y in
                   zip(self.bpoints, other.bpoints))

    def __ne__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return not self == other

    def __getitem__(self, item):
        return self.bpoints[item]

    def __len__(self):
        return len(list(self.bpoints))

    def x_val_intersect(self, x_val):
        t_vals = bezier_x_value_intersections(self, x_val)
        return [Address(t=t) for t in t_vals]

    def y_val_intersect(self, y_val):
        t_vals = bezier_y_value_intersections(self, y_val)
        return [Address(t=t) for t in t_vals]

    def intersect(self, other_seg, tol=1e-12):
        """
        Finds the intersections of two segments. Returns a list of tuples
        (a1, a2) of addresses, such that self.point(a1.t) ==
        other_seg.point(a2.t).

        This should fail if the two segments coincide for more than a
        finite collection of points.
        """
        assert self != other_seg

        if isinstance(self, Line) and isinstance(other_seg, Line):
            t1t2s = line_by_line_intersections(self, other_seg)
            return [address_pair_from_t1t2(t1, t2) for t1, t2 in t1t2s]

        elif isinstance(self, Line) and isinstance(other_seg, BezierSegment):
            t1t2s = bezier_by_line_intersections(other_seg, self)
            return [address_pair_from_t1t2(t2, t1) for t1, t2 in t1t2s]

        elif isinstance(other_seg, Line):
            t1t2s = bezier_by_line_intersections(self, other_seg)
            return [address_pair_from_t1t2(t1, t2) for t1, t2 in t1t2s]

        elif isinstance(other_seg, BezierSegment):
            t1t2s = bezier_intersections(
                self,
                other_seg,
                longer_length=max(self.length(), other_seg.length()),
                tol=sqrt(tol) * 10, tol_deC=tol
            )
            return [address_pair_from_t1t2(t1, t2) for t1, t2 in t1t2s
                    if (0 <= t1 <= 1) and (0 <= t2 <= 1)]

        elif isinstance(other_seg, Arc):
            return [(a1, a2) for a2, a1 in other_seg.intersect(self)]

        elif isinstance(other_seg, Path) or isinstance(other_seg, Subpath):
            raise TypeError(
                "other_seg must be a Segment, not a Path or Subpath; use "
                "Path.intersect() or Subpath.intersect() instead")

        else:
            raise TypeError("other_seg must be a path segment")

    def split(self, t_or_address):
        """returns two segments of same type whose union is this segment and
        which join at self.point(t). (Overwritten by Line.)"""
        t = address2param(self, t_or_address)

        bpoints1, bpoints2 = split_bezier(self.bpoints, t)

        if isinstance(self, QuadraticBezier):
            return QuadraticBezier(*bpoints1), QuadraticBezier(*bpoints2)

        if isinstance(self, CubicBezier):
            return CubicBezier(*bpoints1), CubicBezier(*bpoints2)

        assert False

    def cropped(self, t0_or_address, t1_or_address):
        """returns a cropped copy of the segment which starts at
        self.point(t0) and ends at self.point(t1). Allows t1 >= t0.
        (Overwritten by Line.)"""
        t0 = address2param(self, t0_or_address)
        t1 = address2param(self, t1_or_address)
        return crop_bezier(self, t0, t1)

    def naive_offset(self, amount):
        """
        Returns a cubic bezier segment that approximates the segment's
        offset; requires the segment to have a computable unit tangent and
        second derivative at t = 0, 1.

        (overwritten by Line)
        """

        """
        Let O(t) denote offset at time t. Let UT(t) denote unit tangent at
        time t. We need to compute O'(0), O'(1) in order to set the control
        points to match these values. Have (with d := amount):

        O(t) = p(t) + d * n(t)
        O(t) = p(t) - d * j * UT(t)
        O'(t) = p'(t) - d * j * UT'(t)

        And:

        UT'(t) = (d/dt) p'(t)/|p'(t)|

        So let's first compute d/dt |p'(t)|...

        ...happens to be

             d/dt |p'(t)| = (1 / |p'(t)|) * p'(t).p''(t)

        ...where '.' is a dot product. (Computation ommitted.)

        So (continuing):

        UT'(t) = (p''(t)/|p'(t)|) - p'(t).p''(t)*p'(t)/|p'(t)|^3)

        And (the sought-for derivative):

        O'(t) = p'(t) - d * j * (p''(t)/|p'(t)|) - p'(t).p''(t)*p'(t)/|p'(t)|^3)
        """

        N0 = self.normal(0)
        start = self._start + N0 * amount
        d0 = self.derivative(0)              # p'(0)
        if abs(d0) > 1e-6:
            dd0 = self.derivative(0, n=2)    # p''(0)
            dot = real(d0 * dd0.conjugate())
            ab0 = abs(d0)
            Op0 = d0 - amount * 1j * ((dd0 / ab0) - dot * d0 / ab0**3)
            control1 = start + Op0 / 3
        else:
            control1 = start

        N1 = self.normal(1)
        end = self._end + N1 * amount
        d1 = self.derivative(1)
        if abs(d1) > 1e-6:
            dd1 = self.derivative(1, n=2)
            dot = real(d1 * dd1.conjugate())
            ab1 = abs(d1)
            Op1 = d1 - amount * 1j * ((dd1 / ab1) - dot * d1 / ab1**3)
            control2 = end - Op1 / 3
        else:
            control2 = end

        return CubicBezier(start, control1, control2, end)


def extract_complex(thing):
    if isinstance(thing, Number):
        return thing

    if isinstance(thing, np.ndarray):
        m, n = thing.shape()
        if m >= 2 and n == 1:
            x = float(thing[0, 0])
            y = float(thing[1, 0])
            return x + 1j * y

    try:
        x = float(thing.x)
        y = float(thing.y)
        return x + 1j * y

    except AttributeError:
        pass

    try:
        try:
            x = float(thing['x'])
            y = float(thing['y'])
            return x + 1j * y
        except KeyError:
            raise TypeError
    except TypeError:
        pass

    raise ValueError("Unable to extract complex number from argument")


class Line(BezierSegment):
    _class_repr_options = _load_repr_options_for('line')

    def __init__(self, start, end):
        self._start = extract_complex(start)
        self._end = extract_complex(end)
        self._repr_options_init()
        self._field_names = ['start', 'end']

    def tweaked(self, start=None, end=None):
        start = start if start is not None else self._start
        end = end if end is not None else self._end
        return Line(start, end)

    def point(self, t_or_address):
        """returns the coordinates of the Bezier curve evaluated at t."""
        t = address2param(self, t_or_address)

        # the following fanciness in order not to corrupt x/y coordinates
        # in vertical/horizontal lines, respectively:

        if self._start.real == self._end.real:
            real_part = self._start.real
        else:
            # !!! do not rewrite as combination of the form
            # "start + t * (end - start)":
            real_part = (1 - t) * self._start.real + t * self._end.real

        if self._start.imag == self._end.imag:
            imag_part = self._start.imag
        else:
            # !!! do not rewrite as combination of the form
            # "start + t * (end - start)":
            imag_part = (1 - t) * self._start.imag + t * self._end.imag

        return complex(real_part, imag_part)

    def length(self, t0=0, t1=1, error=None, min_depth=None):
        """
        Returns the signed length of the line segment between t0 and t1, where
        t0, t1 default to 0, 1 and can be given as addresses or as floats.

        The keyword parameters, t0, t1 can also be given as addresses.
        """
        t0 = address2param(self, t0)
        t1 = address2param(self, t1)
        return abs(self._end - self._start) * (t1 - t0)

    @property
    def bpoints(self):
        return self._start, self._end

    @bpoints.setter
    def bpoints(self, val):
        raise Exception("Sorry, segments are immutable!")

    def poly(self, return_coeffs=False):
        """returns the line as a Polynomial object."""
        p = self.bpoints
        coeffs = (p[1] - p[0], p[0])
        return coeffs if return_coeffs else np.poly1d(coeffs)

    def derivative(self, t=None, n=1):
        """returns the nth derivative of the segment at t, which, given that
        the segment is linear, is independent of t"""
        assert self._end != self._start
        if n == 1:
            return self._end - self._start
        elif n > 1:
            return 0
        else:
            raise ValueError("n should be a positive integer")

    def unit_tangent(self, t=None):
        """returns the unit tangent of the segment at t, which, given that
        the segment is linear, is independent of t"""
        assert self._end != self._start
        dseg = self._end - self._start
        return dseg / abs(dseg)

    def curvature(self, t=None):
        """returns the curvature of the line, which is always zero."""
        return 0

    def reversed(self):
        """returns a copy of the Line object with its orientation reversed"""
        return Line(self._end, self._start)

    def xbox(self, stroke_width=None):
        if stroke_width is not None:
            return self.stroke(stroke_width).xbox()

        return \
            min(self._start.real, self._end.real), \
            max(self._start.real, self._end.real)

    def ybox(self, stroke_width=None):
        if stroke_width is not None:
            return self.stroke(stroke_width).ybox()

        return \
            min(self._start.imag, self._end.imag), \
            max(self._start.imag, self._end.imag)

    def parallel_to(self, other):
        if not isinstance(other, Line):
            raise ValueError("expecting Line segment in parallel_to")
        return np.isclose(complex_determinant(self.end - self.start, other.end - other.start), 0)

    def cropped(self, t0_or_address, t1_or_address):
        """returns a cropped copy of this segment which starts at
        self.point(t0) and ends at self.point(t1). Allows t1 >= t0. Allows
        t0, t1 be given as addresses"""
        return Line(self.point(t0_or_address), self.point(t1_or_address))

    def split(self, t_or_address):
        """returns two Lines, whose union is this segment and which join at
        self.point(t)."""
        pt = self.point(t_or_address)
        return Line(self._start, pt), Line(pt, self._end)

    def naive_offset(self, amount):
        """performs a one-sided offset of the line by amount"""
        n = self.normal(0)
        return Line(self._start + amount * n, self._end + amount * n)

    def is_horizontal(self):
        return self._start.imag == self._end.imag

    def is_vertical(self):
        return self._start.real == self._end.real

    def x(self):
        if not self.is_vertical():
            raise ValueError(".x() not defined on non-vertical line segment")
        return self._start.real

    def y(self):
        if not self.is_horizontal():
            raise ValueError(".x() not defined on non-horizontal line segment")
        return self._start.imag


class QuadraticBezier(BezierSegment):
    _class_repr_options = _load_repr_options_for('quadratic')

    def __init__(self, start, control, end):
        self._start = extract_complex(start)
        self._control = extract_complex(control)
        self._end = extract_complex(end)
        self._repr_options_init()
        self._field_names = ['start', 'control', 'end']

        # used to know if self._length needs to be updated:
        self._length_info = {'length': None, 'bpoints': None}

    # def shortname(self):
    #     return 'quadratic'

    def tweaked(self, start=None, control=None, end=None):
        start = start if start is not None else self._start
        control = control if control is not None else self._control
        end = end if end is not None else self._end
        return QuadraticBezier(start, control, end)

    def can_use_T_from_previous(self, previous):
        if isinstance(previous, QuadraticBezier):
            return (self._start == previous.end and
                    (self._control - self._start) == (
                        previous.end - previous.control))
        else:
            return self._control == self._start

    def point(self, t_or_address):
        """returns the coordinates of the Bezier curve evaluated at t."""
        t = address2param(self, t_or_address)
        return \
            (1 - t)**2 * self._start + \
            2 * (1 - t) * t * self._control + \
            t**2 * self._end

    def length(self, t0=0, t1=1, error=None, min_depth=None):
        t0 = address2param(self, t0)
        t1 = address2param(self, t1)
        if t0 == 0 and t1 == 1:
            if self._length_info['bpoints'] == self.bpoints:
                return self._length_info['length']
        a = self._start - 2 * self._control + self._end
        b = 2 * (self._control - self._start)
        a_dot_b = a.real * b.real + a.imag * b.imag

        if abs(a) < 1e-12:
            s = abs(b) * (t1 - t0)
        elif abs(a_dot_b + abs(a) * abs(b)) < 1e-12:
            tstar = abs(b) / (2 * abs(a))
            if t1 < tstar:
                return abs(a) * (t0**2 - t1**2) - abs(b) * (t0 - t1)
            elif tstar < t0:
                return abs(a) * (t1**2 - t0**2) - abs(b) * (t1 - t0)
            else:
                return abs(a) * (t1**2 + t0**2) - abs(b) * (t1 + t0) + \
                    abs(b)**2 / (2 * abs(a))
        else:
            c2 = 4 * (a.real**2 + a.imag**2)
            c1 = 4 * a_dot_b
            c0 = b.real**2 + b.imag**2

            beta = c1 / (2 * c2)
            gamma = c0 / c2 - beta**2

            dq1_mag = sqrt(c2 * t1**2 + c1 * t1 + c0)
            dq0_mag = sqrt(c2 * t0**2 + c1 * t0 + c0)
            logarand = (sqrt(c2) * (t1 + beta) + dq1_mag) / \
                       (sqrt(c2) * (t0 + beta) + dq0_mag)

            s = (t1 + beta) * dq1_mag - (t0 + beta) * dq0_mag + \
                gamma * sqrt(c2) * log(logarand)
            s /= 2

        if t0 == 1 and t1 == 0:
            self._length_info['length'] = s
            self._length_info['bpoints'] = self.bpoints
            return self._length_info['length']
        else:
            return s

    @property
    def bpoints(self):
        return self._start, self._control, self._end

    @bpoints.setter
    def bpoints(self, val):
        raise Exception("Sorry, segments are immutable!")

    def poly(self, return_coeffs=False):
        """returns the quadratic as a Polynomial object."""
        p = self.bpoints
        coeffs = (p[0] - 2 * p[1] + p[2],
                  2 * (p[1] - p[0]),
                  p[0])
        return coeffs if return_coeffs else np.poly1d(coeffs)

    def derivative(self, t_or_address, n=1):
        """returns the nth derivative of the segment at t.
        Note: Bezier curves can have points where their derivative vanishes.
        If you are interested in the tangent direction, use the unit_tangent()
        method instead."""
        t = address2param(self, t_or_address)
        p = self.bpoints
        if n == 1:
            return 2 * ((p[1] - p[0]) * (1 - t) + (p[2] - p[1]) * t)
        elif n == 2:
            return 2 * (p[2] - 2 * p[1] + p[0])
        elif n > 2:
            return 0
        else:
            raise ValueError("n should be a positive integer.")

    def reversed(self):
        """returns a copy of the QuadraticBezier object with its orientation
        reversed."""
        new_quad = QuadraticBezier(self._end, self._control, self._start)
        if self._length_info['length']:
            new_quad._length_info = self._length_info
            new_quad._length_info['bpoints'] = (
                self._end, self._control, self._start)
        return new_quad

    @property
    def control(self):
        return self._control

    @control.setter
    def control(self, val):
        raise Exception("Sorry, segments are immutable!")

    @property
    def control1(self):
        return self._control

    @control1.setter
    def control1(self, val):
        raise Exception("Sorry, segments are immutable!")


class CubicBezier(BezierSegment):
    _class_repr_options = _load_repr_options_for('cubic')

    def __init__(self, start, control1, control2, end):
        self._start = extract_complex(start)
        self._control1 = extract_complex(control1)
        self._control2 = extract_complex(control2)
        self._end = extract_complex(end)
        self._repr_options_init()
        self._field_names = ['start', 'control1', 'control2', 'end']

        # used to know if self._length needs to be updated
        self._length_info = {'length': None,
                             'bpoints': None,
                             'error': None,
                             'min_depth': None}

    def tweaked(self, start=None, control1=None, control2=None, end=None):
        start = start if start is not None else self._start
        control1 = control1 if control1 is not None else self._control1
        control2 = control2 if control2 is not None else self._control2
        end = end if end is not None else self._end
        return CubicBezier(start, control1, control2, end)

    def rotate_c1_by_angle(self, angle):
        self._control1 = self._start + (self._control1 - self._start) * cis_deg(angle)
        return self

    def rotate_c2_by_angle(self, angle):
        self._control2 = self._end + (self._control2 - self._end) * cis_deg(angle)
        return self

    def can_use_S_from_previous(self, previous):
        if isinstance(previous, CubicBezier):
            return (self._start == previous.end and
                    (self._control1 - self._start) == (
                        previous.end - previous.control2))
        else:
            return self._control1 == self._start

    def point(self, t_or_address):
        """
        Evaluate the cubic Bezier curve at t using Horner's rule.
        algebraically equivalent to

        1 * P0 * (1 - t)**3 +
        3 * P1 * t * (1 - t)**2 +
        3 * P2 * (1 - t) * t**2 +
        1 * P3 * t**3

        for (P0, P1, P2, P3) = self.bpoints
        """
        t = address2param(self, t_or_address)
        return self._start + t * (
            3 * (self._control1 - self._start) + t * (
                3 * (self._start + self._control2) -
                6 * self._control1 + t * (
                    - self._start +
                    3 * (self._control1 - self._control2) +
                    self._end)))

    def length(self, t0=0, t1=1, error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH):
        """
        Calculate the length of the path up to a certain position
        """
        t0 = address2param(self, t0)
        t1 = address2param(self, t1)

        if t0 == 0 and t1 == 1:
            if self._length_info['bpoints'] == self.bpoints \
                    and self._length_info['error'] <= error \
                    and self._length_info['min_depth'] >= min_depth:
                return self._length_info['length']

        # using scipy.integrate.quad is quick
        if _quad_available:
            s = quad(lambda u: abs(self.derivative(u)),
                     t0, t1, epsabs=error, limit=1000)[0]
        else:
            s = segment_length(self, t0, t1, self.point(t0), self.point(t1),
                               error, min_depth, 0)

        if t0 == 0 and t1 == 1:
            self._length_info['length'] = s
            self._length_info['bpoints'] = self.bpoints
            self._length_info['error'] = error
            self._length_info['min_depth'] = min_depth
            return self._length_info['length']
        else:
            return s

    @property
    def bpoints(self):
        return self._start, self._control1, self._control2, self._end

    @bpoints.setter
    def bpoints(self, val):
        raise Exception("Sorry, segments are immutable!")

    def poly(self, return_coeffs=False):
        """Returns a the cubic as a Polynomial object."""
        p = self.bpoints
        coeffs = (- p[0] + 3 * (p[1] - p[2]) + p[3],
                  3 * (p[0] - 2 * p[1] + p[2]),
                  3 * (- p[0] + p[1]),
                  p[0])
        return coeffs if return_coeffs else np.poly1d(coeffs)

    def derivative(self, t, n=1):
        """returns the nth derivative of the segment at t.
        Note: Bezier curves can have points where their derivative vanishes.
        If you are interested in the tangent direction, use the unit_tangent()
        method instead."""
        p = self.bpoints

        if n == 1:
            return \
                3 * (p[1] - p[0]) * (1 - t)**2 + \
                6 * (p[2] - p[1]) * (1 - t) * t + \
                3 * (p[3] - p[2]) * t**2

        elif n == 2:
            return \
                6 * (1 - t) * (p[2] - 2 * p[1] + p[0]) + \
                6 * t * (p[3] - 2 * p[2] + p[1])

        elif n == 3:
            return 6 * (p[3] - 3 * (p[2] - p[1]) - p[0])

        elif n > 3:
            return 0

        else:
            raise ValueError("n should be a positive integer.")

    def reversed(self):
        """returns a copy of the CubicBezier object with its orientation
        reversed."""
        new_cub = CubicBezier(self._end, self._control2, self._control1,
                              self._start)
        if self._length_info['length']:
            new_cub._length_info = self._length_info
            new_cub._length_info['bpoints'] = (
                self._end, self._control2, self._control1, self._start)
        return new_cub

    @property
    def control1(self):
        return self._control1

    @control1.setter
    def control1(self, val):
        raise Exception("Sorry, segments are immutable!")

    @property
    def control2(self):
        return self._control2

    @control2.setter
    def control2(self, val):
        raise Exception("Sorry, segments are immutable!")


class Arc(Segment):
    _class_repr_options = _load_repr_options_for('arc')

    def __init__(self, start, radius, rotation, large_arc, sweep, end,
                 autoscale_radius=True):
        """
        This should be thought of as a part of an ellipse connecting two
        points on that ellipse, start and end.
        Parameters
        - - - - - - - - - -
        start : complex
            The start point of the curve. Note: `start` and `end` cannot be
            the same.  To make a full ellipse or circle, use two `Arc`
            objects.
        radius : complex
            rx + 1j * ry, where rx and ry are the radii of the ellipse (also
            known as its semi - major and semi - minor axes, or vice - versa
            or if rx < ry).
            Note: If rx = 0 or ry = 0 then this arc is treated as a
            straight line segment joining the endpoints.
            Note: If rx or ry has a negative sign, the sign is dropped; the
            absolute value is used instead.
            Note:  If no such ellipse exists, the radius will be scaled so
            that one does (unless autoscale_radius is set to False).
        rotation : float
            This is the CCW angle (in degrees) from the positive x-axis of
            the current coordinate system to the x-axis of the ellipse.
        large_arc : bool
            Given two points on an ellipse, there are two elliptical arcs
            connecting those points, the first going the short way around the
            ellipse, and the second going the long way around the ellipse. If
            `large_arc == False`, the shorter elliptical arc will be used. If
            `large_arc == True`, then longer elliptical will be used.
            In other words, `large_arc` should be 0 for arcs spanning less
            than or equal to 180 degrees and 1 for arcs spanning greater than
            180 degrees.
        sweep : bool
            For any acceptable parameters `start`, `end`, `rotation`, and
            `radius`, there are two ellipses with the given major and minor
            axes (radii) which connect `start` and `end`. One which connects
            them in a CCW fashion and one which connected them in a CW
            fashion. If `sweep == True`, the CCW ellipse will be used. If
            `sweep == False`, the CW ellipse will be used.  See note on curve
            orientation below.
        end : complex
            The end point of the curve. Note: `start` and `end` cannot be the
            same.  To make a full ellipse or circle, use two `Arc` objects.
        autoscale_radius : bool
            If `autoscale_radius == True`, then will also scale `self._radius`
            in the case that no ellipse exists with the input parameters
            (see inline comments for further explanation).

        Derived Parameters / Attributes
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        self.theta : float
            This is the phase (in degrees) of self.u1transform(self._start).
            It is $\theta_1$ in the official documentation and ranges from
            - 180 to 180.
        self.delta : float
            This is the angular distance (in degrees) between the start and
            end of the arc after the arc has been sent to the unit circle
            through self.u1transform().
            It is $\Delta \theta$ in the official documentation and ranges
            from - 360 to 360; being positive when the arc travels CCW and
            negative otherwise (i.e. is positive / negative when
            sweep == True / False).
        self.center : complex
            This is the center of the arc's ellipse.
        self.phi : float
            The arc's rotation in radians, i.e. `radians(self._rotation)`.
        self.phi_unit : complex
            Equal to `exp(1j * self.phi)` which is also equal to
            `cos(self.phi) + 1j * sin(self.phi)`.

        Note on curve orientation (CW vs CCW)
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        The notions of clockwise (CW) and counter - clockwise (CCW) are
        reversed in some sense when viewing SVGs (as the y coordinate starts
        at the top of the image and increases towards the bottom).
        """
        assert start != end
        assert radius.real != 0 and radius.imag != 0

        self._start = extract_complex(start)
        r = extract_complex(radius)
        self._radius = abs(r.real) + 1j * abs(r.imag)
        self._rotation = float(rotation)
        self._large_arc = bool(large_arc)
        self._sweep = bool(sweep)
        self._end = extract_complex(end)
        self.autoscale_radius = autoscale_radius

        self._repr_options_init()
        self._field_names = [
            'start', 
            'radius', 
            'rotation', 
            'large_arc', 
            'sweep', 
            'end',
        ]

        # Convenience parameters
        self.phi = radians(self._rotation)
        self.phi_unit = exp(1j * self.phi)

        # Derive derived parameters
        self._parameterize()

    def tweaked(self, start=None, radius=None, rotation=None,
                large_arc=None, sweep=None, end=None, autoscale_radius=None):
        start = start if start is not None else self._start
        radius = radius if radius is not None else self._radius
        rotation = rotation if rotation is not None else self._rotation
        large_arc = large_arc if large_arc is not None else self._large_arc
        sweep = sweep if sweep is not None else self._sweep
        end = end if end is not None else self._end
        return Arc(start, radius, rotation,
                   large_arc, sweep, end, autoscale_radius)

    def __eq__(self, other, tol=0):
        if not isinstance(other, Arc):
            return NotImplemented
        return \
            abs(self._start - other._start) <= tol and \
            abs(self._end - other._end) <= tol and \
            abs(self._radius - other._radius) <= tol and \
            abs(self._rotation - other._rotation) <= tol and \
            self._large_arc == other._large_arc and \
            self._sweep == other._sweep

    def __ne__(self, other):
        if not isinstance(other, Arc):
            return NotImplemented
        return not self == other

    def _parameterize(self):
        # See http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
        # Andy's notation roughly follows theirs
        rx, ry = self._radius.real, self._radius.imag
        rx_sqd, ry_sqd = rx**2, ry**2

        # The transform
        #
        #    z -> (z - (end + start) / 2) / phi_unit                       (1)
        #
        # is an isometric transformation that moves the ellipse so that
        # the midpoint between self._start and self._end lies on the origin
        # and that rotates the ellipse so that its axes align with the
        # xy-coordinate system. The image of self._start under this
        # transformation is
        #
        #    (start - (end + start) / 2) / phi_unit
        #  = (start - end) / (2 * phi_unit)
        #
        # denoted by zp below.

        # Note: zp only depends on start, end and phi_unit, not on rx or ry,
        # hence does not need to be recomputed if rx and ry are rescaled

        # Note: -zp is the image of self._end under the same transformation

        zp = (self._start - self._end) / (2 * self.phi_unit)
        xp, yp = zp.real, zp.imag
        xp_sqd, yp_sqd = xp**2, yp**2

        # The radii rx, ry are feasible if and only if zp is inside an ellipse
        # centered at the origin of x-radius rx and y-radius ry; the equation
        # of such ellipse being x^2 / rx^2 + y^2 / ry^2 = 1, such an ellipse
        # exists if and only if xp^2 / rx^2 + yp^2 / ry^2 <= 1
        check = xp_sqd / rx_sqd + yp_sqd / ry_sqd
        if check > 1:
            if self.autoscale_radius:
                rx_sqd *= check
                ry_sqd *= check
                rx *= sqrt(check)
                ry *= sqrt(check)
                self._radius = rx + 1j * ry
            else:
                raise ValueError("No such elliptic arc exists.")

        # Under transform (1), the equation of our ellipse is
        #
        #     (x - cx)^2 / rx^2 + (y - cy) / ry^2 = 1
        #
        # where c = cx + 1j * cy is the image of the center of the ellipse
        # under (1). Given that both zp = (xp, yp) and -zp = (-xp, -yp) are
        # points on the transformed ellipse, we find
        #
        #     (xp - cx)^2 / rx^2 + (yp - cy)^2 / ry^2 = 1
        #
        # on the one hand, and
        #
        #     (-xp - cx)^2 / rx^2 + (-yp - cy)^2 / ry^2 = 1
        #
        # on the other hand. After a bunch of algebra to solve for cx, cy, one
        # finds
        #
        #     cx = +- (yp rx / ry) * radical                               (*)
        #     cy = -+ (xp ry / rx) * radical                              (**)
        #
        # where
        #
        #     radical = sqrt(radicand), where
        #     radicand = (rx^2 ry^2 - Q) / Q, where
        #     Q = yp^2 rx^2 + xp^2 ry^2
        #
        # and moreover it turns out that the correct selection of signs in
        # (*), (**) is given by the XOR of ._large_arc and ._sweep:
        Q = rx_sqd * yp_sqd + ry_sqd * xp_sqd
        radicand = (rx_sqd * ry_sqd - Q) / Q
        radical = sqrt(max(0, radicand))
        cp = (rx * yp / ry - 1j * ry * xp / rx) * radical
        if self._large_arc == self._sweep:
            cp *= -1

        # For the original center, we pass c through the inverse of (1):
        self.center = self.phi_unit * cp + (self._start + self._end) / 2

        # Now we do a second transformation, from (x', y') to (u_x, u_y)
        # coordinates, which is a translation moving the center of the
        # ellipse to the origin and a dilation stretching the ellipse to be
        # the unit circle
        u1 = (xp - cp.real) / rx + 1j * (yp - cp.imag) / ry    # transformed start
        u2 = (-xp - cp.real) / rx + 1j * (-yp - cp.imag) / ry  # transformed end

        # Computation of self.theta, the starting angle of the ellipse:
        self.theta = degrees(phase(u1))
        assert -180 <= self.theta <= 180

        # Computation of self.delta, the oriented aperture of the ellipse in
        # degrees. The following delta possibly goes the wrong way around the
        # circle, including being 0 instead of 360, so we adjust:
        self.delta = degrees(phase(u2 / u1)) % 360
        if not self._sweep and self.delta != 0:
            self.delta -= 360
        if self.delta == 0 and self._large_arc:
            self.delta = 360 if self._sweep else -360

        # Some checks
        assert -360 <= self.delta <= 360
        if abs(self.delta) != 180:
            assert (abs(self.delta) > 180) == (self._large_arc is True)

    def point(self, t_or_address):
        t = t_or_address
        if isinstance(t_or_address, Address):
            t = t_or_address.t
        if t < 0 or t > 1:
            raise ValueError("t out of bounds in Arc.point")
        if t == 0:
            return self._start
        if t == 1:
            return self._end
        t_unit = exp(1j * radians(self.theta + t * self.delta))
        rx = self._radius.real
        ry = self._radius.imag
        return \
            self.center + \
            self.phi_unit * (rx * t_unit.real + 1j * ry * t_unit.imag)

    def centeriso(self, z):
        """This is an isometry that translates and rotates self so that it
        is centered on the origin and has its axes aligned with the xy
        axes."""
        return (1 / self.phi_unit) * (z - self.center)

    def icenteriso(self, zeta):
        """This is an isometry, the inverse of standardiso()."""
        return self.phi_unit * zeta + self.center

    def phase2t(self, psi):
        """
        Given phase -pi < psi <= pi,
        returns the t value, 0 <= t <= 1, if any, such that
        exp(1j * psi) = self.u1transform(self.point(t))

        Symbolically, note that

              self.u1transform(self.point(t))
            = ((point(t) - center) / phi_unit).scaled_by(1/rx, 1/ry)
            = ((center + phi_unit * t_unit.scaled_by(rx, xy) - center) /
              phi_unit).scaled_by(1/rx, 1/ry)
            = t_unit
            = exp(1j * radians(theta + t * delta))

        so need to solve

           deg(psi) = theta + t * delta mod 360

        for 0 <= t <= 1, if such t exists.
        """
        def _deg_lower_limit(rads, domain_lower_limit):
            degs = degrees(rads)
            if degs < domain_lower_limit:
                degs += 360
            assert domain_lower_limit <= degs < 360 + domain_lower_limit
            return degs

        def _deg_upper_limit(rads, domain_upper_limit):
            degs = degrees(rads)
            if degs > domain_upper_limit:
                degs -= 360
            assert domain_upper_limit >= degs > domain_upper_limit - 360
            return degs

        assert -pi < psi <= pi
        assert -180 <= self.theta <= 180

        if self.delta > 0:
            degs = _deg_lower_limit(psi, domain_lower_limit=self.theta)
        else:
            degs = _deg_upper_limit(psi, domain_upper_limit=self.theta)

        return (degs - self.theta) / self.delta

    def t2phase(self, t):
        return radians(self.theta + t * self.delta)

    def u1transform(self, z):
        """This is an affine transformation (same as used in
        self._parameterize()) that sends self to the unit circle."""
        zeta = (z - self.center) / self.phi_unit  # same as centeriso(z)
        return real(zeta) / self._radius.real + imag(zeta) / self._radius.imag * 1j

    def iu1transform(self, zeta):
        """The inverse of self.u1transform()."""
        z = real(zeta) * self._radius.real + imag(zeta) * self._radius.imag * 1j
        return self.phi_unit * z + self.center

    def x_val_intersect(self, x_val):
        assert isinstance(x_val, Real)
        u1poly = self.u1transform(np.poly1d([1j, x_val]))
        t2s = polyroots(real(u1poly)**2 + imag(u1poly)**2 - 1)
        t1s = [self.phase2t(phase(u1poly(t2))) for t2 in t2s]
        return [Address(t=t1) for t1 in t1s if 0 <= t1 <= 1]

    def y_val_intersect(self, y_val):
        assert isinstance(y_val, Real)
        u1poly = self.u1transform(np.poly1d([1, y_val * 1j]))
        t2s = polyroots(real(u1poly)**2 + imag(u1poly)**2 - 1)
        t1s = [self.phase2t(phase(u1poly(t2))) for t2 in t2s]
        return [Address(t=t1) for t1 in t1s if 0 <= t1 <= 1]

    def intersect(self, other_seg, tol=1e-12):
        """
        NOT FULLY IMPLEMENTED. Supposed to return a list of tuples (t1, t2)
        such that self.point(t1) == other_seg.point(t2).

        Note: This will fail if the two segments coincide for more than a
        finite collection of points.

        Note: Arc related intersections are only partially supported, i.e. are
        only half-heartedly implemented and not well tested. Please feel free
        to let me know if you're interested in such a feature -- or even
        better please submit an implementation if you want to code one."""
        if isinstance(other_seg, BezierSegment):
            u1poly = self.u1transform(other_seg.poly())
            u1poly_norm_sqd = real(u1poly)**2 + imag(u1poly)**2
            t2s = polyroots01(u1poly_norm_sqd - 1)
            t1s = [self.phase2t(phase(u1poly(t2))) for t2 in t2s]
            return [address_pair_from_t1t2(t1, t2) for t1, t2 in zip(t1s, t2s) if (0 <= t1 <= 1)]

        elif isinstance(other_seg, Arc):
            assert other_seg != self
            # This could be made explicit to increase efficiency
            longer_length = max(self.length(), other_seg.length())
            inters = bezier_intersections(self, other_seg,
                                          longer_length=longer_length,
                                          tol=tol, tol_deC=tol)

            # ad hoc fix for redundant solutions
            if len(inters) > 2:
                def keyfcn(tpair):
                    t1, t2 = tpair
                    return abs(self.point(t1) - other_seg.point(t2))
                inters.sort(key=keyfcn)
                for idx in range(1, len(inters) - 1):
                    if abs(inters[idx][0] - inters[idx + 1][0]) < \
                       abs(inters[idx][0] - inters[0][0]):
                        return [address_pair_from_t1t2_tuple(inters[0]),
                                address_pair_from_t1t2_tuple(inters[idx])]
                else:
                    return [address_pair_from_t1t2_tuple(inters[0]),
                            address_pair_from_t1t2_tuple(inters[-1])]
            return [address_pair_from_t1t2_tuple(inters[0]),
                    address_pair_from_t1t2_tuple(inters[1])]

        else:
            raise TypeError("other_seg should be a Arc, Line, "
                            "QuadraticBezier, or CubicBezier object.")

    def length(self, t0=0, t1=1, error=LENGTH_ERROR,
               min_depth=LENGTH_MIN_DEPTH):
        """The length of an elliptical large_arc segment requires numerical
        integration, and in that case it's simpler to just do a geometric
        approximation, as for cubic bezier curves."""
        t0 = address2param(self, t0)
        t1 = address2param(self, t1)
        if _quad_available:
            return quad(lambda u: abs(self.derivative(u)), t0, t1,
                        epsabs=error, limit=1000)[0]
        else:
            return segment_length(self, t0, t1, self.point(t0),
                                  self.point(t1), error, min_depth, 0)

    def derivative(self, t, n=1):
        """returns the nth derivative of the segment at t."""
        angle = radians(self.theta + t * self.delta)
        phi = self.phi
        rx = self._radius.real
        ry = self._radius.imag
        k = (self.delta * 2 * pi / 360)**n  # ((d / dt)angle)**n

        if n % 4 == 0 and n > 0:
            return k * (
                (rx * cos(phi) * cos(angle) + ry * (-sin(phi)) * sin(angle)) +
                (rx * sin(phi) * cos(angle) + ry   * cos(phi)  * sin(angle)) * 1j
            )

        elif n % 4 == 1:
            return k * (
                (-rx * cos(phi) * sin(angle) + ry * (-sin(phi)) * cos(angle)) +
                (-rx * sin(phi) * sin(angle) + ry   * cos(phi)  * cos(angle)) * 1j
            )

        elif n % 4 == 2:
            return k * (
                (-rx * cos(phi) * cos(angle) - ry * (-sin(phi)) * sin(angle)) +
                (-rx * sin(phi) * cos(angle) - ry   * cos(phi)  * sin(angle)) * 1j
            )

        elif n % 4 == 3:
            return k * (
                (rx * cos(phi) * sin(angle) - ry * (-sin(phi)) * cos(angle)) +
                (rx * sin(phi) * sin(angle) - ry   * cos(phi)  * cos(angle)) * 1j
            )

        else:
            raise ValueError("n should be a positive integer.")

    def unit_tangent(self, t):
        """returns the unit tangent vector of the segment at t (centered at
        the origin and expressed as a complex number)."""
        dseg = self.derivative(t)
        return dseg / abs(dseg)

    def reversed(self):
        """returns a copy of the Arc object with its orientation reversed."""
        return Arc(self._end, self._radius, self._rotation, self._large_arc,
                   not self._sweep, self._start)

    def t2lambda(self, t):
        p = self.point(t) - self.center
        return phase(p / self.phi_unit)

    def lambda2eta(self, lamda):
        return atan2(sin(lamda) / self._radius.imag,
                     cos(lamda) / self._radius.real)

    def maisonobe_E(self, eta):
        """(this function is unused; see next function)"""
        return \
            self.center + \
            self.phi_unit * (self.rx * cos(eta) + 1j * self.ry * sin(eta))

    def maisonobe_E_prime(self, eta):
        """see paper 'Drawing an elliptical arc using polylines, quadratic
        or cubic Bezier curves' by L. Maisonobe, 2003, sections 2.2.1 and
        3.4.1"""
        return \
            self.phi_unit * (-self._radius.real * sin(eta) +
                             +self._radius.imag * cos(eta) * 1j)

    def maisonobe_cubic_interpolation(self, t1, t2):
        """see paper 'Drawing an elliptical arc using polylines, quadratic
        or cubic Bezier curves' by L. Maisonobe, 2003, sections 2.2.1 and
        3.4.1

        This interpolation respects slope and curvature at the endpoints of
        the cubic"""
        assert 0 <= t1 < t2 <= 1
        start = self.point(t1)
        end   = self.point(t2)
        eta1  = self.lambda2eta(self.t2lambda(t1))
        eta2  = self.lambda2eta(self.t2lambda(t2))

        discr = 4 + 3 * tan((eta2 - eta1) * 0.5)**2
        alpha = abs(sin(eta2 - eta1)) * (sqrt(discr) - 1) / 3
        if not self._sweep:
            alpha *= -1
        control1 = start + alpha * self.maisonobe_E_prime(eta1)
        control2 = end   - alpha * self.maisonobe_E_prime(eta2)

        return CubicBezier(start, control1, control2, end)

    def midpoint_cubic_interpolation(self, t1, t2):
        """
        This interpolation respects slopes at the endpoints of the cubic,
        and places the midpoint of the cubic in the middle of the interpolated
        arc.

        Note: This interpolation seems to be preferred over Maisonobe's
        interpolation in drawing programs, but Maisonobe's seems to give
        faster convergence of areas.
        """
        assert 0 <= t1 < t2 <= 1
        start = self.point(t1)
        end = self.point(t2)
        psi1 = radians(self.theta + t1 * self.delta)
        psi2 = radians(self.theta + t2 * self.delta)
        aperture = psi2 - psi1
        alpha = (4 / 3) * tan(aperture / 4)
        assert alpha * self.delta > 0
        control1 = (1 + alpha * 1j) * exp(1j * psi1)
        control2 = (1 - alpha * 1j) * exp(1j * psi2)
        control1 = self.iu1transform(control1)
        control2 = self.iu1transform(control2)
        return CubicBezier(start, control1, control2, end)

    def converted_to_bezier_subpath(self, quality=0.01, safety=5,
                                    use_Maisonobe=True):
        assert quality > 0
        safety = int(min(4, safety))
        if use_Maisonobe:
            other = self.maisonobe_cubic_interpolation(0, 1)
        else:
            other = self.midpoint_cubic_interpolation(0, 1)
        assert other.start == self._start
        assert other.end == self._end
        divergence = divergence_of_offset(self, other, 0, safety=safety,
                                          early_return_threshold=quality)
        if divergence <= quality:
            return Subpath(other), [0, 1]
        first_half, secnd_half = self.split(0.5)
        assert first_half.end == secnd_half.start
        P1, ts1 = first_half.converted_to_bezier_subpath(quality, safety)
        P2, ts2 = secnd_half.converted_to_bezier_subpath(quality, safety)
        assert P1.end == P2.start
        allts = [0.5 * t for t in ts1] + [0.5 + 0.5 * t for t in ts2[1:]]
        return P1.extend(P2), allts

    def pro_offset(self, amount, quality=0.01, safety=5, two_sided=False):
        converted, _ = self.converted_to_bezier_subpath(quality * 0.5, safety)
        assert isinstance(converted, Subpath)
        return converted.pro_offset(amount, quality * 0.5, safety, two_sided)

    def offset(self, amount, quality=0.01, safety=5):
        return self.pro_offset(amount, quality, safety)[0]

    def _box_helper(self, atan2_value):
        A = 360 * atan2_value / tau - self.theta
        B = self.delta
        points = [self._start, self._end]
        for k in range(ceil(-A / 180), floor((B - A) / 180) + 1):
            t = (A + k * 180) / B
            if t > 1 and np.isclose(t, 1):
                t = 1
            if t < 0 and np.isclose(t, 0):
                t = 0
            assert 0 <= t <= 1
            points.append(self.point(t))
        return points

    def xbox(self, stroke_width=None):
        if stroke_width is not None:
            return self.stroke(stroke_width).xbox()

        # in-house radian angle as a function of t, 0 <= t <= 1
        # a(t) := (tau/360) * (self.theta + self.delta*t)

        # position as a function of t, pre-rotation:
        # (X, Y) := (rx*cos(a(t)), ry*sin(a(t)))

        # post-rotation, with p := self.phi:
        # (X*cos(p) - Y*sin(p),
        #  X*sin(p) + Y*cos(p))

        # x-coordinate thereof:
        # rx*cos(a(t))*cos(p) - ry*sin(a(t))*sin(p)

        # x'(t) is therefore:
        # -rx*sin(a(t))*cos(p)*a'(t) - ry*cos(a(t))*sin(p)*a'(t)

        # assuming a'(t) =/= 0, x' == 0 if and only if:
        #      rx*sin(a(t))*cos(p) + ry*cos(a(t))*sin(p) == 0
        # <==> det([[cos(a(t)), -rx*cos(p)],
        #           [sin(a(t)), ry*sin(p)]]) == 0

        # so (cos(a(t)), sin(a(t))) is colinear with (-rx*cos(p), ry*sin(p))
        # so a(t) = atan2(ry*sin(p), -rx*cos(p)) + k*(tau/2)

        # and t = ((360/tau)*(atan2(ry*sin(p), -rx*cos(p)) + k*(tau/2)) - self.theta)/self.delta
        #       = ((360/tau)*(atan2(ry*sin(p), -rx*cos(p))) - self.theta + k*180)/self.delta

        # and 0 <= t <= 1 if and only if
        #      0 <= ((360/tau)*(atan2(ry*sin(p), -rx*cos(p))) - self.theta) + k*180 <= self.delta
        # <==> 0 <= A + k*180 <= B
        # with A := (360/tau)*(atan2(ry*sin(p), -rx*cos(p))) - self.theta
        #      B := self.delta

        # also:
        # -A/180 <= k <= (B - A)/180
        # and:
        # t = ((360/tau)*(atan2(ry*sin(p), -rx*cos(p))) - self.theta + k*180)/self.delta
        #   = (A + k*180)/B

        p, rx, ry = self.phi, self._radius.real, self._radius.imag
        candidate_extrema = self._box_helper(atan2(ry * sin(p), -rx * cos(p)))
        return \
            min(p.real for p in candidate_extrema), \
            max(p.real for p in candidate_extrema)

    def ybox(self, stroke_width=None):
        if stroke_width is not None:
            return self.stroke(stroke_width).ybox()

        # in-house radian angle as a function of t, 0 <= t <= 1
        # a(t) := (tau/360) * (self.theta + self.delta*t)

        # position as a function of t, pre-rotation:
        # (X, Y) := (rx*cos(a(t)), ry*sin(a(t)))

        # post-rotation, with p := self.phi:
        # (X*cos(p) - Y*sin(p),
        #  X*sin(p) + Y*cos(p))

        # y-coordinate thereof:
        # rx*cos(a(t))*sin(p) + ry*sin(a(t))*cos(p)

        # y'(t) is therefore:
        # -rx*sin(a(t))*sin(p)*a'(t) + ry*cos(a(t))*cos(p)*a'(t)

        # and y'(t) == 0 implies (assuming self.delta > 0):
        #      -rx*sin(a(t))*sin(p) + ry*cos(a(t))*cos(p) == 0
        # <==> det([[cos(a(t)), rx*sin(p)],
        #           [sin(a(t)), ry*cos(p)]]) == 0

        # so (cos(a(t)), sin(a(t))) is colinear with (rx*sin(p), ry*cos(p))
        # so a(t) = atan2(ry*cos(p), rx*sin(p)) + k*(tau/2)

        # and t = (a(t) * (360/tau) - self.theta)/self.delta
        #       = ((atan2(ry*cos(p), rx*sin(p)) + k*(tau/2)) * (360/tau) - self.theta)/self.delta
        #       = ((360/tau) * atan2(ry*cos(p), rx*sin(p)) - self.theta + k*180)/self.delta

        # and 0 <= t <= 1 if and only if
        #      0 <= (360 / tau) * atan2(ry*cos(p), rx*sin(p)) - self.theta + k*180 <= self.delta
        # <==> 0 <= A + k * 180 <= B
        # with A := (360/tau) * atan2(ry*cos(p), rx*sin(p)) - self.theta
        #      B := self.delta

        # also:
        # -A/180 <= k <= (B-A)/180
        # and:
        # t = ((360/tau) * atan2(ry*cos(p), rx*sin(p)) - self.theta + k*180)/self.delta
        #   = (A + k*180)/B

        p, rx, ry = self.phi, self._radius.real, self._radius.imag
        candidate_extrema = self._box_helper(atan2(ry * cos(p), rx * sin(p)))
        return \
            min(p.imag for p in candidate_extrema), \
            max(p.imag for p in candidate_extrema)

    def split(self, t):
        """returns two segments, whose union is this segment and which join
        at self.point(t)."""
        return self.cropped(0, t), self.cropped(t, 1)

    def cropped(self, t0, t1):
        """returns a cropped copy of this segment which starts at
        self.point(t0) and ends at self.point(t1). Allows t1 > t0 but not
        t0 == t1."""
        t0 = address2param(self, t0)
        t1 = address2param(self, t1)
        new_large_arc = 0 if abs(self.delta * (t1 - t0)) <= 180 else 1
        return Arc(self.point(t0), radius=self._radius,
                   rotation=self._rotation,
                   large_arc=new_large_arc, sweep=self._sweep,
                   end=self.point(t1), autoscale_radius=self.autoscale_radius)

    def radialrange(self, origin, return_all_global_extrema=False):
        """returns the tuples (d_min, t_min) and (d_max, t_max) which minimize
        and maximize, respectively, the distance,
        d = |self.point(t) - origin|.

        And... er... this function is, apparently, not implemented."""

        print("inside Arc.radialrange! ouch! this is not implemented!")

        if return_all_global_extrema:
            raise NotImplementedError

        u1orig = self.u1transform(origin)

        # Transform to a coordinate system where the ellipse is centered
        # at the origin and its axes are horizontal / vertical
        zeta0 = self.centeriso(origin)
        a, b = self._radius.real, self._radius.imag
        x0, y0 = zeta0.real, zeta0.imag

        # Find t s.t. z'(t)
        a2mb2 = (a**2 - b**2)
        if u1orig.imag:  # x != x0
            coeffs = [
                a2mb2**2,
                2 * a2mb2 * b**2 * y0,
                (- a**4 + (2 * a**2 - b**2 + y0**2) * b**2 + x0**2) * b**2,
                - 2 * a2mb2 * b**4 * y0,
                - b**6 * y0**2
            ]
            ys = polyroots(coeffs, realroots=True,
                           condition=lambda r: - b <= r <= b)
            xs = (a * sqrt(1 - y**2 / b**2) for y in ys)

        else:  # This case is very similar, see notes and assume instead y0!=y
            b2ma2 = (b**2 - a**2)
            coeffs = [
                b2ma2**2,
                2 * b2ma2 * a**2 * x0,
                (- b**4 + (2 * b**2 - a**2 + x0**2) * a**2 + y0**2) * a**2,
                - 2 * b2ma2 * a**4 * x0,
                - a**6 * x0**2
            ]
            xs = polyroots(coeffs, realroots=True,
                           condition=lambda r: - a <= r <= a)
            ys = (b * sqrt(1 - x**2 / a**2) for x in xs)

        raise _NotImplemented4ArcException

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, val):
        raise Exception("Sorry, segments are immutable!")

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, val):
        raise Exception("Sorry, segments are immutable!")

    @property
    def large_arc(self):
        return self._large_arc

    @large_arc.setter
    def large_arc(self, val):
        raise Exception("Sorry, segments are immutable!")

    @property
    def sweep(self):
        return self._sweep

    @sweep.setter
    def sweep(self, val):
        raise Exception("Sorry, segments are immutable!")


class Subpath(ContinuousCurve, MutableSequence):
    """
    A subpath is a sequence of end-to-end contiguous path segments.

    A subpath whose end equals its start may or may not be closed via its
    .Z property. See methods Subpath.set_Z(), Subpath.unset_Z()
    for more details. Only subpaths for which .Z == True are considered
    'closed' (or "topologically closed"). Subpaths for which .end == .start
    are said to be "geometrically closed". In other words, topological
    closedness implies geometric closedness, but not vice-versa.

    Note that Subpath has its own d() function and can be used autonomously
    from Path.
    """
    _class_repr_options = _load_repr_options_for('subpath')

    def __init__(self, *things, wiggle_endpoints_into_place=True):
        self.debug = False
        self._Z = False
        self._segments = []
        self._repr_options_init()
        self._field_names = []
        for s in things:
            self.append(s, wiggle_endpoints_into_place)  # ends up calling .insert which itself calls .splice
        self._reset()

    def _reset(self):
        self._length = None
        self._lengths = None
        if self._Z and (len(self) == 0 or self[0].start != self[-1].end):
            self._Z = False
        self._check_health()
        return self  # so that callers who return self can save a line

    def _check_health(self):
        assert all(isinstance(thing, Segment) for thing in self)

        for s, t in zip(self, self[1:]):
            assert s.end == t.start

        if self._Z:
            assert self.start_equals_end()

    def __normalize_index(self, index, max):
        if index < 0:
            index += len(self)
        if not 0 <= index <= max:
            raise ValueError("index out of bounds")
        return index

    def __getitem__(self, index):  # (MutableSequence abstract class)
        return self._segments[index]

    def __setitem__(self, index, value):  # (MutableSequence abstract class)
        if not isinstance(value, Segment):
            raise ValueError("Subpath.__setitem__ given non-Segment value")
        index = self.__normalize_index(index, len(self))
        self.splice(index, index + 1, value)
        return value

    def __delitem__(self, index):  # (MutableSequence abstract class)
        if isinstance(index, int):
            index = self.__normalize_index(index, len(self) - 1)
            self.splice(index, index + 1, None)

        elif isinstance(index, slice):
            if index.step is not None:
                raise NotImplementedError

            start_index = 0 if index.start is None else int(index.start)
            end_index = len(self) if index.stop is None else int(index.stop)
            self.splice(start_index, end_index, Subpath())

        else:
            raise ValueError

    def __len__(self):
        return len(self._segments)  # (MutableSequence abstract class)

    def is_empty(self):
        return len(self) == 0

    def append(self, thing, wiggle_endpoints_into_place=True, bridge_discontinuity=False):
        self.splice(len(self), len(self), thing, wiggle_endpoints_into_place, bridge_discontinuity=bridge_discontinuity)

    def insert(self, index, value, bridge_discontinuity=False):  # (MutableSequence abstract class)
        if index < 0:
            index += len(self)
        if not 0 <= index <= len(self):
            raise ValueError("index out of bounds")
        return self.splice(index, index, value, bridge_discontinuity=bridge_discontinuity)

    # overwrites native extend:
    def extend(self, *things, bridge_discontinuity=False):
        # This is tricky; we have to pre-group curves in things for
        # absolute correctness, but not fall into infinite recursion
        container = Subpath()
        for t in things:
            container.append(t)  # self.append must not call self.extend!
        self.splice(len(self), len(self), container, bridge_discontinuity=bridge_discontinuity)
        return self

    def mod_index(self, index, use_Z):
        return index if not use_Z or not self._Z else index % len(self)

    def first_segment(self):
        if not len(self):
            return None
        assert isinstance(self._segments[0], Segment)
        return self._segments[0]

    def last_segment(self):
        if not len(self):
            return None
        assert isinstance(self._segments[-1], Segment)
        return self._segments[-1]

    def prev_segment(self, index, use_Z=False):
        assert 0 <= index <= len(self)
        prev_index = self.mod_index(index - 1, use_Z)
        return self[prev_index] if prev_index >= 0 else None

    def next_segment(self, index, use_Z=False):
        assert -1 <= index <= len(self) - 1
        next_index = self.mod_index(index + 1, use_Z)
        return self[next_index] if next_index <= len(self) - 1 else None

    # I put this here because I think it's faster than letting python
    # figure out the iterator via __getitem__ etc (?)
    def __iter__(self):
        return self._segments.__iter__()

    def prepend(self, value):
        return self.insert(0, value)

    def splice(self, start_index, end_index, value, wiggle_endpoints_into_place=True, bridge_discontinuity=False):
        """
        replaces segments of indices start_index, ..., end_index - 1 with
        the (segments in) value, which may be a segment, a subpath, or a
        "naively continuous" path, as long as no discontinuity is induced; if
        end_index == start_index, no segments are replaced, and insertion
        occurs right before the segment at start_index == end_index, so that
        the first ted segment has index start_index. Can be used with an
        empty value (of type Path or Subpath), in which case the effect is
        only to delete the segments in the sequence start_index, ...,
        end_index - 1
        """
        # assertions / checking
        if not isinstance(value, Curve) and value is not None:
            raise ValueError("expecting instance of Curve")

        if isinstance(value, Path) and not value.is_naively_continuous():
            raise ValueError("Subpath.splice fed path with discontinuity")

        assert 0 <= start_index <= end_index <= len(self)

        if len(self) > 0:
            prev = self.prev_segment(start_index, use_Z=True)
            next = self.next_segment(end_index - 1, use_Z=True)

        else:
            prev = next = None

        is_empty_insert = \
            value is None or \
            (not isinstance(value, Segment) and len(value) == 0)

        if is_empty_insert:
            if prev is not None and \
               next is not None and \
               prev.start != next.end:
                raise ValueError("Subpath.splice jumpcut discontinuity")

        else:
            if prev and prev.end != value.start:
                if wiggle_endpoints_into_place and \
                   np.isclose(prev.end, value.start):
                    new_prev = prev.matched_end_with_start_of(value.first_segment())
                    new_value = value.matched_start_with_end_of(prev)
                    prev_index = self.mod_index(start_index - 1, use_Z=True)
                    assert 0 <= prev_index < len(self)
                    assert prev is self._segments[prev_index]
                    self._segments[prev_index] = prev = new_prev
                    value = new_value
                    assert prev.end == value.start

                    # reset next, as well...
                    next = self.next_segment(end_index - 1, use_Z=True)

                elif bridge_discontinuity:
                    pass

                else:
                    raise ValueError("Subpath.splice .start discontinuity")

            if next and next.start != value.end:
                if wiggle_endpoints_into_place and \
                   np.isclose(next.start, value.end):
                    new_next = next.matched_start_with_end_of(value.last_segment())
                    new_value = value.matched_end_with_start_of(next)
                    next_index = self.mod_index(end_index, True)
                    assert 0 <= next_index < len(self)
                    assert next is self._segments[next_index]
                    self._segments[next_index] = next = new_next
                    value = new_value
                    assert next.start == value.end  # and...

                    # reset prev, as well...
                    prev = self.prev_segment(start_index, use_Z=True)
                    assert prev is None or prev.end == value.start

                elif bridge_discontinuity:
                    pass

                else:
                    raise ValueError("Subpath.splice .end discontinuity", next.start, value.end)

        # delete
        for i in reversed(range(start_index, end_index)):
            del self._segments[i]

        assert 0 <= start_index <= len(self)

        # just doing a bit of brain farting here
        new_next = self._segments[start_index] if start_index < len(self) else (
            None if not self._Z else (
                self._segments[0] if len(self) > 0 else None
            )
        )
        assert new_next == next

        # insert
        for seg in segment_iterator_of(value, back_to_front=True):
            self._segments.insert(start_index, seg)

        if not is_empty_insert:
            if prev is not None and prev.end != value.start:
                assert bridge_discontinuity

                bridge = Line(prev.end, value.start)
                self._segments.insert(start_index, bridge)

            if next is not None and next.start != value.end:
                assert bridge_discontinuity

                self._segments.insert(start_index + len(value), Line(value.end, next.start))

        self._reset()

        return self

    def smelt_colinear_line_segments(self):
        if len(self) <= 1:
            return self

        new_segments = [self[0]]
        for i in range(1, len(self)):
            next_segment = self[i]
            last_segment = new_segments[-1]
            assert last_segment.end == next_segment.start

            if False:
                new_segments.append(next_segment)
                continue

            if not isinstance(next_segment, Line) or \
               not isinstance(last_segment, Line) or \
               not next_segment.parallel_to(last_segment):
                new_segments.append(next_segment)
                continue

            new_segments[-1] = last_segment.tweaked(end=next_segment.end)

        self._segments = new_segments
        self._check_health()
        return self

    def cyclical_rotate_segments(self, how_much):
        """
        'how_much' will be the index of the new starting segment
        """
        if not self._Z:
            raise ValueError("rotating starting segment requires loop")

        new_segments = []
        for i in range(len(self)):
            new_segments.append(self._segments[(i + how_much) % len(self)])
        assert len(new_segments) == len(self._segments)

        self._segments = new_segments
        self._check_health()
        return self

    def reversed(self):
        to_return = Subpath(*[seg.reversed() for seg in reversed(self)])
        if self._Z:
            to_return.set_Z()
        return to_return

    def __eq__(self, other, tol=0):
        if not isinstance(other, Subpath):
            return NotImplemented
        seg_pairs = zip(self._segments, other._segments)
        return \
            len(self) == len(other) and \
            all(s.__eq__(o, tol) for s, o in seg_pairs) and \
            self._Z == other._Z

    def __ne__(self, other):
        if not isinstance(other, Subpath):
            return NotImplemented
        return not self == other

    def _calc_lengths(self, error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH):
        if self._length is not None:
            return

        lengths = [each.length(error=error, min_depth=min_depth) for each in
                   self._segments]

        self._length = sum(lengths)
        self._lengths = [each / self._length for each in lengths]

    def point(self, T):
        a = self.T2address(T, trust_existing_fields=True)  # (the customer is always right...)
        return self._segments[a.segment_index].point(a.t)

    def length(self, T0=0, T1=1,
               error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH):
        self._calc_lengths(error=error, min_depth=min_depth)
        T0 = address2param(self, T0)
        T1 = address2param(self, T1)

        if T0 == 0 and T1 == 1:
            return self._length

        a0 = self.T2address(T0)
        a1 = self.T2address(T1)

        si0 = a0.segment_index
        si1 = a1.segment_index
        t0 = a0.t
        t1 = a1.t

        if si0 == si1:
            return self[si0].length(t0=t0, t1=t1)

        return \
            self[si0].length(t0=t0) + \
            sum(self[i].length() for i in range(si0 + 1, si1)) + \
            self[si1].length(t1=t1)

    @property
    def start(self):
        if len(self) == 0:
            return None
        return self[0].start

    @start.setter
    def start(self, pt):
        raise Exception("Sorry, segments are immutable!")

    @property
    def end(self):
        if len(self) == 0:
            return None
        return self[-1].end

    @end.setter
    def end(self, pt):
        raise Exception("Sorry, segments are immutable!")

    @property
    def Z(self):
        return self._Z

    @Z.setter
    def Z(self, val):
        raise Exception("Please use .set_Z(), .unset_Z()"
                        "to set Z field of a subpath!")

    def start_equals_end(self):
        return len(self) > 0 and self[0].start == self[-1].end

    def is_bezier_subpath(self):
        return all(isinstance(seg, BezierSegment) for seg in self)

    def is_or_has_arc(self):
        return any(isinstance(seg, Arc) for seg in self)

    def matched_end_with_start_of(self, other):
        if len(self) == 0:
            raise ValueError("Empty subpath cannot tweak its end")
        our_last = self._segments[-1]
        new_last = our_last.matched_end_with_start_of(other)
        assert np.isclose(our_last.end, new_last.end)

        new_segments = list(self._segments)
        new_segments[-1] = new_last
        if self._Z:
            new_segments[0] = new_segments[0].tweaked(start=new_last.end)
            assert new_segments[0].start == new_segments[-1].end

        return Subpath(*new_segments).set_Z(following=self)

    def matched_start_with_end_of(self, other):
        if len(self) == 0:
            raise ValueError("Empty subpath cannot tweak its start")
        our_first = self._segments[0]
        new_first = our_first.matched_start_with_end_of(other)
        assert np.isclose(our_first.start, new_first.start)

        new_segments = list(self._segments)
        new_segments[0] = new_first
        if self._Z:
            new_segments[-1] = new_segments[-1].tweaked(end=new_first.start)
            assert new_segments[0].start == new_segments[-1].end

        return Subpath(*new_segments).set_Z(following=self)

    def d(self,
          previous_segment=None,  # for use with relative coordinates
          options={},
          **kw):
        """
        Returns a path d-string for the subpath object. A dictionary
        of options can be provided, or else individual keywords.
        See the global 'd_string_params' dictionary for valid option
        names.
        """
        op = {}
        op.update(svgpathtools_d_string_params)
        op.update(options)
        op.update(kw)

        for key in op:
            if key not in svgpathtools_d_string_params:
                raise ValueError(f"unknown d-string option: {key}")

        # validating printing options
        for key in _d_string_must_be_strings_params:
            value = op[key]

            if not isinstance(value, str):
                raise ValueError(f"expecting a string for option {key};")

            stripped = value.strip(' ')
            if stripped not in ['', ',']:
                raise ValueError(f"option '{key}' contains non-whitespace-f-or-comma, or more than one comma")

            if key in _d_string_must_be_pure_spaces_params and stripped != '':
                raise ValueError(f"option '{key}' containing non-spaces")

            if key in _d_string_must_be_nonempty_params and len(value) == 0:
                raise ValueError(f"option '{key}' should be length > 0")

        def format_boolean(boolean):
            return str(int(boolean))

        def append_boolean(boolean):
            nonlocal parts
            parts.append(format_boolean(boolean))
            parts.append(' ')

        def format_number(num):
            # it seems that the threshold at which python
            # chooses to print large floats representing
            # integers in scientific notation is 10^16, so
            # we'll use the same here (i.e., avoiding converting
            # something like 10^16 to an int, otherwise it will
            # print as 10000000000000000 instead of as 10^16).
            if int(num) == num and abs(num) < 1e16:
                return str(int(num))

            if op['decimals'] is not None:
                to_places = f"{num:.{op['decimals']}f}"
                if '.' in to_places:
                    while to_places[-1] == '0':
                        to_places = to_places[:-1]

                    if to_places[-1] == '.':
                        to_places = to_places[:-1]

                else:
                    assert op['decimals'] == 0

                if to_places == '-0':
                    to_places = '0'

                assert len(to_places) > 0
                return to_places

            return str(num)

        def append_angle(num):  # so far, we don't distinguish with numbers:
            nonlocal parts
            parts.append(format_number(num))
            parts.append(' ')

        def format_complex(z, x_only=False, y_only=False):
            assert not x_only or not y_only
            # segment_end_x = segment_end and x_only
            # segment_end_y = segment_end and y_only
            ans = ''
            if not y_only:
                ans += format_number(z.real)
            if not x_only and not y_only:
                ans += op['spacing_within_coordinate']
            if not x_only:
                ans += format_number(z.imag)
            return ans

        def append_cor(z, x_only=False, y_only=False, segment_end=False):
            nonlocal parts, last_end, previous_segment
            w = z
            if op['use_relative_cors'] and last_end is not None:
                w = z - last_end

            parts.append(format_complex(w, x_only=x_only, y_only=y_only))

            if not segment_end:
                parts.append(' ')

            else:
                last_end = z

        def append_radius(z):
            nonlocal parts
            parts.append(format_complex(z))
            parts.append(' ')

        def format_command(command):
            assert command in 'MLCQSTVHAZ'

            c = ''
            if command != 'M':
                c += op['spacing_before_command']

            if op['use_relative_cors']:
                c += command.lower()

            else:
                c += command

            if command != 'Z':
                c += op['spacing_after_command']

            return c

        def append_command(command):
            nonlocal parts, previous_command
            if command == 'L' and previous_command == 'L':
                if op['include_elidable_line_commands']:
                    parts.append(format_command(command))
                else:
                    parts.append(op['elided_line_command_replacement'])

            elif command == 'L' and previous_command == 'M':
                if op['include_elidable_first_line']:
                    parts.append(format_command(command))
                else:
                    parts.append(op['elided_first_line_command_replacement'])

            elif command == previous_command:
                if op['include_elidable_commands']:
                    parts.append(format_command(command))
                else:
                    parts.append(op['elided_command_replacement'])

            else:
                parts.append(format_command(command))

            previous_command = command

        if len(self) == 0:
            assert self._Z is False
            return ''

        parts = []
        previous_command = None
        last_end = previous_segment.end if previous_segment is not None else None
        append_command('M')
        append_cor(self.start, segment_end=True)
        assert last_end is not None

        for index, segment in enumerate(self):
            assert \
                index == 0 or \
                previous_segment.end == segment.start

            if isinstance(segment, Line) and \
               (index < len(self) - 1 or not self._Z):
                if op['use_V_and_H'] and last_end.real == segment.end.real:
                    append_command('V')
                    append_cor(segment.end, y_only=True, segment_end=True)

                elif op['use_V_and_H'] and last_end.imag == segment.end.imag:
                    append_command('H')
                    append_cor(segment.end, x_only=True, segment_end=True)

                else:
                    append_command('L')
                    append_cor(segment.end, segment_end=True)

            elif isinstance(segment, CubicBezier):
                if op['use_S_and_T'] and \
                   segment.can_use_S_from_previous(previous_segment):
                    append_command('S')
                    append_cor(segment.control2)
                    append_cor(segment.end, segment_end=True)

                else:
                    append_command('C')
                    append_cor(segment.control1)
                    append_cor(segment.control2)
                    append_cor(segment.end, segment_end=True)

            elif isinstance(segment, QuadraticBezier):
                if op['use_S_and_T'] and \
                   segment.can_use_T_from_previous(previous_segment):
                    append_command('T')
                    append_cor(segment.end, segment_end=True)

                else:
                    append_command('Q')
                    append_cor(segment.control1)
                    append_cor(segment.end, segment_end=True)

            elif isinstance(segment, Arc):
                append_command('A')
                append_radius(segment.radius)
                append_angle(segment.rotation)
                append_boolean(segment.large_arc)
                append_boolean(segment.sweep)
                append_cor(segment.end, segment_end=True)

            else:
                assert isinstance(segment, Line)
                assert index == len(self) - 1
                assert self.Z

            previous_segment = segment

        if self._Z:
            assert previous_segment.end == self[0].start
            append_command('Z')

        return ''.join(parts)

    def subpath_at_address(self, address):
        return self

    def segment_at_address(self, address):
        return self[address.segment_index]

    def T2address(self, T_or_address, trust_existing_fields=False):
        """
        creates a new address from a T-value, or completes a partially
        filled address with the T-value set, to an address with
        .segment_index and .t fields set; if a is the returned address, then

        self.point(a.T) == self[a.segment_index].point(t)

        If 'trust_existing_fields' is True, this function "trusts" an address
        with pre-existing .segment_index, .t fields (though it insists
        for both to exist).

        Note: If the T-value lands on more than 1 segment, the first such
        segment is returned (i.e., we minimize the possible value of
        a.segment_index, among all possibilities). The only exception to
        this is if T_or_address == 1 and the last segment has length 0,
        in which case an address with a.segment_index = len(self) - 1,
        a.t = 1 is returned.
        """
        self._calc_lengths()
        if self.length() == 0:
            raise ValueError("Subpath.T2address() called on length 0 subpath")

        assert isinstance(T_or_address, Number) or isinstance(T_or_address, Address)

        a = T_or_address
        if isinstance(T_or_address, Number):
            a = Address(T=T_or_address)

        T = a.T
        if T is None:
            raise ValueError("no T in Subpath.T2address")

        if a.segment_index is not None or a.t is not None:
            if a.segment_index is None or a.t is None:
                raise ValueError("asymmetrically filled address in T2address")
            if trust_existing_fields:
                return a

        # in the following, we use the class setters when the address should
        # be unique; for non-unique addresses, we allow ourselves to overwrite
        # previously existing, potentionally conflicting fields:

        if T == 1:
            if self._lengths[-1] == 0:
                a._segment_index = len(self) - 1
                a._t = 1
            else:
                a.segment_index = len(self) - 1
                a.t = 1

        elif T == 0:
            if self._lengths[0] == 0:
                a._segment_index = 0
                a._t = 0
            else:
                a.segment_index = 0
                a.t = 0

        else:
            T0 = 0
            found = False
            for seg_idx, seg_length in enumerate(self._lengths):
                T1 = T0 + seg_length  # the T-value the current segment ends on
                if T1 >= T:
                    if T1 == T:
                        a._segment_index = seg_idx
                        a._t = 1
                    else:
                        a.segment_index = seg_idx
                        a.t = (T - T0) / seg_length
                    found = True
                    break
                T0 = T1

            if not found:
                assert 0 <= T <= 1
                raise BugException

        return a

    def index_or_None(self, thing):
        try:
            if thing is None:
                return None
            z = self.index(thing)
            print("z:", z)
            return z
            # return None if thing is None else self.index(thing)
        except ValueError:
            if not isinstance(thing, Segment):
                raise ValueError("provided thing is not a Segment")
            raise ValueError("provided segment not in subpath")

    def t2address(self, t=None, segment=None, segment_index=None):
        """
        Preliminary note: t may be given as an Address object. Also,
        t, segment_index, segment are kept as keywords parameters mainly
        for syntactic consistency with Path.T2address.

        Takes a mandatory value t, 0 <= t <= 1, and an integer segment_index
        (mandatory unless len(self) == 1, and possibly supplied indirectly
        via the 'segment' keyword argument---see previous function---which
        is otherwise not needed), and returns an address a such that

           -- a.t == t
           -- a.segment_index == segment_index
           -- self.point(a.T) == self[segment_index].point(t)
           -- a.subpath_index == ? (pre-existing value or None)
           -- a.W             == ? (pre-existing value or None)
        """
        a = t if isinstance(t, Address) else Address(t=t)

        if a.t is None:
            raise ValueError("t missing in Subpath.t2address")

        if isinstance(segment_index, Segment):
            if segment is not None:
                raise ValueError("segment_index twice supplied")

        a.segment_index = segment_index  # (does not overwrite non-None value)
        a.segment_index = self.index_or_None(segment)

        if a.segment_index is None:
            print("segment:", segment)
            print("segment_index:", segment_index)
            raise ValueError("segment_index missing")
        if not 0 <= a.segment_index <= len(self) - 1:
            raise ValueError("segment_index out of range")

        t = a.t
        segment_index = a.segment_index

        self._calc_lengths()

        T0 = sum(self._lengths[:segment_index])
        T1 = T0 + self._lengths[segment_index]
        assert T1 == sum(self._lengths[:segment_index + 1])
        a.T = T1 if t == 1 else (T1 - T0) * t + T0

        return a

    def derivative(self, T_or_address, n=1):
        """
        returns the tangent vector of the Subpath at T (centered at the
        origin and expressed as a complex number).

        Note: Bezier curves can have points where their derivative vanishes.
        The unit_tangent() method, by contrast, attempts to compute the
        direction of the derivative at those points as well.
        """
        a = self.T2address(param2address(self, T_or_address))
        segment = self[a.segment_index]
        if self._length:
            seg_length = self._lengths[a.segment_index] * self._length
        else:
            seg_length = segment.length()
        return segment.derivative(a.t, n=n) / seg_length**n

    def unit_tangent(self, T_or_address):
        """
        returns the unit tangent vector of the Subpath at T (centered at the
        origin and expressed as a complex number).  If the tangent vector's
        magnitude is zero, this method will attempt to find the limit of
        self.derivative(u) / abs(self.derivative(u)) as u approaches T.
        See the implementation of bezier_unit_tangent for more details.
        """
        a = self.T2address(param2address(self, T_or_address))
        return self._segments[a.segment_index].unit_tangent(a.t)

    def curvature(self, T_or_address):
        """returns the curvature of the subpath at T while checking for
        possible non-differentiability at T. Outputs float('inf') if not
        differentiable at T."""
        a = self.T2address(param2address(self, T_or_address))
        segment_index, t = a.segment_index, a.t

        seg = self[segment_index]

        if np.isclose(t, 0) and (segment_index != 0 or self._Z):
            previous_seg_in_path = self._segments[
                (segment_index - 1) % len(self._segments)]
            if not seg.joins_smoothly_with(previous_seg_in_path):
                return float('inf')

        elif np.isclose(t, 1) and (segment_index != len(self) - 1 or self._Z):
            next_seg_in_path = self._segments[
                (segment_index + 1) % len(self._segments)]
            if not next_seg_in_path.joins_smoothly_with(seg):
                return float('inf')

        # curvature is invariant under reparameterization, so we can
        # use the segment's own parameterization (?):
        return seg.curvature(t)

    def area(self, quality=0.01, safety=3):
        """
        Returns the area enclosed by the subpath, assuming self.end ==
        self.start, after converting arcs to cubic bezier segments.

        The 'quality' and 'safety' parameters control the latter conversion.
        See the docstring for Subpath.converted_bezier_path for more details.

        Note: negative area results from CW (as opposed to CCW)
        parameterization of the Path object.
        """
        if len(self) == 0:
            return 0

        assert self.start_equals_end()

        cubicized = self.converted_to_bezier(quality=quality, safety=safety, reuse_segments=True, use_Maisonobe=True)
        area_enclosed = 0
        for seg in cubicized:
            x         = real(seg.poly())
            dy        = imag(seg.poly()).deriv()
            integrand = x * dy
            integral  = integrand.integ()
            area_enclosed += integral(1) - integral(0)
        return area_enclosed

    def normalize_address(self, a, use_Z=True):
        """
        Expects a fully-formed address a; tweaks a like so: if a.t == 0
        and the preceding segment's t == 1 address points to the same point
        on the subpath (i.e., the previous segment exists!), change a to use
        t == 1 and the segment_index of the previous segment.

        The 'use_Z' option determines whether self[len(self) - 1] counts as a
        "previous segment" of self[0], assuming self.Z == True.
        """
        if not a.is_complete(for_object=self):
            raise ValueError("incomplete Subpath address")

        if a.t == 0:
            prev_index = self.mod_index(a.segment_index - 1, use_Z=use_Z)
            prev = self[prev_index] if prev_index >= 0 else None
            if prev is not None:
                assert prev.end == self[a.segment_index].start
                b = self.t2address(t=1, segment_index=prev_index)
                assert b.T == a.T or (b.T == 1 and a.T == 0 and
                                      self._Z and use_Z)
                a._segment_index = prev_index
                a._t = 1
                a._T = b.T

    def x_val_intersect(self, x_val):
        to_return = []
        for segment_index, seg in enumerate(self):
            seg_adrs = seg.x_val_intersect(x_val)
            for adr in seg_adrs:
                if adr.t == 0 and len(to_return) > 0 and to_return[-1].t == 1 and to_return[-1].segment_index == segment_index - 1:
                    continue
                to_return.append(self.t2address(t=adr.t, segment_index=segment_index))
                assert to_return[-1].segment_index == segment_index
        return to_return

    def intersect(self, other_curve, justonemode=False, tol=1e-12,
                  normalize=True):
        """
        Returns list of pairs of named Intersection objects by taking all the
        pairwise intersections of segments in self and segments in
        other_curve. Here other_curve can be either a segment, a subpath, or a
        path. In the latter case, the latter case, other_curve.intersect(self)
        is called, and intersections are swapped.

        If the two curves coincide for more than a finite set of
        points, this code will (should) fail.

        If justonemode==True, then returns the first intersection found.

        tol is used as a parameter passed to segment.intersect(segment); see
        implementations of segment.intersect

        If normalize==True, remove all 't==0' intersections (beginning-of-
        segment intersections) that can be written as 't==1' intersections
        (end-of-segment intersections). This option can be useful to avoid
        duplicates.


        Note: If the respective subpath is a geometric loop but not a
        topological loop (i.e., _Z has not been set for the subpath), the
        adjacency between the first and last segments of the subpath is ignored
        by the 'normalize' option. Similarly, if other_curve is made up of
        several subpaths that are adjacent at their endpoints, these
        adjacencies are ignored by 'normalize'.)
        """
        subpath1 = self

        if isinstance(other_curve, Subpath):
            subpath2 = other_curve

        elif isinstance(other_curve, Segment):
            subpath2 = Subpath(other_curve)

        elif isinstance(other_curve, Path):
            reversed_intersections = \
                other_curve.intersect(self, normalize=normalize)
            return [(a2, a1) for (a1, a2) in reversed_intersections]

        else:
            raise ValueError("bad other_curve in Subpath.intersect")

        # let...

        def append_new_intersection(a1, a2):
            if normalize:
                subpath1.normalize_address(a1)
                subpath2.normalize_address(a2)
                if (a1, a2) not in intersection_list:
                    intersection_list.append((a1, a2))
            else:
                assert (a1, a2) not in intersection_list
                intersection_list.append((a1, a2))

        # in...

        intersection_list = []
        for si1, seg1 in enumerate(subpath1):
            for si2, seg2 in enumerate(subpath2):
                for a1, a2 in seg1.intersect(seg2, tol=tol):
                    a1 = subpath1.t2address(segment_index=si1, t=a1)
                    a2 = subpath2.t2address(segment_index=si2, t=a2)
                    append_new_intersection(a1, a2)
                    if justonemode:
                        return intersection_list[0]

        return intersection_list

    def even_odd_encloses(self, pt, ref_point=None):
        if not self.Z:
            raise ValueError("Subpath.even_odd_encloses wants loop")
        if ref_point is None:
            ref_point = self.point_outside()
        intersections = \
            Subpath(Line(pt, ref_point)).intersect(self, normalize=False)
        return bool(len({a1.t for a1, a2 in intersections}) % 2)

    def union_encloses(self, pt, ref_point=None):
        raise NotImplementedError

    def rotate(self, degs, origin=0):
        self._segments = [rotate(seg, degs, origin) for seg in self]
        return self._reset()

    def translate(self, x, y=None):
        self._segments = [translate(seg, x, y) for seg in self]
        return self._reset()

    def scale(self, sx, sy=None):
        self._segments = [scale(seg, sx, sy) for seg in self]
        return self._reset()

    def transform(self, tf):
        if isinstance(tf, str):
            tf = parse_transform(tf)
        self._segments = [transform(seg, tf) for seg in self]
        return self._reset()

    def cropped(self, T0_or_address, T1_or_address, drop_small=False):
        """
        Returns a cropped copy of the subpath starting at self.point(T0) and
        ending at self.point(T1); T0 and T1 must be distinct, 0 < T0, T1 < 1.

        If T1 < T0 the crop interpreted as a wraparound crop. In that case the
        subpath must be geometrically closed.

        If drop_small==True, initial and final subpath segments seg such that
        np.isclose(seg.start, seg.eng) are dropped from the final answer.
        This can (theoretically) result in an empty subpath being returned.
        """
        def subpath_greater_than(adr1, adr2):
            if adr1.T > adr2.T:
                return True

            if adr1.T < adr2.T:
                return False

            if adr1.segment_index > adr2.segment_index:
                assert adr1.t == 0 and adr1.t == 1
                return True

            return False

        a0 = param2address(self, T0_or_address)
        if a0.t is None or a0.segment_index is None:
            self.T2address(a0)

        a1 = param2address(self, T1_or_address)
        if a1.t is None or a1.segment_index is None:
            self.T2address(a1)

        assert \
            a0.subpath_index is None or \
            a1.subpath_index is None or \
            a0.subpath_index == a1.subpath_index  # (recently added)

        initial_orientation = subpath_greater_than(a0, a1)

        if a0.T == a1.T:
            raise ValueError("Subpath.cropped called with T0 == T1")

        if a0.t == 1:
            # whether or not self._Z:
            a0._segment_index = (a0._segment_index + 1) % len(self)
            a0._t = 0

        if a1.t == 0:
            # whether or not self._Z:
            a1._segment_index = (a1._segment_index - 1) % len(self)
            a1._t = 1

        if initial_orientation != subpath_greater_than(a0, a1):
            raise ValueError("Subpath.cropped sees endpoints change place"
                             "after preprocessing; corner case?")

        if a0.T < a1.T:
            if a0.segment_index == a1.segment_index:
                to_return = Subpath(self[a0.segment_index].cropped(a0, a1))
            else:
                to_return = Subpath(self[a0.segment_index].cropped(a0, 1))
                # to_return.debug = True
                for index in range(a0.segment_index + 1, a1.segment_index):
                    to_return.append(self[index])
                to_return.append(self[a1.segment_index].cropped(0, a1))

        elif a0.T > a1.T:
            if not self._Z:
                raise ValueError("Subpath will not do wraparound crop of"
                                 "non-closed subpath")

            else:
                assert self.start_equals_end()

            to_return = self.cropped(a0, 1) if a0.T < 1 else Subpath()
            to_return.extend(self.cropped(0, a1) if a1.T > 0 else Subpath())

        else:
            raise ValueError("T0 == T1 in Subpath.cropped().")

        return to_return

    def converted_to_bezier(self, quality=0.01, safety=5,
                            reuse_segments=True, use_Maisonobe=False):
        """
        Warning: reuses same segments when available, unless 'reuse_segments'
        is set to False.
        """
        new_subpath = Subpath()

        for s in self:
            if isinstance(s, Arc):
                cpath, _ = s.converted_to_bezier_subpath(
                    quality,
                    safety,
                    use_Maisonobe=use_Maisonobe
                )
                new_subpath.extend(cpath)

            else:
                p = s if reuse_segments else translate(s, 0)
                new_subpath.append(p)

        if self._Z:
            new_subpath.set_Z()

        return new_subpath

    def path_of(self):
        return Path(self)

    def to_path(self):
        return Path(self)

    def set_Z(self, forceful=False, following=None):
        """Set ._Z value of self to True, unless 'following' is not None, in
        which case self._Z is set to match the value of following._Z.

        If 'forceful' is True, will create, if necessary, a final line
        joining self.start to self.end before setting ._Z to True. If
        'forceful' is False and self.start != self.end, raises a value error.
        """
        if following is not None and not following.Z:
            self._Z = False
            return self

        if len(self) == 0:
            raise ValueError("calling .set_Z() on empty subpath")

        if self._Z:
            assert self.start_equals_end()
            warn("Z already set is Subpath.set_Z(); ignoring")

        else:
            if not self.start_equals_end():
                if forceful:
                    self.append(Line(self.end, self.start))
                    assert self.start_equals_end()

                else:
                    raise ValueError("self.end != self.start (use .set_Z(forceful=True) ?)")

            self._Z = True

        return self

    def unset_Z(self):
        self._Z = False
        return self

    def offset(self, amount, quality=0.01, safety=5,
               join='miter', miter_limit=4):
        return self.pro_offset(amount, quality, safety, join, miter_limit)[0]

    def pro_offset(self, amount, quality=0.01, safety=10,
                   join='miter', miter_limit=4, two_sided=False):
        converted = self.converted_to_bezier(quality, safety, True)
        way_out   = Path()  # don't worry this will soon be a subpath
        way_in    = Path()  # don't worry this will soon be a subpath
        skeleton  = Subpath()

        for seg in converted:
            wo, sk, wi = seg.pro_offset(amount, quality, safety, two_sided)
            assert isinstance(wo, Subpath)
            assert isinstance(wi, Subpath)
            assert isinstance(sk, Subpath)
            way_out.append(wo)
            skeleton.extend(sk)
            way_in.prepend(wi)

        if self._Z:
            skeleton.set_Z()

        # joins
        reversed_skeleton = skeleton.reversed() if two_sided else Subpath()
        assert len(reversed_skeleton) == way_in.num_segments()
        both_results = []
        both_skeleton_offset_pairs = [
            (skeleton, way_out),
            (reversed_skeleton, way_in)
        ]

        for ske_off_pair in both_skeleton_offset_pairs:
            joined = join_offset_segments_into_subpath(
                ske_off_pair[0],
                ske_off_pair[1],
                amount, join, miter_limit
            )
            both_results.append(joined)
            assert joined.Z == self.Z

        way_out = both_results[0]  # Subpath
        way_in = both_results[1]  # Subpath

        assert isinstance(way_out, Subpath)
        assert isinstance(way_in, Subpath)

        return way_out, skeleton, way_in

    def stroke(self, width, quality=0.01, safety=5, join='miter',
               miter_limit=4, cap='butt', reversed=False):
        if reversed:
            width *= -1

        way_out, skeleton, way_in = self.pro_offset(
            width / 2,
            quality,
            safety,
            join,
            miter_limit,
            two_sided=True
        )

        # line caps
        if self._Z:
            result = Path(way_out, way_in)
        else:
            assert cap in ['butt', 'round', 'square']
            c_end = endcap_for_curve(skeleton, width / 2, cap)
            c_start = endcap_for_curve(skeleton.reversed(), width / 2, cap)
            assert c_start.start == way_in.end
            assert c_start.end == way_out.start
            assert c_end.start == way_out.end
            assert c_end.end == way_in.start
            way_out.extend(c_end)
            way_in.extend(c_start)
            way_out.extend(way_in)
            way_out.set_Z(forceful=False)
            result = Path(way_out)

        return result


class Path(Curve, MutableSequence):
    _class_repr_options = _load_repr_options_for('path')

    @staticmethod
    def update_class_repr_options(dictionary):
        for key in dictionary:
            if key not in Path._class_repr_options:
                raise ValueError("unknown repr_option:", key)
        Path._class_repr_options.update(dictionary)

    def __init__(self, *things, extend_by_segments=False,
                 clone_affected_subpaths=True):
        self._subpaths = []

        # in case we want to print during init: ;)
        self._repr_options_init()
        self._field_names = []

        # building the subpaths from 'things'
        self.extend(things,
                    extend_by_segments=extend_by_segments,
                    clone_affected_subpaths=clone_affected_subpaths)

        assert all(isinstance(s, Subpath) for s in self._subpaths)

    # def shortname(self):
    #     return 'path'

    def first_nonempty(self):
        for index, x in enumerate(self):
            if len(x) > 0:
                return index, x
        return None

    def last_nonempty(self):
        for index, x in enumerate(reversed(self)):
            if len(x) > 0:
                return len(self) - 1 - index, x
        return None

    def first_segment(self):
        fn = self.first_nonempty()
        return fn.first_segment() if fn is not None else None

    def last_segment(self):
        ln = self.last_nonempty()
        return ln.last_segment() if ln is not None else None

    def __normalize_index(self, index, max):
        if index < 0:
            index += len(self)
        if not 0 <= index <= max:
            raise ValueError("out of bounds index in Path.__normalize_index")
        return index

    def __getitem__(self, index):  # (MutableSequence abstract method)
        return self._subpaths[index]

    def __setitem__(self, index, value, even_if_empty=True):  # (MutableSequence abstract method)
        if not isinstance(value, Subpath):
            raise ValueError("value not a Subpath in Path.__setitem__")
        if len(value) == 0 and even_if_empty is False:
            return value
        index = self.__normalize_index(index, len(self))
        self._subpaths[index] = value
        return value

    def __delitem__(self, index):  # (MutableSequence abstract method)
        index = self.__normalize_index(index, len(self) - 1)
        del self._subpaths[index]

    def __len__(self):  # (MutableSequence abstract method)
        return len(self._subpaths)

    def insert(self, index, value, even_if_empty=True):  # (MutableSequence abstract method)
        if not isinstance(value, Subpath):
            raise ValueError("value not a Subpath in Path.__setitem__")
        if len(value) > 0 or even_if_empty:
            self._subpaths.insert(index, value)
        return value

    # provided by MutableSequence: __contains__, __iter__, __reversed__
    #                              index, count, append, reverse, pop,
    #                              remove, __iadd__; extend is overriden

    # Unfortunately we still have to provide our own 'append' implementation
    # because of the 'even_if_empty' keyword:
    def append(self, value, even_if_empty=False):
        self.insert(len(self), value, even_if_empty=even_if_empty)

    # I put this here because I think it's faster than letting python
    # figure out the iterator via __getitem__ etc (?)
    def __iter__(self):
        return self._subpaths.__iter__()

    # overrides the default python implementation:
    def extend(self, things, even_if_empty=False,
               extend_by_segments=True, clone_affected_subpaths=True):
        # note: temp_subpath is either None or else has length
        # at least 1 and is not a pre-existing path

        temp_subpath = None

        assert all(isinstance(s, Subpath) for s in self._subpaths)

        def ingest_new_subpath(subpath):
            assert isinstance(subpath, Subpath)
            nonlocal temp_subpath
            if temp_subpath is not None:
                assert isinstance(temp_subpath, Subpath)
                self._subpaths.append(temp_subpath)
                temp_subpath = None
            if even_if_empty or len(subpath) > 0:
                self._subpaths.append(subpath)

        for thing in things:
            if isinstance(thing, Path):
                if thing is self and len(thing) > 0:
                    raise ValueError("attempt to extend path by self")
                for subpath in thing:
                    ingest_new_subpath(subpath)

            elif isinstance(thing, Subpath):
                ingest_new_subpath(thing)

            elif isinstance(thing, Segment):
                if temp_subpath is None:
                    if extend_by_segments and \
                       len(self) > 0 and \
                       len(self[-1]) > 0 and \
                       not self[-1].Z and \
                       self[-1].end == thing.start:
                        if not clone_affected_subpaths:
                            self[-1].append(thing)
                        else:
                            temp_subpath = self.pop().cloned()
                            temp_subpath.append(thing)
                    else:
                        temp_subpath = Subpath(thing)
                elif temp_subpath.end == thing.start:
                    temp_subpath.append(thing)
                else:
                    self._subpaths.append(temp_subpath)
                    temp_subpath = Subpath(thing)

            else:
                raise ValueError("Path constructor takes segments, subpaths")

        if temp_subpath is not None:
            self._subpaths.append(temp_subpath)

        assert all(isinstance(s, Subpath) for s in self._subpaths)

        return self

    def erase(self):
        self._subpaths = []  # yes boss!

    def prepend(self, value, even_if_empty=False):
        self.insert(0, value, even_if_empty=even_if_empty)

    def reversed(self):
        """
        returns a copy of the Path object with its orientation reversed.
        """
        return Path(*[subpath.reversed() for subpath in reversed(self)])

    def __eq__(self, other, tol=0):
        if not isinstance(other, Path):
            return NotImplemented
        return \
            len(self) == len(other) and \
            all(s.__eq__(o, tol) for s, o in zip(self, other))

    def __ne__(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        return not self == other

    def segment_iterator(self, back_to_front=False):
        if back_to_front:
            for t in reversed(self):
                for seg in reversed(t):
                    yield seg
        else:
            for t in self:
                for seg in t:
                    yield seg

    def is_bezier_path(self):
        return all(s.is_bezier_subpath() for s in self)

    def is_or_has_arc(self):
        return any(s.is_or_has_arc() for s in self)

    def is_naively_continuous(self):
        prev = None
        for s in self.segment_iterator():
            if prev and prev.end != s.start:
                return False
            prev = s
        return True

    def _calc_lengths(self, error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH):
        lengths = [each.length(error=error, min_depth=min_depth) for each in
                   self]
        self._length = sum(lengths)
        self._lengths = [each / self._length for each in lengths]

    def point(self, W_or_address):
        # W: name of time parameter for Path
        # T: .......................for Subpath
        # t: .......................for Segment
        a = param2address(self, W_or_address)
        self.W2address(a)
        return self[a.subpath_index][a.segment_index].point(a)

    def length(self, W0=0, W1=1,
               error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH):
        self._calc_lengths(error=error, min_depth=min_depth)

        a0 = self.W2address(W0)
        a1 = self.W2address(W1)

        if a0.W == 0 and a1.W == 1:
            return self._length

        else:
            if a1.W < a0.W:
                raise NotImplementedError

            if len(self) == 1:
                return self[0].length(T0=a0, T1=a1)

            if a0.subpath_index == a1.subpath_index:
                return self[a0.subpath_index].length(T0=a0, T1=a1)

            assert a0.subpath_index < a1.subpath_index

            return \
                self[a0.subpath_index].length(T0=a0) + \
                sum(self[i].length() for i in range(a0.subpath_index + 1,
                                                    a1.subpath_index)) + \
                self[a1.subpath_index].length(T1=a1)

    @property
    def start(self):
        for x in self:
            if len(x) > 0:
                return x.start
        return None

    @start.setter
    def start(self, pt):
        raise Exception("Segments are immutable!")

    @property
    def end(self):
        for x in reversed(self):
            if len(x) > 0:
                return x.end
        return None

    @end.setter
    def end(self, pt):
        raise Exception("Segments are immutable!")

    def d(self,
          options={},
          **kw):
        """
        Returns a path d-string for the path object. See the notes of
        Subpath.d(...) for more details.
        """
        op = {}
        op.update(svgpathtools_d_string_params)
        op.update(options)
        op.update(kw)
        nonempty = [s for s in self if len(s) > 0]
        ans = ''
        for index, s in enumerate(nonempty):
            if index > 0:
                ans += op['spacing_before_new_subpath']
                ans += s.d(previous_segment=nonempty[index - 1][-1], options=op)
            else:
                ans += s.d(options=op)
        return ans

    def index_or_None(self, thing):
        try:
            return None if thing is None else self.index(thing)
        except ValueError:
            if not isinstance(thing, Subpath):
                raise ValueError("provided thing is not a Subpath")
            raise ValueError("provided subpath not in Path")

    def t2address(self, t=None, segment=None, segment_index=None,
                  subpath=None, subpath_index=None):

        a = t if isinstance(t, Address) else Address(t=t)

        if a.t is None:
            raise ValueError("t missing in Path.t2address")

        # resolve subpath_index
        a.subpath_index = subpath_index
        a.subpath_index = self.index_or_None(subpath)

        # let the Subpath method fill in T, segment, segment_index:
        self[a.subpath_index].t2address(
            a,
            segment=segment,
            segment_index=segment_index
        )

        # W (T2address will also check that a.subpath_index is present)
        return self.T2address(a)

    def T2address(self, T=None, subpath=None, subpath_index=None):
        """
        Preliminary note: T may be an address.

        Takes one of:

           --- a T-value and either 'subpath_index' or 'subpath'
           --- an incomplete Address address with at least its T- and
               subpath_index set

        (OK: actually, subpath_index/subpath are not even necessary if self
        has only 1 nonempty subpath) and returns a full address matching the
        above fields, such that

            self.point(a.W) == self[subpath_index].point(T)

        where subpath_index is a.subpath_index or the input parameter
        subpath_index, depending on which is defined.

        Note: The address returned is the same object as the provided
        address, if provided; hence this function has (intentional) side
        effects! """

        a = T if isinstance(T, Address) else Address(T=T)

        if a.T is None:
            raise ValueError("missing T value in Path.T2address")

        # obtaining subpath_index; we are very lenient with the poor user,
        # and make good efforts to reorder messed-up arguments
        if isinstance(subpath, int):
            if subpath_index is not None and subpath_index != subpath:
                raise ValueError("subpath_index twice supplied")
            subpath_index = subpath
            subpath = None

        # now for real (pls see Address.subpath_index property setter to
        # understand how this works):
        a.subpath_index = subpath_index
        a.subpath_index = self.index_or_None(subpath)

        if a.subpath_index is None:
            raise ValueError("subpath_index missing in Path.T2address")
        if not 0 <= a.subpath_index <= len(self) - 1:
            raise ValueError("bad subpath_index in Path.T2address")

        # W
        T = a.T
        subpath_index = a.subpath_index

        self._calc_lengths()

        W0 = sum(self._lengths[:subpath_index])
        W1 = W0 + self._lengths[subpath_index]
        assert W1 == sum(self._lengths[:subpath_index + 1])
        a.W = W1 if T == 1 else (W1 - W0) * T + W0

        if a.segment_index is None or a.t is None:
            self[a.subpath_index].T2address(a)

        return a

    def W2address(self, W):
        """
        Constructs and returns a full address object from a W value,
        or fills in remaining fields of an unfinished address with a W value.
        """
        def compute_T_and_subpath_index():
            self._calc_lengths()

            W0 = 0
            for index, subpath in enumerate(self._subpaths):
                W1 = W0 + self._lengths[index]
                if W <= W1:
                    return (W - W0) / (W1 - W0), index
                W0 = W1

            assert 0 <= W <= W1
            raise BugException

        a = W if isinstance(W, Address) else Address(W=W)

        if a.W is None:
            raise ValueError("W is None in Path.W2address")

        W = a.W

        # shortcuts:
        if W == 1:
            a.subpath_index = self.last_nonempty()[0]
            a.T = 1
            a.segment_index = len(self[a.subpath_index]) - 1
            a.t = 1
            return a

        if W == 0:
            a.subpath_index = self.first_nonempty()[0]
            a.T = 0
            a.segment_index = 0
            a.t = 0
            return a

        le_T, le_index = compute_T_and_subpath_index()

        if a.subpath_index is None and a.T is None:
            a.subpath_index = le_index
            a.T = le_T

        elif a.subpath_index != le_index or a.T != le_T:
            if (
                le_T == 1 and
                le_index == a.subpath_index - 1 and
                a.T == 0 and
                a.subpath_index < len(self) and
                len(self[a.subpath_index]) > 0
            ):
                # pre-existing fields look OK, leave this address alone
                # (the caller may not want to see it change through side effect)
                pass

            else:
                print("le_T, le_index:", le_T, le_index)
                print("a.T, a.subpath_index:", a.T, a.subpath_index)
                raise ValueError("Incompatible-looking pre-existing "
                                 "value of T, subpath_index in Path.W2address")

        if a.segment_index is None or a.t is None:
            self[a.subpath_index].T2address(a)

        return a

    def subpath_at_address(self, address):
        return self[address.subpath_index]

    def segment_at_address(self, address):
        assert address.subpath_index is not None
        assert address.segment_index is not None
        return self[address.subpath_index][address.segment_index]

    def derivative(self, W_or_address, n=1):
        """
        Given an address a or a value W, which is resolved to the default
        address a, returns the n-th derivative of the path's
        W-parameterization at a.
        """
        a = param2address(self, W_or_address)
        self._calc_lengths()
        return \
            self[a.subpath_index].derivative(a, n=n) / \
            self._lengths[a.subpath_index]**n

    def unit_tangent(self, W_or_address):
        """
        Given an address a or a value W, which is resolved to the default
        address a, returns self[a.subpath_index][a.segment_index].unit_tangent(a.t).
        """
        a = param2address(self, W_or_address)
        return self[a.subpath_index][a.segment_index].unit_tangent(a.t)

    def curvature(self, W_or_address):
        """
        Given an address a or a value W, which is resolved to the default
        address a, the curvature of the path at a, outputting float('inf') if
        the path is not differentiable at a.
        """
        a = param2address(self, W_or_address)
        return self[a.subpath_index][a.segment_index].curvature(a.t)

    def area(self, quality=0.01, safety=3):
        """
        Returns the directed area enclosed by the path; requires each subpath
        to be geometrically closed. (But with or without being topologically
        closed.)

        Negative area results from CW (as opposed to CCW) parameterization of
        a Subpath.

        Elliptical arc segments are converted to bezier paths [nb: the
        original segments remain in place]; the 'quality' and 'safety'
        parameters control this conversion, as described in
        Subpath.converted_to_bezier().
        """
        return sum([s.area(quality, safety) for s in self])

    def xray(self, x, normalize=True, justonemode=False, tol=1e-12):
        xmin, xmax, ymin, ymax = self.bbox()
        ray = Line(start=x + 1j * (ymin - 10), end=x + 1j * (ymax + 10))
        return self.intersect(ray), ray

    def yray(self, y, normalize=True, justonemode=False, tol=1e-12):
        xmin, xmax, ymin, ymax = self.bbox()
        return [a1 for (a1, a2) in self.intersect(Line(start=(xmin - 10) + 1j * y,
                                                       end=(xmax + 10) + 1j * y))]

    def intersect(self, other_curve, normalize=True, justonemode=False,
                  tol=1e-12):
        """
        Returns a list of pairs of addresses (a1, a2): for each intersection
        found, a1 is the intersection's adddress with respect to this path, a2
        with respect to the other path.

        If normalize==False, makes a list of intersections between all
        possible pairs of segments in the two paths.

        If normalize==True, normalizes addresses that fall at the beginning
        of a segment (i.e., addresses with t == 0) to be included in the
        subpath's previous segment, if present (so addresses with t == 0
        switch to addresses with t == 1, when possible), while removing
        duplicate intersections with t == 1.

        'other_curve' can be either a path, a subpath or a segment; in the
        latter two cases, the subpath or segment is wrapper in a path resp.
        / subpath and path before computing the intersections.

        If justonemode==True, returns just the first intersection found.

        tol is used to check for redundant intersections (see comment above
        the code block where tol is used).

        Fails if the two path objects coincide for more than a finite set of
        points.
        """
        path1 = self
        if isinstance(other_curve, Path):
            path2 = other_curve

        elif isinstance(other_curve, Subpath):
            path2 = Path(other_curve)

        elif isinstance(other_curve, Segment):
            path2 = Path(Subpath(other_curve))

        else:
            raise ValueError("bad type for other_curve in Path.intersect()")

        assert path1 != path2

        intersection_list = []

        for ind1, sub1 in enumerate(path1):
            for ind2, sub2 in enumerate(path2):
                tweenies = sub1.intersect(sub2,
                                          normalize=normalize,
                                          justonemode=justonemode,
                                          tol=tol)
                for a1, a2 in tweenies:
                    a1.subpath_index = ind1
                    a2.subpath_index = ind2
                    path1.T2address(a1)
                    path2.T2address(a2)

                if justonemode and len(tweenies) > 0:
                    return tweenies[0]

                intersection_list.extend(tweenies)

        return intersection_list

    def even_odd_encloses(self, pt, ref_point_outside=None):
        if any(not s.Z and len(s) > 0 for s in self):
            raise ValueError('path has non-loop subpath')
        if ref_point_outside is None:
            ref_point_outside = self.point_outside()
        intersections = Path(
            Subpath(
                Line(pt, ref_point_outside)
            )
        ).intersect(self)
        return bool(len({a1.t for a1, a2 in intersections}) % 2)

    def union_encloses(self, pt, ref_point_outside=None):
        """Not implemented"""
        return any(s.union_encloses(pt, ref_point_outside) for s in self)

    # It just seemed more intuitive to not clone subpaths by default
    # when all that is changed is their topological closure...
    def smelt_loops(self, clone_affected_subpaths=False):
        """
        Closes subpaths that form geometric loops, if any.

        By default, does not clone affected subpaths. This contrasts
        with the default behavior of selft.smelt_subpaths, which clones
        subpaths before concatenating them.
        """
        newsubpaths = []
        for s in self:
            if len(s) > 0 and s.end == s.start and not s.Z:
                if clone_affected_subpaths:
                    newsubpaths.append(s.cloned().set_Z())
                else:
                    newsubpaths.append(s.set_Z())
            else:
                newsubpaths.append(s)
        self._subpaths = newsubpaths
        return self

    def smelt_subpaths(self, wraparound=False, clone_affected_subpaths=True):
        """ gets rid of empty subpaths and concatenates index-wise consecutive
        subpaths whose endpoints match and whose Z-properties are not set

        If wraparound==True, will also attempt to concatenate the first and
        last (nonempty) subpaths, unless they already coincide. (If such
        concatenation is made, the resulting subpath becomes the new last
        subpath as opposed to becoming the new first subpath, by default.)
        (Note: Maybe one should make the latter behavior should be controllable
        with an extra option?)

        Makes clones of affected subpaths by default, such as not to disturb
        other paths possibly having these subpaths. This contrasts with the
        default behavior of smelt_loops.
        """
        oldsubpaths = [s for s in self if len(s) > 0]
        newsubpaths = []

        if len(oldsubpaths) > 0:
            current_subpath = oldsubpaths[0]

            for this_subpath in oldsubpaths[1:]:
                if not current_subpath.Z and not this_subpath.Z and \
                   current_subpath.end == this_subpath.start:
                    if current_subpath in oldsubpaths and \
                       clone_affected_subpaths:
                        current_subpath = current_subpath.cloned()
                    current_subpath.extend(this_subpath)
                else:
                    newsubpaths.append(current_subpath)
                    current_subpath = this_subpath

            if wraparound and \
               not current_subpath.Z and not newsubpaths[0].Z and \
               current_subpath.end == newsubpaths[0].start:
                if newsubpaths[0] is not current_subpath:
                    if current_subpath in oldsubpaths and \
                       clone_affected_subpaths:
                        current_subpath = current_subpath.cloned()
                    current_subpath.extend(newsubpaths[0])
                    newsubpaths = newsubpaths[1:]
                else:
                    # by default, we don't close loop; leave that
                    # to smelt_loops if desired
                    pass

            newsubpaths.append(current_subpath)

        self._subpaths = newsubpaths
        return self

    def rotate(self, degs, origin=0):
        for subpath in self:
            subpath.rotate(degs, origin)
        return self

    def translate(self, x, y=0):
        for subpath in self:
            subpath.translate(x, y)
        return self

    def scale(self, sx, sy=None):
        for subpath in self:
            subpath.scale(sx, sy)
        return self

    def transform(self, tf):
        if isinstance(tf, str):
            tf = parse_transform(tf)
        for subpath in self:
            subpath.transform(tf)
        return self

    def cropped(self, W0_or_address, W1_or_address):
        a0 = self.W2address(param2address(self, W0_or_address))
        a1 = self.W2address(param2address(self, W1_or_address))

        if a0.W == 1 and a1.W == 0:
            # (we could do this but the test says not not to):
            raise ValueError("bad endpoints in Path.cropped")

        if a0 > a1 and not self.isloop():
            # (we could do this too, but test says...)
            raise ValueError("we could do this crop, but, per the tests,"
                             "wraparound crops only allowed on loops!")

        while a0.T == 1:
            assert a0.t == 1
            assert a0.segment_index == len(self[a0.subpath_index]) - 1
            while True:
                a0._subpath_index = (a0.subpath_index + 1) % len(self)
                if a0.subpath_index == 0:
                    a0._W = 0
                if self[a0.subpath_index].length() > 0:
                    break
                else:
                    assert self.length() > 0
            a0._T = a0._t = 0
            a0._segment_index = 0

        if a0.t == 1:
            assert a0.segment_index < len(self[a0.subpath_index]) - 1
            a0._segment_index = a0.segment_index + 1
            a0._t = 0

        while a1.T == 0:
            assert a1.t == 0
            assert a1.segment_index == 0
            while True:
                a1._subpath_index = (a1.subpath_index - 1) % len(self)
                if a1.subpath_index == len(self) - 1:
                    a1._W = 1
                if self[a1.subpath_index].length() > 0:
                    break
                else:
                    assert self.length() > 0
            a1._T = a1._t = 1
            a1._segment_index = len(self[a1.subpath_index]) - 1
            assert a1._segment_index >= 0

        if a1.t == 0:
            assert a1.segment_index > 0
            a1._segment_index = a1._segment_index - 1
            a1._t = 1

        if a0.W < a1.W:
            if a0.subpath_index == a1.subpath_index:
                to_return = Path(self[a0.subpath_index].cropped(a0, a1))

            else:
                to_return = Path(self[a0.subpath_index].cropped(a0.tweaked(W=None, subpath_index=None), 1))
                for index in range(a0.subpath_index, a1.subpath_index):
                    to_return.append(self[index])
                to_return.append(self[a1.subpath_index].cropped(0, a1.tweaked(W=None, subpath_index=None)))
                if len(self) == 1 and self[0].Z:
                    assert len(to_return) == 2
                    to_return.smelt_subpaths(wraparound=True)
                    assert len(to_return) == 1

        elif a0.W > a1.W:
            if not self.isloop():
                raise ValueError("we could do this crop, but, per the tests,"
                                 "wraparound crops only allowed on loops!")
            to_return = self.cropped(a0, 1)
            to_return.extend(self.cropped(0, a1))
            if self.isloop():
                to_return.smelt_subpaths()

        else:
            raise ValueError("W0 == W1 in Path.cropped().")

        return to_return

    def heuristic_crop(self, other_curve, crop_to_inside=True):
        new_path = crop(self, other_curve, crop_to_inside=crop_to_inside)
        self.erase()
        self.extend(new_path)

    def isloop(self):
        return len(self) == 1 and self[0].Z

    def is_empty(self):
        return all(len(s) == 0 for s in self)

    def set_Z(self, forceful=False, following=None):
        if len(self) == 1:
            self[0].set_Z(forceful, following)
            return self
        raise ValueError("ambiguous set_Z command: more or less than one subpath")

    def unset_Z(self):
        if len(self) == 1:
            self[0].unset_Z()
            return self
        raise ValueError("ambiguous unset_Z command: more or less than one subpath")

    def converted_to_bezier(self, quality=0.01, safety=5,
                            reuse_segments=True):
        """See Subpath method of same name."""
        return Path(*[s.converted_to_bezier(
            quality=quality,
            safety=safety,
            reuse_segments=reuse_segments
        ) for s in self])

    def offset(self, amount, quality=0.01, safety=5, join='miter',
               miter_limit=4):
        return self.pro_offset(amount, quality, safety, join, miter_limit)[0]

    def pro_offset(self, amount, quality=0.01, safety=5, join='miter',
                   miter_limit=4, two_sided=False):
        skeletons = Path()
        way_outs = Path()
        way_ins  = Path()
        for s in self:
            wo, sk, wi = s.pro_offset(amount, quality, safety,
                                      join, miter_limit, two_sided)
            assert all(isinstance(x, Subpath) for x in [wo, sk, wi])
            way_outs.append(wo)
            skeletons.append(sk)
            way_ins.append(wi, even_if_empty=True)
        assert len(way_outs) == len(way_ins)
        return way_outs, skeletons, way_ins

    def stroke(self, width, quality=0.01, safety=5, join='miter',
               miter_limit=4, cap='butt', reversed=False):
        if not isinstance(width, Number):
            raise ValueError("stroke fed a non-number width")
        stroke = Path()
        for s in self:
            assert isinstance(s, Subpath)
            path = s.stroke(width, quality, safety, join, miter_limit, cap,
                            reversed)
            assert isinstance(path, Path)
            stroke.extend(path)
        return stroke
