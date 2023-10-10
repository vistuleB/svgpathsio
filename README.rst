Python library for working with SVG paths
=========================================

Streamlined fork of mathandy/svgpathtools. Focuses 
on providing complete working support SVG path manipulations
without document support. Input/output functionalities are
limited to the parsing and pretty-printing of paths.

Main classes:

- ``Path`` objects correspond to general SVG paths, made up of one more connected ``Subpath`` s
- ``Subpath`` objects correspond to continuous (connected) SVG paths; each subpath can be topologically closed or not, and subpaths that are geometrically closed are not necessarily topologically closed (i.e., just because the subpath ends  where it starts does not mean that ``Z`` property has been set)
- ``Segment`` is the parent class of the constituent segments that make up subpaths; each segment is either an ``Arc`` or a ``BezierCurve``
- ``Arc`` afore-mentioned instance of ``Segment``
- ``BezierCurve`` instance of ``Segment``, parent class of ``Line``, ``QuadraticBezier`` and ``CubicBezier``
- ``CubicBezier`` instance of ``BezierCurve`` (and ``Segment``)
- ``QuadraticBezier`` instance of ``BezierCurve`` (and ``Segment``)
- ``Line`` instance of ``BezierCurve`` (and ``Segment``)
- ``Curve`` base class from which all the above inherit

Thus: 

1. a ``Path`` is a container of ``Subpath`` s
2. a ``Subpath`` is a container of ``Segment`` s
3. each ``Segment`` is either an ``Arc`` or a ``BezierCurve``
4. the atomic ``Segment`` types are ``Arc``, ``Line``, ``QuadraticBezier`` and ``CubicBezier``, with the latter 3 inheriting from ``BezierCurve``
5. besides the inheritances described above, all objects also inherit from ``Curve``, where operations common to all objects such ``.length`` or ``.transform`` are implemented

Note ``Path`` and ``Subpaths`` present much the same API and
are often interchangeable objects.

Features
--------

-  **parse** and **pretty-print** paths
-  **parse** and **pretty-print** SVG transforms, navigating between three format types: transform ``string``, list of tokens (``["translate", 0, 10, "scale", 2.3]``), and numpy ``nd.array``
-  **apply transforms** to paths
-  compute **tangent vectors** and (right-hand rule) **normal vectors**
-  compute **curvature**
-  compute **intersections** between paths and/or segments
-  compute **bounding boxes** of paths, subpaths and segments
-  **reverse** segment/subpath/path orientation
-  **crop** and **split** paths and segments
-  **offset** and **stroke** paths
-  take path **unions**
-  compute **enclosed area**
-  compute **arc length**
-  compute **inverse arc length**
-  convert between path, subpath and segment parameterizations (see notes on ``Address`` object below)
-  convert Arc segments to CubicBezier segments
-  convert Bézier path segments to **numpy.poly1d** (polynomial) objects
-  convert polynomials (in standard form) to their Bézier form
-  some HTML color manipulation convenience functions (conversion from 
   rgb tuples to hexadecimal strings and back, named html color table)

The ``Address`` object
----------------------

While positions on paths, subpaths and segments are specified
parameterically, this easily leads to "parameter ambiguity".
(I.e., does the given real number represent a segment, path or subpath
parameter?) To counter the tendency for confusion, the module wraps parameter
values inside of a larger ``Address`` object that disambiguates the
type of the parameter. ``Address`` objects can be substituted at all 
points in the API for parameter values, and the module will check that the
right type of ``Address`` object is used in the right place.

Prerequisites
-------------

-  **numpy**

Licence
-------

This module is under a MIT License.

