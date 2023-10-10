Python library for working with SVG paths
=========================================

Streamlined fork of mathandy/svgpathtools that focuses 
on providing complete working SVG path manipulations
without support or concern for the enclosing document.

Contains support for stroking and offsetting paths,
paths with multiple subpaths, and the disambiguation of 
topological closure from geometric closure.

Organized by the following OOP hierarchy:

- ``Path`` objects correspond to general SVG paths, made up
one more connected ``Subpath`s
- ``Subpath`` objects correspond to continuous (connected) SVG
paths; each subpath can be topologically closed or not,
and subpaths that are geometrically closed are not necessarily
topologically closed (i.e., just because the subpath ends 
where it starts does not mean that ``Z`` property has been set)
- ``Segment`` is the parent class of the constituent segments
that make up subpaths; each segment is either an ``Arc`` or a 
`BezierCurve`
- ``Arc`` afore-mentioned instance of ``Segment``
- ``BezierCurve`` instance of ``Segment`, parent class of ``Line``,
`QuadraticBezier`` and ``CubicBezier`
- ``CubicBezier`` instance of ``BezierCurve`` (and ``Segment``)
- ``QuadraticBezier`` instance of ``BezierCurve`` (and ``Segment``)
- ``Line`` instance of ``BezierCurve`` (and ``Segment``)
- ``Curve`` base class from which all the above inherit

Thus: 

1. a ``Path`` is a container of ``Subpath`s
2. a ``Subpath`` is a container of ``Segment`s
3. each ``Segment`` is either an ``Arc`` or a ``BezierCurve`
4. the atomic ``Segment`` types are ``Arc``, ``Line`, ``QuadraticBezier`` and ``CubicBezier``, with the latter 3 inheriting from ``BezierCurve``
5. besides the inheritances described above, all objects also inherit from ``Curve``, where operations common to all objects such ``.length`` or ``.transform`` are implemented

Note ``Path`` and ``Subpaths`` present much the same API and
are often interchangeable objects.

Overview of Features
--------------------

Support for:

-  compute **tangent vectors** and (right-hand rule) **normal vectors**
-  compute **curvature**
-  compute **intersections** between paths and/or segments
-  find a **bounding box** for a path, subpath or segment
-  **reverse** segment/path orientation
-  **crop** and **split** paths and segments
-  **smooth** paths (i.e. smooth away kinks to make paths
   differentiable)
-  apply arbitrary **svg transformations** to a path; flatten
   documents
-  compute **area** enclosed by a closed path
-  compute **arc length**
-  compute **inverse arc length**
-  **transition maps** between path, subpath and segment coordinates
-  convert Bézier path segments to **numpy.poly1d** (polynomial) objects
-  convert polynomials (in standard form) to their Bézier form
-  some HTML color manipulation convenience functions (conversion from 
   rgb tuples to hexadecimal strings and back, named html color table)
-  crop path to other path
-  offset and stroke paths
-  path union
-  elliptical arc segment to bezier-based subpaths conversion
-  bounding box computations with and without stroke width

Positions on paths, subpaths or segments are specified
parameterically, but this easily leads to "parameter ambiguity" 
(i.e., does the given real number represent a segment, path or subpath
parameter?). To counter this, the module returns ``Address`` objects
instead of parameter values, where an ``Address`` object is a wrapper
containing both a parameter value and its semantics. Correspondingly,
the API also accepts ``Address`` objects instead of standalone parameters
as inputs, and will check that the right type of ``Address`` object is 
used in the right place.

Prerequisites
-------------

-  **numpy**

Licence
-------

This module is under a MIT License.

