svgpathtools offset and stroke fork
===================================

This is a fork of mathandy/svgpathtools, 
a collection of tools for manipulating and analyzing SVG
Path objects and Bézier curves.

This is an advanced fork that adds support for subpaths, stroke, 
offsets, addresses, path
cropping, as well as path union and intersections. A number of bugs
and incompletenesses have been removed. Some legacy elements of 
the API that maintained backward-compatibility with older versions of the
library have also been removed.

The SaxDocument class included in this fork also has the following 
additional features:

- supports a `<style>` element and CSS manipulation
- supports `<defs>` and `<use>`
- supports `<text>` and `<textPath>`

Note: Much of the following README is taken from the original
mathandy/svgpathtools README.

Features
--------

svgpathtools contains functions designed to **easily read, write and
display SVG files** as well as a selection of
geometrically-oriented tools to **transform and analyze path
elements**.

Additionally, the submodule *bezier.py* contains tools for working
with general **nth order Bezier curves stored as n-tuples**.

Some of the original capabilities of mathandy/svgpathtools:

-  **read**, **write**, and **display** SVG files containing Path (and
   other) SVG elements
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
-  convenience functions, such the generation of hexadecimal color
   codes from RGB color tuples and back
   
And, in this fork:

-  (properly) work with paths made up of **multiple subpaths**
-  generate **path offsets**
-  generate **path strokes** using any combination of the standard
   SVG options for 'cap' and 'join'
-  **convert elliptical arc segments** to bezier-based subpaths, to
   desired accuracy
-  enjoy the support of **Address objects**
-  style your paths, text and dots elements with **PathAndAttributes**,
   **TextAndAttributes** and **DotAndAttributes** classes
-  use the **SaxDocument** class to easily keep track of definitions,
   styles, as well as of the afore-mentioned dot, paths and text elements
-  compute the **bounding box** of a group of PathAndAttribute objects 
   while taking stroke widths into consideration
-  automatically compute the **viewbox** of a document from its
   path contents (warning: as of this writing, text elements are not taken into account)
-  use **classes and css** to efficiently style your SaxDocument elements
-  use **<defs> and <use>** elements in SaxDocument
-  compute **path unions**
-  **crop** paths to the inside/outside of a given window, or to 
   the inside/outside of an arbitrary path

Prerequisites
-------------

-  **numpy**

Installation
------------

Start by installing the original mathandy/svgpathtools. Then
replace the whole contents of the source directory (located e.g. at
`/opt/local/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/svgpathtools/`
on my machine) with the contents of this repo's `svgpathtools` folder ;)

Main Classes
------------

The svgpathtools module is primarily structured around four path segment
classes: ``Line``, ``QuadraticBezier``, ``CubicBezier``, and ``Arc``.
Instances of these classes are called *segments*. These four classes
inherit from an abstract ``Segment`` superclass.

The module also contains two container classes, ``Path`` and ``Subpath``.
A ``Subpath`` object consists of a list of end-to-end segments, possibly
forming a closed loop. Lastly, a ``Path`` consists of an arbitrary list of subpaths.

For example, an SVG path such as 

``M 0,0 L 1,0 1,1 0,1 Z M 2,0 L 3,0 3,1 2,1 Z``  (1)

would end up modeled as a ``Path`` containing two ``Subpath`` s each being a
sequence of four ``Line`` objects. (Each 'Z' command results in an extra line
segment being added to close off that subpath.) 

SVG distinguishes between subpaths that are merely geometrically closed and
those that are closed via a ``Z`` command. For example, (1) renders as

.. figure:: https://user-images.githubusercontent.com/19382247/54197407-ca6c7c00-44fe-11e9-9d59-c4f1d7834897.png

whereas

``M 0,0 L 1,0 1,1 0,1 0,0 M 2,0 L 3,0 3,1 2,1 2,0``  (2)

(with the four missing line segments, but without Z's) renders as

.. figure:: https://user-images.githubusercontent.com/19382247/54197555-351db780-44ff-11e9-92a9-913ee2828399.png

with indented corners, because geometric closure does not equate to
topological closure. Each ``Subpath`` object remembers whether it is topologically
closed or not via an internal ``._Z`` boolean property, which can be set
via the methods ``.set_Z()`` and ``.unset_Z()``, described in more detail below.

Constructors
------------

The constructors for the above-mentioned classes are invoked as follows:

-  ``Line(start, end)``

-  ``QuadraticBezier(start, control, end)``

-  ``CubicBezier(start, control1, control2, end)``

-  ``Arc(start, radius, rotation, large_arc, sweep, end)``  (note:
   large_arc and sweep are boolean)

-  ``Subpath(*segments-or-subpaths-or-paths)``

-  ``Path(*segments-or-subpaths-or-paths)``

Here values ``start``, ``end``, ``control``, etc, denote points encoded as python complex
numbers. For example, the Cartesian point (100, 200) is encoded as the
complex value ``100+200j``.

For the ``Arc`` constructor, ``radius`` encodes the radii ``rx``, ``ry`` of the
ellipse in the form of a complex number ``rx + 1j * ry``, while other arguments have their
usual meaning. (Consult the SVG spec or the ``Arc`` docstring for more details.)

.. code:: ipython2

    from svgpathtools import Path, Subpath, Line, QuadraticBezier, CubicBezier, Arc
    
    seg1 = CubicBezier(300+100j, 100+100j, 200+200j, 200+300j)  # A cubic beginning at (300, 100) and ending at (200, 300)
    seg2 = Line(200+300j, 250+350j)                             # A line beginning at (200, 300) and ending at (250, 350)
    seg3 = QuadraticBezier(0, 100, 100+100j)                    # A quadratic beginning at (0, 0) and ending at (100, 100)
    
    seg1.end  # 200+300j
    seg2.start  # 200+300j
    
    subpath1 = Subpath(seg1, seg2)  # A subpath consisting of seg1 followed by seg2
    
    try:
        subpath2 = Subpath(seg1, seg3)  # Throws an exception because seg1.end != seg3.start, and because subpaths consist of a list of contiguous segments
        assert False
    except ValueError:
        pass
    
    subpath1.Z  # False; subpath1 is not geometrically closed, let alone topologically closed

    try:
        subpath1.set_Z()  # Throws because subpath1 is not geometrically closed
    except ValueError:
        subpath1.set_Z(forceful=True)  # Adds a line segment to subpath1, closes it topologically
        print("\nsubpath1 after forceful closure:")
        print(subpath1)

    subpath1.Z  # True, because we called .set_Z(forceful=True)
    subpath1.unset_Z()  # Now subpath1 is topologically open, but the added line segment remains
    subpath1.Z  # False
    
    print("\nthe open version of subpath1 (still with 3 segments!):")
    print(subpath1)
    
    subpath1.set_Z()  # Because subpath1 is geometrically closed, we don't need `forceful=True` to close it anymore
    subpath1.Z  # True
    
    path1 = Path(subpath1)  # path1 consists of a single subpath
    len(path1)  # 1, because path1 has a single subpath
    len(path1[0])  # 3, because subpath1 has 3 segments
    
    path2 = Path(seg1, seg2)  # The path constructor can accept segments, too
    len(path2)  # 1, because seg1, seg2 are contiguous, they automatically got bundled into the same subpath
    
    print("\nHere's what path2 looks like:")
    print(path2)
    
    path3 = Path(seg1, seg3)
    len(path3)  # 2, because seg1.end != seg3.start, seg1 and seg3 got placed in different subpaths
    path3[0]  # A Subpath object containing only seg1
    path3[1]  # A Subpath object containing only seg2
    assert path3[0] == Subpath(seg1)
    assert path3[1] == Subpath(seg3)
    
    print("\nHere's what path3 looks like:")
    print(path3)
    
    # Construct a path consisting of one closed subpath directly:
    path4 = \
        Path(
            Subpath(
                Line(0, 100),
                Line(100, 100+100j),
                Line(100+100j, 100j),
                Line(100j, 0)
            ).set_Z()  # .set_Z() returns the Subpath object on which it is called
        )
        
    # Another option, using the points2lines Line generator:
    from svgpathtools import points2lines
    path5 = Path(Subpath(*points2lines(0, 100, 100+100j, 100j, 0)).set_Z())
    assert path5 == path4
    
    # Yet another option, providing one less point and using forceful=True :)
    path6 = Path(Subpath(*points2lines(0, 100, 100+100j, 100j)).set_Z(forceful=True))
    assert path6 == path4
    
    # Or, using the Subpath.path_of() function to wrap a Subpath into a Path:
    path7 = Subpath(*points2lines(0, 100, 100+100j, 100j, 0)).set_Z().path_of()
    assert path7 == path4
    
    # Last but not least, creating paths directly from d-strings:
    from svgpathtools import parse_path
    path8 = parse_path('M 0,0 1,0 1,1 0,1 Z m 2,0 1,0 0,1 -1,0 Z')  # (note the second subpath uses relative moveto and lineto commands, because 'm' not 'M')
    
    print("\nLet's take a look at path8 (formatting with 'use_fixed_indent', 'segment.use_oneline'):")
    print(path8.__repr__('use_fixed_indent segment.use_oneline'))  # The 'use_fixed_indent' option indents each new subpath and segment at 4 spaces, 'segment.use_oneline' prevents segment arguments from being similarly indented, keeping them on one line
    
    # How we could construct this directly:
    path9 = \
        Path(
            Subpath(*points2lines(0, 1, 1+1j, 1j, 0)).set_Z(),
            Subpath(*points2lines(2, 3, 3+1j, 2+1j, 2)).set_Z()
        )
    assert path9 == path8
    
    # Or, with a little more code reuse:
    square = Subpath(*points2lines(0, 1, 1+1j, 1j, 0)).set_Z()
    path10 = Path(square, square.translated(2+0j))  # The 'translated' method returns a translated copy of the path, subpath or segment
    assert path10 == path8

    # Another printing option that can be useful is 'constructor_ready', which prints .set_Z()'s instead of .Z's:
    print("\nThe 'constructor_ready' option produces output that is valid python code:")
    print(path9.__repr__('use_fixed_indent segment.use_oneline constructor_ready'))  # 'constructor_ready' has the effect of... see output below!

>>

.. parsed-literal::

    subpath1 after forceful closure:
    Subpath(CubicBezier(300+100j,
                        100+100j,
                        200+200j,
                        200+300j),
            Line(200+300j,
                 250+350j),
            Line(250+350j,
                 300+100j)).Z

    the open version of subpath1 (still with 3 segments!):
    Subpath(CubicBezier(300+100j,
                        100+100j,
                        200+200j,
                        200+300j),
            Line(200+300j,
                 250+350j),
            Line(250+350j,
                 300+100j))

    Here's what path2 looks like:
    Path(Subpath(CubicBezier(300+100j,
                             100+100j,
                             200+200j,
                             200+300j),
                 Line(200+300j,
                      250+350j)))

    Here's what path3 looks like:
    Path(Subpath(CubicBezier(300+100j,
                             100+100j,
                             200+200j,
                             200+300j)),
         Subpath(QuadraticBezier(0,
                                 100,
                                 100+100j)))

    Let's take a look at path8 (formatting with 'use_fixed_indent', 'segment.use_oneline'):
    Path(
        Subpath(
            Line(0j, 1+0j),
            Line(1+0j, 1+1j),
            Line(1+1j, 1j),
            Line(1j, 0j)
        ).Z,
        Subpath(
            Line(2+0j, 3+0j),
            Line(3+0j, 3+1j),
            Line(3+1j, 2+1j),
            Line(2+1j, 2+0j)
        ).Z
    )

    The 'constructor_ready' option produces output that is valid python code:
    Path(
        Subpath(
            Line(0, 1),
            Line(1, 1+1j),
            Line(1+1j, 1j),
            Line(1j, 0)
        ).set_Z(),
        Subpath(
            Line(2, 3),
            Line(3, 3+1j),
            Line(3+1j, 2+1j),
            Line(2+1j, 2)
        ).set_Z()
    )

Appending, Insertions, Deletions, Etc.
--------------------------------------

The ``Path`` behaves much like a
list: its supbaths can be **append**\ ed, **insert**\ ed, set by index,
**del**\ eted, **enumerate**\ d, **slice**\ d out, **pop**\ ped, etc. For example,

.. code:: ipython2

    for subpath in path[1::2]:
        # do stuff
        
traverses the subpaths in Path "path" starting from the second subpath
and skipping every other subpath.

Note that ``Path.append(...)``, ``Path.insert(index, ...)`` and ``Path[i] = ...`` all
require Subpath-type arguments. On the other hand, the function ``Path.extend(...)`` accepts an
arbitrary sequence of segments and subpaths as arguments. (In fact, it even accepts paths,
which it simply swallows subpath-by-subpath.)
If the sequence contains
standalone segments, adjacent segments in the sequence that are geometrically
contiguous are placed into the
same subpath. The ``.extend`` method has default signature

.. code:: ipython2

    Path.extend(*args, even_if_empty=False, extend_by_segments=True, clone_affected_subpaths=True)

where the ``even_if_empty`` option controls whether empty subpaths are added or not, 
and where the ``extend_by_segments`` option controls whether 
the first segment in a sequence of standalone segment is glued on to the path's last
subpath, if that subpath ends where the segment starts and is not topologically closed,
instead of automatically initiating a new subpath. If ``extend_by_segments`` is true,
some existing subpaths may be extended by newly arriving segments–whether such affected
subpaths are cloned afresh to avoid unexpected side effects is controlled by ``clone_affected_subpaths``.
(In fact, the Path constructor itself uses a call to ``.extend`` to process its input
list, with the difference that the constructor sets ``extend_by_segments=False`` by default.
The ``extend_by_segments`` and ``clone_affected_subpaths`` options can 
be passed to the Path constructor as well, e.g., 
``Path(seg1, subpath1, seg2, seg3, extend_by_segments=True)``.)

The ``Subpath`` class has all similar methods and iterators as ``Path``, but throws a 
ValueError if an attempt is made to modify the subpath in a way that would break continuity.

Similarly to ``Path.extend(...)``, ``Subpath.extend(...)`` accepts an arbitrary mix of 
Segment, Subpath and
Path objects as arguments, which are treated as a single long list of segments,
generated in order of the arguments. ``Subpath.extend()`` will only check that
the segments in the proposed list are contiguous, and that appending them will not
break closure, if present. (Specifically, if the subpath is topologically closed,
``Subpath.extend()`` checks that the new endpoint of the subpath would still equal
its old startpoint, before accepting the extension.) Like for Path, a similar mixture can actually be passed to the Subpath constructor as
well. 

(Note that ``Subpath.extend(...)`` and the Subpath constructor are not shy to swallow
topologically closed subpaths, and will indeed entirely ignore the topological closure
of subpaths encountered.)

Similarly to paths, one can iterate over a subpath, which yields a sequence of
segments.

.. code:: ipython2

    from svgpathtools import Path, Subpath, Segment, points2lines
    
    # Construct a building block:
    tooth = Subpath(*points2lines(0, 1+1j, 2))  # a 2-line subpath
    
    # Replicate inside another subpath:
    subpath1 = Subpath(
        tooth,
        tooth.translated(2),
        tooth.translated(4)
    )
    assert len(subpath1) == 6
    assert all(isinstance(thing, Segment) for thing in subpath1)  # An example of iterating over a subpath
    
    # We can also derefence an array, for the same effect:
    subpath2 = Subpath(*[tooth.translated(2*i) for i in range(3)])
    assert subpath1 == subpath2
    
    # Let's mutilate subpath2
    subpath2.pop(0)  # removes first segment of subpath2
    subpath2.pop()  # removes last segment of subpath2
    assert subpath2 == Subpath(tooth, tooth.translated(2)).scaled(1, -1).translated(1+1j)
    
    # Starting from subpath1 again, let's build a square
    subpath3 = subpath1.rotated(-90, origin=0).translated(6)

    subpath1.extend(subpath3)  # We must use 'extend' because the argument is a Subpath, not a Segment
    
    assert len(subpath1) == 12
    
    subpath4 = subpath1.rotated(180, origin=0).translated(6-6j)
    
    assert len(subpath4) == 12
    
    subpath1.extend(subpath4)
    
    assert len(subpath1) == 24
    
    # If we haven't screwed up, our toothy square should be geometrically closed; we can make that topological:
    subpath1.set_Z()
    
    # Print out numbered segments in our square
    for index, seg in enumerate(subpath1):
        print("segment number", index, "is", seg.__repr__('use_oneline'))
        
    print("")
    # Print out every other segment, starting from last and going backwards (look, mom, no hands!):
    for index, seg in enumerate(subpath1[-1::-2]):
        true_index_in_subpath = len(subpath1) - 1 - 2 * index
        print("segment number", true_index_in_subpath, "is", seg)
    
>>

.. parsed-literal::

    segment number 0 is Line(0, 1+1j)
    segment number 1 is Line(1+1j, 2)
    segment number 2 is Line(2, 3+1j)
    segment number 3 is Line(3+1j, 4)
    segment number 4 is Line(4, 5+1j)
    segment number 5 is Line(5+1j, 6)
    segment number 6 is Line(6+0j, 7-1j)
    segment number 7 is Line(7-1j, 6-2j)
    segment number 8 is Line(6-2j, 7-3j)
    segment number 9 is Line(7-3j, 6-4j)
    segment number 10 is Line(6-4j, 7-5j)
    segment number 11 is Line(7-5j, 6-6j)
    segment number 12 is Line(6-6j, 5-7j)
    segment number 13 is Line(5-7j, 4-6j)
    segment number 14 is Line(4-6j, 3-7j)
    segment number 15 is Line(3-7j, 2-6j)
    segment number 16 is Line(2-6j, 1-7j)
    segment number 17 is Line(1-7j, -6j)
    segment number 18 is Line(-6j, -1-5j)
    segment number 19 is Line(-1-5j, -4j)
    segment number 20 is Line(-4j, -1-3j)
    segment number 21 is Line(-1-3j, -2j)
    segment number 22 is Line(-2j, -1-1j)
    segment number 23 is Line(-1-1j, 0j)

    segment number 23 is Line(-1-1j, 0j)
    segment number 21 is Line(-1-3j, -2j)
    segment number 19 is Line(-1-5j, -4j)
    segment number 17 is Line(1-7j, -6j)
    segment number 15 is Line(3-7j, 2-6j)
    segment number 13 is Line(5-7j, 4-6j)
    segment number 11 is Line(7-5j, 6-6j)
    segment number 9 is Line(7-3j, 6-4j)
    segment number 7 is Line(7-1j, 6-2j)
    segment number 5 is Line(5+1j, 6)
    segment number 3 is Line(3+1j, 4)
    segment number 1 is Line(1+1j, 2)

Some examples involving the Path object constructor:

.. code:: ipython2

    from svgpathtools import Path, parse_subpath

    very_simple = parse_subpath('M 0,0 1,0 2,0')  # a subpath consisting of two collinear line segments

    version1 = Path(very_simple, very_simple.translated(2))  # consists of two subpaths of length 2 (the subpaths are end-to-end)
    assert len(version1) == 2 and all(len(x) == 2 for x in version1)
    version2 = Path(very_simple, *very_simple.translated(2))  # the second occurrence of very_simple is atomized into segments before being passed into the constructor, but the constructor will automatically reassemble these segments into a single subpath; ends up the same as version1
    assert version2 == version
    version3 = Path(very_simple, *very_simple.translated(2), extend_by_segments=True)  # this time the atomized segments will glom onto the first subpath, because they are contiguous with it and the 'extend_by_segments' option is set; one ends up with a path containing a single subpath of length 4; the original 'very_simple' subpath is not affected because the constructor clones affected subpaths by default
    assert len(version3) == 1 and len(version3[0]) == 4
    version4 = Path(*very_simple, *very_simple.translated(2))  # boths subpaths are atomized into segments before being passed into the constructor; same result as version3
    assert version4 == version3
    
    print("\nversion1 & version2:")
    print(version1)

    print("\nversion3 & version4:")
    print(version3)
    
>>

.. parsed-literal::
    
    version1 & version2:
    Path(Subpath(Line(0j, 1+0j),
                 Line(1+0j, 2+0j)),
         Subpath(Line(2+0j, 3+0j),
                 Line(3+0j, 4+0j)))

    version3 & version4:
    Path(Subpath(Line(0j, 1+0j),
                 Line(1+0j, 2+0j),
                 Line(2+0j, 3+0j),
                 Line(3+0j, 4+0j)))

Some examples involving deletion/insertion of subpaths:

.. code:: ipython2

    from svgpathtools import Path, parse_subpath
    
    closed_triangle = parse_subpath('M 0,0 1,1 0,2 Z')  # returns a Supath instance
    line = parse_subpath('M 0,0 2,0')  # returns a Subpath instance
    
    path = Path(
        closed_triangle.translated(2+2j),
        line,
        line.translated(3j)
    )
    
    del path[1]  # the 'line' subpath is gone!
    assert len(path) == 2
    assert path == Path(closed_triangle.translated(2+2j), line.translated(3j))
    
    path.insert(0, closed_triangle)  # (we could also have said 'path.prepend(closed_triangle)')
    assert len(path) == 3
    assert path == Path(closed_triangle, closed_triangle.translated(2+2j), line.translated(3j))
    
    path[0].unset_Z().pop()  # opening the triangle and removing its third side
    
    # since path[0] held an original reference to closed_triangle, closed_triangle is now
    altered
    
    print("\nso-called closed_triangle is no longer so closed:")
    print(closed_triangle)
    
    print("\npath:")
    print(path)
    
>>

.. parsed-literal::

    so-called closed_triangle is no longer so closed:
    Subpath(Line(0j, 1+1j), Line(1+1j, 2j))

    path:
    Path(Subpath(Line(0j, 1+1j),
                 Line(1+1j, 2j)),
         Subpath(Line(2+2j, 3+3j),
                 Line(3+3j, 2+4j),
                 Line(2+4j, 2+2j)).Z,
         Subpath(Line(3j, 2+3j)))
         
Editing Segments
----------------

Segments are immutable, in order to protect Subpath objects from losing
their continuity/closure, etc.

However, use ``.tweaked`` to obtain a cloned copy of a segment with
à la carte fields edited. For example

.. code:: ipython2

    my_cubic_bezier2 = my_cubic_bezier1.tweaked(end=101-2.2j, control1=0+5j)
    
will assign to ``my_cubic_bezier2`` an altered copy of ``my_cubic_bezier1``
in which ``end`` and ``control1`` have new values.

Or: Edit the underscore fields directly, at your own risk. E.g., ``my_cubic_bezier1._end = 101-2.2j``.

Writing and Displaying SVGs
---------------------------

The ``SaxDocument`` supports SVG parsing, simple styling and output. A SaxDocument consists of four fields: 

- ``doc.root_attrs`` is a dictionary that holds attributes for the SVG root element, such as viewBox, width and height

- ``doc.elements`` is a list **PathAndAttributes**, **DotAndAttributes** and **TextAndAttributes** objects, explained below

- ``doc.styles`` a dictionary of in-document class styles, if any; the key-value pairs of this dictionary will become the content of the SVG's ``<style>`` element

- ``doc.defs`` a list with the same format as ``doc.elements``, whose elements become the content of the SVG's ``<defs>`` element

Note that PathAndAttributes objects, as well as DotAndAttributes and TextAndAttributes object, observe a dual syntax whereby their fields can be accessed either via .-notation or via [' ']-notation. E.g., the following are all equivalent: 

.. code:: ipython2

    path_aa = PathAndAttributes(d='M 1,1 2,2')
    path_aa.fill = 'red' 

.. code:: ipython2

    path_aa = PathAndAttributes(d='M 1+1j 2+2j')
    path_aa['fill'] = 'red'

.. code:: ipython2

    path_aa = PathAndAttributes(fill='red')
    path_aa.d = 'M 1+1j 2+2j'

.. code:: ipython2

    path_aa = PathAndAttributes()
    path_aa.fill = '#f00'
    path_aa.d = 'M 1+1j 2+2j'

.. code:: ipython2

    path_aa = PathAndAttributes(d='M 1+1j 2+2j', fill='#f00')

.. code:: ipython2

    path_aa = PathAndAttributes()
    path_aa.update({'d': 'M 1+1j 2+2j', 'fill': 'red'})

Some attribute names have workaround aliases due to limitations of the python syntax: "classname" is mapped to "class", and "width" is mapped to "stroke-width". E.g., the first three lines of the following code snippet all (re-)set the "class" attribute of ``path_aa``:

.. code:: ipython2

    path_aa.classname = 'bigshape'
    path_aa['class'] = 'littleshape'
    path_aa['classname'] = 'greenshape'
    
    path_aa['stroke-width'] = 4.2
    path_aa.width = 5.2
    
    print(path_aa.classname)
    print(path_aa['class'])
    print(path_aa.width)
    print(path_aa['stroke-width'])
    
>>

.. parsed-literal::

    greenshape
    greenshape
    5.2
    5.2


For convience, the DotAndAttributes class implements three more aliases: ``x``, ``y`` and ``radius`` map to ``cx``, ``cy`` and ``r`` respectively.

The SaxDocument class observes a similar dual syntax, but only for three standard attributes ``width``, ``height`` and ``viewBox``. Moreover ``viewbox`` serves as an alias for ``viewBox``.

Finally, note that PathAndAttributes objects have both a ``.d`` attribute, which returns the d-string for the path in question, and a ``.object`` attribute, which returns the Path object associated to the same d-string. These fields are automatically synchronized. One can read from ``.object`` when only ``.d`` has been initialized, and vice-versa. When writing to a PathAndAttributes object one can also use the ``path`` key as an alias for either ``d`` or ``object``: which it is will be resolved depending on the type of data provided.

Here is a simple example of creating and populating a SaxDocument from scratch:

.. code:: ipython2

    from svgpathtools import *

    doc = SaxDocument()

    p1 = Path(*points2lines(0, 100, 100j))

    doc.elements.extend([
        PathAndAttributes(path=p1, width=2, fill='AliceBlue', stroke='none'),
        PathAndAttributes(path=Path(p1, p1.translated(200)).translated(200j), classname='very_proper'),  # Here 'classname' is mapped to 'class'. Note that directly writing 'class' would yield a python syntax error
        PathAndAttributes(path='M 20,20 C 100+300j 200+10j 300+200j', width=2, stroke='#000', fill='none')  # This is not a valid d-string because of the complex-number notation, but svgpathtools can parse it none the less!
    ])

    print(doc.elements[0]['stroke-width'])
    print(doc.elements[1]['class'])

    doc.styles['.very_proper'] = 'fill:#a0f'  # (don't forget that period in the class name!!!! just like in css!!!)

    doc.set_background_color(random_color())
    doc.reset_viewbox()
    doc.root_attrs['width'] = 400
    doc.set_height_from_width()  # uses the pre-existing width and the viewbox to find the height
    doc.display()  # Other possibility: doc.save('my_filename.svg')
    
>>

.. parsed-literal::

    2
    very_proper
    
.. figure:: https://user-images.githubusercontent.com/19382247/54968261-f551d800-4fb4-11e9-94ee-dff162ddfc3d.png
    
The call

.. code:: ipython2

    doc.reset_viewbox()
    
recomputes the viewbox automatically from the paths present in ``doc.elements``. One can also assign a viewbox directly via one of these assignment syntaxes:

.. code:: ipython2

    doc.root_attrs['viewBox'] = '0 0 100 100'

.. code:: ipython2

    doc.viewbox = '0 0 100 100'

.. code:: ipython2

    doc.viewBox = '0 0 100 100'
    
Likewise, one might set the width of the document via either of

.. code:: ipython2

    doc.root_attrs['width'] = 400

.. code:: ipython2

    doc.width = 400

and the same for ``height``. The SVG ``width`` and ``height`` fields can also take units, e.g., ``doc.width = '400mm'``.

Note that

.. code:: ipython2

    doc.set_background_color(...)
    
can be useful for visualizing the dimensions of the SVG, as an SVG's boundaries might not otherwise be visible. This feature is implemented by adding an additional ``<rect>`` element to the top of the SVG.

The ``SaxDocument`` can also parse SVGs. Simply use the ``SaxDocument.sax_parse()`` function with the desired file name. Note this will reset the SaxDocument object as per the contents of the file, and can effectively be thought of as a constructor call.

For example, here is a makeshift SVG with some internal css styles and some external (missing) css styles:


.. parsed-literal::

    <svg version="1.1" viewBox="0 0 300 300" width="600" height="600" xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink">
        <style>
            .zoomA {
                stroke: red;
                stroke-width: 4;
            }
        </style>
        <g transform="rotate(-30)">
            <g transform="translate(100, 100)">
                <!-- <rect class="liliputh" x="0" y="0" width="50" height="50"/> -->
                <path class="liliputh" d='M0,0 H 50 V 50 H 0' />
            </g>
        </g>
        <circle class="zoomA" cx="140" cy="100" r="15"/>
        <path class="zoomi" d="M60,80 -60,-80 60,-80 -60,80 Z" transform="translate(150, 150)"/>
        <path class="antigusto" d="m0,0 1,0 0,1 z m 1.6,0 1,0 0,1 z m 1.6,0 1,0 0,1z" transform="translate(300, 50) rotate(70) scale(25)"/>
    </svg>
    
One could process this document as follows, assuming it has been saved to "test2.svg":

.. code:: ipython2

    from svgpathtools import *

    doc = SaxDocument()
    doc.sax_parse('test2.svg')

    for el in doc:  # (equivalent to "for p in doc.elements:")
        print(el.__class__.__name__, el)

    doc.set_background_color(random_color())
    doc.display()  # and/or: doc.save('filename.svg')
    
>>

.. parsed-literal::

    PathAndAttributes {'transform': 'rotate(-30) translate(100, 100)', 'class': 'liliputh', 'd': 'M0,0 H 50 V 50 H 0 V 0', 'original_tag': 'path'}
    PathAndAttributes {'class': 'zoomA', 'cx': 140.0, 'cy': 100.0, 'r': 15.0, 'd': 'M125.0,100.0a15.0,15.0 0 1,0 30.0,0a15.0,15.0 0 1,0 -30.0,0Z', 'original_tag': 'circle'}
    PathAndAttributes {'class': 'zoomi', 'd': 'M60,80 -60,-80 60,-80 -60,80 Z', 'transform': 'translate(150, 150)', 'original_tag': 'path'}
    PathAndAttributes {'class': 'antigusto', 'd': 'm0,0 1,0 0,1 z m 1.6,0 1,0 0,1 z m 1.6,0 1,0 0,1z', 'transform': 'translate(300, 50) rotate(70) scale(25)', 'original_tag': 'path'}

The displayed figure (not to size):

.. figure:: https://user-images.githubusercontent.com/19382247/54864968-877d9480-4d99-11e9-8a48-8613d921900e.png

One of the issues displaying the above SVG is that external styles are missing. Here is a quick plug, assigning randomized styles to paths with missing styles. The key call is ``doc.collect_classnames(prepend_dot=True)``:

.. code:: ipython2

    from svgpathtools import *

    doc = SaxDocument()
    doc.sax_parse('test2.svg')

    for dot_name in doc.collect_classnames(prepend_dot=True):  # yields '.liliputh', '.zoomA', '.zoomi', '.antigusto'
        if dot_name not in doc.styles:  # throws out '.zoomA' which is already in doc.styles
            doc.styles[dot_name] = f"fill:{random_color()};stroke:black;stroke-width:4;opacity:0.5"

    doc.set_background_color(random_color())
    doc.display()
    
This gives us the already-more-legible figure:
    
.. figure:: https://user-images.githubusercontent.com/19382247/54864923-e5f64300-4d98-11e9-8455-17d2708a754d.png

In this figure, the rightmost shape is overwhelmed by its stroke: what is happening is that the stroke is being magnified 25 times due to that path's ``transform`` attribute. To palliate this situation we can incorporate the transform into the path, so that the stroke occurs after the transform, not before. The ``.flatten()`` method of PathAndAttributes instances achieves this:

.. code:: ipython2

    from svgpathtools import *

    doc = SaxDocument()
    doc.sax_parse('test2.svg')

    for dot_name in doc.collect_classnames(prepend_dot=True):
        if dot_name not in doc.styles:
            doc.styles[dot_name] = f"fill:{random_color()};stroke:black;stroke-width:4;opacity:0.5"

    for el in doc:  # (nb: all elements are PathAndAttributes instances, in this document)
        el.flatten()

    doc.set_background_color(random_color())
    doc.display()
    
This time we get:
    
.. figure:: https://user-images.githubusercontent.com/19382247/54865079-d841bd00-4d9a-11e9-90ad-4a15fb867598.png

The offending shape is protruding outside the viewport. In the next iteration, we readjust the viewport to exactly accommodate the paths that are present via a call to ``doc.reset_viewbox()``:

.. code:: ipython2

    from svgpathtools import *

    doc = SaxDocument()
    doc.sax_parse('test2.svg')

    for dot_name in doc.collect_classnames(prepend_dot=True):
        if dot_name not in doc.styles:
            doc.styles[dot_name] = f"fill:{random_color()};stroke:black;stroke-width:4;opacity:0.5"

    for el in doc:
        el.flatten()

    doc.reset_viewbox()  # (<- new!)
    doc.set_background_color(random_color())
    doc.display()

Yielding:

.. figure:: https://user-images.githubusercontent.com/19382247/54865142-f65bed00-4d9b-11e9-8b43-777b08c473a7.png

Some strokes are protruding from the viewbox. (The reason why these offending strokes are displayed at all beyond the viewbox is unknown to the author of this README, but is replicated across three different SVG viewers. Also note this occurs only top and bottom, but not on the left- and right-hand sides of the SVG.) One can pass the ``with_strokes`` option to ``.reset_viewbox()`` to have the viewbox exactly accommodate the strokes, including widths found in the in-document styles:

.. code:: ipython2

    from svgpathtools import *

    doc = SaxDocument()
    doc.sax_parse('test2.svg')

    for dot_name in doc.collect_classnames(prepend_dot=True):
        if dot_name not in doc.styles:
            doc.styles[dot_name] = f"fill:{random_color()};stroke:black;stroke-width:4;opacity:0.5"

    for path in doc:
        path.flatten()

    doc.reset_viewbox(with_strokes=True)  # (<- new!)
    doc.set_background_color(random_color())
    doc.display()
    
>>
    
.. figure:: https://user-images.githubusercontent.com/19382247/54865337-117c2c00-4d9f-11e9-84fc-e1a11138bd47.png

!!!!!! END OF NEW README, START OF OLD README !!!! STILL HAVE TO ADD DESCRIPTION OF .point(), Address(), .intersect(), .offset(), .stroke()
===========================================================================

Reading SVGSs
-------------

| The **svg2paths()** function converts an svgfile to a list of Path
  objects and a separate list of dictionaries containing the attributes
  of each said path.
| Note: Line, Polyline, Polygon, and Path SVG elements can all be
  converted to Path objects using this function.

.. code:: ipython2

    # Read SVG into a list of path objects and list of dictionaries of attributes 
    from svgpathtools import svg2paths, wsvg
    paths, attributes = svg2paths('test.svg')
    
    # Update: You can now also extract the svg-attributes by setting
    # return_svg_attributes=True, or with the convenience function svg2paths2
    from svgpathtools import svg2paths2
    paths, attributes, svg_attributes = svg2paths2('test.svg')
    
    # Let's print out the first path object and the color it was in the SVG
    # We'll see it is composed of two CubicBezier objects and, in the SVG file it 
    # came from, it was red
    redpath = paths[0]
    redpath_attribs = attributes[0]
    print(redpath)
    print(redpath_attribs['stroke'])


.. parsed-literal::

    Path(CubicBezier(start=(10.5+80j), control1=(40+10j), control2=(65+10j), end=(95+80j)),
         CubicBezier(start=(95+80j), control1=(125+150j), control2=(150+150j), end=(180+80j)))
    red


Writing SVGSs (and some geometric functions and methods)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **wsvg()** function creates an SVG file from a list of path. This
function can do many things (see docstring in *paths2svg.py* for more
information) and is meant to be quick and easy to use. Note: Use the
convenience function **disvg()** (or set 'openinbrowser=True') to
automatically attempt to open the created svg file in your default SVG
viewer.

.. code:: ipython2

    # Let's make a new SVG that's identical to the first
    wsvg(paths, attributes=attributes, svg_attributes=svg_attributes, filename='output1.svg')

.. figure:: https://cdn.rawgit.com/mathandy/svgpathtools/master/output1.svg
   :alt: output1.svg

   output1.svg

There will be many more examples of writing and displaying path data
below.

The .point() method and transitioning between path and path segment parameterizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SVG Path elements and their segments have official parameterizations.
These parameterizations can be accessed using the ``Path.point()``,
``Line.point()``, ``QuadraticBezier.point()``, ``CubicBezier.point()``,
and ``Arc.point()`` methods. All these parameterizations are defined
over the domain 0 <= t <= 1.

| **Note:** In this document and in inline documentation and doctrings,
  I use a capital ``T`` when referring to the parameterization of a Path
  object and a lower case ``t`` when referring speaking about path
  segment objects (i.e. Line, QaudraticBezier, CubicBezier, and Arc
  objects).
| Given a ``T`` value, the ``Path.T2t()`` method can be used to find the
  corresponding segment index, ``k``, and segment parameter, ``t``, such
  that ``path.point(T)=path[k].point(t)``.
| There is also a ``Path.t2T()`` method to solve the inverse problem.

.. code:: ipython2

    # Example:
    
    # Let's check that the first segment of redpath starts 
    # at the same point as redpath
    firstseg = redpath[0] 
    print(redpath.point(0) == firstseg.point(0) == redpath.start == firstseg.start)
    
    # Let's check that the last segment of redpath ends on the same point as redpath
    lastseg = redpath[-1] 
    print(redpath.point(1) == lastseg.point(1) == redpath.end == lastseg.end)
    
    # This next boolean should return False as redpath is composed multiple segments
    print(redpath.point(0.5) == firstseg.point(0.5))
    
    # If we want to figure out which segment of redpoint the 
    # point redpath.point(0.5) lands on, we can use the path.T2t() method
    k, t = redpath.T2t(0.5)
    print(redpath[k].point(t) == redpath.point(0.5))


.. parsed-literal::

    True
    True
    False
    True


Bezier curves as NumPy polynomial objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Another great way to work with the parameterizations for ``Line``,
  ``QuadraticBezier``, and ``CubicBezier`` objects is to convert them to
  ``numpy.poly1d`` objects. This is done easily using the
  ``Line.poly()``, ``QuadraticBezier.poly()`` and ``CubicBezier.poly()``
  methods.
| There's also a ``polynomial2bezier()`` function in the pathtools.py
  submodule to convert polynomials back to Bezier curves.

**Note:** cubic Bezier curves are parameterized as

.. math:: \mathcal{B}(t) = P_0(1-t)^3 + 3P_1(1-t)^2t + 3P_2(1-t)t^2 + P_3t^3

where :math:`P_0`, :math:`P_1`, :math:`P_2`, and :math:`P_3` are the
control points ``start``, ``control1``, ``control2``, and ``end``,
respectively, that svgpathtools uses to define a CubicBezier object. The
``CubicBezier.poly()`` method expands this polynomial to its standard
form

.. math:: \mathcal{B}(t) = c_0t^3 + c_1t^2 +c_2t+c3

 where

.. math::

   \begin{bmatrix}c_0\\c_1\\c_2\\c_3\end{bmatrix} = 
   \begin{bmatrix}
   -1 & 3 & -3 & 1\\
   3 & -6 & -3 & 0\\
   -3 & 3 & 0 & 0\\
   1 & 0 & 0 & 0\\
   \end{bmatrix}
   \begin{bmatrix}P_0\\P_1\\P_2\\P_3\end{bmatrix}

``QuadraticBezier.poly()`` and ``Line.poly()`` are `defined
similarly <https://en.wikipedia.org/wiki/B%C3%A9zier_curve#General_definition>`__.

.. code:: ipython2

    # Example:
    b = CubicBezier(300+100j, 100+100j, 200+200j, 200+300j)
    p = b.poly()
    
    # p(t) == b.point(t)
    print(p(0.235) == b.point(0.235))
    
    # What is p(t)?  It's just the cubic b written in standard form.  
    bpretty = "{}*(1-t)^3 + 3*{}*(1-t)^2*t + 3*{}*(1-t)*t^2 + {}*t^3".format(*b.bpoints())
    print("The CubicBezier, b.point(x) = \n\n" + 
          bpretty + "\n\n" + 
          "can be rewritten in standard form as \n\n" +
          str(p).replace('x','t'))


.. parsed-literal::

    True
    The CubicBezier, b.point(x) = 
    
    (300+100j)*(1-t)^3 + 3*(100+100j)*(1-t)^2*t + 3*(200+200j)*(1-t)*t^2 + (200+300j)*t^3
    
    can be rewritten in standard form as 
    
                    3                2
    (-400 + -100j) t + (900 + 300j) t - 600 t + (300 + 100j)


The ability to convert between Bezier objects to NumPy polynomial
objects is very useful. For starters, we can take turn a list of Bézier
segments into a NumPy array

Numpy Array operations on Bézier path segments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Example available
here <https://github.com/mathandy/svgpathtools/blob/master/examples/compute-many-points-quickly-using-numpy-arrays.py>`__

To further illustrate the power of being able to convert our Bezier
curve objects to numpy.poly1d objects and back, lets compute the unit
tangent vector of the above CubicBezier object, b, at t=0.5 in four
different ways.

Tangent vectors (and more on NumPy polynomials)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    t = 0.5
    ### Method 1: the easy way
    u1 = b.unit_tangent(t)
    
    ### Method 2: another easy way 
    # Note: This way will fail if it encounters a removable singularity.
    u2 = b.derivative(t)/abs(b.derivative(t))
    
    ### Method 2: a third easy way 
    # Note: This way will also fail if it encounters a removable singularity.
    dp = p.deriv() 
    u3 = dp(t)/abs(dp(t))
    
    ### Method 4: the removable-singularity-proof numpy.poly1d way  
    # Note: This is roughly how Method 1 works
    from svgpathtools import real, imag, rational_limit
    dx, dy = real(dp), imag(dp)  # dp == dx + 1j*dy 
    p_mag2 = dx**2 + dy**2  # p_mag2(t) = |p(t)|**2
    # Note: abs(dp) isn't a polynomial, but abs(dp)**2 is, and,
    #  the limit_{t->t0}[f(t) / abs(f(t))] == 
    # sqrt(limit_{t->t0}[f(t)**2 / abs(f(t))**2])
    from cmath import sqrt
    u4 = sqrt(rational_limit(dp**2, p_mag2, t))
    
    print("unit tangent check:", u1 == u2 == u3 == u4)
    
    # Let's do a visual check
    mag = b.length()/4  # so it's not hard to see the tangent line
    tangent_line = Line(b.point(t), b.point(t) + mag*u1)
    disvg([b, tangent_line], 'bg', nodes=[b.point(t)])


.. parsed-literal::

    unit tangent check: True


Translations (shifts), reversing orientation, and normal vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    # Speaking of tangents, let's add a normal vector to the picture
    n = b.normal(t)
    normal_line = Line(b.point(t), b.point(t) + mag*n)
    disvg([b, tangent_line, normal_line], 'bgp', nodes=[b.point(t)])
    
    # and let's reverse the orientation of b! 
    # the tangent and normal lines should be sent to their opposites
    br = b.reversed()
    
    # Let's also shift b_r over a bit to the right so we can view it next to b
    # The simplest way to do this is br = br.translated(3*mag),  but let's use 
    # the .bpoints() instead, which returns a Bezier's control points
    br.start, br.control1, br.control2, br.end = [3*mag + bpt for bpt in br.bpoints()]  # 
    
    tangent_line_r = Line(br.point(t), br.point(t) + mag*br.unit_tangent(t))
    normal_line_r = Line(br.point(t), br.point(t) + mag*br.normal(t))
    wsvg([b, tangent_line, normal_line, br, tangent_line_r, normal_line_r], 
         'bgpkgp', nodes=[b.point(t), br.point(t)], filename='vectorframes.svg', 
         text=["b's tangent", "br's tangent"], text_path=[tangent_line, tangent_line_r])

.. figure:: https://cdn.rawgit.com/mathandy/svgpathtools/master/vectorframes.svg
   :alt: vectorframes.svg

   vectorframes.svg

Rotations and Translations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    # Let's take a Line and an Arc and make some pictures
    top_half = Arc(start=-1, radius=1+2j, rotation=0, large_arc=1, sweep=1, end=1)
    midline = Line(-1.5, 1.5)
    
    # First let's make our ellipse whole
    bottom_half = top_half.rotated(180)
    decorated_ellipse = Path(top_half, bottom_half)
    
    # Now let's add the decorations
    for k in range(12):
        decorated_ellipse.append(midline.rotated(30*k))
        
    # Let's move it over so we can see the original Line and Arc object next
    # to the final product
    decorated_ellipse = decorated_ellipse.translated(4+0j)
    wsvg([top_half, midline, decorated_ellipse], filename='decorated_ellipse.svg')

.. figure:: https://cdn.rawgit.com/mathandy/svgpathtools/master/decorated_ellipse.svg
   :alt: decorated\_ellipse.svg

   decorated\_ellipse.svg

arc length and inverse arc length
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we'll create an SVG that shows off the parametric and geometric
midpoints of the paths from ``test.svg``. We'll need to compute use the
``Path.length()``, ``Line.length()``, ``QuadraticBezier.length()``,
``CubicBezier.length()``, and ``Arc.length()`` methods, as well as the
related inverse arc length methods ``.ilength()`` function to do this.

.. code:: ipython2

    # First we'll load the path data from the file test.svg
    paths, attributes = svg2paths('test.svg')
    
    # Let's mark the parametric midpoint of each segment
    # I say "parametric" midpoint because Bezier curves aren't 
    # parameterized by arclength 
    # If they're also the geometric midpoint, let's mark them
    # purple and otherwise we'll mark the geometric midpoint green
    min_depth = 5
    error = 1e-4
    dots = []
    ncols = []
    nradii = []
    for path in paths:
        for seg in path:
            parametric_mid = seg.point(0.5)
            seg_length = seg.length()
            if seg.length(0.5)/seg.length() == 1/2:
                dots += [parametric_mid]
                ncols += ['purple']
                nradii += [5]
            else:
                t_mid = seg.ilength(seg_length/2)
                geo_mid = seg.point(t_mid)
                dots += [parametric_mid, geo_mid]
                ncols += ['red', 'green']
                nradii += [5] * 2
    
    # In 'output2.svg' the paths will retain their original attributes
    wsvg(paths, nodes=dots, node_colors=ncols, node_radii=nradii, 
         attributes=attributes, filename='output2.svg')

.. figure:: https://cdn.rawgit.com/mathandy/svgpathtools/master/output2.svg
   :alt: output2.svg

   output2.svg

Intersections between Bezier curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    # Let's find all intersections between redpath and the other 
    redpath = paths[0]
    redpath_attribs = attributes[0]
    intersections = []
    for path in paths[1:]:
        for (T1, seg1, t1), (T2, seg2, t2) in redpath.intersect(path):
            intersections.append(redpath.point(T1))
            
    disvg(paths, filename='output_intersections.svg', attributes=attributes,
          nodes = intersections, node_radii = [5]*len(intersections))

.. figure:: https://cdn.rawgit.com/mathandy/svgpathtools/master/output_intersections.svg
   :alt: output\_intersections.svg

   output\_intersections.svg

An Advanced Application: Offsetting Paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we'll find the `offset
curve <https://en.wikipedia.org/wiki/Parallel_curve>`__ for a few paths.

.. code:: ipython2

    from svgpathtools import parse_path, Line, Path, wsvg
    def offset_curve(path, offset_distance, steps=1000):
        """Takes in a Path object, `path`, and a distance,
        `offset_distance`, and outputs an piecewise-linear approximation 
        of the 'parallel' offset curve."""
        nls = []
        for seg in path:
            for k in range(steps):
                t = k / float(steps)
                offset_vector = offset_distance * seg.normal(t)
                nl = Line(seg.point(t), seg.point(t) + offset_vector)
                nls.append(nl)
        connect_the_dots = [Line(nls[k].end, nls[k+1].end) for k in range(len(nls)-1)]
        if path.isclosed():
            connect_the_dots.append(Line(nls[-1].end, nls[0].end))
        offset_path = Path(*connect_the_dots)
        return offset_path
    
    # Examples:
    path1 = parse_path("m 288,600 c -52,-28 -42,-61 0,-97 ")
    path2 = parse_path("M 151,395 C 407,485 726.17662,160 634,339").translated(300)
    path3 = parse_path("m 117,695 c 237,-7 -103,-146 457,0").translated(500+400j)
    paths = [path1, path2, path3]
    
    offset_distances = [10*k for k in range(1,51)]
    offset_paths = []
    for path in paths:
        for distances in offset_distances:
            offset_paths.append(offset_curve(path, distances))
    
    # Note: This will take a few moments
    wsvg(paths + offset_paths, 'g'*len(paths) + 'r'*len(offset_paths), filename='offset_curves.svg')

.. figure:: https://cdn.rawgit.com/mathandy/svgpathtools/master/offset_curves.svg
   :alt: offset\_curves.svg

   offset\_curves.svg

Compatibility Notes for users of svg.path (v2.0)
------------------------------------------------

-  renamed Arc.arc attribute as Arc.large\_arc

-  Path.d() : For behavior similar\ `2 <#f2>`__\  to svg.path (v2.0),
   set both useSandT and use\_closed\_attrib to be True.

2 The behavior would be identical, but the string formatting used in
this method has been changed to use default format (instead of the
General format, {:G}), for inceased precision. `↩ <#a2>`__

Licence
-------

This module is under a MIT License.

