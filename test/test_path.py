# External dependencies
from __future__ import division, absolute_import, print_function
import unittest
from math import sqrt, pi
from numbers import Number
import numpy as np

# Internal dependencies
from svgpathtools import *

# An important note for those doing any debugging:
# ------------------------------------------------
# Most of these test points are not calculated separately, as that would
# take too long and be too error prone. Instead the curves have been verified
# to be correct visually with the disvg() function.


class LineTest(unittest.TestCase):
    def test_lines(self):
        # These points are calculated, and not just regression
        # tests.
        line1 = Line(0j, 400 + 0j)
        self.assertAlmostEqual(line1.point(0), 0j)
        self.assertAlmostEqual(line1.point(0.3), (120 + 0j))
        self.assertAlmostEqual(line1.point(0.5), (200 + 0j))
        self.assertAlmostEqual(line1.point(0.9), (360 + 0j))
        self.assertAlmostEqual(line1.point(1), (400 + 0j))
        self.assertAlmostEqual(line1.length(), 400)

        line2 = Line(400 + 0j, 400 + 300j)
        self.assertAlmostEqual(line2.point(0), (400 + 0j))
        self.assertAlmostEqual(line2.point(0.3), (400 + 90j))
        self.assertAlmostEqual(line2.point(0.5), (400 + 150j))
        self.assertAlmostEqual(line2.point(0.9), (400 + 270j))
        self.assertAlmostEqual(line2.point(1), (400 + 300j))
        self.assertAlmostEqual(line2.length(), 300)

        line3 = Line(400 + 300j, 0j)
        self.assertAlmostEqual(line3.point(0), (400 + 300j))
        self.assertAlmostEqual(line3.point(0.3), (280 + 210j))
        self.assertAlmostEqual(line3.point(0.5), (200 + 150j))
        self.assertAlmostEqual(line3.point(0.9), (40 + 30j))
        self.assertAlmostEqual(line3.point(1), 0j)
        self.assertAlmostEqual(line3.length(), 500)

    def test_equality(self):
        # This is to test the __eq__ and __ne__ methods, so we can't use
        # assertEqual and assertNotEqual
        line = Line(0j, 400 + 0j)
        cubic = CubicBezier(600 + 500j, 600 + 350j, 900 + 650j, 900 + 500j)
        self.assertTrue(line == Line(0, 400))
        self.assertTrue(line != Line(100, 400))
        self.assertFalse(line == str(line))
        self.assertTrue(line != str(line))
        self.assertFalse(cubic == line)


class CubicBezierTest(unittest.TestCase):
    def test_approx_circle(self):
        """This is a approximate circle drawn in Inkscape"""

        cub1 = CubicBezier(
            complex(0, 0),
            complex(0, 109.66797),
            complex(-88.90345, 198.57142),
            complex(-198.57142, 198.57142)
        )

        self.assertAlmostEqual(cub1.point(0), 0j)
        self.assertAlmostEqual(cub1.point(0.1), (-2.59896457 + 32.20931647j))
        self.assertAlmostEqual(cub1.point(0.2), (-10.12330256 + 62.76392816j))
        self.assertAlmostEqual(cub1.point(0.3), (-22.16418039 + 91.25500149j))
        self.assertAlmostEqual(cub1.point(0.4), (-38.31276448 + 117.27370288j))
        self.assertAlmostEqual(cub1.point(0.5), (-58.16022125 + 140.41119875j))
        self.assertAlmostEqual(cub1.point(0.6), (-81.29771712 + 160.25865552j))
        self.assertAlmostEqual(cub1.point(0.7), (-107.31641851 + 176.40723961j))
        self.assertAlmostEqual(cub1.point(0.8), (-135.80749184 + 188.44811744j))
        self.assertAlmostEqual(cub1.point(0.9), (-166.36210353 + 195.97245543j))
        self.assertAlmostEqual(cub1.point(1), (-198.57142 + 198.57142j))

        cub2 = CubicBezier(
            complex(-198.57142, 198.57142),
            complex(-109.66797 - 198.57142, 0 + 198.57142),
            complex(-198.57143 - 198.57142, -88.90345 + 198.57142),
            complex(-198.57143 - 198.57142, 0),
        )

        self.assertAlmostEqual(cub2.point(0), (-198.57142 + 198.57142j))
        self.assertAlmostEqual(cub2.point(0.1), (-230.78073675 + 195.97245543j))
        self.assertAlmostEqual(cub2.point(0.2), (-261.3353492 + 188.44811744j))
        self.assertAlmostEqual(cub2.point(0.3), (-289.82642365 + 176.40723961j))
        self.assertAlmostEqual(cub2.point(0.4), (-315.8451264 + 160.25865552j))
        self.assertAlmostEqual(cub2.point(0.5), (-338.98262375 + 140.41119875j))
        self.assertAlmostEqual(cub2.point(0.6), (-358.830082 + 117.27370288j))
        self.assertAlmostEqual(cub2.point(0.7), (-374.97866745 + 91.25500149j))
        self.assertAlmostEqual(cub2.point(0.8), (-387.0195464 + 62.76392816j))
        self.assertAlmostEqual(cub2.point(0.9), (-394.54388515 + 32.20931647j))
        self.assertAlmostEqual(cub2.point(1), (-397.14285 + 0j))

        cub3 = CubicBezier(
            complex(-198.57143 - 198.57142, 0),
            complex(0 - 198.57143 - 198.57142, -109.66797),
            complex(88.90346 - 198.57143 - 198.57142, -198.57143),
            complex(-198.57142, -198.57143)
        )

        self.assertAlmostEqual(cub3.point(0), (-397.14285 + 0j))
        self.assertAlmostEqual(cub3.point(0.1), (-394.54388515 - 32.20931675j))
        self.assertAlmostEqual(cub3.point(0.2), (-387.0195464 - 62.7639292j))
        self.assertAlmostEqual(cub3.point(0.3), (-374.97866745 - 91.25500365j))
        self.assertAlmostEqual(cub3.point(0.4), (-358.830082 - 117.2737064j))
        self.assertAlmostEqual(cub3.point(0.5), (-338.98262375 - 140.41120375j))
        self.assertAlmostEqual(cub3.point(0.6), (-315.8451264 - 160.258662j))
        self.assertAlmostEqual(cub3.point(0.7), (-289.82642365 - 176.40724745j))
        self.assertAlmostEqual(cub3.point(0.8), (-261.3353492 - 188.4481264j))
        self.assertAlmostEqual(cub3.point(0.9), (-230.78073675 - 195.97246515j))
        self.assertAlmostEqual(cub3.point(1), (-198.57142 - 198.57143j))

        cub4 = CubicBezier(
            complex(-198.57142, -198.57143),
            complex(109.66797 - 198.57142, 0 - 198.57143),
            complex(0, 88.90346 - 198.57143),
            complex(0, 0),
        )

        self.assertAlmostEqual(cub4.point(0), (-198.57142 - 198.57143j))
        self.assertAlmostEqual(cub4.point(0.1), (-166.36210353 - 195.97246515j))
        self.assertAlmostEqual(cub4.point(0.2), (-135.80749184 - 188.4481264j))
        self.assertAlmostEqual(cub4.point(0.3), (-107.31641851 - 176.40724745j))
        self.assertAlmostEqual(cub4.point(0.4), (-81.29771712 - 160.258662j))
        self.assertAlmostEqual(cub4.point(0.5), (-58.16022125 - 140.41120375j))
        self.assertAlmostEqual(cub4.point(0.6), (-38.31276448 - 117.2737064j))
        self.assertAlmostEqual(cub4.point(0.7), (-22.16418039 - 91.25500365j))
        self.assertAlmostEqual(cub4.point(0.8), (-10.12330256 - 62.7639292j))
        self.assertAlmostEqual(cub4.point(0.9), (-2.59896457 - 32.20931675j))
        self.assertAlmostEqual(cub4.point(1), 0j)

    def test_svg_examples(self):
        # M100,200 C100,100 250,100 250,200
        path1 = CubicBezier(100 + 200j, 100 + 100j, 250 + 100j, 250 + 200j)
        self.assertAlmostEqual(path1.point(0), (100 + 200j))
        self.assertAlmostEqual(path1.point(0.3), (132.4 + 137j))
        self.assertAlmostEqual(path1.point(0.5), (175 + 125j))
        self.assertAlmostEqual(path1.point(0.9), (245.8 + 173j))
        self.assertAlmostEqual(path1.point(1), (250 + 200j))

        # S400,300 400,200
        path2 = CubicBezier(250 + 200j, 250 + 300j, 400 + 300j, 400 + 200j)
        self.assertAlmostEqual(path2.point(0), (250 + 200j))
        self.assertAlmostEqual(path2.point(0.3), (282.4 + 263j))
        self.assertAlmostEqual(path2.point(0.5), (325 + 275j))
        self.assertAlmostEqual(path2.point(0.9), (395.8 + 227j))
        self.assertAlmostEqual(path2.point(1), (400 + 200j))

        # M100,200 C100,100 400,100 400,200
        path3 = CubicBezier(100 + 200j, 100 + 100j, 400 + 100j, 400 + 200j)
        self.assertAlmostEqual(path3.point(0), (100 + 200j))
        self.assertAlmostEqual(path3.point(0.3), (164.8 + 137j))
        self.assertAlmostEqual(path3.point(0.5), (250 + 125j))
        self.assertAlmostEqual(path3.point(0.9), (391.6 + 173j))
        self.assertAlmostEqual(path3.point(1), (400 + 200j))

        # M100,500 C25,400 475,400 400,500
        path4 = CubicBezier(100 + 500j, 25 + 400j, 475 + 400j, 400 + 500j)
        self.assertAlmostEqual(path4.point(0), (100 + 500j))
        self.assertAlmostEqual(path4.point(0.3), (145.9 + 437j))
        self.assertAlmostEqual(path4.point(0.5), (250 + 425j))
        self.assertAlmostEqual(path4.point(0.9), (407.8 + 473j))
        self.assertAlmostEqual(path4.point(1), (400 + 500j))

        # M100,800 C175,700 325,700 400,800
        path5 = CubicBezier(100 + 800j, 175 + 700j, 325 + 700j, 400 + 800j)
        self.assertAlmostEqual(path5.point(0), (100 + 800j))
        self.assertAlmostEqual(path5.point(0.3), (183.7 + 737j))
        self.assertAlmostEqual(path5.point(0.5), (250 + 725j))
        self.assertAlmostEqual(path5.point(0.9), (375.4 + 773j))
        self.assertAlmostEqual(path5.point(1), (400 + 800j))

        # M600,200 C675,100 975,100 900,200
        path6 = CubicBezier(600 + 200j, 675 + 100j, 975 + 100j, 900 + 200j)
        self.assertAlmostEqual(path6.point(0), (600 + 200j))
        self.assertAlmostEqual(path6.point(0.3), (712.05 + 137j))
        self.assertAlmostEqual(path6.point(0.5), (806.25 + 125j))
        self.assertAlmostEqual(path6.point(0.9), (911.85 + 173j))
        self.assertAlmostEqual(path6.point(1), (900 + 200j))

        # M600,500 C600,350 900,650 900,500
        path7 = CubicBezier(600 + 500j, 600 + 350j, 900 + 650j, 900 + 500j)
        self.assertAlmostEqual(path7.point(0), (600 + 500j))
        self.assertAlmostEqual(path7.point(0.3), (664.8 + 462.2j))
        self.assertAlmostEqual(path7.point(0.5), (750 + 500j))
        self.assertAlmostEqual(path7.point(0.9), (891.6 + 532.4j))
        self.assertAlmostEqual(path7.point(1), (900 + 500j))

        # M600,800 C625,700 725,700 750,800
        path8 = CubicBezier(600 + 800j, 625 + 700j, 725 + 700j, 750 + 800j)
        self.assertAlmostEqual(path8.point(0), (600 + 800j))
        self.assertAlmostEqual(path8.point(0.3), (638.7 + 737j))
        self.assertAlmostEqual(path8.point(0.5), (675 + 725j))
        self.assertAlmostEqual(path8.point(0.9), (740.4 + 773j))
        self.assertAlmostEqual(path8.point(1), (750 + 800j))

        # S875,900 900,800
        inversion = (750 + 800j) + (750 + 800j) - (725 + 700j)
        path9 = CubicBezier(750 + 800j, inversion, 875 + 900j, 900 + 800j)
        self.assertAlmostEqual(path9.point(0), (750 + 800j))
        self.assertAlmostEqual(path9.point(0.3), (788.7 + 863j))
        self.assertAlmostEqual(path9.point(0.5), (825 + 875j))
        self.assertAlmostEqual(path9.point(0.9), (890.4 + 827j))
        self.assertAlmostEqual(path9.point(1), (900 + 800j))

    def test_length(self):
        # A straight line:
        cub = CubicBezier(
            complex(0, 0),
            complex(0, 0),
            complex(0, 100),
            complex(0, 100)
        )

        self.assertAlmostEqual(cub.length(), 100)

        # A diagonal line:
        cub = CubicBezier(
            complex(0, 0),
            complex(0, 0),
            complex(100, 100),
            complex(100, 100)
        )

        self.assertAlmostEqual(cub.length(), sqrt(2 * 100 * 100))

        # A quarter circle large_arc with radius 100
        # http://www.whizkidtech.redprince.net/bezier/circle/
        kappa = 4 * (sqrt(2) - 1) / 3

        cub = CubicBezier(
            complex(0, 0),
            complex(0, kappa * 100),
            complex(100 - kappa * 100, 100),
            complex(100, 100)
        )

        # We can't compare with pi*50 here, because this is just an
        # approximation of a circle large_arc. pi*50 is 157.079632679
        # So this is just yet another "warn if this changes" test.
        # This value is not verified to be correct.
        self.assertAlmostEqual(cub.length(), 157.1016698)

        # A recursive solution has also been suggested, but for CubicBezier
        # curves it could get a false solution on curves where the midpoint is
        # on a straight line between the start and end. For example, the
        # following curve would get solved as a straight line and get the
        # length 300.
        # Make sure this is not the case.
        cub = CubicBezier(
            complex(600, 500),
            complex(600, 350),
            complex(900, 650),
            complex(900, 500)
        )
        self.assertTrue(cub.length() > 300.0)

    def test_equality(self):
        # This is to test the __eq__ and __ne__ methods, so we can't use
        # assertEqual and assertNotEqual
        segment = CubicBezier(complex(600, 500), complex(600, 350),
                              complex(900, 650), complex(900, 500))

        self.assertTrue(segment ==
                        CubicBezier(600 + 500j, 600 + 350j, 900 + 650j, 900 + 500j))
        self.assertTrue(segment !=
                        CubicBezier(600 + 501j, 600 + 350j, 900 + 650j, 900 + 500j))
        self.assertTrue(segment != Line(0, 400))


class QuadraticBezierTest(unittest.TestCase):

    def test_svg_examples(self):
        """These is the path in the SVG specs"""
        # M200,300 Q400,50 600,300 T1000,300
        path1 = QuadraticBezier(200 + 300j, 400 + 50j, 600 + 300j)
        self.assertAlmostEqual(path1.point(0), (200 + 300j))
        self.assertAlmostEqual(path1.point(0.3), (320 + 195j))
        self.assertAlmostEqual(path1.point(0.5), (400 + 175j))
        self.assertAlmostEqual(path1.point(0.9), (560 + 255j))
        self.assertAlmostEqual(path1.point(1), (600 + 300j))

        # T1000, 300
        inversion = (600 + 300j) + (600 + 300j) - (400 + 50j)
        path2 = QuadraticBezier(600 + 300j, inversion, 1000 + 300j)
        self.assertAlmostEqual(path2.point(0), (600 + 300j))
        self.assertAlmostEqual(path2.point(0.3), (720 + 405j))
        self.assertAlmostEqual(path2.point(0.5), (800 + 425j))
        self.assertAlmostEqual(path2.point(0.9), (960 + 345j))
        self.assertAlmostEqual(path2.point(1), (1000 + 300j))

    def test_length(self):
        # expected results calculated with
        # svg.path.segment_length(q, 0, 1, q.start, q.end, 1e-14, 20, 0)
        q1 = QuadraticBezier(200 + 300j, 400 + 50j, 600 + 300j)
        q2 = QuadraticBezier(200 + 300j, 400 + 50j, 500 + 200j)
        closedq = QuadraticBezier(6 + 2j, 5 - 1j, 6 + 2j)
        linq1 = QuadraticBezier(1, 2, 3)
        linq2 = QuadraticBezier(1 + 3j, 2 + 5j, -9 - 17j)
        nodalq = QuadraticBezier(1, 1, 1)
        tests = [(q1, 487.77109389525975),
                 (q2, 379.90458193489155),
                 (closedq, 3.1622776601683795),
                 (linq1, 2),
                 (linq2, 22.73335777124786),
                 (nodalq, 0)]
        for q, exp_res in tests:
            self.assertAlmostEqual(q.length(), exp_res)

        # partial length tests
        tests = [(q1, 212.34775387566032),
                 (q2, 166.22170622052397),
                 (closedq, 0.7905694150420949),
                 (linq1, 1.0),
                 (nodalq, 0.0)]
        t0 = 0.25
        t1 = 0.75
        for q, exp_res in tests:
            self.assertAlmostEqual(q.length(t0=t0, t1=t1), exp_res)

        # linear partial cases
        linq2 = QuadraticBezier(1 + 3j, 2 + 5j, -9 - 17j)
        tests = [(0, 1 / 24, 0.13975424859373725),
                 (0, 1 / 12, 0.1863389981249823),
                 (0, 0.5, 4.844813951249543),
                 (0, 1, 22.73335777124786),
                 (1 / 24, 1 / 12, 0.04658474953124506),
                 (1 / 24, 0.5, 4.705059702655722),
                 (1 / 24, 1, 22.59360352265412),
                 (1 / 12, 0.5, 4.658474953124562),
                 (1 / 12, 1, 22.54701877312288),
                 (0.5, 1, 17.88854381999832)]
        for t0, t1, exp_s in tests:
            self.assertAlmostEqual(linq2.length(t0=t0, t1=t1), exp_s)

    def test_equality(self):
        # This is to test the __eq__ and __ne__ methods, so we can't use
        # assertEqual and assertNotEqual
        segment = QuadraticBezier(200 + 300j, 400 + 50j, 600 + 300j)
        self.assertTrue(segment ==
                        QuadraticBezier(200 + 300j, 400 + 50j, 600 + 300j))
        self.assertTrue(segment !=
                        QuadraticBezier(200 + 301j, 400 + 50j, 600 + 300j))
        self.assertFalse(segment == Arc(0j, 100 + 50j, 0, 0, 0, 100 + 50j))
        self.assertTrue(Arc(0j, 100 + 50j, 0, 0, 0, 100 + 50j) != segment)


class ArcTest(unittest.TestCase):
    def test_trusting_acos(self):
        """`u1.real` is > 1 in this arc due to numerical error."""
        try:
            Arc(
                start=(160.197 + 102.925j),
                radius=(0.025 + 0.025j),
                rotation=0.0,
                large_arc=False,
                sweep=True,
                end=(160.172 + 102.95j)
            )
        except ValueError:
            self.fail("Arc() raised ValueError unexpectedly!")

    def test_points(self):
        arc1 = Arc(0j, 100 + 50j, 0, 0, 0, 100 + 50j)
        self.assertAlmostEqual(arc1.center, 100 + 0j)
        self.assertAlmostEqual(arc1.theta, 180.0)
        self.assertAlmostEqual(arc1.delta, -90.0)

        self.assertAlmostEqual(arc1.point(0.0), 0j)
        self.assertAlmostEqual(arc1.point(0.1), (1.23116594049 + 7.82172325201j))
        self.assertAlmostEqual(arc1.point(0.2), (4.89434837048 + 15.4508497187j))
        self.assertAlmostEqual(arc1.point(0.3), (10.8993475812 + 22.699524987j))
        self.assertAlmostEqual(arc1.point(0.4), (19.0983005625 + 29.3892626146j))
        self.assertAlmostEqual(arc1.point(0.5), (29.2893218813 + 35.3553390593j))
        self.assertAlmostEqual(arc1.point(0.6), (41.2214747708 + 40.4508497187j))
        self.assertAlmostEqual(arc1.point(0.7), (54.6009500260 + 44.5503262094j))
        self.assertAlmostEqual(arc1.point(0.8), (69.0983005625 + 47.5528258148j))
        self.assertAlmostEqual(arc1.point(0.9), (84.3565534960 + 49.3844170298j))
        self.assertAlmostEqual(arc1.point(1.0), (100 + 50j))

        arc2 = Arc(0j, 100 + 50j, 0, 1, 0, 100 + 50j)
        self.assertAlmostEqual(arc2.center, 50j)
        self.assertAlmostEqual(arc2.theta, -90.0)
        self.assertAlmostEqual(arc2.delta, -270.0)

        self.assertAlmostEqual(arc2.point(0.0), 0j)
        self.assertAlmostEqual(arc2.point(0.1), (-45.399049974 + 5.44967379058j))
        self.assertAlmostEqual(arc2.point(0.2), (-80.9016994375 + 20.6107373854j))
        self.assertAlmostEqual(arc2.point(0.3), (-98.7688340595 + 42.178276748j))
        self.assertAlmostEqual(arc2.point(0.4), (-95.1056516295 + 65.4508497187j))
        self.assertAlmostEqual(arc2.point(0.5), (-70.7106781187 + 85.3553390593j))
        self.assertAlmostEqual(arc2.point(0.6), (-30.9016994375 + 97.5528258148j))
        self.assertAlmostEqual(arc2.point(0.7), (15.643446504 + 99.3844170298j))
        self.assertAlmostEqual(arc2.point(0.8), (58.7785252292 + 90.4508497187j))
        self.assertAlmostEqual(arc2.point(0.9), (89.1006524188 + 72.699524987j))
        self.assertAlmostEqual(arc2.point(1.0), (100 + 50j))

        arc3 = Arc(0j, 100 + 50j, 0, 0, 1, 100 + 50j)
        self.assertAlmostEqual(arc3.center, 50j)
        self.assertAlmostEqual(arc3.theta, -90.0)
        self.assertAlmostEqual(arc3.delta, 90.0)

        self.assertAlmostEqual(arc3.point(0.0), 0j)
        self.assertAlmostEqual(arc3.point(0.1), (15.643446504 + 0.615582970243j))
        self.assertAlmostEqual(arc3.point(0.2), (30.9016994375 + 2.44717418524j))
        self.assertAlmostEqual(arc3.point(0.3), (45.399049974 + 5.44967379058j))
        self.assertAlmostEqual(arc3.point(0.4), (58.7785252292 + 9.54915028125j))
        self.assertAlmostEqual(arc3.point(0.5), (70.7106781187 + 14.6446609407j))
        self.assertAlmostEqual(arc3.point(0.6), (80.9016994375 + 20.6107373854j))
        self.assertAlmostEqual(arc3.point(0.7), (89.1006524188 + 27.300475013j))
        self.assertAlmostEqual(arc3.point(0.8), (95.1056516295 + 34.5491502813j))
        self.assertAlmostEqual(arc3.point(0.9), (98.7688340595 + 42.178276748j))
        self.assertAlmostEqual(arc3.point(1.0), (100 + 50j))

        arc4 = Arc(0j, 100 + 50j, 0, 1, 1, 100 + 50j)
        self.assertAlmostEqual(arc4.center, 100 + 0j)
        self.assertAlmostEqual(arc4.theta, 180.0)
        self.assertAlmostEqual(arc4.delta, 270.0)

        self.assertAlmostEqual(arc4.point(0.0), 0j)
        self.assertAlmostEqual(arc4.point(0.1), (10.8993475812 - 22.699524987j))
        self.assertAlmostEqual(arc4.point(0.2), (41.2214747708 - 40.4508497187j))
        self.assertAlmostEqual(arc4.point(0.3), (84.3565534960 - 49.3844170298j))
        self.assertAlmostEqual(arc4.point(0.4), (130.901699437 - 47.5528258148j))
        self.assertAlmostEqual(arc4.point(0.5), (170.710678119 - 35.3553390593j))
        self.assertAlmostEqual(arc4.point(0.6), (195.105651630 - 15.4508497187j))
        self.assertAlmostEqual(arc4.point(0.7), (198.768834060 + 7.82172325201j))
        self.assertAlmostEqual(arc4.point(0.8), (180.901699437 + 29.3892626146j))
        self.assertAlmostEqual(arc4.point(0.9), (145.399049974 + 44.5503262094j))
        self.assertAlmostEqual(arc4.point(1.0), (100 + 50j))

        arc5 = Arc((725.307482225571 - 915.5548199281527j),
                   (202.79421639137703 + 148.77294617167183j),
                   225.6910319606926, 1, 1,
                   (-624.6375539637027 + 896.5483089399895j))
        self.assertAlmostEqual(arc5.point(0.0), (725.307482226 - 915.554819928j))
        self.assertAlmostEqual(arc5.point(0.0909090909091),
                               (1023.47397369 - 597.730444283j))
        self.assertAlmostEqual(arc5.point(0.181818181818),
                               (1242.80253007 - 232.251400124j))
        self.assertAlmostEqual(arc5.point(0.272727272727),
                               (1365.52445614 + 151.273373978j))
        self.assertAlmostEqual(arc5.point(0.363636363636),
                               (1381.69755131 + 521.772981736j))
        self.assertAlmostEqual(arc5.point(0.454545454545),
                               (1290.01156757 + 849.231748376j))
        self.assertAlmostEqual(arc5.point(0.545454545455),
                               (1097.89435807 + 1107.12091209j))
        self.assertAlmostEqual(arc5.point(0.636363636364),
                               (820.910116547 + 1274.54782658j))
        self.assertAlmostEqual(arc5.point(0.727272727273),
                               (481.49845896 + 1337.94855893j))
        self.assertAlmostEqual(arc5.point(0.818181818182),
                               (107.156499251 + 1292.18675889j))
        self.assertAlmostEqual(arc5.point(0.909090909091),
                               (-271.788803303 + 1140.96977533j))

    def test_length(self):
        # I'll test the length calculations by making a circle, in two parts.
        arc1 = Arc(0j, 100 + 100j, 0, 0, 0, 200 + 0j)
        arc2 = Arc(200 + 0j, 100 + 100j, 0, 0, 0, 0j)
        self.assertAlmostEqual(arc1.length(), pi * 100)
        self.assertAlmostEqual(arc2.length(), pi * 100)

    def test_equality(self):
        # This is to test the __eq__ and __ne__ methods, so we can't use
        # assertEqual and assertNotEqual
        segment = Arc(0j, 100 + 50j, 0, 0, 0, 100 + 50j)
        self.assertTrue(segment == Arc(0j, 100 + 50j, 0, 0, 0, 100 + 50j))
        self.assertTrue(segment != Arc(0j, 100 + 50j, 0, 1, 0, 100 + 50j))


class TestPath(unittest.TestCase):
    def test_circle(self):
        arc1 = Arc(0j, 100 + 100j, 0, 0, 0, 200 + 0j)
        arc2 = Arc(200 + 0j, 100 + 100j, 0, 0, 0, 0j)
        path = Path(arc1, arc2)
        self.assertAlmostEqual(path.point(0.0), 0j)
        self.assertAlmostEqual(path.point(0.25), (100 + 100j))
        self.assertAlmostEqual(path.point(0.5), (200 + 0j))
        self.assertAlmostEqual(path.point(0.75), (100 - 100j))
        self.assertAlmostEqual(path.point(1.0), 0j)
        self.assertAlmostEqual(path.length(), pi * 200)

    def test_svg_specs(self):
        """The paths that are in the SVG specs"""

        # Big pie: M300,200 h-150 a150,150 0 1,0 150,-150 z
        path = Path(Line(300 + 200j, 150 + 200j),
                    Arc(150 + 200j, 150 + 150j, 0, 1, 0, 300 + 50j),
                    Line(300 + 50j, 300 + 200j))
        # The points and length for this path are calculated and not
        # regression tests.
        self.assertAlmostEqual(path.point(0.0), (300 + 200j))
        self.assertAlmostEqual(path.point(0.14897825542), (150 + 200j))
        self.assertAlmostEqual(path.point(0.5), (406.066017177 + 306.066017177j))
        self.assertAlmostEqual(path.point(1 - 0.14897825542), (300 + 50j))
        self.assertAlmostEqual(path.point(1.0), (300 + 200j))
        # The errors seem to accumulate. Still 6 decimal places is more
        # than good enough.
        self.assertAlmostEqual(path.length(), pi * 225 + 300, places=6)

        # Little pie: M275,175 v-150 a150,150 0 0,0 -150,150 z
        path = Path(Line(275 + 175j, 275 + 25j),
                    Arc(275 + 25j, 150 + 150j, 0, 0, 0, 125 + 175j),
                    Line(125 + 175j, 275 + 175j))
        # The points and length for this path are calculated and not
        # regression tests.
        self.assertAlmostEqual(path.point(0.0), (275 + 175j))
        self.assertAlmostEqual(path.point(0.2800495767557787), (275 + 25j))
        self.assertAlmostEqual(path.point(0.5),
                               (168.93398282201787 + 68.93398282201787j))
        self.assertAlmostEqual(path.point(1 - 0.2800495767557787), (125 + 175j))
        self.assertAlmostEqual(path.point(1.0), (275 + 175j))
        # The errors seem to accumulate. Still 6 decimal places is more
        # than good enough.
        self.assertAlmostEqual(path.length(), pi * 75 + 300, places=6)

    def test_repr(self):
        path = Path(
            Line(start=600 + 350j, end=650 + 325j),
            Arc(start=650 + 325j, radius=25 + 25j, rotation=-30,
                large_arc=0, sweep=1, end=700 + 300j),
            CubicBezier(start=700 + 300j, control1=800 + 400j,
                        control2=750 + 200j, end=600 + 100j),
            QuadraticBezier(start=600 + 100j, control=600, end=600 + 300j))
        self.assertEqual(eval(repr(path)), path)

    def test_equality(self):
        # This is to test the __eq__ and __ne__ methods, so we can't use
        # assertEqual and assertNotEqual
        path1 = Path(
            Line(start=600 + 350j, end=650 + 325j),
            Arc(start=650 + 325j, radius=25 + 25j, rotation=-30,
                large_arc=0, sweep=1, end=700 + 300j),
            CubicBezier(start=700 + 300j, control1=800 + 400j,
                        control2=750 + 200j, end=600 + 100j),
            QuadraticBezier(start=600 + 100j, control=600, end=600 + 300j))
        path2 = Path(
            Line(start=600 + 350j, end=650 + 325j),
            Arc(start=650 + 325j, radius=25 + 25j, rotation=-30,
                large_arc=0, sweep=1, end=700 + 300j),
            CubicBezier(start=700 + 300j, control1=800 + 400j,
                        control2=750 + 200j, end=600 + 100j),
            QuadraticBezier(start=600 + 100j, control=600, end=600 + 300j))

        self.assertTrue(path1 == path2)
        # Modify path2:
        path2[0][0]._start = 601 + 350j
        self.assertTrue(path1 != path2)

        # Modify back:
        path2[0][0]._start = 600 + 350j
        self.assertFalse(path1 != path2)

        # Get rid of the last segment:
        del path2[-1]
        self.assertFalse(path1 == path2)

        # It's not equal to a list of it's segments
        self.assertTrue(path1 != path1[:])
        self.assertFalse(path1 == path1[:])

    def test_path_constructor(self):
        """Test that contiguous segments end up glommed together in same
        subpath, etc"""

        a = Line(0, 1)
        b = Line(1, 1 + 1j)
        c = Line(3, 2)
        d = Line(2, 2 + 2j)
        e = Line(2 + 2j, 0)

        path = Path(a, b, c, d, e)

        self.assertEqual(len(path), 2)
        self.assertEqual(path[0], Subpath(a, b))
        self.assertEqual(path[1], Subpath(c, d, e))

        with self.assertRaises(ValueError):
            Subpath(a, b, c, d, e)

    def test_cropped(self):
        p_closed = Subpath(Line(0, 1), Line(1, 1 + 1j), Line(1 + 1j, 1j),
                           Line(1j, 0)).set_Z().path_of()
        first_half = Path(Line(0, 1), Line(1, 1 + 1j))
        second_half = Path(Line(1 + 1j, 1j), Line(1j, 0))
        middle_half = Path(Line(1, 1 + 1j), Line(1 + 1j, 1j))
        other_middle_half = Path(Line(1j, 0), Line(0, 1))
        self.assertTrue(p_closed.cropped(0, 0.5) == first_half)
        self.assertTrue(p_closed.cropped(1, 0.5) == first_half)
        self.assertTrue(p_closed.cropped(.5, 1) == second_half)
        self.assertTrue(p_closed.cropped(0.25, 0.75) == middle_half)
        self.assertTrue(p_closed.cropped(0.75, 0.25) == other_middle_half)
        with self.assertRaises(ValueError):
            p_closed.cropped(1, 0)
        with self.assertRaises(ValueError):
            p_closed.cropped(.5, 1.1)
        with self.assertRaises(ValueError):
            p_closed.cropped(-0.1, 0.1)

        p_open = Path(Line(0, 1), Line(1, 1 + 1j), Line(1 + 1j, 1j),
                      Line(1j, 2j))

        self.assertTrue(p_open.cropped(0, 0.5) == first_half)

        with self.assertRaises(ValueError):
            p_open.cropped(.75, .25)
        with self.assertRaises(ValueError):
            p_open.cropped(1, .25)
        with self.assertRaises(ValueError):
            p_open.cropped(1, 0)

    def test_transform_scale(self):
        line1 = Line(600.5 + 350.5j, 650.5 + 325.5j)
        arc1 = Arc(650 + 325j, 25 + 25j, -30, 0, 1, 700 + 300j)
        arc2 = Arc(650 + 325j, 30 + 25j, -30, 0, 0, 700 + 300j)
        cub1 = CubicBezier(650 + 325j, 25 + 25j, -30, 700 + 300j)
        cub2 = CubicBezier(700 + 300j, 800 + 400j, 750 + 200j, 600 + 100j)
        quad3 = QuadraticBezier(600 + 100j, 600, 600 + 300j)
        linez = Line(600 + 300j, 600 + 350j)
        line2 = Line(100j, 0)
        line3 = Line(0, 100)

        lilimine = Subpath(line2, line3)
        carrot = Subpath(cub1, cub2, quad3)
        bezpath = Path(line1, cub1, cub2, quad3)
        bezpathz = Path(line1, cub1, cub2, quad3, linez)
        path = Path(line1, arc1, cub2, quad3)
        pathz = Path(line1, arc1, cub2, quad3, linez)
        lpath = Path(linez)
        qpath = Path(quad3)
        cpath = Path(cub1)
        apath = Path(arc1, arc2)

        test_curves = [
            lilimine, carrot, bezpath, bezpathz, path, pathz, lpath,
            apath, line1, arc1, arc2, quad3, cub1, cub2, linez, qpath, cpath
        ]

        def scale_a_point(pt, sx, sy=None, origin=0j):
            if sy is None:
                sy = sx

            zeta = pt - origin
            pt_vec = [[zeta.real],
                      [zeta.imag],
                      [1]]
            transform = [[sx, 0, origin.real],
                         [0, sy, origin.imag]]

            return complex(*np.dot(transform, pt_vec).ravel())

        for curve in test_curves:
            # generate a random point and a random scaling
            t = np.random.rand()
            pt = curve.point(t)

            # random diagonal transformation
            sx = 2 * np.random.rand()
            sy = 2 * np.random.rand()

            # random origin
            origin = (10  * (np.random.rand() - 0.5) +
                      10j * (np.random.rand() - 0.5))

            # find seg which t lands on for failure reporting
            address = param2address(curve, t)
            if isinstance(curve, Path):
                assert address.subpath_index is not None
            seg = curve.segment_at_address(address)
            _fail_msg = "\nFailure!\non segment  {}\n".format(seg)

            # case where no `sy` and no `origin` given
            curve_scaled = curve.scaled(sx)
            res = curve_scaled.point(t)
            ans = scale_a_point(pt, sx, None)
            fail_msg = _fail_msg + ("curve.scaled({}, {}, {}) = \n{}\n"
                                    "".format(sx, None, None, curve_scaled))
            fail_msg += "seg_scaled.point({}) = {}\n".format(t, res)
            fail_msg += "ans = {}".format(ans)
            self.assertAlmostEqual(ans, res, places=4, msg=fail_msg)

            # case where random `origin` given but no `sy`
            ans = scale_a_point(pt, sx, None, origin)
            curve_scaled = curve.scaled(sx, origin=origin)
            res = curve_scaled.point(t)
            fail_msg = _fail_msg + ("curve.scaled({}, {}, {}) = \n{}\n"
                                    "".format(sx, None, origin, curve_scaled))
            fail_msg += "seg_scaled.point({}) = {}\n".format(t, res)
            fail_msg += "ans = {}".format(ans)
            self.assertAlmostEqual(ans, res, places=4, msg=fail_msg)

            # the following tests don't in general hold for curves with
            # > 1 segments
            if curve.num_segments() > 0:
                continue

            # case where `sx != sy`, and no `origin` given
            ans = scale_a_point(pt, sx, sy)
            curve_scaled = curve.scaled(sx, sy)
            res = curve_scaled.point(t)
            fail_msg = _fail_msg + ("curve.scaled({}, {}) = \n{}\n"
                                    "".format(sx, sy, curve_scaled))
            fail_msg += "seg_scaled.point({}) = {}\n".format(t, res)
            fail_msg += "ans = {}\n".format(ans)
            fail_msg += "pt = {}\n".format(pt)
            fail_msg += "curve._lengths: {}\n".format(curve._lengths)
            fail_msg += "scaled_curve._lengths: {}".format(curve_scaled._lengths)
            self.assertAlmostEqual(ans, res, places=4, msg=fail_msg)

            # case where `sx != sy`, and random `origin` given
            ans = scale_a_point(pt, sx, sy, origin)
            curve_scaled = curve.scaled(sx, sy, origin)
            res = curve_scaled.point(t)
            fail_msg = _fail_msg + ("curve.scaled({}, {}, {}) = \n{}\n"
                                    "".format(sx, sy, origin, curve_scaled))
            fail_msg += "seg_scaled.point({}) = {}\n".format(t, res)
            fail_msg += "ans = {}".format(ans)
            self.assertAlmostEqual(ans, res, places=4, msg=fail_msg)

        # more tests for scalar (i.e. `sx == sy`) case
        for curve in test_curves:
            # scale by 2 around (100, 100)
            o = 0
            scaled_curve = curve.scaled(2.0, origin=o)

            # expected length
            len_orig = curve.length()
            len_trns = scaled_curve.length()

            self.assertAlmostEqual(len_orig * 2.0, len_trns)

            # expected positions
            for T in np.linspace(0.0, 1.0, num=100):
                pt_orig = curve.point(T)
                pt_trns = scaled_curve.point(T)
                pt_xpct = (pt_orig - o) * 2.0 + o
                self.assertAlmostEqual(pt_xpct, pt_trns, delta=0.00001)

            # scale by 0.3 around (0, -100)
            # the 'almost equal' test fails at the 7th decimal place for
            # some length and position tests here.
            scaled_curve = curve.scaled(0.3, origin=complex(0, -100))

            # expected length
            len_orig = curve.length()
            len_trns = scaled_curve.length()
            self.assertAlmostEqual(len_orig * 0.3, len_trns, delta=0.000001)

            # expected positions
            for T in np.linspace(0.0, 1.0, num=100):
                pt_orig = curve.point(T)
                pt_trns = scaled_curve.point(T)
                pt_xpct = (pt_orig - complex(0, -100)) * 0.3 + complex(0, -100)
                self.assertAlmostEqual(pt_xpct, pt_trns, delta=0.000001)


class Test_ilength(unittest.TestCase):
    # See svgpathtools.notes.inv_arclength.py for information on how these
    # test values were generated (using the .length() method).
    ##############################################################

    def test_ilength_lines(self):
        l = Line(1, 3 - 1j)
        # nodall = Line(1 + 1j, 1 + 1j)

        tests = [
            (l, 0.01, 0.022360679774997897),
            (l, 0.1, 0.223606797749979),
            (l, 0.5, 1.118033988749895),
            (l, 0.9, 2.012461179749811),
            (l, 0.99, 2.213707297724792)]

        for (l, t, s) in tests:
            self.assertAlmostEqual(l.ilength(s).t, t)

    def test_ilength_quadratics(self):
        q1 = QuadraticBezier(200 + 300j, 400 + 50j, 600 + 300j)
        q2 = QuadraticBezier(200 + 300j, 400 + 50j, 500 + 200j)
        closedq = QuadraticBezier(6 + 2j, 5 - 1j, 6 + 2j)
        linq = QuadraticBezier(1 + 3j, 2 + 5j, -9 - 17j)
        # nodalq = QuadraticBezier(1, 1, 1)

        tests = [
            (q1, 0.01, 6.364183310105577),
            (q1, 0.1, 60.23857499635088),
            (q1, 0.5, 243.8855469477619),
            (q1, 0.9, 427.53251889917294),
            (q1, 0.99, 481.40691058541813),
            (q2, 0.01, 6.365673533661836),
            (q2, 0.1, 60.31675895732397),
            (q2, 0.5, 233.24592830045907),
            (q2, 0.9, 346.42891253298706),
            (q2, 0.99, 376.32659156736844),
            (closedq, 0.01, 0.06261309767133393),
            (closedq, 0.1, 0.5692099788303084),
            (closedq, 0.5, 1.5811388300841898),
            (closedq, 0.9, 2.5930676813380713),
            (closedq, 0.99, 3.0996645624970456),
            (linq, 0.01, 0.04203807797699605),
            (linq, 0.1, 0.19379255804998186),
            (linq, 0.5, 4.844813951249544),
            (linq, 0.9, 18.0823363780483),
            (linq, 0.99, 22.24410609777091)]

        for q, t, s in tests:
            try:
                self.assertAlmostEqual(q.ilength(s).t, t)
            except:
                print(q)
                print(s)
                print(t)
                raise

    def test_ilength_cubics(self):
        c1 = CubicBezier(200 + 300j, 400 + 50j, 600 + 100j, -200)
        symc = CubicBezier(1 - 2j, 10 - 1j, 10 + 1j, 1 + 2j)
        closedc = CubicBezier(1 - 2j, 10 - 1j, 10 + 1j, 1 - 2j)

        tests = [(c1, 0.01, 9.53434737943073),
                 (c1, 0.1, 88.89941848775852),
                 (c1, 0.5, 278.5750942713189),
                 (c1, 0.9, 651.4957786584646),
                 (c1, 0.99, 840.2010603832538),
                 (symc, 0.01, 0.2690118556702902),
                 (symc, 0.1, 2.45230693868727),
                 (symc, 0.5, 7.256147083644424),
                 (symc, 0.9, 12.059987228602886),
                 (symc, 0.99, 14.243282311619401),
                 (closedc, 0.01, 0.26901140075538765),
                 (closedc, 0.1, 2.451722765460998),
                 (closedc, 0.5, 6.974058969750422),
                 (closedc, 0.9, 11.41781741489913),
                 (closedc, 0.99, 13.681324783697782)]

        for (c, t, s) in tests:
            self.assertAlmostEqual(c.ilength(s).t, t)

    def test_ilength_arcs(self):
        arc1 = Arc(0j, 100 + 50j, 0, 0, 0, 100 + 50j)
        arc2 = Arc(0j, 100 + 50j, 0, 1, 0, 100 + 50j)
        arc3 = Arc(0j, 100 + 50j, 0, 0, 1, 100 + 50j)
        arc4 = Arc(0j, 100 + 50j, 0, 1, 1, 100 + 50j)
        arc5 = Arc(0j, 100 + 100j, 0, 0, 0, 200 + 0j)
        arc6 = Arc(200 + 0j, 100 + 100j, 0, 0, 0, 0j)
        arc7 = Arc(0j, 100 + 50j, 0, 0, 0, 100 + 50j)

        tests = [(arc1, 0.01, 0.785495042476231),
                 (arc1, 0.1, 7.949362877455911),
                 (arc1, 0.5, 48.28318721111137),
                 (arc1, 0.9, 105.44598206942156),
                 (arc1, 0.99, 119.53485487631241),
                 (arc2, 0.01, 4.71108115728524),
                 (arc2, 0.1, 45.84152747676626),
                 (arc2, 0.5, 169.38878996795734),
                 (arc2, 0.9, 337.44707303579696),
                 (arc2, 0.99, 360.95800139278765),
                 (arc3, 0.01, 1.5707478805335624),
                 (arc3, 0.1, 15.659620687424416),
                 (arc3, 0.5, 72.82241554573457),
                 (arc3, 0.9, 113.15623987939003),
                 (arc3, 0.99, 120.3201077143697),
                 (arc4, 0.01, 2.3588068777503897),
                 (arc4, 0.1, 25.869735234740887),
                 (arc4, 0.5, 193.9280183025816),
                 (arc4, 0.9, 317.4752807937718),
                 (arc4, 0.99, 358.6057271132536),
                 (arc5, 0.01, 3.141592653589793),
                 (arc5, 0.1, 31.415926535897935),
                 (arc5, 0.5, 157.07963267948966),
                 (arc5, 0.9, 282.7433388230814),
                 (arc5, 0.99, 311.01767270538954),
                 (arc6, 0.01, 3.141592653589793),
                 (arc6, 0.1, 31.415926535897928),
                 (arc6, 0.5, 157.07963267948966),
                 (arc6, 0.9, 282.7433388230814),
                 (arc6, 0.99, 311.01767270538954),
                 (arc7, 0.01, 0.785495042476231),
                 (arc7, 0.1, 7.949362877455911),
                 (arc7, 0.5, 48.28318721111137),
                 (arc7, 0.9, 105.44598206942156),
                 (arc7, 0.99, 119.53485487631241)]

        for (c, t, s) in tests:
            self.assertAlmostEqual(c.ilength(s).t, t)

    def test_ilength_paths(self):
        line1 = Line(600 + 350j, 650 + 325j)
        arc1 = Arc(650 + 325j, 25 + 25j, -30, 0, 1, 700 + 300j)
        cub1 = CubicBezier(650 + 325j, 25 + 25j, -30, 700 + 300j)
        cub2 = CubicBezier(700 + 300j, 800 + 400j, 750 + 200j, 600 + 100j)
        quad3 = QuadraticBezier(600 + 100j, 600, 600 + 300j)
        linez = Line(600 + 300j, 600 + 350j)

        bezpath = Path(line1, cub1, cub2, quad3)
        bezpathz = Path(line1, cub1, cub2, quad3, linez)
        path = Path(line1, arc1, cub2, quad3)
        pathz = Path(line1, arc1, cub2, quad3, linez)
        lpath = Path(linez)
        qpath = Path(quad3)
        cpath = Path(cub1)
        apath = Path(arc1)

        tests = [(bezpath, 0.0, 0.0),
                 (bezpath, 0.1111111111111111, 286.2533595149515),
                 (bezpath, 0.2222222222222222, 503.8620222915423),
                 (bezpath, 0.3333333333333333, 592.6337135346268),
                 (bezpath, 0.4444444444444444, 644.3880677233315),
                 (bezpath, 0.5555555555555556, 835.0384185011363),
                 (bezpath, 0.6666666666666666, 1172.8729938994575),
                 (bezpath, 0.7777777777777778, 1308.6205983178952),
                 (bezpath, 0.8888888888888888, 1532.8473168900994),
                 (bezpath, 1.0, 1758.2427369258733),
                 (bezpathz, 0.0, 0.0),
                 (bezpathz, 0.1111111111111111, 294.15942308605435),
                 (bezpathz, 0.2222222222222222, 512.4295461513882),
                 (bezpathz, 0.3333333333333333, 594.0779370040138),
                 (bezpathz, 0.4444444444444444, 658.7361976564598),
                 (bezpathz, 0.5555555555555556, 874.1674336581542),
                 (bezpathz, 0.6666666666666666, 1204.2371344392693),
                 (bezpathz, 0.7777777777777778, 1356.773042865213),
                 (bezpathz, 0.8888888888888888, 1541.808492602876),
                 (bezpathz, 1.0, 1808.2427369258733),
                 (path, 0.0, 0.0),
                 (path, 0.1111111111111111, 81.44016397108298),
                 (path, 0.2222222222222222, 164.72556816469307),
                 (path, 0.3333333333333333, 206.71343564679154),
                 (path, 0.4444444444444444, 265.4898349999353),
                 (path, 0.5555555555555556, 367.5420981413199),
                 (path, 0.6666666666666666, 487.29863861165995),
                 (path, 0.7777777777777778, 511.84069655405284),
                 (path, 0.8888888888888888, 579.9530841780238),
                 (path, 1.0, 732.9614757397469),
                 (pathz, 0.0, 0.0),
                 (pathz, 0.1111111111111111, 86.99571952663854),
                 (pathz, 0.2222222222222222, 174.33662608180325),
                 (pathz, 0.3333333333333333, 214.42194393858466),
                 (pathz, 0.4444444444444444, 289.94661033436205),
                 (pathz, 0.5555555555555556, 408.38391100702125),
                 (pathz, 0.6666666666666666, 504.4309373835351),
                 (pathz, 0.7777777777777778, 533.774834546298),
                 (pathz, 0.8888888888888888, 652.931321760894),
                 (pathz, 1.0, 782.9614757397469),
                 (lpath, 0.0, 0.0),
                 (lpath, 0.1111111111111111, 5.555555555555555),
                 (lpath, 0.2222222222222222, 11.11111111111111),
                 (lpath, 0.3333333333333333, 16.666666666666664),
                 (lpath, 0.4444444444444444, 22.22222222222222),
                 (lpath, 0.5555555555555556, 27.77777777777778),
                 (lpath, 0.6666666666666666, 33.33333333333333),
                 (lpath, 0.7777777777777778, 38.88888888888889),
                 (lpath, 0.8888888888888888, 44.44444444444444),
                 (lpath, 1.0, 50.0),
                 (qpath, 0.0, 0.0),
                 (qpath, 0.1111111111111111, 17.28395061728395),
                 (qpath, 0.2222222222222222, 24.69135802469136),
                 (qpath, 0.3333333333333333, 27.777777777777786),
                 (qpath, 0.4444444444444444, 40.12345679012344),
                 (qpath, 0.5555555555555556, 62.3456790123457),
                 (qpath, 0.6666666666666666, 94.44444444444446),
                 (qpath, 0.7777777777777778, 136.41975308641975),
                 (qpath, 0.8888888888888888, 188.27160493827154),
                 (qpath, 1.0, 250.0),
                 (cpath, 0.0, 0.0),
                 (cpath, 0.1111111111111111, 207.35525375551356),
                 (cpath, 0.2222222222222222, 366.0583590267552),
                 (cpath, 0.3333333333333333, 474.34064293812787),
                 (cpath, 0.4444444444444444, 530.467036317684),
                 (cpath, 0.5555555555555556, 545.0444351253911),
                 (cpath, 0.6666666666666666, 598.9767847757622),
                 (cpath, 0.7777777777777778, 710.4080903390646),
                 (cpath, 0.8888888888888888, 881.1796899225557),
                 (cpath, 1.0, 1113.0914444911352),
                 (apath, 0.0, 0.0),
                 (apath, 0.1111111111111111, 9.756687033889872),
                 (apath, 0.2222222222222222, 19.51337406777974),
                 (apath, 0.3333333333333333, 29.27006110166961),
                 (apath, 0.4444444444444444, 39.02674813555948),
                 (apath, 0.5555555555555556, 48.783435169449355),
                 (apath, 0.6666666666666666, 58.54012220333922),
                 (apath, 0.7777777777777778, 68.2968092372291),
                 (apath, 0.8888888888888888, 78.05349627111896),
                 (apath, 1.0, 87.81018330500885)]

        for (c, W, s) in tests:
            try:
                self.assertAlmostEqual(c.ilength(s).W, W, msg=str((c, W, s)))

            except:
                # These test case values were generated using a system
                # with scipy installed -- if scipy is not installed,
                # then in cases where `t == 1`, `s` may be slightly
                # greater than the length computed previously.
                # Thus this try/except block exists as a workaround.
                if c.length() < s:
                    with self.assertRaises(ValueError):
                        c.ilength(s)
                else:
                    raise

    # Exceptional Cases
    def test_ilength_exceptions(self):
        nodalq = QuadraticBezier(1, 1, 1)
        with self.assertRaises(AssertionError):
            nodalq.ilength(1)

        lin = Line(0, 0.5j)
        with self.assertRaises(ValueError):
            lin.ilength(1)


class Test_intersect(unittest.TestCase):
    def test_intersect(self):
        ###################################################################
        # test that `some_seg.intersect(another_seg)` will produce properly
        # ordered tuples, i.e. the first element in each tuple refers to
        # `some_seg` and the second element refers to `another_seg`.
        # Also tests that the correct number of intersections is found.
        a = Line(0 + 200j, 300 + 200j)
        b = QuadraticBezier(40 + 150j, 70 + 200j, 210 + 300j)
        c = CubicBezier(60 + 150j, 40 + 200j, 120 + 250j, 200 + 160j)
        d = Arc(70 + 150j, 50 + 100j, 0, 0, 0, 200 + 100j)

        segdict = {'line': a, 'quadratic': b, 'cubic': c, 'arc': d}

        # test each segment type against each other type
        for a, b in [(x, y) for x in segdict for y in segdict]:
            if a is b:
                continue
            x = segdict[a]
            y = segdict[b]
            xiy = sorted(x.intersect(y, tol=1e-15), key=(lambda z: z[0].t))
            yix = sorted(y.intersect(x, tol=1e-15), key=(lambda z: z[1].t))
            self.assertEqual(len(xiy), len(yix))
            for xy, yx in zip(xiy, yix):
                self.assertAlmostEqual(xy[0].t, yx[1].t)
                self.assertAlmostEqual(xy[1].t, yx[0].t)
                self.assertAlmostEqual(x.point(xy[0]), y.point(yx[0]))
            if {a, b} == {'line', 'quadratic'}:
                self.assertEqual(len(xiy), 1)
            else:
                if len(xiy) != 2:
                    print("a, b, len():", a, b, len(xiy))
                self.assertEqual(len(xiy), 2)

        # test each segment against another segment of same type
        for x in segdict:
            if x == 'arc':
                # this is an example of the Arc.intersect method not working
                # in call cases.  See docstring for a note on its
                # incomplete implementation.
                continue
            x = segdict[x]
            o = x.center if isinstance(x, Arc) else x.point(0.5)
            y = x.rotated(90, origin=o).translated(5)
            xiy = sorted(x.intersect(y, tol=1e-15), key=(lambda z: z[0].t))
            yix = sorted(y.intersect(x, tol=1e-15), key=(lambda z: z[1].t))
            for xy, yx in zip(xiy, yix):
                self.assertAlmostEqual(xy[0].t, yx[1].t)
                self.assertAlmostEqual(xy[1].t, yx[0].t)
                self.assertAlmostEqual(x.point(xy[0]), y.point(yx[0]))
            self.assertEqual(len(xiy), len(yix))
            self.assertEqual(len(xiy), 1)
            self.assertEqual(len(yix), 1)
        ###################################################################

    def test_line_line_0(self):
        l0 = Line(start=(25.389999999999997 + 99.989999999999995j),
                  end=(25.389999999999997 + 90.484999999999999j))
        l1 = Line(start=(25.390000000000001 + 84.114999999999995j),
                  end=(25.389999999999997 + 74.604202137430320j))
        i = l0.intersect(l1)
        self.assertEqual(len(i), 0)

    def test_line_line_1(self):
        l0 = Line(start=(-124.705378549 + 327.696674827j),
                  end=(12.4926214511 + 121.261674827j))
        l1 = Line(start=(-12.4926214511 + 121.261674827j),
                  end=(124.705378549 + 327.696674827j))
        i = l0.intersect(l1)
        self.assertEqual(len(i), 1)
        self.assertLess(abs(l0.point(i[0][0]) - l1.point(i[0][1])), 1e-9)


class TestPathTools(unittest.TestCase):
    # moved from test_pathtools.py

    def setUp(self):
        self.arc1 = Arc(650 + 325j, 25 + 25j, -30.0, False, True, 700 + 300j)
        self.line1 = Line(0, 100 + 100j)
        self.quadratic1 = QuadraticBezier(100 + 100j, 150 + 150j, 300 + 200j)
        self.cubic1 = CubicBezier(300 + 200j, 350 + 400j, 400 + 425j, 650 + 325j)
        self.path_of_all_seg_types = Path(self.line1, self.quadratic1,
                                          self.cubic1, self.arc1)
        self.path_of_bezier_seg_types = Path(self.line1, self.quadratic1,
                                             self.cubic1)

    def test_is_bezier_segment(self):
        # False
        self.assertFalse(isinstance(self.arc1, BezierSegment))
        self.assertFalse(isinstance(self.path_of_bezier_seg_types, BezierSegment))

        # True
        self.assertTrue(isinstance(self.line1, BezierSegment))
        self.assertTrue(isinstance(self.quadratic1, BezierSegment))
        self.assertTrue(isinstance(self.cubic1, BezierSegment))

    def test_is_bezier_path(self):
        # False
        self.assertFalse(self.path_of_all_seg_types.is_bezier_path())

        # True
        self.assertTrue(self.path_of_bezier_seg_types.is_bezier_path())
        self.assertTrue(Path().is_bezier_path())

    def test_polynomial2bezier(self):
        def distfcn(tup1, tup2):
            assert len(tup1) == len(tup2)
            return sum((tup1[i] - tup2[i])**2 for i in range(len(tup1)))**0.5

        # Case: Line
        pcoeffs = [(-1.7 - 2j), (6 + 2j)]
        p = np.poly1d(pcoeffs)
        correct_bpoints = [(6 + 2j), (4.3 + 0j)]

        # Input np.poly1d object
        bez = poly2bez(p)
        bpoints = bez.bpoints
        self.assertAlmostEqual(distfcn(bpoints, correct_bpoints), 0)

        # Input list of coefficients
        bpoints = poly2bez(pcoeffs, return_bpoints=True)
        self.assertAlmostEqual(distfcn(bpoints, correct_bpoints), 0)

        # Case: Quadratic
        pcoeffs = [(29.5 + 15.5j), (-31 - 19j), (7.5 + 5.5j)]
        p = np.poly1d(pcoeffs)
        correct_bpoints = [(7.5 + 5.5j), (-8 - 4j), (6 + 2j)]

        # Input np.poly1d object
        bez = poly2bez(p)
        bpoints = bez.bpoints
        self.assertAlmostEqual(distfcn(bpoints, correct_bpoints), 0)

        # Input list of coefficients
        bpoints = poly2bez(pcoeffs, return_bpoints=True)
        self.assertAlmostEqual(distfcn(bpoints, correct_bpoints), 0)

        # Case: Cubic
        pcoeffs = [(-18.5 - 12.5j), (34.5 + 16.5j), (-18 - 6j), (6 + 2j)]
        p = np.poly1d(pcoeffs)
        correct_bpoints = [(6 + 2j), 0j, (5.5 + 3.5j), (4 + 0j)]

        # Input np.poly1d object
        bez = poly2bez(p)
        bpoints = bez.bpoints
        self.assertAlmostEqual(distfcn(bpoints, correct_bpoints), 0)

        # Input list of coefficients object
        bpoints = poly2bez(pcoeffs, return_bpoints=True)
        self.assertAlmostEqual(distfcn(bpoints, correct_bpoints), 0)

    def test_bpoints2bezier(self):
        cubic_bpoints = [(6 + 2j), 0, (5.5 + 3.5j), (4 + 0j)]
        quadratic_bpoints = [(6 + 2j), 0, (5.5 + 3.5j)]
        line_bpoints = [(6 + 2j), 0]
        self.assertTrue(isinstance(bpoints2bezier(cubic_bpoints), CubicBezier))
        self.assertTrue(isinstance(bpoints2bezier(quadratic_bpoints),
                                   QuadraticBezier))
        self.assertTrue(isinstance(bpoints2bezier(line_bpoints), Line))
        self.assertSequenceEqual(bpoints2bezier(cubic_bpoints).bpoints,
                                 cubic_bpoints)
        self.assertSequenceEqual(bpoints2bezier(quadratic_bpoints).bpoints,
                                 quadratic_bpoints)
        self.assertSequenceEqual(bpoints2bezier(line_bpoints).bpoints,
                                 line_bpoints)

    def test_closest_point_in_path(self):
        # Note: currently the radiialrange method is not implemented for Arc
        # objects
        # test_path = self.path_of_all_seg_types
        # origin = -123 - 123j
        # expected_result = ???
        # self.assertAlmostEqual(min_radius(origin, test_path),
        # expected_result)

        # generic case (where is_bezier_path(test_path) == True)
        test_path = self.path_of_bezier_seg_types
        pt = 300 + 300j
        expected_distance = 29.382522853493143
        expected_point = 327.28326882229123 + 289.0933097776349j
        distance, address = test_path.closest_point_to(pt)
        self.assertAlmostEqual(distance, expected_distance)
        self.assertAlmostEqual(test_path.point(address), expected_point)

        # cubic test with multiple valid solutions
        test_path = Path(CubicBezier(1 - 2j, 10 - 1j, 10 + 1j, 1 + 2j))
        pt = 3
        expected_distance = 1.7191878932122302
        expected_points = [3.270512052592931 + 1.6977721406506427j,
                           3.270512052592931 - 1.6977721406506427j]
        distance, address = test_path.closest_point_to(pt)
        self.assertAlmostEqual(distance, expected_distance)
        self.assertAlmostEqual(min(abs(expected_points[0] - test_path.point(address)),
                                   abs(expected_points[1] - test_path.point(address))), 0)

    def test_farthest_point_in_path(self):
        # Note: currently the radialrange method is not implemented for Arc
        # objects
        # test_path = self.path_of_all_seg_types
        # origin = -123 - 123j
        # expected_result = ???
        # self.assertAlmostEqual(min_radius(origin, test_path),
        # expected_result)

        # boundary test
        test_path = self.path_of_bezier_seg_types
        pt = 300 + 300j
        expected_distance = 424.26406871192853
        expected_point = test_path.point(0)
        # distance, address = test_path.farthest_point_from(pt)
        distance, address = test_path.farthest_point_from(pt)
        self.assertAlmostEqual(distance, expected_distance)
        self.assertAlmostEqual(test_path.point(address), expected_point)

        # non-boundary test
        test_path = Path(CubicBezier(1 - 2j, 10 - 1j, 10 + 1j, 1 + 2j))
        pt = 3
        expected_distance = 4.75
        expected_point = test_path.point(0.5)
        distance, address = test_path.farthest_point_from(pt)
        self.assertAlmostEqual(distance, expected_distance)
        self.assertAlmostEqual(test_path.point(address), expected_point)

    def test_path_encloses_pt(self):
        line1 = Line(0, 100 + 100j)
        quadratic1 = QuadraticBezier(100 + 100j, 150 + 150j, 300 + 200j)
        cubic1 = CubicBezier(300 + 200j, 350 + 400j, 400 + 425j, 650 + 325j)
        line2 = Line(650 + 325j, 650 + 10j)
        line3 = Line(650 + 10j, 0)
        open_bez_path = Path(line1, quadratic1, cubic1)
        closed_bez_path = Subpath(line1, quadratic1, cubic1, line2, line3).set_Z().path_of()

        inside_pt = 200 + 20j
        outside_pt1 = 1000 + 1000j
        outside_pt2 = 800 + 800j
        # boundary_pt = 50 + 50j

        # Note: currently the intersect() method is not implemented for Arc
        # objects
        # arc1 = Arc(650+325j, 25+25j, -30.0, False, True, 700+300j)
        # closed_path_with_arc = Path(line1, quadratic1, cubic1, arc1)
        # self.assertTrue(
        #     path_encloses_pt(inside_pt, outside_pt2, closed_path_with_arc))

        # True cases
        self.assertTrue(
            closed_bez_path.even_odd_encloses(inside_pt))
        self.assertTrue(
            closed_bez_path.even_odd_encloses(inside_pt))

        # False cases
        self.assertFalse(
            closed_bez_path.even_odd_encloses(outside_pt1))
        self.assertFalse(
            closed_bez_path.even_odd_encloses(outside_pt2))

        # Exception Cases
        with self.assertRaises(ValueError):
            open_bez_path.even_odd_encloses(inside_pt)
            # path_encloses_pt(inside_pt, outside_pt2, open_bez_path)

        # Display test paths and points
        # ns2d = [inside_pt, outside_pt1, outside_pt2, boundary_pt]
        # ncolors = ['green', 'red', 'orange', 'purple']
        # disvg(closed_path_with_arc, nodes=ns2d, node_colors=ncolors,
        #       openinbrowser=True)
        # disvg(open_bez_path, nodes=ns2d, node_colors=ncolors,
        #       openinbrowser=True)
        # disvg(closed_bez_path, nodes=ns2d, node_colors=ncolors,
        #       openinbrowser=True)


class TestStroke(unittest.TestCase):
    def setUp(self):
        pass

    def test_joins(self):
        p = Path(Line(0, 0.5 + 1j), Line(0.5 + 1j, 1))
        answers = {
            1:
                Path(
                    Subpath(
                        Line(
                            0.22360679774997896 - 0.11180339887498948j,
                            0.5 + 0.44098300562505266j
                        ),
                        Line(
                            0.5 + 0.44098300562505266j,
                            0.7763932022500211 - 0.11180339887498948j
                        ),
                        Line(
                            0.7763932022500211 - 0.11180339887498948j,
                            1.223606797749979 + 0.11180339887498948j
                        ),
                        Line(
                            1.223606797749979 + 0.11180339887498948j,
                            0.7236067977499789 + 1.1118033988749896j
                        ),
                        Line(
                            0.7236067977499789 + 1.1118033988749896j,
                            0.27639320225002106 + 1.1118033988749896j
                        ),
                        Line(
                            0.27639320225002106 + 1.1118033988749896j,
                            -0.22360679774997896 + 0.11180339887498948j
                        ),
                        Line(
                            -0.22360679774997896 + 0.11180339887498948j,
                            0.22360679774997896 - 0.11180339887498948j
                        )
                    ).set_Z()
                ),

            4:
                Path(
                    Subpath(
                        Line(
                            0.22360679774997896 - 0.11180339887498948j,
                            0.5 + 0.44098300562505266j
                        ),
                        Line(
                            0.5 + 0.44098300562505266j,
                            0.7763932022500211 - 0.11180339887498948j
                        ),
                        Line(
                            0.7763932022500211 - 0.11180339887498948j,
                            1.223606797749979 + 0.11180339887498948j
                        ),
                        Line(
                            1.223606797749979 + 0.11180339887498948j,
                            0.7236067977499789 + 1.1118033988749896j
                        ),
                        Line(
                            0.7236067977499789 + 1.1118033988749896j,
                            0.5000000000000001 + 1.5590169943749472j
                        ),
                        Line(
                            0.5000000000000001 + 1.5590169943749472j,
                            0.27639320225002106 + 1.1118033988749896j
                        ),
                        Line(
                            0.27639320225002106 + 1.1118033988749896j,
                            -0.22360679774997896 + 0.11180339887498948j
                        ),
                        Line(
                            -0.22360679774997896 + 0.11180339887498948j,
                            0.22360679774997896 - 0.11180339887498948j
                        )
                    ).set_Z()
                ),

            'round':
                Path(
                    Subpath(
                        Line(
                            0.22360679774997896 - 0.11180339887498948j,
                            0.5 + 0.44098300562505266j
                        ),
                        Line(
                            0.5 + 0.44098300562505266j,
                            0.7763932022500211 - 0.11180339887498948j
                        ),
                        Line(
                            0.7763932022500211 - 0.11180339887498948j,
                            1.223606797749979 + 0.11180339887498948j
                        ),
                        Line(
                            1.223606797749979 + 0.11180339887498948j,
                            0.7236067977499789 + 1.1118033988749896j
                        ),
                        Arc(
                            0.7236067977499789 + 1.1118033988749896j,
                            0.25 + 0.25j,
                            0,
                            False,
                            True,
                            0.27639320225002106 + 1.1118033988749896j
                        ),
                        Line(
                            0.27639320225002106 + 1.1118033988749896j,
                            -0.22360679774997896 + 0.11180339887498948j
                        ),
                        Line(
                            -0.22360679774997896 + 0.11180339887498948j,
                            0.22360679774997896 - 0.11180339887498948j
                        )
                    ).set_Z()
                )
        }
        for (key, item) in answers.items():
            if isinstance(key, Number):
                q = p.stroke(0.5, miter_limit=key)
            else:
                q = p.stroke(0.5, join=key)
            self.assertTrue(q.__eq__(answers[key], 1e-12))

    def test_caps(self):
        p = Path(Line(0, 1j), Line(1j, 1 + 1j))
        answers = {
            'butt':
                Path(
                    Subpath(
                        Line(0.25 + 0j, 0.25 + 0.75j),
                        Line(0.25 + 0.75j, 1 + 0.75j),
                        Line(1 + 0.75j, 1 + 1.25j),
                        Line(1 + 1.25j, 1.25j),
                        Line(1.25j, -0.25 + 1.25j),
                        Line(-0.25 + 1.25j, -0.25 + 1j),
                        Line(-0.25 + 1j, -0.25 + 0j),
                        Line(-0.25 + 0j, 0.25 + 0j)
                    ).set_Z()
                ),
            'round':
                Path(
                    Subpath(
                        Line(0.25 + 0j, 0.25 + 0.75j),
                        Line(0.25 + 0.75j, 1 + 0.75j),
                        Arc(1 + 0.75j, 0.25 + 0.25j, 0, False, True, 1.25 + 1j),
                        Arc(1.25 + 1j, 0.25 + 0.25j, 0, False, True, 1 + 1.25j),
                        Line(1 + 1.25j, 1.25j),
                        Line(1.25j, -0.25 + 1.25j),
                        Line(-0.25 + 1.25j, -0.25 + 1j),
                        Line(-0.25 + 1j, -0.25 + 0j),
                        Arc(-0.25 + 0j, 0.25 + 0.25j, 0, False, True, -0.25j),
                        Arc(-0.25j, 0.25 + 0.25j, 0, False, True, 0.25 + 0j)
                    ).set_Z()
                ),
            'square':
                Path(
                    Subpath(
                        Line(0.25 + 0j, 0.25 + 0.75j),
                        Line(0.25 + 0.75j, 1 + 0.75j),
                        Line(1 + 0.75j, 1.25 + 0.75j),
                        Line(1.25 + 0.75j, 1.25 + 1.25j),
                        Line(1.25 + 1.25j, 1 + 1.25j),
                        Line(1 + 1.25j, 1.25j),
                        Line(1.25j, -0.25 + 1.25j),
                        Line(-0.25 + 1.25j, -0.25 + 1j),
                        Line(-0.25 + 1j, -0.25 + 0j),
                        Line(-0.25 + 0j, -0.25 - 0.25j),
                        Line(-0.25 - 0.25j, 0.25 - 0.25j),
                        Line(0.25 - 0.25j, 0.25 + 0j)
                    ).set_Z()
                )
        }
        for cap in ['butt', 'round', 'square']:
            stroke = p.stroke(0.5, cap=cap)
            self.assertTrue(stroke.__eq__(answers[cap], 1e-12))

    def test_complex_stroke(self):
        p = parse_path('M 0,0 0,2 2,2 2,1 A 1,2 45 1 0 2,0 M 5,0 l 0,2 2,0 0,-2 Z m 3,0 l 0,2 2,0 0,-2 Z')

        answer = Path(
            Subpath(
                Line(
                    0.125 + 0j,
                    0.125 + 1.875j
                ),
                Line(
                    0.125 + 1.875j,
                    1.875 + 1.875j
                ),
                Line(
                    1.875 + 1.875j,
                    1.875 + 1j
                ),
                Arc(
                    1.875 + 1j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    2.0978820228628625 + 0.9222561282140254j
                ),
                CubicBezier(
                    2.0978820228628625 + 0.9222561282140254j,
                    2.3205481727120874 + 1.2025999375669743j,
                    2.988121061619792 + 1.1177576986469826j,
                    3.6954937452349133 + 0.5767408933374979j
                ),
                CubicBezier(
                    3.6954937452349133 + 0.5767408933374979j,
                    4.402866428850035 + 0.0357240880280133j,
                    4.9085075849119715 - 0.7960114356570552j,
                    4.9085075849119715 - 1.3201045509471827j
                ),
                CubicBezier(
                    4.9085075849119715 - 1.3201045509471827j,
                    4.9085075849119715 - 1.5821511085922468j,
                    4.811086603957486 - 1.7196610121455456j,
                    4.632427639941231 - 1.7878424400655046j
                ),
                CubicBezier(
                    4.632427639941231 - 1.7878424400655046j,
                    4.453768675924975 - 1.8560238679854641j,
                    4.168349986797831 - 1.8381540001259937j,
                    3.8213110337557974 - 1.6871315653889372j
                ),
                CubicBezier(
                    3.8213110337557974 - 1.6871315653889372j,
                    3.1272331276717287 - 1.3850866959148236j,
                    2.3950721789340106 - 0.6396716102022778j,
                    2.1157915954889828 + 0.04708828319354834j
                ),
                Arc(
                    2.1157915954889828 + 0.04708828319354834j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    1.9529117168064516 + 0.11579159548898267j
                ),
                Arc(
                    1.9529117168064516 + 0.11579159548898267j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    1.8842084045110172 - 0.04708828319354834j
                ),
                CubicBezier(
                    1.8842084045110172 - 0.04708828319354834j,
                    2.195835552333496 - 0.8133894136834745j,
                    2.9514932638632168 - 1.5812555913799502j,
                    3.721553996291689 - 1.9163662343995045j
                ),
                CubicBezier(
                    3.721553996291689 - 1.9163662343995045j,
                    4.106584362505925 - 2.0839215559092814j,
                    4.452203208322896 - 2.124207804425575j,
                    4.721564449436687 - 2.0214117642349487j
                ),
                CubicBezier(
                    4.721564449436687 - 2.0214117642349487j,
                    4.990925690550477 - 1.9186157240443218j,
                    5.1585075849119715 - 1.6560145750006847j,
                    5.1585075849119715 - 1.3201045509471827j
                ),
                CubicBezier(
                    5.1585075849119715 - 1.3201045509471827j,
                    5.1585075849119715 - 0.6482845028401784j,
                    4.604137239709988 + 0.19652482198206755j,
                    3.8473712848125694 + 0.775318870393962j
                ),
                CubicBezier(
                    3.8473712848125694 + 0.775318870393962j,
                    3.218470181177721 + 1.2563185949030597j,
                    2.54572840536869 + 1.4669148198187465j,
                    2.1249999999999996 + 1.2582566813778187j
                ),
                Line(
                    2.1249999999999996 + 1.2582566813778187j,
                    2.125 + 2j
                ),
                Arc(
                    2.125 + 2j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    2 + 2.125j
                ),
                Line(
                    2 + 2.125j,
                    2.125j
                ),
                Arc(
                    2.125j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    -0.125 + 2j
                ),
                Line(
                    -0.125 + 2j,
                    -0.125 + 0j
                ),
                Arc(
                    -0.125 + 0j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    -0.125j
                ),
                Arc(
                    -0.125j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    0.125 + 0j
                )
            ).set_Z(),
            Subpath(
                Line(
                    5.125 + 0.125j,
                    5.125 + 1.875j
                ),
                Line(
                    5.125 + 1.875j,
                    6.875 + 1.875j
                ),
                Line(
                    6.875 + 1.875j,
                    6.875 + 0.12499999999999997j
                ),
                Line(
                    6.875 + 0.12499999999999997j,
                    5.125 + 0.125j
                )
            ).set_Z(),
            Subpath(
                Line(
                    5 - 0.125j,
                    7 - 0.125j
                ),
                Arc(
                    7 - 0.125j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    7.125 + 0j
                ),
                Line(
                    7.125 + 0j,
                    7.125 + 2j
                ),
                Arc(
                    7.125 + 2j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    7 + 2.125j
                ),
                Line(
                    7 + 2.125j,
                    5 + 2.125j
                ),
                Arc(
                    5 + 2.125j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    4.875 + 2j
                ),
                Line(
                    4.875 + 2j,
                    4.875 + 0j
                ),
                Arc(
                    4.875 + 0j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    5 - 0.125j
                )
            ).set_Z(),
            Subpath(
                Line(
                    8.125 + 0.125j,
                    8.125 + 1.875j
                ),
                Line(
                    8.125 + 1.875j,
                    9.875 + 1.875j
                ),
                Line(
                    9.875 + 1.875j,
                    9.875 + 0.12499999999999997j
                ),
                Line(
                    9.875 + 0.12499999999999997j,
                    8.125 + 0.125j
                )
            ).set_Z(),
            Subpath(
                Line(
                    8 - 0.125j,
                    10 - 0.125j
                ),
                Arc(
                    10 - 0.125j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    10.125 + 0j
                ),
                Line(
                    10.125 + 0j,
                    10.125 + 2j
                ),
                Arc(
                    10.125 + 2j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    10 + 2.125j
                ),
                Line(
                    10 + 2.125j,
                    8 + 2.125j
                ),
                Arc(
                    8 + 2.125j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    7.875 + 2j
                ),
                Line(
                    7.875 + 2j,
                    7.875 + 0j
                ),
                Arc(
                    7.875 + 0j,
                    0.125 + 0.125j,
                    0,
                    False,
                    True,
                    8 - 0.125j
                )
            ).set_Z()
        )
        # note: if we're unlucky, the following test could fail
        # depending on the platform because of numerical rounding
        # issues (but would need a 1e-15 difference to cross over a
        # 1e-2 threshold, should be unlikely)
        stroke = p.stroke(0.25, join='round', cap='round')
        self.assertTrue(stroke.__eq__(answer, 1e-12))


if __name__ == '__main__':
    unittest.main()
