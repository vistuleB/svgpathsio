import unittest
from svgpathtools import *
from os.path import join, dirname


class TestSVG2Paths(unittest.TestCase):
    def test_svg2paths_polygons(self):
=        paths, _ = svg2paths(join(dirname(__file__), 'polygons.svg'))

        # triangular polygon test
        path = paths[0]
        path_correct = Path(Subpath(Line(55.5 + 0j, 55.5 + 50j),
                                    Line(55.5 + 50j, 105.5 + 50j),
                                    Line(105.5 + 50j, 55.5 + 0j)).set_Z())
        self.assertTrue(isinstance(path, Path))
        self.assertTrue(len(path) == 1)
        self.assertTrue(isinstance(path[0], Subpath))
        self.assertTrue(len(path[0]) == 3)
        self.assertTrue(path[0].Z)
        self.assertTrue(path == path_correct)

        # triangular quadrilateral (with a redundant 4th "closure" point)
        path = paths[1]
        path_correct = Path(Subpath(Line(0 + 0j, 0 - 100j),
                                    Line(0 - 100j, 0.1 - 100j),
                                    Line(0.1 - 100j, 0 + 0j),
                                    Line(0 + 0j, 0 + 0j)).set_Z())  # result of redundant point
        self.assertTrue(isinstance(path, Path))
        self.assertTrue(len(path) == 1)
        self.assertTrue(isinstance(path[0], Subpath))
        self.assertTrue(len(path[0]) == 4)
        self.assertTrue(path[0].Z)
        self.assertTrue(path == path_correct)

    def test_svg2paths_ellipses(self):
        paths, _ = svg2paths(join(dirname(__file__), 'ellipse.svg'))

        # ellipse tests
        path_ellipse = paths[0]
        path_ellipse_correct = Path(Subpath(Arc(50 + 100j, 50 + 50j, 0.0, True, False, 150 + 100j),
                                            Arc(150 + 100j, 50 + 50j, 0.0, True, False, 50 + 100j)).set_Z())
        self.assertTrue(isinstance(path_ellipse, Path))
        self.assertTrue(len(path_ellipse) == 1)
        self.assertTrue(isinstance(path_ellipse[0], Subpath))
        self.assertTrue(len(path_ellipse[0]) == 2)
        self.assertTrue(path_ellipse[0].Z)
        self.assertTrue(path_ellipse == path_ellipse_correct)

        # circle tests
        paths, _ = svg2paths(join(dirname(__file__), 'circle.svg'))

        path_circle = paths[0]
        path_circle_correct = Path(Subpath(Arc(50 + 100j, 50 + 50j, 0.0, True, False, 150 + 100j),
                                           Arc(150 + 100j, 50 + 50j, 0.0, True, False, 50 + 100j)).set_Z())
        self.assertTrue(isinstance(path_circle, Path))
        self.assertTrue(len(path_circle) == 1)
        self.assertTrue(isinstance(path_circle[0], Subpath))
        self.assertTrue(len(path_circle[0]) == 2)
        self.assertTrue(path_circle[0].Z)
        self.assertTrue(path_circle == path_circle_correct)


if __name__ == '__main__':
    unittest.main()
