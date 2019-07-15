from __future__ import division, absolute_import, print_function
import unittest
from svgpathtools import *


class TestGeneration(unittest.TestCase):

    def test_path_parsing(self):
        """Examples from the SVG spec"""

        path_strings = [
            'M 100,100 L 300,100 L 200,300 Z',
            'M 0,0 L 50,20 M 100,100 L 300,100 L 200,300 Z',
            'M 100,100 L 200,200',
            'M 100,200 L 200,100 L -100,-200',
            'M 100,200 C 100,100 250,100 250,200 S 400,300 400,200',
            'M 100,200 C 100,100 400,100 400,200',
            'M 100,500 C 25,400 475,400 400,500',
            'M 100.0,800.0 C 175.00,700.0 325.0,700.0 400,800',
            'M 600,200 C 675,100 975,100 900,200',
            'M 600,500 C 600,350 900,650 900,500',
            'M 600,800 C 625,700 725,700 750,800 S 875,900 900,800',
            'M 200,300 Q 400,50 600,300 T 1000,300',
            'M -3.4E+38,3.4E+38 L -3.4E-38,3.4E-38',
            'M 0,0 L 50,20 M 50,20 L 200,100 Z',
            'M 600,350 L 650,325 A 25,25 -30 0,1 700,300 L 750,275',
        ]

        expected_answers = [
            ('M 100,100 L 300,100 200,300 Z', 'M 100,100 H 300 L 200,300 Z'),
            ('M 0,0 L 50,20 M 100,100 L 300,100 200,300 Z', 'M 0,0 L 50,20 M 100,100 H 300 L 200,300 Z'),
            'M 100,100 L 200,200',
            'M 100,200 L 200,100 -100,-200',
            'M 100,200 C 100,100 250,100 250,200 C 250,300 400,300 400,200',
            'M 100,200 C 100,100 400,100 400,200',
            'M 100,500 C 25,400 475,400 400,500',
            'M 100,800 C 175,700 325,700 400,800',
            'M 600,200 C 675,100 975,100 900,200',
            'M 600,500 C 600,350 900,650 900,500',
            'M 600,800 C 625,700 725,700 750,800 C 775,900 875,900 900,800',
            'M 200,300 Q 400,50 600,300 Q 800,550 1000,300',
            'M -3.4e+38,3.4e+38 L -3.4e-38,3.4e-38',
            'M 0,0 L 50,20 M 50,20 L 200,100 Z',
            ('M 600,350 L 650,325 A 27.9508497187,27.9508497187 -30 0 1 700,300 L 750,275',            # Python 2
             'M 600,350 L 650,325 A 27.950849718747367,27.950849718747367 -30 0 1 700,300 L 750,275')  # Python 3
        ]

        for string, expected_answer1 in zip(path_strings, expected_answers):
            answer = parse_path(string).d()

            if isinstance(expected_answer1, tuple):
                expected_answer2 = expected_answer1[1]
                expected_answer1 = expected_answer1[0]
            else:
                expected_answer2 = expected_answer1

            option1 = answer == expected_answer1
            option2 = answer == expected_answer2

            msg = ('\nstring = {}\nanswer = {}\na1     = {}\na2     = {}'
                   ''.format(string, answer, expected_answer1, expected_answer2))

            self.assertTrue(option1 or option2, msg)

    def test_normalizing(self):
        # Relative paths are made absolute, some spacing and number formatting
        # changes
        string = 'M0 0L3.4E2-10L100.0,100M100,100l100,-100'
        expected_answer = "M 0,0 L 340,-10 100,100 M 100,100 L 200,0"
        answer = parse_path(string).d()
        self.assertTrue(answer == expected_answer)


if __name__ == '__main__':
    unittest.main()
