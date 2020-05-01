# Beginning tests. Much more should be written.
#
# Dov Grobgeld <dov.grobgeld@gmail.com>
# 2020-05-01 Fri

import unittest
import numpy as np
from npeuclid import Vec2, Vec2Array, Affine2, Vec3, Vec3Array, Affine3


class TestNPEuclid(unittest.TestCase):
    def test_vec2_construction(self):
        p = Vec2(2,3)
        self.assertAlmostEqual(p[0], 2)
        self.assertAlmostEqual(p[1], 3)

    def test_aff2_translate(self):
        p = Vec2(2,3)
        t = Affine2.new_translate(10,20)
        q = t*p
        self.assertAlmostEqual(q[0], 12)
        self.assertAlmostEqual(q[1], 23)

    def test_aff2_new_rotate(self):
        p = Vec2(2,3)
        t = Affine2.new_rotate(np.pi/2)
        q = t*p
        self.assertAlmostEqual(q[0], -3)
        self.assertAlmostEqual(q[1], 2)

    def test_aff2_rotate(self):
        p = Vec2(2,3)
        t = Affine2()
        q = t.rotate(np.pi/2)*p
        self.assertAlmostEqual(q[0], -3)
        self.assertAlmostEqual(q[1], 2)

    def test_aff3_linear(self):
        p = Vec2(2,3)
        t = Affine2.new_translate(10,20)
        q = t.linear()*p
        self.assertAlmostEqual(q[0], 2)
        self.assertAlmostEqual(q[1], 3)

    def test_aff3_translate(self):
        p = Vec3(2,3,4)
        t = Affine3.new_translate(10,20,30)
        q = t*p
        self.assertAlmostEqual(q[0], 12)
        self.assertAlmostEqual(q[1], 23)
        self.assertAlmostEqual(q[2], 34)

    def test_aff3_rotatex(self):
        p = Vec3(2,3,4)
        t = Affine3()
        q = t.rotatex(np.pi/2)*p
        self.assertAlmostEqual(q[0], 2)
        self.assertAlmostEqual(q[1], -4)
        self.assertAlmostEqual(q[2], 3)

    def test_aff3_rotatey(self):
        p = Vec3(2,3,4)
        t = Affine3()
        q = t.rotatey(np.pi/2)*p
        self.assertAlmostEqual(q[0], 4)
        self.assertAlmostEqual(q[1], 3)
        self.assertAlmostEqual(q[2], -2)

    def test_aff3_rotatez(self):
        p = Vec3(2,3,4)
        t = Affine3()
        q = t.rotatez(np.pi/2)*p
        self.assertAlmostEqual(q[0], -3)
        self.assertAlmostEqual(q[1], 2)
        self.assertAlmostEqual(q[2], 4)

    def test_aff3_linear(self):
        p = Vec3(2,3,4)
        t = Affine3.new_translate(10,20,30)
        q = t.linear()*p
        self.assertAlmostEqual(q[0], 2)
        self.assertAlmostEqual(q[1], 3)
        self.assertAlmostEqual(q[2], 4)

if __name__ == '__main__':
    unittest.main()
    
