# Copyright (c) 2018 Dov Grobgeld <dov.grobgeld@gmail.com>
#
# euclid graphics maths module with numpy
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2.1 of the License, or (at your
# option) any later version.
# 
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
# for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA

import numpy as np
import math 

class Vec2(np.ndarray):
    '''An (x,y) pair'''
    def __new__(cls, x=0,y=0):
        obj = np.array([x,y],dtype=np.float)
        return super(Vec2,cls).__new__(cls, shape=(2,), buffer=obj,dtype=np.float)
  
    def __repr__(self):
        return 'Vec2({0},{1})'.format(self[0],self[1])
  
    def __str__(self):
        return self.__repr__()
  
    def __mul__(self, other):
        if isinstance(other,int) or isinstance(other,float):
            return Vec2(self[0]*other,self[1]*other)
        else:
            raise TypeError('No support for mul operator for other operand of type ' + str(type(other)))
  
    def __div__(self, other):
        if isinstance(other,int) or isinstance(other,float):
            return Vec2(self[0]/other,self[1]/other)
        else:
            raise TypeError('No support for div operator for other operand of type ' + str(type(other)))
  
    def reflect(self, normal):
        # assume normal is normalized
        assert isinstance(normal, Vec2)
        d = 2 * (self.x * normal.x + self.y * normal.y)
        return Vec2(self.x - d * normal.x,
                   self.y - d * normal.y)
  
    def magnitude_squared(self):
        return self[0]**2+self[1]**2
  
    def magnitude(self):
        return math.sqrt(self[0]**2+self[1]**2)
  
    def normalize(self):
        d = self.magnitude()
        return Vec2(self[0]/d,self[1]/d)
  
    def angle(self, other):
        """Return the angle to the vector other w.r.t. to the origin"""
        return math.acos(self.dot(other) / (self.magnitude()*other.magnitude()))
  
    # Use getattr to support vec.x and vec.y
    def __getattr__(self, attr):
        if attr=='x':
            return self[0]
        elif attr=='y':
            return self[1]
        else:
            raise ValueError('No such attribute '+attr)

class Vec2Array(np.ndarray):
    def __new__(cls, list=None, matrix=None):
        if list is not None:
            obj = np.matrix(np.array(list)).astype(np.float)
        else:
            obj = matrix
        self = super(Vec2Array,cls).__new__(cls, shape=obj.shape, buffer=matrix,dtype=obj.dtype)
        self[:] = obj[:]  # Why do I need this??
        return self
  
    def __str__(self):
        return ('Vec2Array('
                + ','.join('[{0},{1}]'.format(self[i,0],self[i,1])
                           for i in range(self.shape[0]))
                + ')'
                )
  
    def __getitem__(self, i):
        if isinstance(i, int):
            x = np.ndarray.__getitem__(self,(i,0))
            y = np.ndarray.__getitem__(self,(i,1))
            p = Vec2(x,y)
        else:
            p = np.ndarray.__getitem__(self,i)
        return p

class Affine2(np.ndarray):
    def __new__(cls, input_array=None):
        if input_array is not None:
            obj = np.asarray(input_array).view(cls)
        else:
            obj = np.eye(3)
        return super(Affine2,cls).__new__(cls, shape=obj.shape, buffer=obj, dtype=obj.dtype)
  
    @classmethod
    def new_scale(cls, sx, sy):
        obj = np.diag([sx,sy,1]).astype(float)
        return cls.__new__(cls, obj)
  
    @classmethod
    def new_translate(cls, tx, ty):
        obj = np.array([[1,0,tx],[0,1,ty],[0,0,1]]).astype(float)
        return cls.__new__(cls, obj)
  
    @classmethod
    def new_rotate(cls, angle):
        return cls.new_rotate_around(angle)
  
    @classmethod
    def new_rotate_around(cls, angle, px=0, py=0):
        st = math.sin(angle)
        ct = math.cos(angle)
        obj = cls.__new__(cls)
    
        # Rotation matrix
        obj[0,0]=ct
        obj[0,1]=-st
        obj[1,0]=st
        obj[1,1]=ct
      
        # Shifting
        obj[0,2] = px - ct*px + st*py
        obj[1,2] = py - st*px - ct*py
    
        return obj
  
    @classmethod
    def new_affine(cls, list):
        '''A new matrix from a list of 6 members in the order xx,xy,yx,yy,tx,ty'''
        xx,xy,yx,yy,tx,ty = list
        return cls.__new__(cls, np.array([[xx,xy,tx],
                                          [yx,yy,ty],
                                          [0,0,1]]))
  
    def tolist(self):
        return [self[i] for i in [(0,0),(0,1),(1,0),(1,1),(0,2),(1,2)]]
  
    def scale(self, sx, sy):
        return self * Affine2.new_scale(sx,sy)
  
    def translate(self, tx, ty):
        return self * Affine2.new_translate(tx,ty)
      
    def pre_translate(self, tx, ty):
        tmat = self.copy()
        tmat[0,2] += tx
        tmat[1,2] += ty
        return tmat
  
    def rotate(self, angle):
        return self * Affine2.new_rotate(angle)
  
    def rotate_around(self, angle, x,y):
        return self * Affine2.new_rotate_around(angle,x,y)
  
    def inverse(self):
        return np.linalg.inv(self)
  
    def __mul__(self, other):
        if isinstance(other, Vec2) or isinstance(other, list) or isinstance(other, tuple):
            res = super(self.__class__, self).dot(
                np.array([[other[0],other[1],1]]).T)
            return Vec2(res[0,0],res[1,0])
        elif isinstance(other, Vec2Array):
            # Add dummy value for affine multiplication
            v = np.vstack((other.T,
                           np.ones((other.shape[0]))))
            res = self.dot(v)[0:2,:].T
            return Vec2Array(matrix=res)
        elif isinstance(other, Affine2):
            return super(self.__class__, self).dot(other)
        else:
            raise TypeError('Unsupported Affine2 multiplication type: ' + str(type(other)))
  
    def __str__(self):
        return 'Affine2('+super(Affine2, self).__str__().replace('\n','\n        ')
    
if __name__=='__main__':
    # Some testing
    p = Vec2(2,3)
    q = Vec2(5,6)
    angle = math.radians(33)
    t = Affine2.new_translate(p.x,p.y).rotate(angle).translate(-p.x,-p.y)
    print t*q
    t = Affine2.new_rotate_around(angle, p.x, p.y)
    print t*q
    t = Affine2().rotate_around(angle, p.x, p.y)
    print t*q
    pp = Vec2Array([[5,6],[7,8],[3,4]])
    print (t*pp)[0]


