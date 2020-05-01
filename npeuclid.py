#!/usr/bin/env python
#
# euclid graphics maths module
# Copyright (c) 2006 Alex Holkner <Alex.Holkner@mail.google.com>
# Copyright (c) 2011 Eugen Zagorodniy <https://github.com/ezag/>
# Copyright (c) 2012 Lorenzo Riano <https://github.com/lorenzoriano>
# Copyright (c) 2018 Dov Grobgeld <dov.grobgeld@gmail.com>
#
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

from __future__ import print_function
import numpy as np
import math 

class Vec2(np.ndarray):
    '''An (x,y) pair'''
    def __new__(cls, x=0,y=0):
        obj = np.array([x,y],dtype=np.float)
        return super(Vec2,cls).__new__(cls, shape=(2,), buffer=obj,dtype=np.float)
  
    def __repr__(self):
        return 'Vec2({0:.5f},{1:.5f})'.format(self[0],self[1])
  
    def __str__(self):
        return self.__repr__()
  
    def magnitude_squared(self):
        return (self**2).sum()
  
    def magnitude(self):
        return math.sqrt((self**2).sum())
  
    def normalize(self):
        d = self.magnitude()
        return self/d
  
    def reflect(self, normal):
        # assume normal is normalized
        assert isinstance(normal, Vec2)
        d = 2*self.dot(normal)
        return self - d*normal

    def angle(self, other):
        """Return the angle to the vector other w.r.t. to the origin"""
        return math.acos(self.dot(other) / (self.magnitude()*other.magnitude()))
  
    def project(self, other):
        """Return one vector projected on the vector other"""
        n = other.normalize()
        return self.dot(n)*n

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
                + ','.join('[{0:.5f},{1:.5f}]'.format(self[i,0],self[i,1])
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

    # Use getattr to support vec.x and vec.y
    def __getattr__(self, attr):
        if attr=='x':
            return self.toarray()[:,0]
        elif attr=='y':
            return self.toarray()[:,1]
        else:
            raise ValueError('No such attribute '+attr)

    def toarray(self):
        '''"cast" self to nd array'''
        return np.ndarray(shape=self.shape,
                          buffer=self.data,
                          dtype=self.dtype)

    def reflect(self, normal):
        # assume normal is normalized
        assert isinstance(normal, Vec2)
        d = 2*self.toarray().dot(normal)
        return self - (d * normal.reshape(2,1)).transpose()

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
  
    def linear(self):
        '''Return a linear matrix'''
        ret = self.copy()
        ret[0:2,2]=0
        return ret

    def tolist(self):
        # Return in the order of new_affine
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

class Vec3(np.ndarray):
    '''An (x,y,z) triplet'''
    def __new__(cls, x=0,y=0,z=0):
        obj = np.array([x,y,z],dtype=np.float)
        return super(Vec3,cls).__new__(cls, shape=(3,), buffer=obj,dtype=np.float)
  
    def __repr__(self):
        return 'Vec3({0:.5f},{1:.5f},{2:.5f})'.format(self[0],self[1],self[2])
  
    def __str__(self):
        return self.__repr__()
  
    def magnitude_squared(self):
        return (self**2).sum()
  
    def magnitude(self):
        return math.sqrt((self**2).sum())
  
    def normalize(self):
        d = self.magnitude()
        return self/d
  
    def reflect(self, normal):
        # assume normal is normalized
        assert isinstance(normal, Vec3)
        d = 2 * self.dot(normal)
        return self - d * normal
  
    def rotate_around(self, axis, theta):
        """Return the vector rotated around axis through angle theta. Right hand rule applies"""

        # Adapted from equations published by Glenn Murray.
        # http://inside.mines.edu/~gmurray/ArbitraryAxisRotation/ArbitraryAxisRotation.html
        x, y, z = self[:]
        u, v, w = axis[:]

        # Extracted common factors for simplicity and efficiency
        r2 = u**2 + v**2 + w**2
        r = math.sqrt(r2)
        ct = math.cos(theta)
        st = math.sin(theta) / r
        dt = (u*x + v*y + w*z) * (1 - ct) / r2
        return Vec3((u * dt + x * ct + (-w * y + v * z) * st),
                    (v * dt + y * ct + ( w * x - u * z) * st),
                    (w * dt + z * ct + (-v * x + u * y) * st))

    def angle(self, other):
        """Return the angle to the vector other w.r.t. to the origin"""
        return math.acos(self.dot(other) / (self.magnitude()*other.magnitude()))
  
    def project(self, other):
        """Return one vector projected on the vector other"""
        n = other.normalize()
        return self.dot(n)*n

    # Use getattr to support vec.x, vec.y, and vec.z
    def __getattr__(self, attr):
        if attr=='x':
            return self[0]
        elif attr=='y':
            return self[1]
        elif attr=='z':
            return self[2]
        else:
            raise ValueError('No such attribute '+attr)

class Vec3Array(np.ndarray):
    def __new__(cls, list=None, matrix=None):
        if list is not None:
            obj = np.matrix(np.array(list)).astype(np.float)
        else:
            obj = matrix
        self = super(Vec3Array,cls).__new__(cls, shape=obj.shape, buffer=matrix,dtype=obj.dtype)
        self[:] = obj[:]  # Why do I need this??
        return self
  
    def __str__(self):
        return ('Vec3Array('
                + ','.join('[{0:.5f},{1:.5f},{2:.5f}]'.format(self[i,0],self[i,1],self[i,2])
                           for i in range(self.shape[0]))
                + ')'
                )
  
    def __getitem__(self, i):
        if isinstance(i, int):
            x = np.ndarray.__getitem__(self,(i,0))
            y = np.ndarray.__getitem__(self,(i,1))
            z = np.ndarray.__getitem__(self,(i,2))
            p = Vec3(x,y,z)
        else:
            p = np.ndarray.__getitem__(self,i)
        return p
    
    # Use getattr to support vec.x and vec.y
    def __getattr__(self, attr):
        if attr=='x':
            return self.toarray()[:,0]
        elif attr=='y':
            return self.toarray()[:,1]
        elif attr=='z':
            return self.toarray()[:,2]
        else:
            raise ValueError('No such attribute '+attr)

    def toarray(self):
        '''"cast" self to nd array'''
        return np.ndarray(shape=self.shape,
                          buffer=self.data,
                          dtype=self.dtype)

    def reflect(self, normal):
        # assume normal is normalized
        assert isinstance(normal, Vec2)
        d = 2*self.toarray().dot(normal)
        return self - (d * normal.reshape(3,1)).transpose()

class Affine3(np.ndarray):
    def __new__(cls, input_array=None):
        if input_array is not None:
            obj = np.asarray(input_array).view(cls)
        else:
            obj = np.eye(4)
        return super(Affine3,cls).__new__(cls, shape=obj.shape, buffer=obj, dtype=obj.dtype)
  
    @classmethod
    def new_scale(cls, sx, sy, sz):
        obj = np.diag([sx,sy,sz,1]).astype(float)
        return cls.__new__(cls, obj)
  
    @classmethod
    def new_translate(cls, tx, ty, tz):
        obj = np.array([[1,0,0,tx],
                        [0,1,0,ty],
                        [0,0,1,tz],
                        [0,0,0,1]]).astype(float)
        return cls.__new__(cls, obj)
  
    def scale(self, sx, sy, sz):
        return self * Affine3.new_scale(x, y, z)

    def translate(self, sx, sy, sz):
        return self * Affine3.new_translate(x, y, z)

    def linear(self):
        '''Return the linear (upper 3x3) matrix'''
        '''Return a linear matrix'''
        ret = self.copy()
        ret[0:3,2]=0
        return ret

    def pre_translate(self, tx, ty, tz):
        tmat = self.copy()
        tmat[0,3] += tx
        tmat[1,3] += ty
        tmat[2,3] += tz
        return tmat

    def rotatex(self, angle):
        return self * Affine3.new_rotatex(angle)

    def rotatey(self, angle):
        return self * Affine3.new_rotatey(angle)

    def rotatez(self, angle):
        return self * Affine3.new_rotatez(angle)

    def rotate_axis(self, angle, axis):
        return self * Affine3.new_rotate_axis(angle, axis)

    def rotate_euler(self, heading, attitude, bank):
        return self * Affine3.new_rotate_euler(heading, attitude, bank)

    def rotate_triple_axis(self, x, y, z):
        return self * Affine3.new_rotate_triple_axis(x, y, z)

    @classmethod
    def new_rotatex(cls, angle):
        self = cls()
        s = math.sin(angle)
        c = math.cos(angle)
        self[1,1]=self[2,2] = c
        self[1,2] = -s
        self[2,1] = s
        return self

    @classmethod 
    def new_rotatey(cls, angle):
        self = cls()
        s = math.sin(angle)
        c = math.cos(angle)
        self[0,0] = self[2,2] = c
        self[0,2] = s
        self[2,0] = -s
        return self    

    @classmethod
    def new_rotatez(cls, angle):
        self = cls()
        s = math.sin(angle)
        c = math.cos(angle)
        self[0,0] = self[1,1] = c
        self[0,1] = -s
        self[1,0] = s
        return self

    @classmethod
    def new_rotate_axis(cls, angle, axis):
        assert(isinstance(axis, Vector3))
        vector = axis.normalize()
        x = vector.x
        y = vector.y
        z = vector.z

        self = cls()
        s = math.sin(angle)
        c = math.cos(angle)
        c1 = 1. - c
        
        # from the glRotate man page
        self[0,0] = x * x * c1 + c
        self[0,1] = x * y * c1 - z * s
        self[0,2] = x * z * c1 + y * s
        self[1,0] = y * x * c1 + z * s
        self[1,1] = y * y * c1 + c
        self[1,2] = y * z * c1 - x * s
        self[2,0] = x * z * c1 - y * s
        self[2,1] = y * z * c1 + x * s
        self[2,2] = z * z * c1 + c
        return self

    @classmethod
    def new_rotate_euler(cls, heading, attitude, bank):
        # from http://www.euclideanspace.com/
        ch = math.cos(heading)
        sh = math.sin(heading)
        ca = math.cos(attitude)
        sa = math.sin(attitude)
        cb = math.cos(bank)
        sb = math.sin(bank)

        self = cls()
        self[0,0] = ch * ca
        self[0,1] = sh * sb - ch * sa * cb
        self[0,2] = ch * sa * sb + sh * cb
        self[1,0] = sa
        self[1,1] = ca * cb
        self[1,2] = -ca * sb
        self[2,0] = -sh * ca
        self[2,1] = sh * sa * cb + ch * sb
        self[2,2] = -sh * sa * sb + ch * cb
        return self

    @classmethod
    def new_rotate_triple_axis(cls, x, y, z):
      m = cls()
      
      m[0,0], m[0,1], m[0,2] = x.x, y.x, z.x
      m[1,0], m[1,1], m[1,2] = x.y, y.y, z.y
      m[2,0], m[2,1], m[2,2] = x.z, y.z, z.z
      
      return m

    @classmethod
    def new_look_at(cls, eye, at, up):
      z = (eye - at).normalized()
      x = up.cross(z).normalized()
      y = z.cross(x)
      
      m = cls.new_rotate_triple_axis(x, y, z)
      m.transpose()
      m[0,3], m[1,3], m[2,3] = -x.dot(eye), -y.dot(eye), -z.dot(eye)
      return m
    
    @classmethod
    def new_perspective(cls, fov_y, aspect, near, far):
        # from the gluPerspective man page
        f = 1 / math.tan(fov_y / 2)
        self = cls()
        assert near != 0.0 and near != far
        self[0,0] = f / aspect
        self[1,1] = f
        self[2,2] = (far + near) / (near - far)
        self[2,3] = 2 * far * near / (near - far)
        self[3,2] = -1
        self[3,3] = 0
        return self

    def __mul__(self, other):
        if isinstance(other, Vec3) or isinstance(other, list) or isinstance(other, tuple):
            res = super(self.__class__, self).dot(
                np.array([[other[0],other[1],other[2],1]]).T)
            return Vec3(res[0,0],res[1,0],res[2,0])
        elif isinstance(other, Vec3Array):
            # Add dummy value for affine multiplication
            v = np.vstack((other.T,
                           np.ones((other.shape[0]))))
            res = self.dot(v)[0:3,:].T
            return Vec3Array(matrix=res)
        elif isinstance(other, Affine3):
            return super(self.__class__, self).dot(other)
        else:
            raise TypeError('Unsupported Affine3 multiplication type: ' + str(type(other)))
  
    def __str__(self):
        return 'Affine3('+super(Affine3, self).__str__().replace('\n','\n        ')

