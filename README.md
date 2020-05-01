# Description

npeuclid is a library that provides vector and matrix classes for 2D and 3D 
graphics applications.

npeuclid is a port of the https://github.com/ezag/pyeuclid library to numpy.
Instead of doing matrix multiplications in native python as euclid is doing,
npeuclid is delegating all calculations to numpy. 

# Differences to pyeuclid

- The base type of all pyeuclid types are numpy ndarrays.
- 2D and 3D geometry is supported. Currenly there is no quaternion support.
- All operaters return their transformed values, i.e. there are no inline operators. E.g. `t.scale(sx,sy)` will return a scaled version of `t`, and won't modify t. 
- There is no vector point and vector support like in euclid. There is only one type `Vec2`, which behaves like a point under transformation. To get a vector transformation do `t.linear()*p` (like in Eigen).

# Types

Here are the types defined by npeuclid:

## 2D 

- `Vec2` - Holds a 2D vector (or point).
- `Vec2Array` - Holds a list of points. All operations that can be done on a signal should be available on a Vec2Array as well. In particular, an `Affine2` operation may be applied on a `Vec2Array` which will transform all the points. 
- `Affine2` - An 2D affine transformation.

## 3D

- `Vec3` - Holds a 3D vector (or point).
- `Vec3Array` - Holds a list of points. All operations that can be done on a signal should be available on a Vec3Array as well. In particular, an `Affine3` operation may be applied on a `Vec3Array` which will transform all the points. 
- `Affine3` - An 3D affine transformation.

# Usage

## 3D

    p = Vec2(2,3)
    q = Vec2(5,6)
    angle = math.radians(33)
    t = Affine2.new_translate(p.x,p.y).rotate(angle).translate(-p.x,-p.y)
    print(t*q)
    t = Affine2.new_rotate_around(angle, p.x, p.y)
    print(t*q)
    t = Affine2.rotate_around(angle, p.x, p.y)
    print(t*q)
    t = Affine2.new_translate(p.x,p.y)
    print(t*p, t.linear()*p) 
    
    # Use of Vec2Array is using numpy and is *much* faster than looping.
    pp = Vec2Array([[5,6],[7,8],[3,4]])
    print (t*pp)[0]
