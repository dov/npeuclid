# Compare euclid vs npeuclid and do testing
import euclid
import npeuclid ; reload(npeuclid)
import numpy as np

eps = 1e-10

def qvl(a,b):
  return abs(a-b) < eps

def aseq(a,b,label):
  if isinstance(a, euclid.Point2) and isinstance(b, npeuclid.Vec2):
    if not (qvl(a.x,b.x) and qvl(a.y,b.y)):
      print 'Error (: '+label+'):', a,'vs',b
    return
  elif isinstance(a, euclid.Matrix3) and isinstance(b, npeuclid.Affine2):
    alist = [a[i] for i in [0,3,6,1,4,7,2,5,8]]
    blist = [b[i//3,i%3] for i in range(9)]
    if not all(qvl(v,w) for v,w in zip(alist,blist)):
      print 'Error ('+label+'):', alist,'vs',blist
    return
  print type(a), 'vs', type(b)
  raise Exception('Unsupported types!')

ev1 = euclid.Point2(1,2)
nv1 = npeuclid.Vec2(1,2)
aseq(ev1,nv1,'1. Point')

raff = np.random.random(6)

em1 = euclid.Matrix3.new_affine(raff)
nm1 = npeuclid.Affine2.new_affine(raff)
aseq(em1,nm1,'1. aff')

aseq(euclid.Matrix3.new_rotate(1),
     npeuclid.Affine2.new_rotate(1),
     '2. rotate')

aseq(euclid.Matrix3.new_rotate(np.pi/3) * ev1,
     npeuclid.Affine2.new_rotate(np.pi/3) * nv1,
     '3. rotate apply'
     )

aseq(euclid.Matrix3.new_scale(5,6) * ev1,
     npeuclid.Affine2.new_scale(5,6) * nv1,
     '4. scale apply'
     )

# Note that this changes em1, but not nm1!
em1_copy = em1.copy()
aseq(em1.pre_translate(8,7) * ev1,
     nm1.pre_translate(8,7) * nv1,
     '5 translate apply'
     )
em1 = em1_copy
aseq(em1,nm1, 'foo')

# Test that Vec2Array application works
rvals = [(np.random.random(), np.random.random()) for i in range (6)]
nmv = nm1 * npeuclid.Vec2Array(rvals)
for i,v in enumerate(rvals):
  aseq(em1 * euclid.Point2(*v), nmv[i], 'Vec2 array apply: Point %d'%i)

print 'ok!'
