from scipy import misc
from skimage import color
from skimage import measure
import matplotlib.pyplot as plt
from skimage.filter import threshold_otsu
from skimage.morphology import skeletonize
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
import numpy as np

import sys
import Queue

processed = set()
trace_tasks = Queue.Queue()

def gen_neighbours_coords(pt):
  yield (pt[0]-1, pt[1])
  yield (pt[0]-1, pt[1]-1)
  yield (pt[0]-1, pt[1]+1)
  yield (pt[0], pt[1]-1)
  yield (pt[0], pt[1]+1)
  yield (pt[0]+1, pt[1])
  yield (pt[0]+1, pt[1]-1)
  yield (pt[0]+1, pt[1]+1)

def gen_neighbours(img, pt):
  return filter(lambda x: x[0] >= 0 and x[1] >= 0 and x[0] <= len(img) - 1 and x[1] <= len(img[0]) - 1 and img[x[0], x[1]], set(gen_neighbours_coords(pt)))

def distance(p1, p2):
  return ((p1[0]-p2[0])**2.0 + (p1[1]-p2[1])**2.0)**0.5


def middle_point(p1, p2):
  return ((p1[0]+p2[0])/2.0,(p1[1]+p2[1])/2.0)

def trace_line(img, pt, from_pt, line_accu=[]):
  global processed
  global trace_tasks

  if (len(line_accu) == 0):
    line_accu = [from_pt]

  if (pt in processed):
    # this will lead to several 1-pixel traces we should remove them afterwards
    
    return line_accu + [pt]
  processed.add(pt)
  neighbours = gen_neighbours(img, pt)
  neighbours.remove(from_pt)

  if (len(neighbours) == 0): 
    return line_accu + [pt]
  if (len(neighbours) == 1): 
    return trace_line(img, neighbours[0], pt, line_accu + [pt])
  if (len(neighbours) >= 2): 
    for npt in neighbours: trace_tasks.put((npt, pt))
    return line_accu + [pt]
  
def concat_lines(line1, line2):
  rline1 = line1[::-1]
  sline2 = line2[1:]
  return rline1+sline2

def find_lines(img):
  global processed
  global trace_tasks
  for y in xrange(0, len(img)):
    for x in xrange(0, len(img[y])):
      pt = (y, x)
      if (pt in processed): continue
      if (img[pt[0], pt[1]]):
        processed.add((y,x))
        neighbours = gen_neighbours(img, pt)
        # pt is in the end of line
        if (len(neighbours) == 1):
          trace_tasks.put((neighbours[0], pt))
        # pt is in the middle of line
        if (len(neighbours) == 2):
          line1 = trace_line(img, neighbours[0], pt)
          line2 = trace_line(img, neighbours[1], pt)
          yield concat_lines(line1, line2)
        # pt is in a branching point or crossing
        if (len(neighbours) >= 3):
          for npt in neighbours: trace_tasks.put((npt, pt))
        while (not trace_tasks.empty()):
          task = trace_tasks.get()
          yield trace_line(img, task[0], task[1])

        
  
  

fimg = misc.imread("wms_simple_small.png")

gimg = 1 - color.colorconv.rgb2grey(fimg)

thresh = threshold_otsu(gimg)
binary = gimg > thresh
 
skeleton = skeletonize(binary)

lines = list(find_lines(skeleton))

# merge neighbour points
def merge_np(lines):
  t = 3
  for l1 in lines: 
    for l2 in lines:
      if (distance(l1[0], l2[0]) < t):
        mp = middle_point(l1[0], l2[0])
        l1[0] = mp
        l2[0] = mp
      if (distance(l1[0], l2[-1]) < t):
        mp = middle_point(l1[0], l2[-1])
        l1[0] = mp
        l2[-1] = mp
      if (distance(l1[-1], l2[0]) < t):
        mp = middle_point(l1[-1], l2[0])
        l1[-1] = mp
        l2[0] = mp
      if (distance(l1[-1], l2[-1]) < t):
        mp = middle_point(l1[-1], l2[-1])
        l1[-1] = mp
        l2[-1] = mp

for i in xrange(30):
  merge_np(lines)

# remove singular lines
lines = filter(lambda x: not (len(x) < 2 or (len(x) == 2 and distance(x[0],x[1]) < 2)), lines)

appr_lines = map(lambda x : approximate_polygon(np.array(x), tolerance=1.5), lines)

# polygonize
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize, polygonize_full, transform

pair_lines = []
sz = 255

def round_snap_bound(v):
  r = int(round(v))
  if (r <= 2): return 0
  if (r >= sz - 2): return sz
  return r

for l in appr_lines:
  for i in xrange(len(l) - 1):
    p1 = l[i]
    p2 = l[i+1]
    p1x = round_snap_bound(p1[0])
    p1y = round_snap_bound(p1[1])
    p2x = round_snap_bound(p2[0])
    p2y = round_snap_bound(p2[1])
    pair_lines.append(((p1x, p1y), (p2x, p2y)))

# add bounding lines needed by polygonize (it's easy to do per pixel. TODO refactor to create only necessary lines)
#for i in xrange(sz):
#  pair_lines.append(((0,i),(0,i+1)))
#  pair_lines.append(((sz,i),(sz,i+1)))
#  pair_lines.append(((i,0),(i+1,0)))
#  pair_lines.append(((i,sz),(i+1,sz)))

#ps = list(polygonize(pair_lines))


# simple polygonize doesn't work. it is a trick (see http://gis.stackexchange.com/questions/58245/generate-polygons-from-a-set-of-intersecting-lines)
M = MultiLineString(pair_lines)
MB = M.buffer(0.001)
P = Polygon([(0, 0), (0, sz), (sz, sz), (sz, 0)])
pso = P.difference(MB)

# round vertices coords
ps = []
for p in pso:
  pb = p.buffer(0.001)
  pbt = transform(lambda x, y, z=None: (int(round(x)), int(round(y))), pb)
  ps.append(pbt)



# display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.imshow(fimg, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('original', fontsize=20)

ax2.imshow(skeleton, cmap=plt.cm.gray)

if False:
  for al in appr_lines:
    ax2.plot(al[:, 1], al[:, 0])

if True:
  for p in ps:
    coords = list(p.exterior.coords)
    npc = np.array(coords)
    ax2.plot(npc[:, 1], npc[:, 0])


ax2.axis('off')
ax2.set_title('lines', fontsize=20)

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.98,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()

