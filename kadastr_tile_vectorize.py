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
  return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


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
  if (len(neighbours) == 2): 
    
    for npt in neighbours: trace_tasks.put((npt, pt))
    return line_accu + [pt]
  if (len(neighbours) > 2): 
    #print "WTF!!! I thougth the maximum value is 2"
    #print pt, neighbours
    #sys.exit()
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
        # pt is in a branching point
        if (len(neighbours) == 3):
          for npt in neighbours: trace_tasks.put((npt, pt))
        # crossing?
        if (len(neighbours) > 3):
          # there can be T-crossings
          #print "WTF!!! I thought the maximum value is 3"
          #print pt, neighbours
          #sys.exit()
          for npt in neighbours: trace_tasks.put((npt, pt))
        while (not trace_tasks.empty()):
          task = trace_tasks.get()
          yield trace_line(img, task[0], task[1])

        
  
  

fimg = misc.imread("wms_simple_small.png")

gimg = 1 - color.colorconv.rgb2grey(fimg)

thresh = threshold_otsu(gimg)
binary = gimg > thresh
 
skeleton = skeletonize(binary)

print "find_lines"
lines = list(find_lines(skeleton))
print "find_lines end"

print len(lines)

# merge neighbour points
for l1 in lines: 
  for l2 in lines:
    if (distance(l1[0], l2[0]) < 2):
      mp = middle_point(l1[0], l2[0])
      l1[0] = mp
      l2[0] = mp
    if (distance(l1[0], l2[-1]) < 2):
      mp = middle_point(l1[0], l2[-1])
      l1[0] = mp
      l2[-1] = mp
    if (distance(l1[-1], l2[0]) < 2):
      mp = middle_point(l1[-1], l2[0])
      l1[-1] = mp
      l2[0] = mp
    if (distance(l1[-1], l2[-1]) < 2):
      mp = middle_point(l1[-1], l2[-1])
      l1[-1] = mp
      l2[-1] = mp

# remove singular lines
lines = filter(lambda x: not (len(x) < 2 or (len(x) == 2 and distance(x[0],x[1]) < 2)), lines)

# display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.imshow(fimg, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('original', fontsize=20)

ax2.imshow(skeleton, cmap=plt.cm.gray)

print len(lines)

for l in lines: 
  if (not None == l):
    npl = np.array(l)
    #ax2.plot(npl[:, 1], npl[:, 0])
    # approximate with douglas-peucker
    appr_npl = approximate_polygon(npl, tolerance=1.5)

    ax2.plot(appr_npl[:, 1], appr_npl[:, 0])

ax2.axis('off')
ax2.set_title('lines', fontsize=20)

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.98,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()

