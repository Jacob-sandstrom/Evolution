
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math
import sys

import neural_network as nn

# Initializing number of dots
N = 10

FRAMES = 100
INTERVAL = 10

XAREA = 100.0
YAREA = 100.0

XGOAL = 50.0
YGOAL = 50.0
RADIUSGOAL = 10.0

NNSIZE = [2, 64, 128, 64, 2]

# Creating dot class
class dot(object):
    def __init__(self):
        self.neural_network = nn.Network(NNSIZE)

        self.x = XAREA * np.random.random_sample()
        self.y = YAREA * np.random.random_sample()
        self.velx = 0.0
        self.vely = 0.0

    def generate_new_vel(self):
        return (np.random.random_sample() - 0.5) 

    def move(self, frame):
        print(self.x)
        def distance(x1, y1, x2, y2):
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        def inside(x1, y1):
            if distance(x1, y1, XGOAL, YGOAL) <= RADIUSGOAL:
                return True
            else:
                return False


        # if dot is inside the circle it tries to maximize the distances to
        # other dots inside circle
        if not inside(self.x, self.y):

            n_val, z = self.neural_network.feedforward(np.reshape( [self.x,self.y], (2, 1)))

            velx, vely = n_val[-1]

            self.velx = velx[0]*2-1
            self.vely = vely[0]*2-1

            self.x = self.x + self.velx
            self.y = self.y + self.vely

            # Border patrol
            if self.x >= XAREA:
                self.x = XAREA
                self.velx = -1 * self.velx
            if self.x <= 0:
                self.x = 0.0
                self.velx = -1 * self.velx
            if self.y >= YAREA:
                self.y = YAREA
                self.vely = -1 * self.vely
            if self.y <= 0:
                self.y = 0.0
                self.vely = -1 * self.vely




# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, XAREA), ylim=(0, YAREA))
ax.set_aspect(1)
circle = plt.Circle((XGOAL, YGOAL), RADIUSGOAL, color='b', fill=False)
ax.add_artist(circle)


class dots(object):
  def __init__(self):
    self.dots = [dot() for i in range(N)]

  def new(self):
    self.dots = [dot() for i in range(N)]

# Initializing dots
# dots = [dot() for i in range(N)]
dots = dots()

d, = ax.plot([dot.x for dot in dots.dots],
            [dot.y for dot in dots.dots], 'ro')


def restart():
    dots.new()


# animation function.  This is called sequentially
def animate(frame):
  
  # if (frame == FRAMES):
  #   plt.close(fig)
    # sys.exit(0)
  print(dots.dots[0])
  for dot in dots.dots:
    dot.move(frame)
    d.set_data([dot.x for dot in dots.dots],
              [dot.y for dot in dots.dots])
  return d,

def infinite_sequence():
    num = 0
    while True:

        if (num == FRAMES):
            # plt.close(fig)
            num = 0
            restart()

        print(num)
        yield num
        num += 1


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=infinite_sequence, interval=INTERVAL, repeat=False)

plt.show()