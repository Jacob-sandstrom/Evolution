
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math
import sys

import neural_network as nn

# Initializing number of dots
N = 10

NEWRANDOMRATE = 0.8  # 0.75 means 1/4 of of dots are always new

FRAMES = 100
INTERVAL = 10

XAREA = 200
YAREA = 200

XGOAL = 100
YGOAL = 100
RADIUSGOAL = 10

NNSIZE = [2, 64, 128, 64, 2]

# Creating dot class


class dot(object):
  def __init__(self):
    self.neural_network = nn.Network(NNSIZE)

    self.x = XAREA * np.random.random_sample()
    self.y = YAREA * np.random.random_sample()
    self.velx = 0.0
    self.vely = 0.0

    self.time_to_goal = None

  def rand_position(self):
    self.x = XAREA * np.random.random_sample()
    self.y = YAREA * np.random.random_sample()

  def generate_new_vel(self):
    return (np.random.random_sample() - 0.5)

  def evaluate(self):
    return self.inside()

  def mutate(self):
    self.neural_network.mutate()

  def distance(self, x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

  def inside(self):
    if self.distance(self.x, self.y, XGOAL, YGOAL) <= RADIUSGOAL:
      return True
    else:
      return False

  def move(self, frame):
    # if dot is inside the circle it stops moving
    if not self.inside():
      n_val, z = self.neural_network.feedforward(
        np.reshape([self.x, self.y], (2, 1)))

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
    else:
      # print(self.time_to_goal)
      if self.time_to_goal == None:
        self.time_to_goal = frame


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
    successful_dots = []
    for d in self.dots:
      if d.evaluate():
        successful_dots.append(d)
    successful_dots.sort(key=lambda d: d.time_to_goal)

    self.dots = [dot() for i in range(N)]

    for i in range(len(successful_dots)):
      successful_dots[i].rand_position()

      successful_dots[i].mutate()

      self.dots[i] = successful_dots[i]

      if i > N/NEWRANDOMRATE:
        break


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

    # print(num)
    yield num
    num += 1


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=infinite_sequence, interval=INTERVAL, repeat=False)

plt.show()
