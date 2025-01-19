import random

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math

from scipy.optimize import curve_fit

RADIUS = 1


@dataclass
class Vector2:
    x: float = 0
    y: float = 0

    def __mul__(self, other):
        return Vector2(self.x * other, self.y * other)

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def rotate(self, rad):
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        return Vector2(cos_a * self.x - sin_a * self.y,
                       sin_a * self.x + cos_a * self.y)

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def norm(self):
        return Vector2(self.x / self.length(), self.y / self.length())

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def angle_to(self, other):
        return math.acos(self.dot(other) / (self.length() * other.length()))

    def to_tuple(self):
        return self.x, self.y

    def __repr__(self):
        return f"Vector -> {self.x}, {self.y}"


@dataclass
class Vector2Gleichung:
    step: Vector2 = Vector2()

    start: Vector2 = Vector2()

    def position(self, factor):
        return self.start + self.step * factor


def intersection_with_sphere(vec: Vector2Gleichung, fig, ax):
    # f^2 * (x_step^2 + y_step^2) - f * 2(x_step * x_start + y_step * y_start) + (x_start^2 + y_start^2) - RADIUS^2 = 0
    a = vec.step.x ** 2 + vec.step.y ** 2
    b = 2 * (vec.step.x * vec.start.x) + 2 * (vec.step.y * vec.start.y)
    c = vec.start.x ** 2 + vec.start.y ** 2 - RADIUS ** 2

    # f = (-b +- sqrt(b^2 - 4*a*c)) / (2 * a)

    unter_wurzel = b ** 2 - 4 * a * c

    if unter_wurzel < 0:
        return None

    f_0 = (-b + math.sqrt(unter_wurzel)) / (2 * a)
    f_1 = (-b - math.sqrt(unter_wurzel)) / (2 * a)

    f = f_0 if vec.position(f_0).x >= vec.position(f_1).x else f_1

    intersection = vec.position(f)
    print(intersection)

    foci_1 = intersection - Vector2(0, 0)

    normal_vec = foci_1.norm()
    tangent = Vector2(-normal_vec.y, normal_vec.x)

    # print(vec.step, normal_vec)
    alpha_in = math.pi - vec.step.angle_to(normal_vec)
    # print(alpha_in / math.pi)

    alpha_out = math.asin(math.sin(alpha_in) / 1.5)
    print(alpha_out)

    out_vector = (normal_vec * -1).rotate(-alpha_out if normal_vec.y > 0 else alpha_out)
    print("OUT", out_vector)

    # print(out_vector)

    tangent_2 = intersection + normal_vec
    tangent_1 = intersection - normal_vec

    out_vector_1 = intersection
    f = intersection.x / out_vector.x
    out_vector_2 = intersection + out_vector * -f

    ax.plot([vec.start.x, intersection.x], [vec.start.y, intersection.y], linestyle="-")
    ax.plot([tangent_1.x, tangent_2.x], [tangent_1.y, tangent_2.y], linestyle="dashed")
    ax.plot([out_vector_1.x, out_vector_2.x], [out_vector_1.y, out_vector_2.y], linestyle="-")
    return out_vector_2.y, intersection.y


l = []
q = []

fig, ax = plt.subplots()
lens = plt.Circle((0, 0), RADIUS, color="blue", fill=False)


def func(x, a, d, f, e):
    y = e * x ** 7 + a * x ** 5 + d * x ** 3 + f * x

    return y


n = 100
for y in range(n):
    _y = (random.random() - 0.5) * RADIUS * 2
    r, i = intersection_with_sphere(Vector2Gleichung(start=Vector2(10, _y), step=Vector2(-1, 0)), fig, ax)
    print(r, i)
    l.append(r)
    q.append(i)

ax.add_patch(lens)
ax.axis("equal")

plt.show()
plt.clf()

Z = [x for _, x in sorted(zip(l, q))]
l = np.array(sorted(l))
_Z = np.diff(Z) / np.diff(l)

plt.plot(l, Z)

popt = [-0.16854429, -0.02005218, 1.4920612, 0.03456291]
popt, pcov = curve_fit(func, l, Z, maxfev=10000000, p0=popt, method="dogbox")

print(list(popt))

plt.plot(l[:-1], func(l[:-1], *popt))

plt.show()
