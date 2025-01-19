import cupy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy
import math


def curve_correct(x, a, d, f, e):
    y = e * x ** 7 + a * x ** 5 + d * x ** 3 + f * x

    return y


def to_cm(a):
    return (a - SIZE / 2) * RADIUS / (SIZE / 2)


def to_pixel(a):
    return int((a / RADIUS) * (SIZE / 2) + SIZE / 2)


RADIUS = 1.35
SIZE = 512
OPT = [-0.23052158335378167, 0.008782290054837563, 1.4378341463152469, 0.04907011811719243]

LOOKUP = None
CUTOUT = None


def stretch(target):
    global LOOKUP

    if LOOKUP is None:
        _LOOKUP = numpy.zeros((SIZE, SIZE, 3), dtype="i")
        for x in range(SIZE):
            for y in range(SIZE):
                _x = to_cm(x)
                _y = to_cm(y)

                dist = math.sqrt(_x ** 2 + _y ** 2)
                if dist > RADIUS:
                    _LOOKUP[x, y, 2] = 0
                    continue

                if dist != 0:
                    factor = curve_correct(dist, *OPT) / dist

                    _x *= factor
                    _y *= factor

                pixel_x = to_pixel(_x)
                pixel_y = to_pixel(_y)

                # new_image[x, y] = target[pixel_x, pixel_y]
                _LOOKUP[x, y, 0] = pixel_x
                _LOOKUP[x, y, 1] = pixel_y
                _LOOKUP[x, y, 2] = 255

        LOOKUP = np.array(_LOOKUP)

    x_coords = LOOKUP[:, :, 0]
    y_coords = LOOKUP[:, :, 1]
    new_image = target[x_coords, y_coords]
    return new_image, LOOKUP[:, :, 2]


if __name__ == "__main__":
    test_image = np.array(Image.open("calibration.jpg").resize((SIZE, SIZE), Image.LANCZOS).convert("RGB"), dtype="int16")
    generated_image,mask = stretch(test_image)
    generated_image, mask = stretch(test_image)
    # print(generated_image.astype("i,i,i,i")[:, :, 0:3].flatten().tolist())
    gen_image = numpy.uint8(generated_image.get())
    gen_mask = numpy.uint8(mask.get())
    # Image.fromarray(gen_mask, "L").save("calibration_test.png")
    plt.imshow(gen_image)
    plt.show()
