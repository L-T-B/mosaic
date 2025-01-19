import math
import random
import sys
import os, os.path
from PIL import Image, ImageOps, ImageDraw
from multiprocessing import Process, Queue, cpu_count, Pool
import cupy as np
import numpy
import strecher
from tqdm import tqdm

# Change these 3 config parameters to suit your needs...
TILE_SIZE = 512  # height/width of mosaic tiles in pixels
TILE_MATCH_RES = 50  # tile matching resolution (higher values give better fit but require more processing)
ROWS = 33
COLUMNS = 33
SPHERE_SIZE = 2.7  # cm
PADDING = 0.25
PIXEL_PADDING = math.floor(PADDING / SPHERE_SIZE * TILE_SIZE)
PIXEL_PER_CM = TILE_SIZE / SPHERE_SIZE
A4_WIDTH = 17 * PIXEL_PER_CM
A4_HEIGHT = 24.7 * PIXEL_PER_CM

TILE_BLOCK_SIZE = TILE_SIZE / max(min(TILE_MATCH_RES, TILE_SIZE), 1)
WORKER_COUNT = max(cpu_count() - 2, 1)
OUT_FILE = 'mosaic.png'
low_file = "mosaic-low.png"
PDF_FILE = 'mosaic_print.pdf'
EOQ_VALUE = None

S_SIZE = math.floor(TILE_SIZE / 2 * math.sqrt(3))


def process_tile(tile_path):
    try:
        img = Image.open(tile_path)
        img = ImageOps.exif_transpose(img)

        # tiles must be square, so get the largest square that fits inside the image
        w = img.size[0]
        h = img.size[1]
        min_dimension = min(w, h)
        w_crop = (w - min_dimension) // 2
        h_crop = (h - min_dimension) // 2
        img = img.crop((w_crop, h_crop, w - w_crop, h - h_crop))

        large_tile_img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
        small_tile_img = img.resize((int(TILE_SIZE / TILE_BLOCK_SIZE), int(TILE_SIZE / TILE_BLOCK_SIZE)),
                                    Image.LANCZOS)

        return numpy.array(large_tile_img.convert('RGB'), dtype="i"), \
            numpy.array(small_tile_img.convert('RGB')).reshape((2500, 3))
    except Exception as e:
        print(repr(e))
        return None, None


class TileProcessor:
    def __init__(self, tiles_directory):
        self.tiles_directory = tiles_directory

    def get_tiles(self):
        large_tiles = []
        small_tiles = []

        print('Reading tiles from {}...'.format(self.tiles_directory))

        paths = []
        # search the tiles directory recursively
        for root, subFolders, files in os.walk(self.tiles_directory):
            for tile_name in files:
                print('Reading {:40.40}'.format(tile_name), flush=True, end='\r')
                tile_path = os.path.join(root, tile_name)
                paths.append(tile_path)

        with Pool() as p:
            imap_pictures = p.map(process_tile, paths)

        for large_tile, small_tile in imap_pictures:
            if large_tile is not None:
                large_tiles.append(large_tile)
                small_tiles.append(small_tile)

        print('Processed {} tiles.'.format(len(large_tiles)))

        return large_tiles, small_tiles


class TargetImage:
    def __init__(self, image_path):
        self.image_path = image_path

    def get_data(self):
        print('Processing main image...')
        img = Image.open(self.image_path)
        x = COLUMNS * TILE_SIZE + (COLUMNS - 1) * PIXEL_PADDING
        y = (ROWS - 1) * S_SIZE + (ROWS - 1) * PIXEL_PADDING + TILE_SIZE
        img = ImageOps.fit(img, (x, y), Image.LANCZOS)

        small_img = img.resize(
            (
                int(x / TILE_BLOCK_SIZE),
                int(y / TILE_BLOCK_SIZE)
            ),
            Image.ANTIALIAS)

        image_data = (img.convert('RGB'), small_img.convert('RGB'))

        print('Main image processed.')

        return image_data


class TileFitter:
    def __init__(self, tiles_data):
        self.tiles_data = tiles_data

    def __get_tile_diff(self, t1, t2, bail_out_value):
        dist = numpy.power(t1 - t2, 2)
        diff = numpy.sum(numpy.matmul(dist, numpy.array([0.3, 0.59, 0.11])))
        factor = random.random() / 2 + 0.1
        return diff * factor

    def get_best_fit_tile(self, img_data, helper):
        best_fit_tile_index = None
        min_diff = sys.maxsize
        tile_index = helper

        # go through each tile in turn looking for the best match for the part of the image represented by 'img_data'
        while tile_index < len(self.tiles_data):
            diff = self.__get_tile_diff(img_data, self.tiles_data[tile_index], min_diff)
            if diff < min_diff:
                min_diff = diff
                best_fit_tile_index = tile_index
            tile_index += 3

        return best_fit_tile_index


def fit_tiles(work_queue, result_queue, tiles_data):
    # this function gets run by the worker processes, one on each CPU core
    tile_fitter = TileFitter(tiles_data)

    while True:
        try:
            img_data, img_coords, helper = work_queue.get(True)
            if img_data is None:
                break
            tile_index = tile_fitter.get_best_fit_tile(img_data, helper)
            result_queue.put((img_coords, tile_index))
        except KeyboardInterrupt:
            pass

    # let the result handler know that this worker has finished everything
    result_queue.put((EOQ_VALUE, EOQ_VALUE))


class ProgressCounter:
    def __init__(self, total):
        self.total = total
        self.counter = 0

    def update(self):
        self.counter += 1
        print("Progress: {:04.1f}%".format(100 * self.counter / self.total), flush=True, end='\r')


class MosaicImage:
    def __init__(self, original_img):
        self.image = Image.new("RGB", original_img.size, color="white")
        self.x_tile_count = COLUMNS
        self.y_tile_count = ROWS

        for y in tqdm(range(self.y_tile_count)):
            for x in range(self.x_tile_count - y % 2):
                box = calculate_large_box(x, y)
                cropped_image = np.array(original_img.crop(box).convert("RGB"))
                _image, _mask = strecher.stretch(cropped_image)
                gen_image = numpy.uint8(_image.get())
                gen_mask = numpy.uint8(_mask.get())
                img = Image.fromarray(gen_image)
                mask = Image.fromarray(gen_mask, "L")
                self.image.paste(img, box, mask=mask)

    def add_tile(self, tile_data, coords):
        _image, _mask = strecher.stretch(np.array(tile_data))
        clean = numpy.uint8(_image.get())
        img = Image.fromarray(clean, 'RGB')
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, TILE_SIZE, TILE_SIZE), fill=170)
        # mask = mask.filter(ImageFilter.GaussianBlur(TILE_SIZE/16))
        self.image.paste(img, coords, mask)

    def save(self, path):
        self.image.save(path)

        picture_width, picture_height = self.image.size


        images = []
        for x in range(int(picture_width // A4_width + 1)):
            for y in range(int(picture_height // A4_height + 1)):
                box = (
                    int(x * A4_width),
                    int(y * A4_height),
                    min(int((x + 1) * A4_width + 2.7 * PIXEL_PER_CM), picture_width),
                    min(int((y + 1) * A4_height + 2.7 * PIXEL_PER_CM), picture_height)
                )
                n_image = Image.new("RGB", (int(A4_width+ 2.7 * PIXEL_PER_CM), int(A4_height+ 2.7 * PIXEL_PER_CM)), color="white")
                n_image.paste(self.image.crop(box), (0, 0))
                images.append(n_image)

        images[0].save(PDF_FILE, save_all=True, append_images=images[1:], dpi=(483, 483))

        self.image.thumbnail((4096, 4096), Image.LANCZOS)
        self.image.save(low_file)


def build_mosaic(result_queue, all_tile_data_large, mosaic):
    active_workers = WORKER_COUNT
    while True:
        try:
            img_coords, best_fit_tile_index = result_queue.get()

            if img_coords == EOQ_VALUE:
                active_workers -= 1
                if not active_workers:
                    break
            else:
                tile_data = all_tile_data_large[best_fit_tile_index]
                mosaic.add_tile(tile_data, img_coords)

        except KeyboardInterrupt:
            pass

    mosaic.save(OUT_FILE)
    print('\nFinished, output is in', OUT_FILE)


def calculate_large_box(x, y):
    return (
        x * (TILE_SIZE + PIXEL_PADDING) + y % 2 * (TILE_SIZE + PIXEL_PADDING) // 2,
        int(y * (S_SIZE + PIXEL_PADDING)),
        x * (TILE_SIZE + PIXEL_PADDING) + TILE_SIZE + y % 2 * (TILE_SIZE + PIXEL_PADDING) // 2,
        int(y * (S_SIZE + PIXEL_PADDING) + TILE_SIZE)
    )


def compose(original_img, tiles):
    print('Building mosaic, press Ctrl-C to abort...')
    original_img_large, original_img_small = original_img
    all_tile_data_large, all_tile_data_small = tiles

    mosaic = MosaicImage(original_img_large)

    work_queue = Queue(WORKER_COUNT)
    result_queue = Queue()

    try:
        # start the worker processes that will build the mosaic image
        Process(target=build_mosaic, args=(result_queue, all_tile_data_large, mosaic)).start()

        # start the worker processes that will perform the tile fitting
        for n in range(WORKER_COUNT):
            Process(target=fit_tiles, args=(work_queue, result_queue, all_tile_data_small)).start()

        progress = ProgressCounter(mosaic.x_tile_count * mosaic.y_tile_count - mosaic.x_tile_count // 2)
        for y in range(mosaic.y_tile_count):
            for x in range(mosaic.x_tile_count - y % 2):
                large_box = calculate_large_box(x, y)
                small_box = (
                    large_box[0] / TILE_BLOCK_SIZE,
                    large_box[1] / TILE_BLOCK_SIZE,
                    large_box[2] / TILE_BLOCK_SIZE,
                    large_box[3] / TILE_BLOCK_SIZE
                )
                # img_data, coords, helper
                work_queue.put(
                    (
                        numpy.array(original_img_small.crop(small_box).getdata()),
                        large_box,
                        (x - y % 2) % 3
                    )
                )
                progress.update()

    except KeyboardInterrupt:
        print('\nHalting, saving partial image please wait...')

    finally:
        # put these special values onto the queue to let the workers know they can terminate
        for n in range(WORKER_COUNT):
            work_queue.put((EOQ_VALUE, EOQ_VALUE, EOQ_VALUE))


def show_error(msg):
    print('ERROR: {}'.format(msg))


def mosaic(img_path, tiles_path):
    image_data = TargetImage(img_path).get_data()
    tiles_data = TileProcessor(tiles_path).get_tiles()
    if tiles_data[0]:
        compose(image_data, tiles_data)
    else:
        show_error("No images found in tiles directory '{}'".format(tiles_path))


if __name__ == '__main__':
    source_image = "test-6.png"
    tile_dir = "files/"
    mosaic(source_image, tile_dir)
