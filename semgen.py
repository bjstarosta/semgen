import os
import re
import logging
import json

import click
import numpy as np
import skimage.external.tifffile as tifffile

import generator
import distorter


VERSION = 0.1
IMG_PTRN = "semgen-{0:04d}"
IMG_EXT = '.tif'
PARAM_FILE = 'semgen-prm.txt'

def find_filenames(dir, format=None):
    if format is None:
        format = IMG_PTRN

    ret = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            if not f.endswith(IMG_EXT):
                continue
            d = string_to_dict(f, format)
            if d is None:
                continue
            ret.append(os.path.join(dir, f))

    return ret

def get_filenames(dir, n, format=None, overwrite=False):
    if format is None:
        format = IMG_PTRN

    format = format + IMG_EXT
    if overwrite == True:
        return _get_filenames(dir, n, 0, format)

    i = 0
    for root, dirs, files in os.walk(dir):
        for f in files:
            if not f.endswith(IMG_EXT):
                continue
            d = string_to_dict(f, format)
            j = int(d['0'])
            if j >= i:
                i = j + 1

    return _get_filenames(dir, n, i, format)

def _get_filenames(dir, n, i, format):
    ret = []
    for i in range(i, i+n):
        ret.append(os.path.join(dir, format.format(i)))
    return ret

def string_to_dict(string, pattern):
    regex = re.sub(r'{(.+?)(?:\:.+)}', r'(?P<_\1>.+)', pattern)
    values = re.search(regex, string)
    if values is None:
        return None
    values = list(values.groups())
    keys = re.findall(r'{(.+?)(?:\:.+)}', pattern)
    return dict(zip(keys, values))

def write_params(dir, data):
    data['_version'] = VERSION
    with click.open_file(os.path.join(dir, PARAM_FILE), 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)

def read_params(dir):
    with click.open_file(os.path.join(dir, PARAM_FILE), 'r') as f:
        data = json.load(f)
    return data

def feature_scale(img, a, b, min=None, max=None, type=None):
    """Min-max normalises a 2-dim array containing image data.

    See: https://en.wikipedia.org/wiki/Feature_scaling

    Args:
        img (ndarray): Array containing image data.
        a (float): Minimum value of the result.
        b (float): Maximum value of the result.
        min (float): Minimum allowed value of the input.
        max (float): Maximum allowed value of the input.
        type (object): Convert output to this value type. E.g. np.uint8

    Returns:
        ndarray: Array containing normalised image data.

    """
    img = np.asarray(img, dtype=np.float16)
    if min is None:
        min = np.min(img)
    if max is None:
        max = np.max(img)
    img = a + ((img - min) * (b - a)) / (max - min)
    if type is not None:
        img = img.astype(type)
    return img

def load_image(path):
    with tifffile.TiffFile(path) as tif:
        return tif.asarray()


@click.group()
@click.version_option(prog_name="SEMGen", version=VERSION)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help="""Logs debug messages during script run."""
)
@click.option(
    '-q',
    '--quiet',
    is_flag=True,
    help="""Only logs error level messages and above."""
)
@click.option(
    '-p',
    '--progress-bar',
    is_flag=True,
    help="""Displays a progress bar."""
)
@click.pass_context
def main(ctx, **kwargs):
    """Generates new images or transforms existing images with SEM noise and
    distortion.
    """
    LOG_FORMAT = '[%(levelname)s] %(message)s'
    if kwargs['quiet'] == True:
        LOG_LEVEL = 'ERROR'
    elif kwargs['verbose'] == True:
        LOG_LEVEL = 'DEBUG'
    else:
        LOG_LEVEL = 'INFO'

    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)

    ctx.obj['quiet'] = kwargs['quiet']
    ctx.obj['verbose'] = kwargs['verbose']
    ctx.obj['pbar'] = kwargs['progress_bar']

@main.group()
@click.argument(
    'destination_dir',
    nargs=1,
    type=click.Path(exists=True)
)
@click.option(
    '-d',
    '--dim',
    type=(int, int),
    default=(1000, 1000),
    help="""Dimensions (W, H) of the generated images. Default: 1000 1000""",
    metavar="W H"
)
@click.option(
    '-n',
    '--number',
    type=int,
    default=1,
    help="""Number of images to generate. Default: 1"""
)
@click.option(
    '-o',
    '--overwrite',
    is_flag=True,
    help="""If set, will overwrite sequential images in a folder starting
        from 0, instead of adding to them."""
)
@click.option(
    '-l',
    '--log-params',
    is_flag=True,
    help="""If set, will write a text file into the destination folder
        containing all the parameters required to recreate the generated
        images."""
)
@click.option(
    '-u',
    '--use-params',
    type=click.Path(exists=True),
    help="""If pointed at a previously generated parameters file, will use
        it to attempt to exactly recreate those conditions during this
        batch generation."""
)
@click.pass_context
def generate(ctx, **kwargs):
    """Generates new images that mimic dislocations on GaN surfaces.

    Output format will be TIFF, and generated images will have sequential
    names.
    """
    ctx.obj['image_path'] = kwargs['destination_dir']
    ctx.obj['image_dim'] = kwargs['dim']
    ctx.obj['image_n'] = kwargs['number']
    ctx.obj['to_write'] = get_filenames(
        ctx.obj['image_path'],
        ctx.obj['image_n'],
        overwrite=kwargs['overwrite']
    )
    ctx.obj['log_params'] = kwargs['log_params']
    ctx.obj['use_params'] = kwargs['use_params']

    logging.info("Enqueued {0:d} images.".format(ctx.obj['image_n']))
    logging.info("Images will be generated starting from index: {0}".format(
        click.format_filename(ctx.obj['to_write'][0])
    ))

@generate.command()
@click.option(
    '-gn',
    '--grain-number',
    type=(int, int),
    default=(4, 8),
    help="""Number grains to generate per image. Operates as a range from
        which a random number will be chosen. Default: 4 to 8"""
)
# TODO: Expand options
@click.pass_context
def gold(ctx, **kwargs):
    """Generate a gold on carbon SEM test case image.
    """
    logging.info("Generator: gold-on-carbon SEM test sample")

    gen = generator.GoldOnCarbonGenerator()
    gen.dim = ctx.obj['image_dim']
    gen.queue_images(ctx.obj['image_n'])

    prm_log = {
        'generator': 'gold',
        'global': {}, # TODO: dump generator class properties into this
        'params': []
    }

    # Disable progress bar if verbose or quiet is enabled
    if ctx.obj['pbar'] == True and ctx.obj['quiet'] == False and ctx.obj['verbose'] == False:
        pbar = click.progressbar(gen,
            label='Generating images...',
            length=ctx.obj['image_n']
        )
    else:
        pbar = gen

    logging.info("Generation begins...")
    with pbar as gen_:
        i = 0
        for im, params in gen_:
            im = feature_scale(im, 0, 255, 0., 1., 'uint8')
            tifffile.imsave(ctx.obj['to_write'][i], im)
            prm_log['params'].append(params)
            i = i + 1

    logging.info("{0:d} images generated in '{1}'.".format(
        i,
        click.format_filename(os.path.abspath(ctx.obj['image_path']))
    ))

    if ctx.obj['log_params'] == True:
        logging.info("Param file written to '{0}'.".format(
            click.format_filename(os.path.abspath(ctx.obj['image_path']))
        ))
        write_params(ctx.obj['image_path'], prm_log)

@main.group()
@click.argument(
    'source_dir',
    nargs=1,
    type=click.Path(exists=True)
)
@click.argument(
    'destination_dir',
    nargs=1,
    type=click.Path(exists=True)
)
@click.option(
    '-o',
    '--overwrite',
    is_flag=True,
    help="""If set, will overwrite sequential images in a folder starting
        from 0, instead of adding to them."""
)
@click.option(
    '-l',
    '--log-params',
    is_flag=True,
    help="""If set, will write a text file into the destination folder
        containing all the parameters required to recreate the generated
        images."""
)
@click.option(
    '-u',
    '--use-params',
    type=click.Path(exists=True),
    help="""If pointed at a previously generated parameters file, will use
        it to attempt to exactly recreate those conditions during this
        batch generation."""
)
@click.pass_context
def distort(ctx, **kwargs):
    """Transforms existing images with SEM noise and distortion.

    Output format will be TIFF, and generated images will have sequential
    names.
    """
    ctx.obj['src_path'] = kwargs['source_dir']
    ctx.obj['dst_path'] = kwargs['destination_dir']
    ctx.obj['to_read'] = find_filenames(ctx.obj['src_path'])
    ctx.obj['to_write'] = get_filenames(
        ctx.obj['dst_path'],
        len(ctx.obj['to_read']),
        overwrite=kwargs['overwrite']
    )
    ctx.obj['log_params'] = kwargs['log_params']
    ctx.obj['use_params'] = kwargs['use_params']

    logging.info("Enqueued {0:d} images.".format(len(ctx.obj['to_read'])))
    logging.info("Images will be generated starting from index: {0}".format(
        click.format_filename(ctx.obj['to_write'][0])
    ))

@distort.command()
# TODO: Expand options
@click.pass_context
def semnoise(ctx, **kwargs):
    """Simulate SEM time-dependent distortions and noise on an existing image.
    """
    logging.info("Distorter: SEM noise generator")

    dst = distorter.SEMNoiseGenerator()
    dst.queue_images(ctx.obj['to_read'])

    prm_log = {
        'distorter': 'semnoise',
        'global': {}, # TODO: dump generator class properties into this
        'params': []
    }

    # Disable progress bar if verbose or quiet is enabled
    if ctx.obj['pbar'] == True and ctx.obj['quiet'] == False and ctx.obj['verbose'] == False:
        pbar = click.progressbar(dst,
            label='Distorting images...',
            length=len(ctx.obj['to_read'])
        )
    else:
        pbar = dst

    logging.info("Distortion begins...")
    with pbar as dst_:
        i = 0
        for im, params in dst_:
            #im = feature_scale(im, 0, 255, 0., 1., 'uint8')
            tifffile.imsave(ctx.obj['to_write'][i], im)
            prm_log['params'].append(params)
            i = i + 1

    logging.info("{0:d} images generated in '{1}'.".format(
        i,
        click.format_filename(os.path.abspath(ctx.obj['image_path']))
    ))

    if ctx.obj['log_params'] == True:
        logging.info("Param file written to '{0}'.".format(
            click.format_filename(os.path.abspath(ctx.obj['image_path']))
        ))
        write_params(ctx.obj['image_path'], prm_log)

if __name__ == '__main__':
    main(obj={})
