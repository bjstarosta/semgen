# -*- coding: utf-8 -*-
"""SEMGen - SEM synthetic image generator.

Simulation code is based primarily off of independent research conducted by
other teams. This package in particular owes a lot to Cizmar et al (2008) for
it's implementation of the gold on carbon generator and the SEM distortion.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import logging
import click

import generators
import distorters
import utils


VERSION = 0.1
IMG_PTRN = "semgen-{0:04d}"
IMG_EXT = '.tif'
PARAM_FILE = 'semgen-prm.txt'


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
    """Generate new images or transform existing images with SEM distortion."""
    LOG_FORMAT = '[%(levelname)s] %(message)s'
    if kwargs['quiet'] is True:
        LOG_LEVEL = 'ERROR'
    elif kwargs['verbose'] is True:
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
    type=click.Path(exists=True, file_okay=False, readable=True)
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
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="""If pointed at a previously generated parameters file, will use
        it to attempt to exactly recreate those conditions during this
        batch generation."""
)
@click.pass_context
def generate(ctx, **kwargs):
    """Generate synthetic images of SEM scanned surfaces.

    Output format will be TIFF, and generated images will have sequential
    names.
    """
    ctx.obj['image_path'] = kwargs['destination_dir']
    ctx.obj['image_dim'] = kwargs['dim']
    ctx.obj['image_n'] = kwargs['number']
    ctx.obj['to_write'] = utils.get_filenames(
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
    '-g',
    '--grey-range',
    type=float,
    default=1.,
    show_default=True,
    help="""Range of greys present in the gradient. Default setting generates
        a black-white gradient, number lower than "1" causes gradients to be
        generated with a smaller range between the brightest and darkest
        colour, with the starting point of the range generated randomly."""
)
@click.option(
    '-l',
    '--grey-limit',
    type=(float, float),
    default=(0., 1.),
    help="""Sets the darkest and brightest possible grey on a floating point scale.
        Default: 0 1""",
    metavar="D B"
)
@click.pass_context
def gradient(ctx, **kwargs):
    """Generate gradient images."""
    logging.info("Generator: gradient")

    gen = generators.factory('gradient', 'GradientGenerator')
    gen.dim = ctx.obj['image_dim']
    gen.grey_range = kwargs['grey_range']
    gen.grey_limit = kwargs['grey_limit']
    gen.queue_images(ctx.obj['image_n'])

    prm_log = {
        'generator': 'gradient',
        'global': {},  # TODO: dump generator class properties into this
        'params': []
    }

    generate_images(ctx, gen, prm_log)


@generate.command()
@click.option(
    '-g',
    '--grey-range',
    type=float,
    default=1.,
    show_default=True,
    help="""Range of greys present in the gradient. Default setting generates
        a black-white gradient, number lower than "1" causes gradients to be
        generated with a smaller range between the brightest and darkest
        colour, with the starting point of the range generated randomly."""
)
@click.option(
    '-l',
    '--grey-limit',
    type=(float, float),
    default=(0., 1.),
    show_default=True,
    help="""Sets the darkest and brightest possible grey on a floating point
        scale. The result image will be losslessly feature scaled to this
        range.""",
    metavar="D B"
)
@click.option(
    '-c',
    '--clip',
    type=(float, float),
    default=(1., 1.),
    show_default=True,
    help="""Sets the boundaries of a randomised, symmetric colour value clip if
        the first value is set to be less than the second one. The difference
        between this option and grey limit is that no feature scaling is
        applied in this step. Default values mean no clipping is applied.""",
    metavar="L H"
)
@click.pass_context
def dipole(ctx, **kwargs):
    """Generate dipole-like images."""
    logging.info("Generator: dipole")

    gen = generators.factory('dipole', 'DipoleGenerator')
    gen.dim = ctx.obj['image_dim']
    gen.grey_range = kwargs['grey_range']
    gen.grey_limit = kwargs['grey_limit']
    gen.clip = kwargs['clip']
    gen.queue_images(ctx.obj['image_n'])

    prm_log = {
        'generator': 'dipole',
        'global': {},  # TODO: dump generator class properties into this
        'params': []
    }

    generate_images(ctx, gen, prm_log)


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
    """Generate a gold on carbon SEM test case image."""
    logging.info("Generator: gold-on-carbon SEM test sample")

    gen = generators.factory('goldoncarbon', 'GoldOnCarbonGenerator')
    gen.dim = ctx.obj['image_dim']
    gen.queue_images(ctx.obj['image_n'])

    prm_log = {
        'generator': 'gold',
        'global': {},  # TODO: dump generator class properties into this
        'params': []
    }

    generate_images(ctx, gen, prm_log)


@main.group()
@click.argument(
    'source_dir',
    nargs=1,
    type=click.Path(exists=True, file_okay=False, readable=True)
)
@click.argument(
    'destination_dir',
    nargs=1,
    type=click.Path(exists=True, file_okay=False, writable=True)
)
@click.option(
    '-d',
    '--dim',
    type=(int, int),
    default=(0, 0),
    show_default=True,
    help="""Dimensions (W, H) to resize the distorted images to.
        Defaults to no resize applied.""",
    metavar="W H"
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
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="""If pointed at a previously generated parameters file, will use
        it to attempt to exactly recreate those conditions during this
        batch generation."""
)
@click.pass_context
def distort(ctx, **kwargs):
    """Transform existing images with SEM noise and distortion.

    Output format will be TIFF, and generated images will have sequential
    names.
    """
    ctx.obj['src_path'] = kwargs['source_dir']
    ctx.obj['dst_path'] = kwargs['destination_dir']
    ctx.obj['image_resize'] = kwargs['dim']
    ctx.obj['to_read'] = utils.find_filenames(ctx.obj['src_path'])
    """ctx.obj['to_write'] = utils.get_filenames(
        ctx.obj['dst_path'],
        len(ctx.obj['to_read']),
        overwrite=kwargs['overwrite']
    )"""
    ctx.obj['to_write'] = utils.remap_filenames(
        ctx.obj['to_read'], ctx.obj['dst_path'])
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
    """Simulate time-dependent distortions and noise on an existing image."""
    logging.info("Distorter: SEM noise generator")

    dst = distorters.factory('semnoise', 'SEMNoiseGenerator')
    dst.queue_images(ctx.obj['to_read'])

    prm_log = {
        'distorter': 'semnoise',
        'global': {},  # TODO: dump generator class properties into this
        'params': []
    }

    distort_images(ctx, dst, prm_log)


def generate_images(ctx, gen, prm_log):
    """Iterate through passed Generator object to generate synthetic images.

    This function serves as a controller between the command line and the
    actual generation code contained in the passed Generator object.

    Args:
        ctx (dict): Context object passed from click.
            Contains parameter/option data passed from the command line.
        gen (generators.Generator): Generator object to iterate.
        prm_log (dict): Serialisable object containing parameters required
            to recreate the currently simulated batch of images.

    """
    logging.info("Generation begins...")

    with click.progressbar(
        label='Generating images...',
        length=ctx.obj['image_n'],
        show_pos=True
    ) as pbar:
        i = 0
        for im in gen:
            utils.save_image(ctx.obj['to_write'][i], im)
            prm_log['params'].append(gen.params_current)
            i = i + 1
            # Disable progress bar if verbose or quiet is enabled
            if (ctx.obj['pbar'] is True
            and ctx.obj['quiet'] is False
            and ctx.obj['verbose'] is False):
                pbar.update(1)

    logging.info("{0:d} images generated in '{1}'.".format(
        i,
        click.format_filename(os.path.abspath(ctx.obj['image_path']))
    ))

    if ctx.obj['log_params'] is True:
        logging.info("Param file written to '{0}'.".format(
            click.format_filename(os.path.abspath(ctx.obj['image_path']))
        ))
        utils.write_params(ctx.obj['image_path'], prm_log)


def distort_images(ctx, dst, prm_log):
    """Iterate through passed Distorter object to alter images.

    This function serves as a controller between the command line and the
    actual distortion code contained in the passed Distorter object.

    Args:
        ctx (dict): Context object passed from click.
            Contains parameter/option data passed from the command line.
        gen (distorters.Distorter): Generator object to iterate.
        prm_log (dict): Serialisable object containing parameters required
            to recreate the currently simulated batch of images.

    """
    if ctx.obj['image_resize'] != (0, 0):
        logging.info("Images will be resized to {0:d}:{1:d} pixels.".format(
            ctx.obj['image_resize'][0], ctx.obj['image_resize'][1]))

    logging.info("Distortion begins...")

    with click.progressbar(
        label='Distorting images...',
        length=len(ctx.obj['to_read']),
        show_pos=True
    ) as pbar:
        i = 0
        for im in dst:
            # im = feature_scale(im, 0, 255, 0., 1., 'uint8')
            if ctx.obj['image_resize'] != (0, 0):
                im = utils.resize_image(im, ctx.obj['image_resize'])

            utils.save_image(ctx.obj['to_write'][i], im)
            prm_log['params'].append(dst.params_current)
            i = i + 1
            # Disable progress bar if verbose or quiet is enabled
            if (ctx.obj['pbar'] is True
            and ctx.obj['quiet'] is False
            and ctx.obj['verbose'] is False):
                pbar.update(1)

    logging.info("{0:d} images generated in '{1}'.".format(
        i,
        click.format_filename(os.path.abspath(ctx.obj['dst_path']))
    ))

    if ctx.obj['log_params'] is True:
        logging.info("Param file written to '{0}'.".format(
            click.format_filename(os.path.abspath(ctx.obj['dst_path']))
        ))
        utils.write_params(ctx.obj['dst_path'], prm_log)


if __name__ == '__main__':
    main(obj={})
