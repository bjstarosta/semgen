import logging
import click

import generator
import distorter
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
            utils.save_image(ctx.obj['to_write'][i], im)
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
        utils.write_params(ctx.obj['image_path'], prm_log)

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
    ctx.obj['to_read'] = utils.find_filenames(ctx.obj['src_path'])
    ctx.obj['to_write'] = utils.get_filenames(
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
            utils.save_image(ctx.obj['to_write'][i], im)
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
        utils.write_params(ctx.obj['image_path'], prm_log)

if __name__ == '__main__':
    main(obj={})
