import subprocess
from pathlib import Path
from typing import Union

import click

from mapa import conf, convert_tiff_to_stl


@click.command(help="🌍 Convert DEM data into STL files 🌏")
@click.option("--input", help="Path to input TIFF file.")
@click.option("--output", help="Path to output STL file.")
@click.option(
    "--as-ascii", is_flag=True, help="Save output STL as ascii file. If not provided, output file will be binary."
)
@click.option(
    "--model-size",
    default=conf.DEFAULT_MODEL_OUTPUT_SIZE_IN_MM,
    help="Desired size of the (larger side of the) generated 3d model in millimeter.",
)
@click.option(
    "--max-res",
    is_flag=True,
    help=(
        "Whether maximum resolution should be used. Note, that this flag potentially increases compute time "
        "dramatically. The default behavior (i.e. max_res=False) should return 3d models with sufficient "
        "resolution, while the output stl file should be < ~400 MB."
    ),
)
@click.option(
    "--z-offset",
    default=conf.DEFAULT_Z_OFFSET,
    help=(
        f"Offset distance in millimeter to be put below the 3d model. Defaults to {conf.DEFAULT_Z_OFFSET}. "
        f"Is not influenced by z-scale."
    ),
)
@click.option(
    "--z-scale",
    default=conf.DEFAULT_Z_SCALE,
    help=(
        f"Value to be multiplied to the z-axis elevation data to scale up the height of the model. "
        f"Defaults to {conf.DEFAULT_Z_SCALE}."
    ),
)
@click.option("--demo", is_flag=True, help="Converts a demo tif of Hawaii into a STL file.")
@click.option(
    "--cut-to-format-ratio",
    default=None,
    help=(
        "Cut the input tiff file to a specified format. Set to `1` if you want the output model to be squared. Set to "
        "`0.5` if you want one side to be half the length of the other side. Omit this flag to keep the input format. "
        "This option is particularly useful when an exact output format ratio is required for example when planning to "
        "put the 3d printed model into a picture frame. Using this option will always try to cut the shorter side of "
        "the input tiff."
    ),
)
@click.version_option()
def dem2stl(
    input: str,
    output: Union[str, None],
    z_offset: float,
    z_scale: float,
    max_res: bool = False,
    demo: bool = False,
    as_ascii: bool = False,
    model_size: int = conf.DEFAULT_MODEL_OUTPUT_SIZE_IN_MM,
    cut_to_format_ratio: Union[None, float] = None,
) -> None:
    if demo is False and input is None:
        click.echo("💥  Either of --input or --demo is required, try --help.")
        raise click.Abort()
    if demo and input:
        click.echo("💥  Only one of --input or --demo is allowed, try --help.")
        raise click.Abort()
    if demo:
        input = conf.DEMO_TIFF_PATH
        max_res = True
        z_scale = 2.5

    convert_tiff_to_stl(input, as_ascii, model_size, output, max_res, z_offset, z_scale, cut_to_format_ratio)


@click.command(help="🗺 Draw a bounding box on a map and turn it into a STL file 🗺")
@click.version_option()
def mapa() -> None:
    notebook_path = Path(__file__).parent / "mapa.ipynb"
    subprocess.run(["jupyter", "notebook", str(notebook_path)])


if __name__ == "__main__":
    dem2stl()
