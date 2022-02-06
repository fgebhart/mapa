import subprocess
from pathlib import Path
from typing import Union

import click

from mapa import conf, convert_tif_to_stl


@click.command(help="🌍 Convert DEM data into STL files 🌏")
@click.option("--input", help="Path to input TIFF file.")
@click.option("--output", help="Path to output STL file.")
@click.option(
    "--as-ascii", is_flag=True, help="Save output STL as ascii file. If not provided, output file will be binary."
)
@click.option(
    "--model-size",
    default=conf.DEFAULT_MODEL_OUTPUT_SIZE_IN_MM,
    help="Desired size of the generated 3d model in millimeter.",
)
@click.option(
    "--max-res",
    is_flag=True,
    help=(
        "Whether maximum resolution should be used. Note, that this flag potentially increases compute time "
        "dramatically. The default behavior (i.e. max_res=False) should return 3d models with sufficient "
        "resolution, while the output stl file should be <= 200 MB."
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
    "--make-square",
    is_flag=True,
    help="If the input tiff is a rectangle and not a square, cut the longer side to make the output STL file a square.",
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
    make_square: bool = False,
) -> None:
    if demo is False and input is None:
        click.echo("💥  Either of --input or --demo is required, try --help.")
        exit(1)
    if demo and input:
        click.echo("💥  Only one of --input or --demo is allowed, try --help.")
        exit(1)
    if demo:
        input = conf.DEMO_TIF_PATH
        max_res = True

    convert_tif_to_stl(input, as_ascii, model_size, output, max_res, z_offset, z_scale, make_square)


@click.command(help="🗺 Draw a bounding box on a map and turn it into a STL file 🗺")
@click.version_option()
def mapa() -> None:
    notebook_path = Path(__file__).parent / "mapa.ipynb"
    subprocess.run(["jupyter", "notebook", str(notebook_path)])


if __name__ == "__main__":
    dem2stl()
