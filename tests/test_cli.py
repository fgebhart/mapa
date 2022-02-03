from pathlib import Path

import pytest
from click.testing import CliRunner
from rasterio.errors import RasterioIOError

from mapa.cli import dem2stl
from mapa.geometry import get_dimensions_of_stl_file


def test_dem2stl__version() -> None:
    cli = CliRunner()

    result = cli.invoke(dem2stl, ["--version"])
    assert result.exit_code == 0, result.stdout
    assert "dem2stl, version" in result.stdout


def test_dem2stl__no_input_file_no_demo() -> None:
    cli = CliRunner()
    result = cli.invoke(dem2stl)
    assert result.exit_code == 1, result.stdout
    assert "💥  Either of --input or --demo is required, try --help." in result.stdout


def test_dem2stl__not_a_tiff_file(corrupted_tiff, tmpdir) -> None:
    cli = CliRunner()
    output_file = tmpdir / "foo.stl"
    result = cli.invoke(dem2stl, ["--input", str(corrupted_tiff), "--output", output_file])
    assert result.exit_code == 1, result.stdout
    with pytest.raises(RasterioIOError, match="not recognized as a supported file format"):
        raise result.exception


def _dimensions_are_equal(stl_a, stl_b) -> bool:
    dim_a = get_dimensions_of_stl_file(stl_a)
    dim_b = get_dimensions_of_stl_file(stl_b)
    return dim_a == dim_b


def test_dem2stl__binary(test_tiff, tmpdir, test_stl_binary) -> None:
    cli = CliRunner()
    output_file = tmpdir / "output.stl"
    result = cli.invoke(dem2stl, ["--input", str(test_tiff), "--output", output_file])
    assert result.exit_code == 0, result.stdout
    assert f"successfully generated STL file: {Path(output_file).absolute()}" in result.stdout
    assert Path(output_file).is_file()

    assert _dimensions_are_equal(test_stl_binary, output_file)


def test_dem2stl__ascii(test_tiff, tmpdir, test_stl_ascii) -> None:
    cli = CliRunner()
    output_file = tmpdir / "output.stl"
    result = cli.invoke(dem2stl, ["--input", str(test_tiff), "--as-ascii", "--output", output_file])
    assert result.exit_code == 0, result.stdout
    assert f"successfully generated STL file: {Path(output_file).absolute()}" in result.stdout
    assert Path(output_file).is_file()

    assert _dimensions_are_equal(test_stl_ascii, output_file)


def test_dem2stl__model_size(test_tiff, tmpdir, test_stl_binary) -> None:
    cli = CliRunner()

    # create output file of model size 200
    output_file_200 = tmpdir / "output_100.stl"
    result = cli.invoke(
        dem2stl,
        ["--input", str(test_tiff), "--output", output_file_200, "--model-size", 200],
    )
    assert result.exit_code == 0, result.stdout
    assert _dimensions_are_equal(test_stl_binary, output_file_200)

    output_200_size = Path(output_file_200).stat().st_size
    expected_size = Path(test_stl_binary).stat().st_size
    assert output_200_size == expected_size

    # create output file of model size 100
    output_file_100 = tmpdir / "output_200.stl"
    result = cli.invoke(
        dem2stl,
        ["--input", str(test_tiff), "--output", output_file_100, "--model-size", 100],
    )
    assert result.exit_code == 0, result.stdout
    assert not _dimensions_are_equal(test_stl_binary, output_file_100)

    output_100_size = Path(output_file_100).stat().st_size
    # even though the files are different, their sizes (and thus the number of triangles) should be the same
    assert output_100_size == expected_size
