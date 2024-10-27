from pathlib import Path

import pytest
from click.testing import CliRunner
from rasterio.errors import RasterioIOError

from mapa.cli import dem2stl, mapa
from mapa.stl_file import get_dimensions_of_stl_file
from mapa.utils import md5_sum


def test_dem2stl__version() -> None:
    cli = CliRunner()

    result = cli.invoke(dem2stl, ["--version"])
    assert result.exit_code == 0, result.stdout
    assert "dem2stl, version" in result.stdout

    result = cli.invoke(mapa, ["--version"])
    assert result.exit_code == 0, result.stdout
    assert "mapa, version" in result.stdout


def test_dem2stl__no_input_file_no_demo(caplog) -> None:
    cli = CliRunner()
    result = cli.invoke(dem2stl)
    assert result.exit_code == 1, result.stdout
    assert "💥  Either of --input or --demo is required, try --help." in caplog.text
    assert "Aborted!" in result.stdout


def test_dem2stl__demo(caplog) -> None:
    cli = CliRunner()
    result = cli.invoke(dem2stl, ["--demo"])
    assert result.exit_code == 0, result.stdout
    assert "successfully generated STL file:" in caplog.text


def test_dem2stl__not_a_tiff_file(corrupted_tiff, tmpdir) -> None:
    cli = CliRunner()
    output_file = tmpdir / "foo.stl"
    result = cli.invoke(
        dem2stl, ["--input", str(corrupted_tiff), "--output", output_file]
    )
    assert result.exit_code == 1, result.stdout
    with pytest.raises(RasterioIOError, match="not recognized"):
        raise result.exception


def _assert_dimensions_equal(stl_a: Path, stl_b: Path, negate: bool = False) -> bool:
    dim_a = get_dimensions_of_stl_file(stl_a)
    dim_b = get_dimensions_of_stl_file(stl_b)
    if negate:
        assert not dim_a == dim_b, f"dimensions not equal: {dim_a} != {dim_b}"
    else:
        assert dim_a == dim_b, f"dimensions not equal: {dim_a} != {dim_b}"


def test_dem2stl__binary(test_tiff, tmpdir, test_stl_binary, caplog) -> None:
    cli = CliRunner()
    output_file = tmpdir / "output.stl"
    result = cli.invoke(dem2stl, ["--input", str(test_tiff), "--output", output_file])
    assert result.exit_code == 0, result.stdout
    assert (
        f"successfully generated STL file: {Path(output_file).absolute()}"
        in caplog.text
    )
    assert Path(output_file).is_file()

    _assert_dimensions_equal(test_stl_binary, output_file)


def test_dem2stl__ascii(test_tiff, tmpdir, test_stl_ascii, caplog) -> None:
    cli = CliRunner()
    output_file = tmpdir / "hawaii_ascii.stl"
    result = cli.invoke(
        dem2stl, ["--input", str(test_tiff), "--as-ascii", "--output", output_file]
    )
    assert result.exit_code == 0, result.stdout
    assert (
        f"successfully generated STL file: {Path(output_file).absolute()}"
        in caplog.text
    )
    assert Path(output_file).is_file()

    _assert_dimensions_equal(test_stl_ascii, output_file)
    assert md5_sum(test_stl_ascii) == md5_sum(output_file)


def test_dem2stl__model_size(test_tiff, tmpdir, test_stl_binary) -> None:
    cli = CliRunner()

    # create output file of model size 200
    output_file_200 = tmpdir / "output_200.stl"
    result = cli.invoke(
        dem2stl,
        ["--input", str(test_tiff), "--output", output_file_200, "--model-size", 200],
    )
    assert result.exit_code == 0, result.stdout
    _assert_dimensions_equal(test_stl_binary, output_file_200)

    output_200_size = Path(output_file_200).stat().st_size
    expected_size = Path(test_stl_binary).stat().st_size
    assert output_200_size == expected_size

    # create output file of model size 100
    output_file_100 = tmpdir / "output_100.stl"
    result = cli.invoke(
        dem2stl,
        ["--input", str(test_tiff), "--output", output_file_100, "--model-size", 100],
    )
    assert result.exit_code == 0, result.stdout
    _assert_dimensions_equal(test_stl_binary, output_file_100, negate=True)

    output_100_size = Path(output_file_100).stat().st_size
    # even though the files are different, their sizes (and thus the number of triangles) should be the same
    assert output_100_size == expected_size
