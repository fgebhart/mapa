"""Helper script for updating the stl files within test/stl/"""

from pathlib import Path

from click.testing import CliRunner

from mapa.cli import dem2stl

if __name__ == "__main__":
    cli = CliRunner()
    test_tiff = Path(__file__).parent / "tiff" / "hawaii_low_res.tiff"

    ascii_output = Path(__file__).parent / "stl" / "hawaii_ascii.stl"
    print(f"regenerating ascii stl file: {ascii_output}")
    cli.invoke(dem2stl, ["--input", str(test_tiff), "--as-ascii", "--output", ascii_output, "--model-size", 200])

    binary_output = Path(__file__).parent / "stl" / "hawaii_binary.stl"
    print(f"regenerating binary stl file: {binary_output}")
    cli.invoke(dem2stl, ["--input", str(test_tiff), "--output", binary_output, "--model-size", 200])
