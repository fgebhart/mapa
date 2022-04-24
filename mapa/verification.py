from pathlib import Path

from mapa import conf


def verify_input_and_output_are_valid(input: str, output: str) -> Path:
    _verify_input_is_valid(input)
    if output is None:
        output = Path.home() / str(Path(input).name).replace(".tiff", ".stl").replace(".tif", ".stl")
    _verify_output_is_valid(output)
    return output


def _verify_input_is_valid(input: str):
    input_path = Path(input)
    if not input_path.is_file():
        raise FileNotFoundError(f"input file: '{input}' does not seem to be a file")
    if input_path.suffix in conf.SUPPORTED_INPUT_FORMAT:
        pass  # ok
    else:
        raise IOError(
            f"input file '{input}' does not seem to be a tiff file, only {conf.SUPPORTED_INPUT_FORMAT} are supported."
        )


def _verify_output_is_valid(output: str):
    output_path = Path(output)
    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"parent directory of output file '{output_path.parent}' does not seem to be a valid directory."
        )
