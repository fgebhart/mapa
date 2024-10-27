from pathlib import Path
from zipfile import ZipFile

import pytest

from mapa.utils import md5_sum
from mapa.zip import create_zip_archive


@pytest.fixture
def file_a(tmp_path) -> Path:
    p = tmp_path / "file_a.stl"
    p.write_text("some content")
    assert p.is_file()
    yield p


@pytest.fixture
def file_b(tmp_path) -> Path:
    p = tmp_path / "file_b.stl"
    p.write_text("some other content")
    assert p.is_file()
    yield p


def test_create_zip_archive(file_a, file_b, tmp_path) -> None:
    # get checksums
    checksum_a = md5_sum(file_a)
    checksum_b = md5_sum(file_b)

    # get file content
    with open(file_a) as f:
        content_a = f.read()
    with open(file_b) as f:
        content_b = f.read()

    # zip files to archive
    output = Path(tmp_path) / "output.zip"
    create_zip_archive(files=[file_a, file_b], output_file=output)
    assert output.is_file()

    with ZipFile(output) as zip_file:
        # read content of files
        with zip_file.open(file_a.name) as f:
            content_a_zip = f.read().decode()
        with zip_file.open(file_b.name) as f:
            content_b_zip = f.read().decode()

        assert content_a == content_a_zip
        assert content_b == content_b_zip

        # extract members
        zip_file.extract(file_a.name, tmp_path)
        zip_file.extract(file_b.name, tmp_path)

    a_unzipped = tmp_path / file_a.name
    b_unzipped = tmp_path / file_b.name
    assert a_unzipped.is_file()
    assert b_unzipped.is_file()

    assert checksum_a == md5_sum(a_unzipped)
    assert checksum_b == md5_sum(b_unzipped)


def test_create_zip_archive__compression_impact(test_stl_binary, output_file) -> None:
    uncompressed = test_stl_binary.stat().st_size
    compressed = (
        create_zip_archive(files=[test_stl_binary], output_file=f"{output_file}.zip")
        .stat()
        .st_size
    )

    # compressing a usual STL file reduces the size by more than a factor of 4
    assert compressed < uncompressed / 4
