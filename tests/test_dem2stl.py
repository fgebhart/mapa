import math

from mapa import convert_tif_to_stl
from mapa.geometry import get_dimensions_of_stl_file


def test_verify_model_size(clipped_tiff, tmpdir) -> None:
    # using a non squared clipped image
    output_file = tmpdir / "output.stl"

    model_size = 100
    convert_tif_to_stl(
        input_file=clipped_tiff,
        as_ascii=False,
        model_size=model_size,
        output_file=output_file,
        max_res=True,
        z_offset=0.1,
        z_scale=1.0,
        make_square=False,  # don't enforce square
    )
    dims = get_dimensions_of_stl_file(output_file)
    assert model_size in dims
    x, y, z = dims
    assert x == model_size
    assert x != y
    assert x != z
    assert y != z

    # now enforce squaring
    convert_tif_to_stl(
        input_file=clipped_tiff,
        as_ascii=False,
        model_size=model_size,
        output_file=output_file,
        max_res=True,
        z_offset=0.0,
        z_scale=1.0,
        make_square=True,  # enforce square
    )
    dims = get_dimensions_of_stl_file(output_file)
    assert model_size in dims
    x_100, y_100, z_100 = dims
    assert x_100 == model_size
    assert x_100 == y_100  # i.e. model is square
    assert x_100 != z_100
    assert y_100 != z_100

    # doubling the model size should also double the dimensions
    model_size = 200
    convert_tif_to_stl(
        input_file=clipped_tiff,
        as_ascii=False,
        model_size=model_size,
        output_file=output_file,
        max_res=True,
        z_offset=0.0,
        z_scale=1.0,
        make_square=True,
    )
    dims = get_dimensions_of_stl_file(output_file)
    assert model_size in dims
    x_200, y_200, z_200 = dims
    assert x_200 == model_size
    assert x_200 == y_200
    assert x_200 != z_200
    assert y_200 != z_200
    assert x_200 == 2 * x_100
    assert y_200 == 2 * y_100
    assert math.isclose(z_200, 2 * z_100, rel_tol=0.01)
    z_scale_1 = z_200

    # increasing z_scale should increase the models z dimension
    convert_tif_to_stl(
        input_file=clipped_tiff,
        as_ascii=False,
        model_size=model_size,
        output_file=output_file,
        max_res=True,
        z_offset=0.0,
        z_scale=2.0,
        make_square=True,
    )
    x, y, z_scale_2 = get_dimensions_of_stl_file(output_file)
    assert z_scale_2 > z_scale_1
    assert math.isclose(z_scale_2, 2 * z_scale_1, rel_tol=0.01)

    # changing the coarseness parameter should not affect the output model dimensions
    convert_tif_to_stl(
        input_file=clipped_tiff,
        as_ascii=False,
        model_size=model_size,
        output_file=output_file,
        max_res=True,
        z_offset=0.0,
        z_scale=1.0,
        make_square=True,
    )
    x_coarse_1, y_coarse_1, z_coarse_1 = get_dimensions_of_stl_file(output_file)
    convert_tif_to_stl(
        input_file=clipped_tiff,
        as_ascii=False,
        model_size=model_size,
        output_file=output_file,
        max_res=False,
        z_offset=0.0,
        z_scale=1.0,
        make_square=True,
    )
    x_coarse_2, y_coarse_2, z_coarse_2 = get_dimensions_of_stl_file(output_file)
    assert x_coarse_1 == x_coarse_2
    assert y_coarse_1 == y_coarse_2
    assert math.isclose(z_coarse_1, z_coarse_2, rel_tol=0.01)
