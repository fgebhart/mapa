# mapa 🌍

[![PyPI](https://badge.fury.io/py/mapa.svg)](https://badge.fury.io/py/mapa)
[![Python](https://img.shields.io/pypi/pyversions/mapa.svg?style=plastic)](https://badge.fury.io/py/mapa)
[![Downloads](https://pepy.tech/badge/mapa/month)](https://pepy.tech/project/mapa)
[![Python Tests](https://github.com/fgebhart/mapa/actions/workflows/test.yml/badge.svg)](https://github.com/fgebhart/mapa/actions/workflows/test.yml)

`mapa` let's you create 3d-printable [STL](https://en.wikipedia.org/wiki/STL_(file_format)) files from satellite
elevation data (using [DEM](https://en.wikipedia.org/wiki/Digital_elevation_model) data).

Under the hood `mapa` uses:
* [numpy](https://numpy.org/) and [numba](https://numba.pydata.org/) to crunch large amounts of data in little time
* [ALOS DEM](https://planetarycomputer.microsoft.com/dataset/alos-dem) satellite data (max resolution of 30m) provided by
  [Planetary Computer](https://planetarycomputer.microsoft.com/)


## Installation
```
pip install mapa
```

## Usage
`mapa` provides the following approaches for creating STL files:

### 1. Using the `mapa` interactive map
The easiest way is using the `mapa` cli. Simply type
```
mapa
```
A [jupyter notebook](https://jupyter.org/) will be started with an interactive map. Follow the described steps by
executing the cells to create a 3d model of whatever place you like.

 Choose bounding box    | Create STL file
:-------------------------:|:-------------------------:
![](https://i.imgur.com/76hcx9N.jpg)  |  ![](https://i.imgur.com/llvxlrk.png)

 Slice STL file         | 3D print
:-------------------------:|:-------------------------:
![](https://i.imgur.com/AKSRHbK.jpg)  |  ![](https://i.imgur.com/WQn9nwY.png)

### 2. Using the dem2stl cli
The `dem2stl` cli lets you create a 3d-printable STL file based on your tiff file. You can run a demo computation to get
a feeling of how the output STL will look like:
```
dem2stl demo
```
If you have your tiff file ready, you may run something like
```
dem2stl --input your_file.tiff --output output.stl --model-size 200 --z-offset 3.0 --z-scale 1.5
```
The full list of options and their intention can be found with `dem2stl --help`:
```
Usage: dem2stl [OPTIONS]

  🌍 Convert DEM data into STL files 🌏

Options:
  --input TEXT                Path to input TIFF file.
  --output TEXT               Path to output STL file.
  --as-ascii                  Save output STL as ascii file. If not provided,
                              output file will be binary.
  --model-size INTEGER        Desired size of the generated 3d model in
                              millimeter.
  --max-res                   Whether maximum resolution should be used. Note,
                              that this flag potentially increases compute
                              time dramatically. The default behavior (i.e.
                              max_res=False) should return 3d models with
                              sufficient resolution, while the output stl file
                              should be < ~400 MB.
  --z-offset FLOAT            Offset distance in millimeter to be put below
                              the 3d model. Defaults to 4.0. Is not influenced
                              by z-scale.
  --z-scale FLOAT             Value to be multiplied to the z-axis elevation
                              data to scale up the height of the model.
                              Defaults to 1.0.
  --demo                      Converts a demo tif of Hawaii into a STL file.
  --cut-to-format-ratio TEXT  Cut the input tiff file to a specified format.
                              Set to `1` if you want the output model to be
                              squared. Set to `0.5` if you want one side to be
                              half the length of the other side. Omit this
                              flag to keep the input format. This option is
                              particularly useful when an exact output format
                              ratio is required for example when planning to
                              put the 3d printed model into a picture frame.
                              Using this option will always try to cut the
                              shorter side of the input tiff.
  --version                   Show the version and exit.
  --help                      Show this message and exit.
```

### 3. Using `mapa` as python library
In case you are building your own application you can simply use `mapa`'s functionality as a within your application by importing the modules functions.
```python
from mapa import convert_tiff_to_stl

path_to_stl = convert_tiff_to_stl(
    input_file: "path/to/your/input_file.tiff",
    as_ascii: False,
    model_size: 200,
    output_file: "path/to/your/output_file.stl",
    max_res: False,
    z_offset: 3.0,
    z_scale: 1.5,
    cut_to_format_ratio: None,
)
```

## Changelog

See [Releases](https://github.com/fgebhart/mapa/releases).


## Contributing

Contributions, feedback or issues are welcome.
