# mapa 🌍
Create 3d-printable STLs from satellite elevation data


## Installation
```
pip install mapa
```

## Usage
mapa uses numpy and numba under the hood to crunch large amounts of data in little time.

mapa provides the following approaches for creating STL files:

### 2. Using the mapa interactive map
The easiest way is using the `mapa` cli. Simply type
```
mapa
```
A [jupyter notebook](https://jupyter.org/) will be started with an interactive map. Follow the described steps by
executing the cells to create a 3d model of whatever place you like.

### 1. Using the dem2stl cli
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
  --input TEXT          Path to input TIFF file.
  --output TEXT         Path to output STL file.
  --as-ascii            Save output STL as ascii file. If not provided, output
                        file will be binary.
  --model-size INTEGER  Desired size of the generated 3d model in millimeter.
  --max-res             Whether maximum resolution should be used. Note, that
                        this flag potentially increases compute time
                        dramatically. The default behavior (i.e.
                        max_res=False) should return 3d models with sufficient
                        resolution, while the output stl file should be <= 200
                        MB.
  --z-offset FLOAT      Offset distance in millimeter to be put below the 3d
                        model. Defaults to 4.0. Is not influenced by z-scale.
  --z-scale FLOAT       Value to be multiplied to the z-axis elevation data to
                        scale up the height of the model. Defaults to 1.0.
  --demo                Converts a demo tif of Hawaii into a STL file.
  --make-square         If the input tiff is a rectangle and not a square, cut
                        the longer side to make the output STL file a square.
  --version             Show the version and exit.
  --help                Show this message and exit.
```

### 3. Using mapa as python library
In case you are building your own application you can simply use mapa's functionality as a within your application by importing the modules functions.
```python
from mapa import convert_tif_to_stl

path_to_stl = convert_tif_to_stl(...)
```


## Contributing

Contributions are welcome.
