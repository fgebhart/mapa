# mapa 🌍
Create 3d-printable STLs from satellite elevation data


## Installation
```
pip install mapa
```

## Usage
mapa uses numpy and numba under the hood to crunch large amounts of data in little time.

### 1. Using the dem2stl cli
The `dem2stl` cli lets you create a 3d-printable STL file based on your tiff file. You can run a demo computation to get a feeling of how the output STL will look like:
```
dem2stl demo
```
If you have your tiff file ready, you may run something like
```
dem2stl --input your_file.tiff --output output.stl --model-size 200 --z-offset 3.0 --z-scale 1.5
```
For more details on the different options, check out the [docs](TODO).

### 2. Using the mapa interactive map
If you don't have a tiff file handy, you may simple select your favorite region using the `mapa` cli. Simply type
```
mapa
```
A jupyter notebook will be started with an interactive map. Follow the described steps by executing the cells to create a 3d model of whatever place you like.

### 3. Using mapa as python library
In case you are building your own application you can simply use mapa's functionality as a within your application by importing the modules functions.
```python
from mapa import convert_tif_to_stl

path_to_stl = convert_tif_to_stl(...)
```
Refer to the [docs](TODO) for more details.


## Documentation
[docs](TODO)
