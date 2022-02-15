from pathlib import Path

SUPPORTED_INPUT_FORMAT = {".tiff", ".tif"}
MAXIMUM_RESOLUTION = 1_000
PERFORMANCE_WARNING_THRESHOLD = 5_000 * 5_000

# default params
DEFAULT_MODEL_OUTPUT_SIZE_IN_MM = 200
DEFAULT_Z_OFFSET = 4.0
DEFAULT_Z_SCALE = 1.0

# path to demo tiff
DEMO_TIFF_PATH = Path(__file__).parent.parent / "tests" / "tiff" / "hawaii.tiff"

# stac catalogue
PLANETARY_COMPUTER_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
PLANETARY_COMPUTER_COLLECTION = "alos-dem"
