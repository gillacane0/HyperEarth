import os
import glob
import warnings
import rasterio
from rasterio.errors import NotGeoreferencedWarning


def read_tif(path):
    """Read a (non-georeferenced) TIFF as a (bands, rows, cols) array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            return src.read()

def find_file(root, filename):
    """Find a file by name anywhere inside the downloaded dataset folder."""
    matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
    if not matches:
        raise FileNotFoundError(f"{filename} not found in {root}")
    return matches[0]
