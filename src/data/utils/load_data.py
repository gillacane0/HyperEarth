import warnings
import rasterio
from rasterio.errors import NotGeoreferencedWarning


def _read_tif(path):
    """Read a (non-georeferenced) TIFF as a (bands, rows, cols) array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            return src.read()