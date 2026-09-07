"""
The tilt series is assumed to be already aligned; the only preprocessing
offered here is intensity normalization to ``[0, 1]``.
"""

import numpy as np
import os


class Sinogram:
    """Container for a tilt series of projections.

    Args:
        data: projection stack, shape (n_projections, x, y).
        angles: tilt angles in degrees, length n_projections.
        times: acquisition times, length n_projections.
        pixelsize: physical pixel size (nm). Stored for reference only.
    """

    def __init__(self, data, angles, times=None, pixelsize=1.0, metadata=None):
        self.data = np.asarray(data)
        self.angles = np.asarray(angles)
        if times is None:
            n = self.data.shape[0]
            times = np.linspace(1, n, n)
        self.times = np.asarray(times)
        self.pixelsize = pixelsize
        self.metadata = {} if metadata is None else metadata

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Load a tilt series from an .mrc / .ali / .rec or .mat file."""

        ext = os.path.splitext(filename)[1].lower().lstrip('.')
        if ext == 'mat':
            return cls.from_mat(filename, **kwargs)
        if ext not in ('mrc', 'ali', 'rec'):
            raise ValueError(
                f"dipster_bare.Sinogram.from_file only supports MRC files "
                f"(.mrc/.ali/.rec) and MATLAB .mat files; got '.{ext}'. Build "
                f"the Sinogram directly from a numpy array instead."
            )
        try:
            import mrcz
        except ImportError as e:
            raise ImportError("Reading MRC files requires the 'mrcz' package.") from e

        data, metadata = mrcz.readMRC(filename)
        data = np.asarray(data)

        pixelsize = metadata.get('pixelsize', 1.0)
        if isinstance(pixelsize, (list, tuple, np.ndarray)):
            pixelsize = pixelsize[0]

        if 'angles' not in metadata:
            raise KeyError(
                "MRC metadata has no 'angles' field; cannot build a Sinogram. "
                "Provide angles explicitly via the constructor."
            )
        angles = np.asarray(metadata['angles'])
        times = np.asarray(metadata['times']) if 'times' in metadata else None
        return cls(data, angles, times=times, pixelsize=pixelsize)

def normalize(sino):
    """Scale ``sino.data`` to the range ``[0, 1]`` in place.

    Args:
        sino (Sinogram): the projection data.

    Returns:
        Sinogram: the same object, with ``data`` normalized.
    """
    d = sino.data
    sino.data = (d - d.min()) / (d.max() - d.min())
    return sino
