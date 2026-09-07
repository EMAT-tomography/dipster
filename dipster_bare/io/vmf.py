"""Self-contained ``.vmf`` (Volume-Movie-Format) reader/writer.

Extracted from tomo-ndt
"""

import os
import struct
import pickle
import bz2

import numpy as np
import pandas as pd
import blosc


# --------------------------------------------------------------------------- #
# v1 lookup tables
# --------------------------------------------------------------------------- #

# Supported compression algorithms for the v1 vmf file format.
compression_codes = [
    'none',      # No compression
    'blosclz',   # Blosc default
    'lz4',       # LZ4 compression library
    'lz4hc',     # High-compression variant of LZ4
    'snappy',    # fast library
    'zlib',      # high-compression library
    'zstd',
]

# Supported dtypes for the v1 vmf file format (struct/numpy type chars).
type_codes = [
    'b',  # signed byte
    'B',  # unsigned byte
    'h',  # signed short
    'H',  # unsigned short
    'i',  # signed int
    'I',  # unsigned int
    'l',  # signed long
    'L',  # unsigned long
    'q',  # signed long long
    'Q',  # unsigned long long
    'f',  # 32-bit float
    'd',  # 64-bit (double precision) float
    '?',  # boolean
]


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #

class VolumeNDt:
    """Base class for ``.vmf`` volume-series readers/writers.

    Holds the file-level metadata (version, volume shape, voxel size and the
    per-record time arrays) and the low-level header read/write helpers shared
    by every version. Volumes are handled as plain numpy arrays.
    """

    def __init__(self, dir):
        self.dir = dir
        self.version = [0, 0, 0]
        self.shape = [0, 0, 0]
        self.voxelsize = 1.0

        self.times = np.array([], dtype=np.float32)
        self.times_start = np.array([], dtype=np.float32)
        self.times_end = np.array([], dtype=np.float32)

        self.write_index = 28
        self.empty = not os.path.isfile(self.dir)

    def _new_file(self, shape):
        """Create a new vmf file and write the base (28-byte) header."""
        with open(self.dir, 'wb') as file:
            file.write(struct.pack('3I', *self.version))
            file.write(struct.pack('3I', *shape))
            file.write(struct.pack('f', self.voxelsize))

        self.shape = shape
        self.empty = False

    def _read_metadata(self):
        """Read the base (28-byte) header: version, shape, voxel size."""
        with open(self.dir, 'rb') as file:
            self.version = tuple(struct.unpack('3I', file.read(12)))
            self.shape = tuple(struct.unpack('3I', file.read(12)))
            self.voxelsize = struct.unpack('f', file.read(4))[0]


# --------------------------------------------------------------------------- #
# Version 0 -- bz2 + pickle (standard library only)
# --------------------------------------------------------------------------- #

class VolumeNDt_0v(VolumeNDt):
    """Version 0 of the vmf format. Volumes are pickled and bz2-compressed.

    Metadata (a pandas DataFrame of times / sizes / positions) is appended to
    the end of the file after every write.
    """

    def __init__(self, dir, obj=None):
        super().__init__(dir)
        self.metadata = pd.DataFrame(columns=['times', 'times_start', 'times_end', 'dsize', 'position'])
        self.empty = not os.path.isfile(self.dir)

        if not self.empty:
            if obj is None:
                self._read_metadata()
            else:
                raise Exception("001: Cannot overwrite existing directory with new file")
        else:
            if obj is None:
                pass
            else:
                self.write_record(obj, 0)

    def write_record(self, obj, time, time_start=None, time_end=None):
        if time_start is None or time_end is None:
            time_start, time_end = time, time

        if self.empty:
            self._new_file(obj.shape)

        self._clear_metadata()

        self.times = np.append(self.times, time)
        self.times_start = np.append(self.times_start, time_start)
        self.times_end = np.append(self.times_end, time_end)

        obj_comp = bz2.compress(pickle.dumps(obj))
        dsize = len(obj_comp)

        arr = [time, time_start, time_end, dsize, self.write_index]
        self.metadata.loc[len(self.metadata)] = arr
        with open(self.dir, 'ab') as file:
            file.seek(self.write_index)
            file.write(obj_comp)
            self.write_index += int(dsize)

            index = file.tell()
            md = pickle.dumps(self.metadata)
            file.write(md)
            file.write(struct.pack('Q', index))
            file.write(struct.pack('I', len(md)))

    def read_record(self, i, indexbytime=True):
        if indexbytime:
            try:
                i = self.metadata.loc[self.metadata.times == i].index[0]
            except:
                raise Exception('001: the time index does not exist in the set of volumes')
        with open(self.dir, 'rb') as file:
            dsize = self.metadata['dsize'].iloc[i]
            position = self.metadata['position'].iloc[i]
            file.seek(int(position))
            obj = file.read(int(dsize))
            obj = pickle.loads(bz2.decompress(obj))
            return obj

    def _new_file(self, shape):
        self.version = (0, 0, 1)
        super()._new_file(shape)

    def _clear_metadata(self):
        with open(self.dir, 'rb+') as file:
            file.seek(self.write_index)
            file.truncate()

    def _read_metadata(self):
        super()._read_metadata()

        with open(self.dir, 'rb') as file:
            file.seek(-12, 2)
            index = struct.unpack('Q', file.read(8))[0]
            size = struct.unpack('I', file.read(4))[0]
            file.seek(int(index))
            self.metadata = pickle.loads(file.read(size))
            file.seek(int(index))

            self.times = np.array(self.metadata.times)
            self.times_start = np.array(self.metadata.times_start)
            self.times_end = np.array(self.metadata.times_end)
            a, b = np.array(self.metadata.dsize), np.array(self.metadata.position)
            self.write_index = int(a[-1] + b[-1])


# --------------------------------------------------------------------------- #
# Version 1 -- blosc compression (tomo-ndt default)
# --------------------------------------------------------------------------- #

class VolumeNDt_1v(VolumeNDt):
    """Version 1 of the vmf format. Volumes are blosc-compressed raw buffers.

    Each record is stored as a 20-byte header (three float times + an unsigned
    long byte-count) followed by the compressed volume. ``read_record`` returns
    a numpy array reshaped to the stored volume shape.
    """

    def __init__(self, dir, obj=None):
        super().__init__(dir)

        self.metadata = pd.DataFrame(columns=['times', 'times_start', 'times_end', 'dsize', 'position'])
        self.read_info = [1, 8]
        self.timeunit = 's'
        self.spaceunit = 'voxels'
        self.notes = ''

        if not self.empty:
            if obj is None:
                self._read_metadata()
            else:
                raise Exception("001: Cannot overwrite existing directory with new file")
        else:
            if obj is None:
                pass
            else:
                self.write_record(obj, 0)

    def write_record(self, obj, time, time_start=None, time_end=None):
        if time_start is None or time_end is None:
            time_start, time_end = time, time

        if self.empty:
            self._new_file(obj)

        self.times = np.append(self.times, time)
        self.times_start = np.append(self.times_start, time_start)
        self.times_end = np.append(self.times_end, time_end)

        obj = self._binarize_obj(obj)

        arr = [time, time_start, time_end, len(obj), self.write_index + 20]
        self.metadata.loc[len(self.metadata)] = arr

        with open(self.dir, 'ab') as file:
            file.seek(self.write_index)
            file.write(struct.pack('fff', time, time_start, time_end))
            file.write(struct.pack('Q', len(obj)))
            self.write_index = file.tell()
            file.write(obj)

            self.write_index = file.tell()

    def read_record(self, i, indexbytime=True):
        if indexbytime:
            try:
                i = self.metadata.loc[self.metadata.times == i].index[0]
            except:
                raise Exception('001: the time index does not exist in the set of volumes')

        with open(self.dir, 'rb') as file:
            dsize = self.metadata['dsize'].iloc[i]
            position = self.metadata['position'].iloc[i]
            file.seek(int(position))
            obj = file.read(int(dsize))
            obj = self._debinarize_obj(obj)
            return obj

    def _new_file(self, obj):
        self.version = (1, 0, 0)
        super()._new_file(obj.shape)
        dtype = obj.dtype.char
        dtype = type_codes.index(dtype)
        self.read_info = tuple([self.read_info[0], dtype])
        with open(self.dir, 'ab') as file:
            file.seek(self.write_index)
            file.write(struct.pack('2I', *self.read_info))
            file.write(struct.pack('2I', len(self.timeunit), len(self.spaceunit)))
            file.write(self.timeunit.encode('utf-8'))
            file.write(self.spaceunit.encode('utf-8'))

            file.write(struct.pack('I', len(self.notes)))
            if len(self.notes) > 0:
                file.write(struct.pack(str(len(self.notes)) + 's', self.notes.encode('utf-8')))

            self.write_index = file.tell()

    def _read_metadata(self):
        super()._read_metadata()

        end = os.path.getsize(self.dir)
        with open(self.dir, 'rb') as file:
            file.seek(28)
            self.read_info = tuple(struct.unpack('2I', file.read(8)))
            ntime, nspace = struct.unpack('2I', file.read(8))
            self.timeunit, self.spaceunit = file.read(ntime).decode('utf-8'), file.read(nspace).decode('utf-8')
            nnote = struct.unpack('I', file.read(4))[0]
            if nnote > 0:
                self.notes = file.read(nnote).decode('utf-8')

            while file.tell() < end:
                self.write_index = file.tell()
                time, time_start, time_end = struct.unpack('fff', file.read(12))
                dsize = struct.unpack('Q', file.read(8))[0]
                self.metadata.loc[len(self.metadata)] = [time, time_start, time_end, dsize, self.write_index + 20]
                self.times = np.array(self.metadata.times)
                self.times_start = np.array(self.metadata.times_start)
                self.times_end = np.array(self.metadata.times_end)
                file.seek(int(dsize) + self.write_index + 20)

    def _binarize_obj(self, obj):
        obj = obj.astype(type_codes[self.read_info[1]]).tobytes()
        if 0:
            return obj
        elif 1 <= self.read_info[0] <= 6:
            return blosc.compress(obj, cname=compression_codes[self.read_info[0]])
        else:
            raise Exception('001: Unsupported Compression Algorithm - see docstring for details')

    def _debinarize_obj(self, obj):
        if 0:
            pass
        elif 1 <= self.read_info[0] <= 6:
            obj = blosc.decompress(obj)
        else:
            raise Exception('001: Unsupported Compression Algorithm - see docstring for details')

        obj = np.frombuffer(obj, dtype=type_codes[self.read_info[1]])
        return obj.reshape(self.shape)


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #

def read_vmf(path, version=(1, 0, 0)):
    """Open an existing ``.vmf`` file for reading.

    The on-disk version header selects the implementation. Returns a
    ``VolumeNDt_0v`` / ``VolumeNDt_1v`` whose ``.times`` lists the stored
    timepoints and whose ``.read_record(time)`` returns a numpy volume.
    """
    if os.path.isfile(path):
        with open(path, 'rb') as file:
            version = struct.unpack('3I', file.read(12))
    else:
        raise Exception('File does not Exist: ' + path)
    match version[0]:
        case 0:
            return VolumeNDt_0v(path)
        case 1:
            return VolumeNDt_1v(path)
        case _:
            raise Exception('Unsupported version or corrupted file format')


def new_vmf(path, obj=None, version=(1, 0, 0)):
    """Create a new ``.vmf`` file for writing.

    Defaults to the v1 (blosc) format. Append volumes with
    ``.write_record(volume, time)``. ``obj`` optionally seeds the file with a
    first volume at time 0.
    """
    if os.path.isfile(path):
        raise Exception('File already exists: ' + path)
    if obj is None:
        match version[0]:
            case 0:
                obj = VolumeNDt_0v(path)
            case 1:
                obj = VolumeNDt_1v(path)
            case _:
                raise Exception('Unsupported version or corrupted file format')
    else:
        match version[0]:
            case 0:
                obj = VolumeNDt_0v(path, obj)
            case 1:
                obj = VolumeNDt_1v(path, obj)
            case _:
                raise Exception('Unsupported version or corrupted file format')
    return obj
