"""
Experimental geospatial-agumented wrapper around a ``pyarrow.dataset``.

>>> import geoarrow.pyarrow.dataset as gads
"""

import hashlib
import json
import posixpath
import uuid
from concurrent.futures import ThreadPoolExecutor, wait

from geoarrow.pyarrow._kernel import Kernel
from geoarrow.pyarrow._type import GeometryExtensionType, wkb, wkt
from geoarrow.types import CoordType

import pyarrow as _pa
import pyarrow.compute as _compute
import pyarrow.dataset as _ds
import pyarrow.fs as _fs
import pyarrow.ipc as _ipc
import pyarrow.types as _types


_FRAGMENT_INDEX_COLUMN = "_fragment_index"
_BOX_FIELD_NAMES = ("xmin", "ymin", "xmax", "ymax")
_BOX_TYPE = _pa.struct(
    [_pa.field(name, _pa.float64(), nullable=False) for name in _BOX_FIELD_NAMES]
)

# Identity columns used by a saved row-group index. These are intentionally
# separate from the columns returned by `index_fragments()` so that the public
# index table keeps its documented shape.
_FILE_PATH_COLUMN = "_file_path"
_ROW_GROUP_INDEX_COLUMN = "_row_group_index"
_ROW_GROUP_NUM_ROWS_COLUMN = "_row_group_num_rows"
_ROW_GROUP_NUM_BYTES_COLUMN = "_row_group_num_bytes"
_ROW_GROUP_DIGEST_COLUMN = "_row_group_digest"
_ROW_GROUP_DIGEST_WEAK_COLUMN = "_row_group_digest_weak"
_BOX_DIGEST_COLUMN = "_box_digest"
_SCHEMA_DIGEST_COLUMN = "_schema_digest"
_FILE_SIZE_COLUMN = "_file_size"
_FILE_MTIME_COLUMN = "_file_mtime_ns"
_BOX_COLUMN_PREFIX = "_box."

_IDENTITY_COLUMNS = [
    _FILE_PATH_COLUMN,
    _ROW_GROUP_INDEX_COLUMN,
    _ROW_GROUP_NUM_ROWS_COLUMN,
    _ROW_GROUP_NUM_BYTES_COLUMN,
    _ROW_GROUP_DIGEST_COLUMN,
    _ROW_GROUP_DIGEST_WEAK_COLUMN,
    _BOX_DIGEST_COLUMN,
    _SCHEMA_DIGEST_COLUMN,
    _FILE_SIZE_COLUMN,
    _FILE_MTIME_COLUMN,
]

_INDEX_METADATA_KEY = b"geoarrow.pyarrow.row_group_index"
_INDEX_FORMAT = "geoarrow.pyarrow.row-group-index"
_INDEX_VERSION = 2
_DEFAULT_INDEX_FILENAME = ".geoarrow-index.arrow"


class GeoDataset:
    """Geospatial-augmented Dataset

    EXPERIMENTAL

    The GeoDataset wraps a pyarrow.Dataset containing one or more geometry columns
    and provides indexing and IO capability. If `geometry_columns` is `None`,
    it will include all columns that inherit from `geoarrow.pyarrow.GeometryExtensionType`.
    The `geometry_columns` are not required to be geoarrow extension type columns:
    text columns will be parsed as WKT; binary columns will be parsed as WKB
    (but are not detected automatically).

    Note that the ``GeoDataset`` is only useful in a context where each fragment
    has been written such that features in the fragment are close together
    in space (e.g., one file or row group per state).
    """

    def __init__(self, parent, geometry_columns=None):
        self._index = None
        self._index_rows = None
        self._index_path = None
        self._index_root = None
        self._index_saving = False
        self._index_was_loaded = False
        self._geometry_columns = geometry_columns
        self._geometry_types = None
        self._fragments = None

        if not isinstance(parent, _ds.Dataset):
            raise TypeError("parent must be a pyarrow.dataset.Dataset")
        self._parent = parent

    @property
    def parent(self):
        """Get the parent Dataset

        Returns the (non geo-aware) parent pyarrow.Dataset.

        >>> import geoarrow.pyarrow.dataset as gads
        >>> import geoarrow.pyarrow as ga
        >>> import pyarrow as pa
        >>> table = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
        >>> dataset = gads.dataset(table)
        >>> type(dataset.parent)
        <class 'pyarrow._dataset.InMemoryDataset'>
        """
        return self._parent

    def to_table(self):
        return self.parent.to_table()

    @property
    def schema(self):
        """Get the dataset schema

        The schema of a GeoDataset is identical to that of its parent.

        >>> import geoarrow.pyarrow.dataset as gads
        >>> import geoarrow.pyarrow as ga
        >>> import pyarrow as pa
        >>> table = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
        >>> dataset = gads.dataset(table)
        >>> dataset.schema
        geometry: extension<geoarrow.wkt<WktType>>
        """
        return self._parent.schema

    def get_fragments(self):
        """Resolve the list of fragments in the dataset

        This is identical to the list of fragments of its parent."""
        if self._fragments is None:
            self._fragments = tuple(self._parent.get_fragments())

        return self._fragments

    @property
    def geometry_columns(self):
        """Get a tuple of geometry column names

        >>> import geoarrow.pyarrow.dataset as gads
        >>> import geoarrow.pyarrow as ga
        >>> import pyarrow as pa
        >>> table = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
        >>> dataset = gads.dataset(table)
        >>> dataset.geometry_columns
        ('geometry',)
        """
        if self._geometry_columns is None:
            schema = self.schema
            geometry_columns = []
            for name, type in zip(schema.names, schema.types):
                if isinstance(type, GeometryExtensionType):
                    geometry_columns.append(name)
            self._geometry_columns = tuple(geometry_columns)

        return self._geometry_columns

    @property
    def geometry_types(self):
        """Resolve a tuple of geometry column types

        This will convert any primitive types to the corresponding
        geo-enabled type (e.g., binary to wkb) and check that geometry
        columns actually refer a field that can be interpreted as
        geometry.

        >>> import geoarrow.pyarrow.dataset as gads
        >>> import geoarrow.pyarrow as ga
        >>> import pyarrow as pa
        >>> table = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
        >>> dataset = gads.dataset(table)
        >>> dataset.geometry_types
        (WktType(geoarrow.wkt),)
        """
        if self._geometry_types is None:
            geometry_types = []
            for col in self.geometry_columns:
                type = self.schema.field(col).type
                if isinstance(type, GeometryExtensionType):
                    geometry_types.append(type)
                elif _types.is_binary(type):
                    geometry_types.append(wkb())
                elif _types.is_string(type):
                    geometry_types.append(wkt())
                else:
                    raise TypeError(f"Unsupported type for geometry column: {type}")

            self._geometry_types = tuple(geometry_types)

        return self._geometry_types

    @property
    def index_path(self):
        """The path of the saved row-group index, if any (EXPERIMENTAL)"""
        return self._index_path

    def save_index(self, path=None, num_threads=None):
        """Save the fragment index (EXPERIMENTAL)

        Only :class:`ParquetRowGroupGeoDataset` supports a saved index.
        """
        raise TypeError(
            f"{type(self).__name__} does not support a saved fragment index"
        )

    def load_index(self, path=None, num_threads=None):
        """Load a saved fragment index (EXPERIMENTAL)

        Only :class:`ParquetRowGroupGeoDataset` supports a saved index.
        """
        raise TypeError(
            f"{type(self).__name__} does not support a saved fragment index"
        )

    def clear_index(self):
        """Forget the in-memory index so that it is rebuilt on next use"""
        self._index = None
        self._index_rows = None
        self._index_was_loaded = False
        return self

    def index_fragments(self, num_threads=None):
        """Resolve a simplified geometry for each fragment

        Currently the simplified geometry is a box in the form of a
        struct array with fields xmin, xmax, ymin, and ymax. The
        fragment index is curently a table whose first column is the fragment
        index and whose subsequent columns are named with the geometry column
        name. A future implementation may handle spherical edges using a type
        of simplified geometry more suitable to a spherical comparison.

        >>> import geoarrow.pyarrow.dataset as gads
        >>> import geoarrow.pyarrow as ga
        >>> import pyarrow as pa
        >>> table = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
        >>> dataset = gads.dataset(table)
        >>> dataset.index_fragments().to_pylist()
        [{'_fragment_index': 0, 'geometry': {'xmin': 0.5, 'ymin': 1.5, 'xmax': 0.5, 'ymax': 1.5}}]
        """
        if self._index is None:
            self._index = self._build_index(
                self.geometry_columns, self.geometry_types, num_threads
            )
            self._maybe_refresh_saved_index()

        return self._index

    def _maybe_refresh_saved_index(self):
        """Persist a refreshed sidecar after some entries were recomputed

        This makes reuse persistent across startups: the next time the dataset
        is opened, the entries recomputed here already match. It only ever
        rewrites a sidecar that was actually loaded and matched this dataset,
        so merely opening a dataset never creates files and an unrelated or
        mismatched index is never silently replaced. Failures (e.g., a
        read-only filesystem) are ignored because a saved index is only an
        optimization.
        """
        if not self._index_was_loaded:
            return
        if not isinstance(self._index_path, str) or self._index_rows is None:
            return
        if self._index_saving:
            return
        if all(row["reused"] for row in self._index_rows):
            return

        try:
            if self._index_filesystem().get_file_info(self._index_path).type != (
                _fs.FileType.File
            ):
                return
        except Exception:
            return

        try:
            self.save_index(self._index_path)
        except Exception:
            pass

    def _build_index(self, geometry_columns, geometry_types, num_threads=None):
        return GeoDataset._index_fragments(
            self.get_fragments(),
            geometry_columns,
            geometry_types,
            num_threads=num_threads,
        )

    def filter_fragments(self, target):
        """Push down a spatial query into a GeoDataset

        Returns a potentially simplified dataset based on the geometry of
        target. Currently this uses `geoarrow.pyarrow.box_agg()` on `target`
        and performs a simple envelope comparison with each fragment. A future
        implementation may handle spherical edges using a type of simplified
        geometry more suitable to a spherical comparison. For datasets with
        more than one geometry column, the filter will be applied to all columns
        and include fragments that intersect the simplified geometry from any
        of the columns.

        Fragments whose simplified geometry is unknown (for example, because
        a saved index was built for a different version of the file) are always
        included so that a stale or mismatched index can never hide data.

        Note that datasets with large row groups/fragments and/or datasets that
        were not written with fragments with spatial significance may return
        most or all of the fragments in the parent dataset.

        >>> import geoarrow.pyarrow.dataset as gads
        >>> import geoarrow.pyarrow as ga
        >>> import pyarrow as pa
        >>> table = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
        >>> dataset = gads.dataset(table)
        >>> dataset.filter_fragments("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))").to_table()
        pyarrow.Table
        geometry: extension<geoarrow.wkt<WktType>>
        ----
        geometry: []
        >>> dataset.filter_fragments("POLYGON ((0 1, 0 2, 1 2, 1 1, 0 1))").to_table()
        pyarrow.Table
        geometry: extension<geoarrow.wkt<WktType>>
        ----
        geometry: [["POINT (0.5 1.5)"]]
        """
        from ._compute import box_agg

        if isinstance(target, str):
            target = [target]
        target_box = box_agg(target).as_py()

        fragments = self.get_fragments()
        index = self.index_fragments()

        # Defensive: if for any reason the index does not describe exactly one
        # row per fragment, do not risk dropping data.
        if (
            index.num_rows != len(fragments)
            or _FRAGMENT_INDEX_COLUMN not in index.column_names
        ):
            fragment_indices = list(range(len(fragments)))
        else:
            maybe_intersects = GeoDataset._index_box_intersects(
                index, target_box, self.geometry_columns
            )
            fragment_indices = [scalar.as_py() for scalar in maybe_intersects]

        filtered_parent = self._filter_parent_fragment_indices(fragment_indices)
        return self._wrap_parent(filtered_parent, fragment_indices)

    def _wrap_parent(self, filtered_parent, fragment_indices):
        new_wrapped = GeoDataset(
            filtered_parent, geometry_columns=self._geometry_columns
        )
        new_wrapped._geometry_types = self.geometry_types

        new_index = self.index_fragments().take(
            _pa.array(fragment_indices, type=_pa.int64())
        )
        new_wrapped._index = new_index.set_column(
            0, _FRAGMENT_INDEX_COLUMN, _pa.array(range(new_index.num_rows))
        )
        new_wrapped._index_path = self._index_path
        new_wrapped._index_root = self._index_root
        if self._index_rows is not None:
            new_wrapped._index_rows = [self._index_rows[i] for i in fragment_indices]

        return new_wrapped

    def _filter_parent_fragment_indices(self, fragment_indices):
        fragments = self.get_fragments()
        fragments_filtered = [fragments[i] for i in fragment_indices]

        if isinstance(self._parent, _ds.FileSystemDataset):
            return _ds.FileSystemDataset(
                fragments_filtered, self.schema, self._parent.format
            )
        else:
            tables = [fragment.to_table() for fragment in fragments_filtered]
            return _ds.InMemoryDataset(tables, schema=self.schema)

    @staticmethod
    def _index_fragment(fragment, column, type):
        scanner = fragment.scanner(columns=[column])
        reader = scanner.to_reader()
        kernel = Kernel.box_agg(type)
        for batch in reader:
            kernel.push(batch.column(0))
        return kernel.finish().storage

    @staticmethod
    def _index_fragments(fragments, columns, types, num_threads=None):
        columns = list(columns)
        if num_threads is None:
            num_threads = _pa.cpu_count()

        num_fragments = len(fragments)
        metadata = [_pa.array(range(num_fragments))]
        if not columns:
            return _pa.table(metadata, names=[_FRAGMENT_INDEX_COLUMN])

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for column, type in zip(columns, types):
                for fragment in fragments:
                    future = executor.submit(
                        GeoDataset._index_fragment, fragment, column, type
                    )
                    futures.append(future)

            wait(futures)

            results = []
            for i, column in enumerate(columns):
                results.append(
                    [
                        futures[i * num_fragments + j].result()
                        for j in range(num_fragments)
                    ]
                )

            result_arrays = [_pa.concat_arrays(result) for result in results]
            return _pa.table(
                metadata + result_arrays, names=[_FRAGMENT_INDEX_COLUMN] + columns
            )

    @staticmethod
    def _index_box_intersects(index, box, columns):
        xmin, ymin, xmax, ymax = box.values()

        # If any geometry column is absent from the index we cannot reason about
        # it and must conservatively include every fragment.
        if not set(columns).issubset(set(index.column_names)):
            return index.column(0)

        expressions = []
        for col in columns:
            field = lambda name: _compute.field(col, name)  # noqa: E731
            intersects = (
                (field("xmin") <= xmax)
                & (field("xmax") >= xmin)
                & (field("ymin") <= ymax)
                & (field("ymax") >= ymin)
            )
            # A null/unknown box means "we don't know where this fragment is".
            # Include it instead of silently dropping it. Non-finite bounds
            # (e.g., the inf/-inf envelope of an empty row group) are treated
            # the same way so that no fragment can be hidden by the index.
            unknown = None
            for name in _BOX_FIELD_NAMES:
                is_missing = _compute.is_null(field(name)) | ~_compute.is_finite(
                    field(name)
                )
                unknown = is_missing if unknown is None else (unknown | is_missing)
            expressions.append(intersects | unknown)

        if not expressions:
            return index.column(0)

        expr = expressions[0]
        for i in range(1, len(expressions)):
            expr = expr | expressions[i]

        result = _ds.dataset(index).filter(expr).to_table()
        return result.column(0)


class ParquetRowGroupGeoDataset(GeoDataset):
    """Geospatial-augmented Parquet dataset using row groups

    An implementation of the GeoDataset that can leverage potentially
    more efficient indexing and more specific filtering. Notably, this
    implementation can (1) split a Parquet dataset into potentially more
    smaller fragments and (2) use column statistics added by most Parquet
    writers to more efficiently build the fragment index for types that support
    this capability.

    This implementation can also persist the row-group index to a sidecar
    file (see :meth:`save_index()` and :meth:`load_index()`). A saved index
    records the identity of each file and row group it describes; when a
    dataset is opened, only the row groups whose identity still matches are
    reused and the rest are recomputed.
    """

    def __init__(
        self,
        parent,
        row_group_fragments,
        row_group_ids,
        geometry_columns=None,
        use_column_statistics=True,
        index_path=None,
    ):
        super().__init__(parent, geometry_columns=geometry_columns)
        self._fragments = row_group_fragments
        self._row_group_ids = row_group_ids
        self._use_column_statistics = use_column_statistics
        self._index_path = index_path
        self._index_root = (
            posixpath.dirname(index_path) if isinstance(index_path, str) else None
        )

    @staticmethod
    def create(
        parent,
        geometry_columns=None,
        use_column_statistics=True,
        index_path=None,
        save_index=False,
    ):
        if not isinstance(parent, _ds.FileSystemDataset) or not isinstance(
            parent.format, _ds.ParquetFileFormat
        ):
            raise TypeError(
                "ParquetRowGroupGeoDataset() is only supported for Parquet datasets"
            )

        # `None` means "derive the default sidecar path"; `False` disables it.
        if index_path is None:
            index_path = ParquetRowGroupGeoDataset._default_index_path(parent)
        elif index_path is False:
            index_path = None

        row_group_fragments = []
        row_group_ids = []

        for file_fragment in parent.get_fragments():
            for row_group_fragment in file_fragment.split_by_row_group():
                row_group_fragments.append(row_group_fragment)
                # Keep track of the row group IDs so we can accellerate
                # building an index later where column statistics are supported
                row_group_ids.append(row_group_fragment.row_groups[0].id)

        parent = _ds.FileSystemDataset(
            row_group_fragments, parent.schema, parent.format
        )
        out = ParquetRowGroupGeoDataset(
            parent,
            row_group_fragments,
            row_group_ids,
            geometry_columns=geometry_columns,
            use_column_statistics=use_column_statistics,
            index_path=index_path,
        )
        if save_index:
            out.save_index()
        return out

    def _wrap_parent(self, filtered_parent, fragment_indices):
        base_wrapped = super()._wrap_parent(filtered_parent, fragment_indices)

        new_row_group_fragments = [self._fragments[i] for i in fragment_indices]
        new_row_group_ids = [self._row_group_ids[i] for i in fragment_indices]
        new_wrapped = ParquetRowGroupGeoDataset(
            base_wrapped._parent,
            new_row_group_fragments,
            new_row_group_ids,
            geometry_columns=base_wrapped.geometry_columns,
            use_column_statistics=self._use_column_statistics,
            index_path=self._index_path,
        )
        new_wrapped._index = base_wrapped._index
        new_wrapped._index_rows = base_wrapped._index_rows
        new_wrapped._index_root = self._index_root
        return new_wrapped

    # ----- Index persistence ------------------------------------------------

    @staticmethod
    def _default_index_path(parent):
        try:
            files = list(parent.files)
        except (AttributeError, NotImplementedError):
            return None

        if not files:
            return None

        root = ParquetRowGroupGeoDataset._data_root_for_files(files)
        if root is None:
            return None

        return posixpath.join(root, _DEFAULT_INDEX_FILENAME)

    @staticmethod
    def _data_root_for_files(files):
        if len(files) == 1:
            root = posixpath.dirname(files[0])
        else:
            try:
                root = posixpath.commonpath(files)
            except ValueError:
                return None

            # commonpath() returns the shared file if paths are identical
            if root in files:
                root = posixpath.dirname(root)

        # Never place a sidecar at the filesystem root (e.g., for datasets
        # assembled from files in unrelated directories).
        if root in ("", "/", posixpath.sep):
            return None

        return root or None

    def _index_filesystem(self):
        try:
            return self._parent.filesystem
        except AttributeError:
            return _fs.LocalFileSystem()

    @staticmethod
    def _relative_key(path, root):
        if root is None:
            return path

        try:
            return posixpath.relpath(path, root)
        except ValueError:
            # e.g., different drives on Windows
            return path

    def _row_key(self, identity):
        return (
            self._relative_key(identity["path"], self._index_root),
            identity["row_group_index"],
        )

    def save_index(self, path=None, num_threads=None):
        """Save the row-group index to a sidecar file

        Writes an Arrow IPC file containing one row per fragment with its
        identity (file path, row-group index, row-group digest, and schema
        digest) along with the computed geometry envelope(s). If an index
        already exists at ``path``, rows describing fragments that are not
        part of this dataset are preserved, which makes it safe to save from a
        spatially filtered dataset.

        Returns the path that was written.
        """
        if path is None:
            path = self._index_path
        if path is None:
            path = self._default_index_path(self._parent)
        if path is None:
            raise TypeError("No index path is available for this dataset")

        root = posixpath.dirname(path)

        # If we are asked to save somewhere other than where the in-memory
        # index was loaded from, rebuild so that keys/reuse are resolved
        # against the target location.
        if path != self._index_path or self._index_rows is None:
            self._index_path = path
            self._index_root = root
            self._index = None
            self._index_rows = None

        # Make sure the in-memory index (and hence _index_rows) is current.
        # Suppress the automatic refresh triggered by index_fragments() while
        # we are already in the process of writing the sidecar.
        self._index_saving = True
        try:
            self.index_fragments(num_threads=num_threads)
        finally:
            self._index_saving = False
        if self._index_rows is None:
            raise ValueError("Could not build a persisted index for this dataset")

        fs = self._index_filesystem()

        new_rows = []
        for row in self._index_rows:
            identity = row["identity"]
            new_rows.append(
                {
                    "file_path": self._relative_key(identity["path"], root),
                    "row_group_index": identity["row_group_index"],
                    "row_group_num_rows": identity["num_rows"],
                    "row_group_num_bytes": identity["num_bytes"],
                    "row_group_digest": identity["digest"],
                    "row_group_digest_weak": identity["digest_weak"],
                    "schema_digest": identity["schema_digest"],
                    "file_size": identity["file_size"],
                    "file_mtime_ns": identity["file_mtime_ns"],
                    "boxes": dict(row["boxes"]),
                }
            )
        new_keys = {(row["file_path"], row["row_group_index"]) for row in new_rows}
        current_files = {row["file_path"] for row in new_rows}

        # Preserve rows from an existing index that belong to other datasets
        # stored in the same sidecar (for example, when saving from a filtered
        # dataset or when several single-file datasets share a directory).
        # Rows describing a file that is part of this dataset but whose row
        # group no longer exists (e.g., the file was rewritten with fewer row
        # groups) are dropped so they cannot shadow the fresh entries.
        existing_rows = self._read_index_rows(path, fs)
        if existing_rows is not None:
            merged = []
            for row in existing_rows:
                key = (row["file_path"], row["row_group_index"])
                if key in new_keys:
                    continue
                if self._index_was_loaded and row["file_path"] in current_files:
                    continue
                merged.append(row)
            new_rows = merged + new_rows

        table = self._index_table_from_rows(
            new_rows, self._geometry_columns_key(), self._geometry_types_key()
        )

        tmp_path = posixpath.join(
            root,
            "." + posixpath.basename(path) + ".tmp-" + uuid.uuid4().hex,
        )
        try:
            with fs.open_output_stream(tmp_path) as sink:
                with _ipc.new_stream(sink, table.schema) as writer:
                    writer.write_table(table)
            fs.move(tmp_path, path)
        except Exception:
            try:
                fs.delete_file(tmp_path)
            except Exception:
                pass
            raise

        self._index_path = path
        self._index_root = root
        return path

    def load_index(self, path=None, num_threads=None):
        """Load a saved row-group index and reuse valid entries

        Entries whose file/row-group identity still matches are reused; any
        fragment that is missing from the saved index, or whose identity has
        changed, is recomputed. An index that cannot be read or that does not
        match this dataset's geometry columns is ignored.

        Returns ``self``.
        """
        if path is None:
            path = self._index_path
        if path is None:
            raise TypeError("No index path is available for this dataset")

        self._index_path = path
        self._index_root = posixpath.dirname(path)
        self._index = None
        self._index_rows = None
        self._index_was_loaded = False
        self.index_fragments(num_threads=num_threads)
        return self

    def _geometry_columns_key(self):
        return [str(col) for col in self.geometry_columns]

    def _geometry_types_key(self):
        return [str(type) for type in self.geometry_types]

    def _read_index_table(self, path, fs=None):
        if fs is None:
            fs = self._index_filesystem()

        try:
            info = fs.get_file_info(path)
            if info.type != _fs.FileType.File:
                return None
            with fs.open_input_stream(path) as src:
                with _ipc.open_stream(src) as reader:
                    return reader.read_all()
        except Exception:
            # A corrupt/inaccessible index must never prevent reading data.
            return None

    def _read_index_rows(self, path, fs=None):
        table = self._read_index_table(path, fs=fs)
        if table is None:
            return None

        try:
            metadata = json.loads(table.schema.metadata[_INDEX_METADATA_KEY])
        except Exception:
            return None

        if not self._metadata_matches(metadata):
            return None

        rows = []
        try:
            columns = {
                name: table.column(name).to_pylist()
                for name in table.column_names
            }
            num_rows = table.num_rows
            geometry_columns = metadata["geometry_columns"]
            for i in range(num_rows):
                boxes = {}
                for col in geometry_columns:
                    boxes[col] = columns[_BOX_COLUMN_PREFIX + col][i]
                rows.append(
                    {
                        "file_path": columns[_FILE_PATH_COLUMN][i],
                        "row_group_index": columns[_ROW_GROUP_INDEX_COLUMN][i],
                        "row_group_num_rows": columns[_ROW_GROUP_NUM_ROWS_COLUMN][i],
                        "row_group_num_bytes": columns[_ROW_GROUP_NUM_BYTES_COLUMN][i],
                        "row_group_digest": columns[_ROW_GROUP_DIGEST_COLUMN][i],
                        "row_group_digest_weak": columns[_ROW_GROUP_DIGEST_WEAK_COLUMN][
                            i
                        ],
                        "box_digest": columns[_BOX_DIGEST_COLUMN][i],
                        "schema_digest": columns[_SCHEMA_DIGEST_COLUMN][i],
                        "file_size": columns[_FILE_SIZE_COLUMN][i],
                        "file_mtime_ns": columns[_FILE_MTIME_COLUMN][i],
                        "boxes": boxes,
                    }
                )
        except Exception:
            return None

        return rows

    def _metadata_matches(self, metadata):
        try:
            if metadata.get("format") != _INDEX_FORMAT:
                return False
            if metadata.get("version") != _INDEX_VERSION:
                return False
            geometry_columns_key = self._geometry_columns_key()
            if list(metadata.get("geometry_columns", [])) != geometry_columns_key:
                return False
            geometry_types_key = self._geometry_types_key()
            if list(metadata.get("geometry_types", [])) != geometry_types_key:
                return False
        except Exception:
            return False

        return True

    def _index_metadata(self, geometry_columns, geometry_types):
        return {
            "format": _INDEX_FORMAT,
            "version": _INDEX_VERSION,
            "geometry_columns": list(geometry_columns),
            "geometry_types": list(geometry_types),
        }

    def _index_table_from_rows(self, rows, geometry_columns, geometry_types):
        columns = [
            _pa.array([row["file_path"] for row in rows], type=_pa.string()),
            _pa.array(
                [row["row_group_index"] for row in rows], type=_pa.int64()
            ),
            _pa.array(
                [row["row_group_num_rows"] for row in rows], type=_pa.int64()
            ),
            _pa.array(
                [row["row_group_num_bytes"] for row in rows], type=_pa.int64()
            ),
            _pa.array([row["row_group_digest"] for row in rows], type=_pa.string()),
            _pa.array(
                [
                    bool(row["row_group_digest_weak"])
                    if row["row_group_digest_weak"] is not None
                    else None
                    for row in rows
                ],
                type=_pa.bool_(),
            ),
            _pa.array(
                [
                    ParquetRowGroupGeoDataset._box_digest(
                        {col: row["boxes"].get(col) for col in geometry_columns}
                    )
                    for row in rows
                ],
                type=_pa.string(),
            ),
            _pa.array([row["schema_digest"] for row in rows], type=_pa.string()),
            _pa.array([row["file_size"] for row in rows], type=_pa.int64()),
            _pa.array([row["file_mtime_ns"] for row in rows], type=_pa.int64()),
        ]
        names = list(_IDENTITY_COLUMNS)
        for col in geometry_columns:
            columns.append(
                _pa.array(
                    [row["boxes"].get(col) for row in rows], type=_BOX_TYPE
                )
            )
            names.append(_BOX_COLUMN_PREFIX + col)

        schema_metadata = {
            _INDEX_METADATA_KEY: json.dumps(
                self._index_metadata(geometry_columns, geometry_types)
            ).encode()
        }
        schema = _pa.schema(
            [_pa.field(name, column.type) for name, column in zip(names, columns)],
            metadata=schema_metadata,
        )
        return _pa.table(columns, schema=schema)

    @staticmethod
    def _box_digest(boxes):
        """Digest the stored envelope values so tampering is detectable"""
        digest = hashlib.sha256()
        digest.update(b"boxes-v1")
        for col in sorted(boxes):
            box = boxes[col]
            digest.update(b"|")
            digest.update(str(col).encode())
            if box is None:
                digest.update(b",none")
                continue
            for name in _BOX_FIELD_NAMES:
                value = box.get(name)
                digest.update(b",")
                if value is None:
                    digest.update(b"none")
                else:
                    digest.update(repr(float(value)).encode())
        return digest.hexdigest()

    # ----- Identity ---------------------------------------------------------

    def _load_saved_rows(self):
        if not isinstance(self._index_path, str):
            return None

        rows = self._read_index_rows(self._index_path)
        if rows is None:
            return None

        # The sidecar exists and matched this dataset's format/columns, so it
        # is safe to refresh it in place after recomputing changed entries.
        self._index_was_loaded = True
        return {
            (row["file_path"], row["row_group_index"]): row for row in rows
        }

    @staticmethod
    def _schema_digest(schema):
        return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()

    @staticmethod
    def _row_group_digest(row_group_metadata):
        digest = hashlib.sha256()
        digest.update(b"row-group-v2")
        digest.update(str(row_group_metadata.num_rows).encode())
        weak = True
        for i in range(row_group_metadata.num_columns):
            column = row_group_metadata.column(i)
            data_page_offset = int(column.data_page_offset or 0)
            if data_page_offset != 0:
                weak = False
            digest.update(b"|")
            digest.update(column.path_in_schema.encode())
            digest.update(b",")
            digest.update(str(data_page_offset).encode())
            digest.update(b",")
            digest.update(str(column.file_offset).encode())
            digest.update(b",")
            digest.update(str(column.total_compressed_size).encode())
            digest.update(b",")
            digest.update(str(column.total_uncompressed_size).encode())
            digest.update(b",")
            digest.update(str(column.num_values).encode())
            digest.update(b",")
            digest.update(str(column.compression).encode())

            # Include the column statistics, which reflect the actual values in
            # the row group. This is what lets us notice a rewrite whose layout
            # (offsets and sizes) is byte-for-byte identical but whose geometry
            # has changed. Writers that omit statistics leave this out; the
            # file size/mtime fallback below covers that case.
            try:
                statistics = column.statistics
                if statistics is not None and statistics.has_min_max:
                    digest.update(b",stats,")
                    digest.update(repr(statistics.min).encode())
                    digest.update(b",")
                    digest.update(repr(statistics.max).encode())
                    digest.update(b",")
                    digest.update(str(statistics.null_count).encode())
                else:
                    digest.update(b",nostats,")
            except Exception:
                digest.update(b",nostats,")

        return digest.hexdigest(), weak

    def _identify_fragment(self, fragment, file_info_cache):
        row_groups = fragment.row_groups
        metadata = fragment.metadata

        if len(row_groups) == 1:
            row_group = row_groups[0]
            row_group_index = row_group.id
            row_group_metadata = row_group.metadata
            num_rows = row_group.num_rows
            num_bytes = row_group.total_byte_size
            digest, weak = self._row_group_digest(row_group_metadata)
        else:
            # Not a single-row-group fragment (should not happen for datasets
            # created by `dataset()`), so describe the fragment as a whole.
            row_group_index = -1
            num_rows = fragment.count_rows()
            num_bytes = sum(rg.total_byte_size for rg in row_groups)
            digest_hash = hashlib.sha256(b"multi-row-group-v1")
            weak = True
            for rg in row_groups:
                piece, piece_weak = self._row_group_digest(rg.metadata)
                digest_hash.update(piece.encode())
                weak = weak and piece_weak
            digest = digest_hash.hexdigest()

        # Cheap file-level identity. This is always recorded (not only for
        # weak digests) because a rewrite can preserve every layout field.
        path = fragment.path
        if path not in file_info_cache:
            try:
                info = fragment.filesystem.get_file_info(path)
                if info.type == _fs.FileType.NotFound:
                    file_info_cache[path] = (None, None)
                else:
                    file_info_cache[path] = (info.size, info.mtime_ns)
            except Exception:
                file_info_cache[path] = (None, None)
        file_size, file_mtime_ns = file_info_cache[path]

        return {
            "path": path,
            "row_group_index": row_group_index,
            "num_rows": num_rows,
            "num_bytes": num_bytes,
            "digest": digest,
            "digest_weak": weak,
            "schema_digest": self._schema_digest(fragment.physical_schema),
            "file_size": file_size,
            "file_mtime_ns": file_mtime_ns,
        }

    @staticmethod
    def _identity_matches(identity, saved_row):
        if identity["digest"] != saved_row["row_group_digest"]:
            return False
        if identity["num_rows"] != saved_row["row_group_num_rows"]:
            return False
        if identity["num_bytes"] != saved_row["row_group_num_bytes"]:
            return False
        if identity["schema_digest"] != saved_row["schema_digest"]:
            return False

        # Always require a matching file-level identity when it was recorded.
        # If it could not be recorded at save time, refuse to reuse the entry
        # rather than risk hiding changed data.
        saved_size = saved_row["file_size"]
        saved_mtime = saved_row["file_mtime_ns"]
        if saved_size is None and saved_mtime is None:
            return False
        if saved_size is not None and identity["file_size"] != saved_size:
            return False
        if saved_mtime is not None and identity["file_mtime_ns"] != saved_mtime:
            return False

        return True

    # ----- Index construction ----------------------------------------------

    def _build_index(self, geometry_columns, geometry_types, num_threads=None):
        fragments = list(self.get_fragments())
        geometry_columns = list(geometry_columns)
        geometry_types = list(geometry_types)

        if num_threads is None:
            num_threads = _pa.cpu_count()

        saved_rows = self._load_saved_rows()

        use_stats = [
            self._use_column_statistics and type.coord_type == CoordType.SEPARATED
            for type in geometry_types
        ]
        stats_columns = {
            col for col, can_use in zip(geometry_columns, use_stats) if can_use
        }

        file_info_cache = {}
        rows = []
        scan_tasks = []
        any_matched = False

        for i, fragment in enumerate(fragments):
            identity = self._identify_fragment(fragment, file_info_cache)
            key = self._row_key(identity)
            saved_row = saved_rows.get(key) if saved_rows is not None else None

            if saved_row is not None and self._identity_matches(identity, saved_row):
                boxes = {col: saved_row["boxes"].get(col) for col in geometry_columns}
                if saved_row["box_digest"] == self._box_digest(boxes):
                    any_matched = True
                    rows.append({"identity": identity, "boxes": boxes, "reused": True})
                    continue
                # The stored envelope does not match its own checksum (e.g.,
                # the file was corrupted or hand-edited): recompute it.
                any_matched = True

            boxes = {}
            scan_columns = []
            for col in geometry_columns:
                if col in stats_columns:
                    box = self._fragment_box_from_stats(fragment, col)
                    if box is not None:
                        boxes[col] = box
                        continue
                scan_columns.append(col)

            rows.append({"identity": identity, "boxes": boxes, "reused": False})
            if scan_columns:
                scan_tasks.append((i, fragment, scan_columns))

        # Compute any envelopes that column statistics could not provide.
        if scan_tasks:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for i, fragment, scan_columns in scan_tasks:
                    for col in scan_columns:
                        type = geometry_types[geometry_columns.index(col)]
                        futures.append(
                            (
                                i,
                                col,
                                executor.submit(
                                    GeoDataset._index_fragment, fragment, col, type
                                ),
                            )
                        )

                wait([future for _, _, future in futures])

                for i, col, future in futures:
                    value = future.result()
                    rows[i]["boxes"][col] = value[0].as_py() if len(value) else None

        self._index_rows = rows
        self._index_was_loaded = self._index_was_loaded and any_matched
        return self._index_table(rows, geometry_columns)

    def _index_table(self, rows, geometry_columns):
        columns = [_pa.array(range(len(rows)), type=_pa.int64())]
        for col in geometry_columns:
            columns.append(
                _pa.array([row["boxes"].get(col) for row in rows], type=_BOX_TYPE)
            )
        return _pa.table(
            columns, names=[_FRAGMENT_INDEX_COLUMN] + list(geometry_columns)
        )

    # ----- Column statistics -----------------------------------------------

    def _build_index_using_stats(self, geometry_columns):
        geometry_columns = list(geometry_columns)
        boxes_by_column = [[] for _ in geometry_columns]
        for fragment in self.get_fragments():
            for i, col in enumerate(geometry_columns):
                boxes_by_column[i].append(
                    self._fragment_box_from_stats(fragment, col)
                )

        return [
            _pa.array(boxes, type=_BOX_TYPE) for boxes in boxes_by_column
        ]

    def _fragment_box_from_stats(self, fragment, column):
        try:
            metadata = fragment.row_groups[0].metadata
        except Exception:
            return None

        x_index, y_index = ParquetRowGroupGeoDataset._xy_leaf_indices(
            fragment.physical_schema, column
        )
        if x_index is None:
            return None

        try:
            stats_x = metadata.column(x_index).statistics
            stats_y = metadata.column(y_index).statistics
            if stats_x is None or stats_y is None:
                return None
            if not (stats_x.has_min_max and stats_y.has_min_max):
                return None

            box = {
                "xmin": float(stats_x.min),
                "ymin": float(stats_y.min),
                "xmax": float(stats_x.max),
                "ymax": float(stats_y.max),
            }
        except (TypeError, ValueError, IndexError):
            return None

        # A NaN bounds a single ordinate; a full scan produces a different
        # envelope, so do not trust statistics in that case.
        if any(value != value for value in box.values()):  # NaN check
            return None

        return box

    @staticmethod
    def _xy_leaf_indices(schema, column):
        fields_before = ParquetRowGroupGeoDataset._count_fields_before(schema)

        x_index = None
        y_index = None
        for path, count in fields_before:
            if not path or path[0] != column:
                continue
            if path[-1] == "x":
                x_index = count
            elif path[-1] == "y":
                y_index = count

        if x_index is None or y_index is None:
            return None, None

        return x_index, y_index

    @staticmethod
    def _count_fields_before(field, fields_before=None, path=(), count=0):
        """Helper to find the parquet column index of a given field path"""

        if isinstance(field, _pa.Schema):
            fields_before = []
            for i in range(len(field.types)):
                count = ParquetRowGroupGeoDataset._count_fields_before(
                    field.field(i), fields_before, path, count
                )
            return fields_before

        if isinstance(field.type, _pa.ExtensionType):
            field = _pa.field(field.name, field.type.storage_type)

        if _types.is_nested(field.type):
            path = path + (field.name,)
            fields_before.append((path, count))
            for i in range(field.type.num_fields):
                count = ParquetRowGroupGeoDataset._count_fields_before(
                    field.type.field(i), fields_before, path, count
                )
            return count
        else:
            fields_before.append((path + (field.name,), count))
            return count + 1


# Use a lazy import here to avoid requiring pyarrow.dataset
def dataset(
    *args,
    geometry_columns=None,
    use_row_groups=None,
    index_path=None,
    save_index=False,
    **kwargs,
):
    """Construct a GeoDataset

    This constructor is intended to mirror `pyarrow.dataset()`, adding
    geo-specific arguments. See :class:`geoarrow.pyarrow.dataset.GeoDataset` for
    details.

    For Parquet datasets, ``index_path`` controls the sidecar file used to
    persist the row-group index. ``None`` (the default) derives a hidden path
    next to the data, ``False`` disables the saved index, and a string uses the
    given path. When a saved index exists, it is validated on first use and
    only the entries whose file/row-group identity still matches are reused.
    Pass ``save_index=True`` to build and write the index immediately.

    >>> import geoarrow.pyarrow.dataset as gads
    >>> import geoarrow.pyarrow as ga
    >>> import pyarrow as pa
    >>> table = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
    >>> dataset = gads.dataset(table)
    """

    parent = _ds.dataset(*args, **kwargs)

    if use_row_groups is None:
        use_row_groups = isinstance(parent, _ds.FileSystemDataset) and isinstance(
            parent.format, _ds.ParquetFileFormat
        )
    if use_row_groups:
        return ParquetRowGroupGeoDataset.create(
            parent,
            geometry_columns=geometry_columns,
            index_path=index_path,
            save_index=save_index,
        )
    else:
        return GeoDataset(parent, geometry_columns=geometry_columns)
