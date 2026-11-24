"""
Experimental geospatial-agumented wrapper around a ``pyarrow.dataset``.

>>> import geoarrow.pyarrow.dataset as gads
"""

from concurrent.futures import ThreadPoolExecutor, wait

from geoarrow.pyarrow._kernel import Kernel
from geoarrow.pyarrow._type import GeometryExtensionType, wkb, wkt
from geoarrow.types import CoordType

import pyarrow as _pa
import pyarrow.compute as _compute
import pyarrow.dataset as _ds
import pyarrow.parquet as _pq
import pyarrow.types as _types

#: Value of the fragment index status column when the fragment's envelope
#: was taken from Parquet column statistics.
ENVELOPE_STATISTICS = "statistics"

#: Value of the fragment index status column when the fragment's envelope
#: was computed by reading the fragment (e.g., because column statistics
#: were missing or unusable).
ENVELOPE_COMPUTED = "computed"

#: Value of the fragment index status column when the fragment contains no
#: non-empty geometries. The envelope of these fragments is null: an empty
#: geometry has no coordinates, and its "envelope" must not be confused
#: with a point at the origin.
ENVELOPE_EMPTY = "empty"

#: Internal status used while building the index for fragments whose
#: envelope is not yet known (e.g., missing or unusable Parquet
#: statistics). These fragments are always resolved by reading them before
#: the index is returned, such that this value never occurs in a finished
#: index.
_ENVELOPE_UNKNOWN = "unknown"

_BOX_FIELD_NAMES = ("xmin", "ymin", "xmax", "ymax")
_BOX_STORAGE_TYPE = _pa.struct(
    [_pa.field(name, _pa.float64(), nullable=False) for name in _BOX_FIELD_NAMES]
)


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

    def index_fragments(self, num_threads=None):
        """Resolve a simplified geometry for each fragment

        Currently the simplified geometry is a box in the form of a
        struct array with fields xmin, xmax, ymin, and ymax. The
        fragment index is a table whose first column is the fragment
        index and whose subsequent columns are named with the geometry
        column name, each followed by a ``_<geometry column>_status``
        column recording how the envelope was obtained:

        - ``"statistics"``: the envelope was taken from Parquet column
          statistics (only for a :class:`ParquetRowGroupGeoDataset`).
        - ``"computed"``: the envelope was computed by reading the
          fragment (e.g., because statistics were missing or unusable).
        - ``"empty"``: the fragment contains no non-empty geometries.
          The envelope of these fragments is null: an empty geometry has
          no coordinates, and its envelope must not be confused with a
          point at the origin.

        A future implementation may handle spherical edges using a type
        of simplified geometry more suitable to a spherical comparison.

        >>> import geoarrow.pyarrow.dataset as gads
        >>> import geoarrow.pyarrow as ga
        >>> import pyarrow as pa
        >>> table = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
        >>> dataset = gads.dataset(table)
        >>> dataset.index_fragments().to_pylist()
        [{'_fragment_index': 0, 'geometry': {'xmin': 0.5, 'ymin': 1.5, 'xmax': 0.5, 'ymax': 1.5}, '_geometry_status': 'computed'}]
        """
        if self._index is None:
            self._index = self._build_index(
                self.geometry_columns, self.geometry_types, num_threads
            )

        return self._index

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

        Note that datasets with large row groups/fragments and/or datasets that
        were not written with fragments with spatial significance may return
        most or all of the fragments in the parent dataset.

        Use :meth:`explain_filter` to inspect the per-fragment decisions
        made by this filter.

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
        maybe_intersects = GeoDataset._index_box_intersects(
            self.index_fragments(), target_box, self.geometry_columns
        )
        fragment_indices = [scalar.as_py() for scalar in maybe_intersects]
        filtered_parent = self._filter_parent_fragment_indices(fragment_indices)
        return self._wrap_parent(filtered_parent, fragment_indices)

    def explain_filter(self, target):
        """Explain the per-fragment decisions made by filter_fragments()

        Returns a table with one row per fragment (one row per row group
        for a :class:`ParquetRowGroupGeoDataset`): the fragment index, the
        decision that :meth:`filter_fragments` would make for the fragment
        ("keep" or "prune"), and a human-readable reason per geometry
        column (whether its envelope intersects the target and where the
        envelope came from).

        >>> import geoarrow.pyarrow.dataset as gads
        >>> import geoarrow.pyarrow as ga
        >>> import pyarrow as pa
        >>> table = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
        >>> dataset = gads.dataset(table)
        >>> dataset.explain_filter("POLYGON ((0 1, 0 2, 1 2, 1 1, 0 1))").to_pylist()
        [{'_fragment_index': 0, 'decision': 'keep', 'reason': "envelope of column 'geometry' intersects target (envelope source: computed)"}]
        """
        from ._compute import box_agg

        if isinstance(target, str):
            target = [target]
        target_box = box_agg(target).as_py()

        index = self.index_fragments()
        decisions = []
        reasons = []
        for row in index.to_pylist():
            fragment_keep = False
            fragment_reasons = []
            for col in self.geometry_columns:
                status = row[GeoDataset._status_column_name(col)]
                if status == ENVELOPE_EMPTY:
                    fragment_reasons.append(f"column '{col}' is empty")
                    continue

                box = row[col]
                intersects = (
                    box["xmin"] <= target_box["xmax"]
                    and box["xmax"] >= target_box["xmin"]
                    and box["ymin"] <= target_box["ymax"]
                    and box["ymax"] >= target_box["ymin"]
                )
                if intersects:
                    fragment_keep = True
                    fragment_reasons.append(
                        f"envelope of column '{col}' intersects target "
                        f"(envelope source: {status})"
                    )
                else:
                    fragment_reasons.append(
                        f"envelope of column '{col}' does not intersect target "
                        f"(envelope source: {status})"
                    )

            decisions.append("keep" if fragment_keep else "prune")
            reasons.append("; ".join(fragment_reasons))

        return _pa.table(
            [
                index.column(0),
                _pa.array(decisions, type=_pa.string()),
                _pa.array(reasons, type=_pa.string()),
            ],
            names=["_fragment_index", "decision", "reason"],
        )

    def _wrap_parent(self, filtered_parent, fragment_indices):
        new_wrapped = GeoDataset(
            filtered_parent, geometry_columns=self._geometry_columns
        )
        new_wrapped._geometry_types = self.geometry_types

        new_index = self.index_fragments().take(
            _pa.array(fragment_indices, type=_pa.int64())
        )
        new_wrapped._index = new_index.set_column(
            0, "_fragment_index", _pa.array(range(new_index.num_rows))
        )

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
    def _status_column_name(column):
        return f"_{column}_status"

    @staticmethod
    def _index_fragment(fragment, column, type):
        scanner = fragment.scanner(columns=[column])
        reader = scanner.to_reader()
        kernel = Kernel.box_agg(type)
        for batch in reader:
            kernel.push(batch.column(0))
        return kernel.finish().storage

    @staticmethod
    def _classify_box(box):
        """Classify an envelope dict as empty or as a valid envelope.

        Returns (is_empty, box) where box is None for empty envelopes.
        Empty geometries have no coordinates: an aggregate over zero
        coordinates is null, NaN, or inverted (e.g., xmin=inf, xmax=-inf),
        none of which may be treated as a point at the origin.
        """
        if box is None:
            return True, None

        values = [box[name] for name in _BOX_FIELD_NAMES]
        if any(value is None or value != value for value in values):
            return True, None

        xmin, ymin, xmax, ymax = values
        if xmin > xmax or ymin > ymax:
            return True, None

        return False, dict(zip(_BOX_FIELD_NAMES, values))

    @staticmethod
    def _classify_scanned_box(box):
        empty, box = GeoDataset._classify_box(box)
        if empty:
            return ENVELOPE_EMPTY, None
        return ENVELOPE_COMPUTED, box

    @staticmethod
    def _statistics_all_null(stats):
        return stats.has_null_count and stats.null_count == stats.num_values

    @staticmethod
    def _classify_statistics(stats_x, stats_y):
        """Classify a row group from the Parquet statistics of its x/y columns.

        Returns (status, box): ENVELOPE_STATISTICS with a valid envelope,
        ENVELOPE_EMPTY for row groups that certainly contain no coordinates,
        or _ENVELOPE_UNKNOWN when the statistics are missing or unusable
        (e.g., written without min/max values by an older writer).
        """
        if stats_x is None or stats_y is None:
            return _ENVELOPE_UNKNOWN, None

        if stats_x.has_min_max and stats_y.has_min_max:
            empty, box = GeoDataset._classify_box(
                {
                    "xmin": stats_x.min,
                    "ymin": stats_y.min,
                    "xmax": stats_x.max,
                    "ymax": stats_y.max,
                }
            )
            if not empty:
                return ENVELOPE_STATISTICS, box
            # Statistics whose min/max are missing, NaN, or inverted can't
            # be used for pruning; recompute the envelope from the data.
            return _ENVELOPE_UNKNOWN, None

        # Statistics without min/max values: an all-null column chunk
        # certainly contains no coordinates. Anything else may have been
        # written by an old writer that omitted min/max values, which is
        # indistinguishable from a row group full of empty geometries
        # without reading the data.
        if GeoDataset._statistics_all_null(stats_x) and GeoDataset._statistics_all_null(
            stats_y
        ):
            return ENVELOPE_EMPTY, None

        return _ENVELOPE_UNKNOWN, None

    @staticmethod
    def _scan_boxes(fragments, tasks, types_by_name, num_threads=None):
        """Compute envelopes for (column, fragment index) pairs by reading them.

        Returns a dict mapping (column, fragment_index) to (status, box).
        """
        results = {}
        if not tasks:
            return results

        if num_threads is None:
            num_threads = _pa.cpu_count()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {}
            for column, fragment_index in tasks:
                future = executor.submit(
                    GeoDataset._index_fragment,
                    fragments[fragment_index],
                    column,
                    types_by_name[column],
                )
                futures[(column, fragment_index)] = future

            wait(futures.values())

            for key, future in futures.items():
                scanned = future.result()
                box = scanned[0].as_py() if len(scanned) > 0 else None
                results[key] = GeoDataset._classify_scanned_box(box)

        return results

    @staticmethod
    def _envelope_column(classified):
        """Build the (envelope, status) index columns for one geometry column."""
        envelopes = _pa.array([box for _, box in classified], type=_BOX_STORAGE_TYPE)
        status = _pa.array([status for status, _ in classified], type=_pa.string())
        return envelopes, status

    @staticmethod
    def _index_table(num_fragments, columns, classified_by_column):
        fragment_index = _pa.array(range(num_fragments))
        if not columns:
            return _pa.table([fragment_index], names=["_fragment_index"])

        arrays = []
        names = []
        for column in columns:
            envelopes, status = GeoDataset._envelope_column(
                classified_by_column[column]
            )
            arrays.extend([envelopes, status])
            names.extend([column, GeoDataset._status_column_name(column)])

        return _pa.table([fragment_index] + arrays, names=["_fragment_index"] + names)

    @staticmethod
    def _index_fragments(fragments, columns, types, num_threads=None):
        columns = list(columns)
        num_fragments = len(fragments)
        types_by_name = dict(zip(columns, types))

        tasks = [
            (column, fragment_index)
            for column in columns
            for fragment_index in range(num_fragments)
        ]
        scanned = GeoDataset._scan_boxes(fragments, tasks, types_by_name, num_threads)

        classified_by_column = {
            column: [scanned[(column, j)] for j in range(num_fragments)]
            for column in columns
        }
        return GeoDataset._index_table(num_fragments, columns, classified_by_column)

    @staticmethod
    def _index_box_intersects(index, box, columns):
        xmin, ymin, xmax, ymax = box.values()
        expressions = []
        for col in columns:
            # Fragments without an envelope (i.e., empty fragments, whose
            # status is "empty") evaluate to null here and are excluded.
            expr = (
                (_compute.field(col, "xmin") <= xmax)
                & (_compute.field(col, "xmax") >= xmin)
                & (_compute.field(col, "ymin") <= ymax)
                & (_compute.field(col, "ymax") >= ymin)
            )
            expressions.append(expr)

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

    When column statistics are used, each row group is classified as having
    a valid envelope (``"statistics"``), as certainly containing no
    coordinates (``"empty"``), or as unknown. Row groups whose statistics
    are missing or unusable (e.g., written without min/max values by an
    older writer) are read to compute their envelope, such that they are
    never confused with empty row groups nor silently dropped by
    :meth:`filter_fragments`.
    """

    def __init__(
        self,
        parent,
        row_group_fragments,
        row_group_ids,
        geometry_columns=None,
        use_column_statistics=True,
    ):
        super().__init__(parent, geometry_columns=geometry_columns)
        self._fragments = row_group_fragments
        self._row_group_ids = row_group_ids
        self._use_column_statistics = use_column_statistics

    @staticmethod
    def create(parent, geometry_columns=None, use_column_statistics=True):
        if not isinstance(parent, _ds.FileSystemDataset) or not isinstance(
            parent.format, _ds.ParquetFileFormat
        ):
            raise TypeError(
                "ParquetRowGroupGeoDataset() is only supported for Parquet datasets"
            )

        row_group_fragments = []
        row_group_ids = []

        for file_fragment in parent.get_fragments():
            for i, row_group_fragment in enumerate(file_fragment.split_by_row_group()):
                row_group_fragments.append(row_group_fragment)
                # Keep track of the row group IDs so we can accellerate
                # building an index later where column statistics are supported
                row_group_ids.append(i)

        parent = _ds.FileSystemDataset(
            row_group_fragments, parent.schema, parent.format
        )
        return ParquetRowGroupGeoDataset(
            parent,
            row_group_fragments,
            row_group_ids,
            geometry_columns=geometry_columns,
            use_column_statistics=use_column_statistics,
        )

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
        )
        new_wrapped._index = base_wrapped._index
        return new_wrapped

    def explain_filter(self, target):
        explained = super().explain_filter(target)
        return explained.append_column(
            "_row_group_id", _pa.array(self._row_group_ids, type=_pa.int64())
        )

    def _build_index(self, geometry_columns, geometry_types, num_threads=None):
        can_use_statistics = [
            type.coord_type == CoordType.SEPARATED for type in self.geometry_types
        ]

        if not self._use_column_statistics or not any(can_use_statistics):
            return super()._build_index(geometry_columns, geometry_types, num_threads)

        fragments = self.get_fragments()
        num_fragments = len(fragments)
        types_by_name = dict(zip(geometry_columns, geometry_types))

        # Classify row groups using Parquet column statistics where possible.
        # This distinguishes row groups with a valid envelope, row groups that
        # certainly contain no coordinates, and row groups whose envelope is
        # unknown (e.g., statistics that were never written or that were
        # written without min/max values by an older writer).
        classified_by_column = {}
        scan_tasks = []

        stats_columns = [
            col
            for col, use_stats in zip(geometry_columns, can_use_statistics)
            if use_stats
        ]
        stats_results = self._build_index_using_stats(stats_columns)
        for col in stats_columns:
            classified = stats_results[col]
            for fragment_index, (status, box) in enumerate(classified):
                if status == _ENVELOPE_UNKNOWN:
                    scan_tasks.append((col, fragment_index))
            classified_by_column[col] = classified

        # Columns whose type can't use column statistics are indexed by
        # reading every row group.
        for col, use_stats in zip(geometry_columns, can_use_statistics):
            if not use_stats:
                classified_by_column[col] = [(_ENVELOPE_UNKNOWN, None)] * num_fragments
                scan_tasks.extend((col, j) for j in range(num_fragments))

        # Row groups with missing or unusable statistics are read to compute
        # their envelope (rather than being silently dropped or conservatively
        # kept by the filter).
        scanned = GeoDataset._scan_boxes(
            fragments, scan_tasks, types_by_name, num_threads
        )

        for col in geometry_columns:
            classified_by_column[col] = [
                scanned[(col, j)] if status == _ENVELOPE_UNKNOWN else (status, box)
                for j, (status, box) in enumerate(classified_by_column[col])
            ]

        return GeoDataset._index_table(
            num_fragments, list(geometry_columns), classified_by_column
        )

    def _build_index_using_stats(self, geometry_columns):
        parquet_fields_before = ParquetRowGroupGeoDataset._count_fields_before(
            self.schema
        )
        parquet_fields_before = {k: v for k, v in parquet_fields_before}
        parquet_fields_before = [
            parquet_fields_before[(col,)] for col in geometry_columns
        ]
        return self._parquet_field_boxes(parquet_fields_before, geometry_columns)

    def _parquet_field_boxes(self, parquet_indices, geometry_columns):
        classified = {col: [] for col in geometry_columns}
        pq_file = None
        last_path = None

        # Note: probably worth parallelizing by file
        for row_group, fragment in zip(self._row_group_ids, self.get_fragments()):
            if pq_file is None or fragment.path != last_path:
                pq_file = _pq.ParquetFile(
                    fragment.path, filesystem=self._parent.filesystem
                )
                last_path = fragment.path

            metadata = pq_file.metadata.row_group(row_group)
            for col, parquet_index in zip(geometry_columns, parquet_indices):
                stats_x = metadata.column(parquet_index).statistics
                stats_y = metadata.column(parquet_index + 1).statistics
                classified[col].append(
                    GeoDataset._classify_statistics(stats_x, stats_y)
                )

        return classified

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
def dataset(*args, geometry_columns=None, use_row_groups=None, **kwargs):
    """Construct a GeoDataset

    This constructor is intended to mirror `pyarrow.dataset()`, adding
    geo-specific arguments. See :class:`geoarrow.pyarrow.dataset.GeoDataset` for
    details.

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
            parent, geometry_columns=geometry_columns
        )
    else:
        return GeoDataset(parent, geometry_columns=geometry_columns)
