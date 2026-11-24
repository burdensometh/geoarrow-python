"""
Experimental geospatial-agumented wrapper around a ``pyarrow.dataset``.

>>> import geoarrow.pyarrow.dataset as gads
"""

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

from geoarrow.pyarrow._compute import box as _ga_box
from geoarrow.pyarrow._kernel import Kernel
from geoarrow.pyarrow._type import GeometryExtensionType, wkb, wkt
from geoarrow.types import CoordType

import pyarrow as _pa
import pyarrow.compute as _compute
import pyarrow.dataset as _ds
import pyarrow.parquet as _pq
import pyarrow.types as _types


def _fragment_row_count(fragment):
    """Number of rows a fragment scan yields.

    ``ParquetFileFragment.count_rows()`` ignores the row group filter of a
    fragment produced by ``split_by_row_group()``, so sum the referenced
    row groups explicitly where present.
    """
    row_groups = getattr(fragment, "row_groups", None)
    if row_groups:
        return sum(row_group.num_rows for row_group in row_groups)
    return fragment.count_rows()


def _byte_budget_ranges(n_rows, nbytes, max_bytes):
    """Split ``range(n_rows)`` into consecutive ``(offset, n_rows)`` ranges.

    ``nbytes(offset, n_rows)`` must return the byte size of a row range.
    Each emitted range fits within ``max_bytes`` unless it is a single row
    that exceeds ``max_bytes`` on its own (which is emitted as-is).
    """
    ranges = []
    _byte_budget_ranges_inner(0, n_rows, nbytes, max_bytes, ranges)
    return ranges


def _byte_budget_ranges_inner(offset, n_rows, nbytes, max_bytes, out):
    if n_rows == 0:
        return
    if n_rows == 1 or nbytes(offset, n_rows) <= max_bytes:
        out.append((offset, n_rows))
        return
    half = n_rows // 2
    _byte_budget_ranges_inner(offset, half, nbytes, max_bytes, out)
    _byte_budget_ranges_inner(offset + half, n_rows - half, nbytes, max_bytes, out)


@dataclass
class EnvelopeIdBatch:
    """One budget-bounded batch of computed envelope and business ID columns.

    A single piece of the output of
    :meth:`GeoDataset.iter_envelope_id_batches`, carrying its provenance:
    the source fragment (for :class:`ParquetRowGroupGeoDataset`, the source
    row group) and the half-open row range ``[row_start, row_end)`` within
    that fragment. ``table`` contains exactly the computed envelope and
    business ID columns and has ``row_end - row_start`` rows, such that
    concatenating the tables of all batches in iteration order restores
    columns aligned row-for-row with the source dataset.

    ``oversized`` is True when the batch holds a single geometry whose
    serialized size alone exceeds the batching budget. These batches are
    the independent strategy for oversized geometries: they are always
    isolated from other rows so that the caller can route them to a
    dedicated path (e.g., persist and release immediately).
    """

    fragment_index: int
    row_start: int
    row_end: int
    table: _pa.Table
    oversized: bool = False

    def __post_init__(self):
        if self.table.num_rows != self.row_end - self.row_start:
            raise ValueError(
                f"Batch table has {self.table.num_rows} rows but row range "
                f"[{self.row_start}, {self.row_end}) implies "
                f"{self.row_end - self.row_start}"
            )

    @property
    def num_rows(self):
        """Number of source rows covered by this batch."""
        return self.table.num_rows


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

    def _resolve_geometry_column(self, geometry_column):
        if geometry_column is None:
            geometry_columns = self.geometry_columns
            if len(geometry_columns) != 1:
                raise ValueError(
                    "geometry_column must be specified for a dataset with "
                    f"{len(geometry_columns)} geometry columns"
                )
            geometry_column = geometry_columns[0]

        if geometry_column not in self.geometry_columns:
            raise ValueError(
                f"geometry_column '{geometry_column}' is not a geometry column "
                "of this dataset"
            )

        geometry_type = self.geometry_types[
            self.geometry_columns.index(geometry_column)
        ]
        return geometry_column, geometry_type

    def iter_envelope_id_batches(
        self,
        geometry_column=None,
        id_column=None,
        max_bytes=2**26,
        batch_size=None,
        envelope_name="envelope",
        id_name="id",
    ):
        """Iterate over computed envelope and business ID columns in batches.

        EXPERIMENTAL

        Computes two derived columns for this dataset --- a Cartesian 2D
        envelope (see :func:`geoarrow.pyarrow.box`) and a business ID ---
        and yields them as :class:`EnvelopeIdBatch` objects whose source
        geometry size is bounded by ``max_bytes``. Reading a very large
        Arrow chunk and computing over it in a single step can exhaust
        memory, so the computation is instead split into consumable
        batches: each batch carries the index of its source fragment (for
        :class:`ParquetRowGroupGeoDataset`, its source row group) and the
        half-open row range ``[row_start, row_end)`` within that fragment.

        This is a lazy generator: the next batch is only read and computed
        when the caller requests it, so the caller can confirm each batch
        (e.g., persist or forward it) before any further work happens. At
        most about ``max_bytes`` of pending source data plus the scanner's
        read-ahead is held in memory at a time.

        Batches are yielded in dataset row order and are contiguous and
        non-overlapping, so concatenating the ``table`` of every batch
        (e.g., with ``pyarrow.concat_tables()``) restores columns that
        align row-for-row with ``dataset.to_table()``. Use
        :meth:`collect_envelope_id_table` to do this with coverage
        validation.

        A single geometry whose serialized size exceeds ``max_bytes``
        cannot be split further. These are handled with an independent
        strategy: each is emitted on its own as a single-row batch flagged
        with ``oversized=True`` (its envelope is still computed), so the
        caller can route it to a dedicated path.

        Spherical edges are not supported (see :func:`geoarrow.pyarrow.box`).

        Parameters
        ----------
        geometry_column : str, optional
            The geometry column to compute envelopes from. Defaults to the
            sole geometry column of the dataset.
        id_column : str, optional
            An existing column to carry through as the business ID. If
            omitted, a deterministic business ID is generated as a uint64
            sequence ``0..N-1`` following dataset row order (which doubles
            as a row-alignment key back to the source table).
        max_bytes : int
            Maximum number of source geometry/ID bytes per batch. This
            bounds the memory of a single compute step; it is not a limit
            on the (much smaller) computed output.
        batch_size : int, optional
            Record batch size used when scanning fragments. Defaults to the
            pyarrow scanner default.
        envelope_name, id_name : str
            Names of the two output columns in each batch's ``table``.

        Returns
        -------
        iterator of :class:`EnvelopeIdBatch`

        >>> import geoarrow.pyarrow.dataset as gads
        >>> import geoarrow.pyarrow as ga
        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     [ga.array(["POINT (0.5 1.5)", "POINT (2.5 3.5)"])], ["geometry"]
        ... )
        >>> dataset = gads.dataset(table)
        >>> for batch in dataset.iter_envelope_id_batches():
        ...     print(batch.fragment_index, batch.row_start, batch.row_end, batch.oversized)
        ...     print(batch.table.to_pydict())
        0 0 2 False
        {'envelope': [{'xmin': 0.5, 'ymin': 1.5, 'xmax': 0.5, 'ymax': 1.5}, {'xmin': 2.5, 'ymin': 3.5, 'xmax': 2.5, 'ymax': 3.5}], 'id': [0, 1]}

        A tiny budget splits the rows; a single geometry that alone exceeds
        the budget is isolated and flagged as oversized:

        >>> batches = list(dataset.iter_envelope_id_batches(max_bytes=1))
        >>> [(b.row_start, b.row_end, b.oversized) for b in batches]
        [(0, 1, True), (1, 2, True)]
        """
        geometry_column, geometry_type = self._resolve_geometry_column(
            geometry_column
        )
        if id_column is not None:
            if id_column == geometry_column:
                raise ValueError("id_column must differ from geometry_column")
            if id_column not in self.schema.names:
                raise ValueError(
                    f"id_column '{id_column}' is not a column of this dataset"
                )
        if max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        return self._iter_envelope_id_batches(
            geometry_column,
            geometry_type,
            id_column,
            max_bytes,
            batch_size,
            envelope_name,
            id_name,
        )

    def _iter_envelope_id_batches(
        self,
        geometry_column,
        geometry_type,
        id_column,
        max_bytes,
        batch_size,
        envelope_name,
        id_name,
    ):
        columns = [geometry_column]
        if id_column is not None:
            columns.append(id_column)
        scan_kwargs = {"columns": columns}
        if batch_size is not None:
            scan_kwargs["batch_size"] = batch_size

        next_id = 0
        for fragment_index, fragment in enumerate(self.get_fragments()):
            row_cursor = 0
            pending = []
            pending_bytes = 0

            def build_batch(pieces, oversized):
                nonlocal next_id
                row_start = pieces[0][0]
                row_end = pieces[-1][0] + pieces[-1][1]
                envelopes = []
                id_slices = []
                n_total = 0
                for _, n_rows, geometry_slice, id_slice in pieces:
                    envelopes.append(_ga_box(geometry_slice))
                    if id_slice is not None:
                        id_slices.append(id_slice)
                    n_total += n_rows

                if len(envelopes) == 1:
                    envelope = envelopes[0]
                else:
                    envelope = _pa.concat_arrays(envelopes)

                if id_column is not None:
                    if len(id_slices) == 1:
                        ids = id_slices[0]
                    else:
                        ids = _pa.concat_arrays(id_slices)
                else:
                    ids = _pa.array(
                        range(next_id, next_id + n_total), type=_pa.uint64()
                    )
                    next_id += n_total

                table = _pa.Table.from_arrays(
                    [envelope, ids], names=[envelope_name, id_name]
                )
                return EnvelopeIdBatch(
                    fragment_index, row_start, row_end, table, oversized
                )

            reader = fragment.scanner(**scan_kwargs).to_reader()
            for record_batch in reader:
                n_batch = record_batch.num_rows
                if n_batch == 0:
                    continue

                geometry = record_batch.column(geometry_column)
                if not isinstance(geometry.type, GeometryExtensionType):
                    geometry = geometry_type.wrap_array(geometry)
                ids = (
                    record_batch.column(id_column)
                    if id_column is not None
                    else None
                )

                def slice_nbytes(offset, n_rows):
                    total = geometry.slice(offset, n_rows).nbytes
                    if ids is not None:
                        total += ids.slice(offset, n_rows).nbytes
                    return total

                ranges = _byte_budget_ranges(n_batch, slice_nbytes, max_bytes)
                for offset, n_rows in ranges:
                    piece = (
                        row_cursor + offset,
                        n_rows,
                        geometry.slice(offset, n_rows),
                        ids.slice(offset, n_rows) if ids is not None else None,
                    )
                    piece_bytes = slice_nbytes(offset, n_rows)

                    if n_rows == 1 and piece_bytes > max_bytes:
                        # Independent strategy for a single oversized
                        # geometry: flush pending rows, then emit it
                        # isolated in its own flagged batch.
                        if pending:
                            yield build_batch(pending, oversized=False)
                            pending = []
                            pending_bytes = 0
                        yield build_batch([piece], oversized=True)
                    else:
                        if pending and pending_bytes + piece_bytes > max_bytes:
                            yield build_batch(pending, oversized=False)
                            pending = []
                            pending_bytes = 0
                        pending.append(piece)
                        pending_bytes += piece_bytes

                row_cursor += n_batch

            if pending:
                yield build_batch(pending, oversized=False)

    def collect_envelope_id_table(self, **kwargs):
        """Compute envelope and business ID columns for the entire dataset.

        EXPERIMENTAL

        Runs :meth:`iter_envelope_id_batches` (all keyword arguments are
        passed through), validates that the batches form a gapless, ordered
        coverage of every fragment (row group), and concatenates them into
        a single table whose rows align with ``dataset.to_table()``.

        >>> import geoarrow.pyarrow.dataset as gads
        >>> import geoarrow.pyarrow as ga
        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     [ga.array(["POINT (0.5 1.5)", "POINT (2.5 3.5)"])], ["geometry"]
        ... )
        >>> dataset = gads.dataset(table)
        >>> dataset.collect_envelope_id_table().to_pydict()
        {'envelope': [{'xmin': 0.5, 'ymin': 1.5, 'xmax': 0.5, 'ymax': 1.5}, {'xmin': 2.5, 'ymin': 3.5, 'xmax': 2.5, 'ymax': 3.5}], 'id': [0, 1]}
        """
        batches = list(self.iter_envelope_id_batches(**kwargs))
        self._validate_envelope_id_coverage(batches)

        if batches:
            return _pa.concat_tables([batch.table for batch in batches])

        # No rows: rebuild the empty output table with the correct schema
        geometry_column, geometry_type = self._resolve_geometry_column(
            kwargs.get("geometry_column")
        )
        id_column = kwargs.get("id_column")
        envelope_name = kwargs.get("envelope_name", "envelope")
        id_name = kwargs.get("id_name", "id")

        empty_geometry = geometry_type.wrap_array(
            _pa.array([], type=geometry_type.storage_type)
        )
        envelope = _ga_box(empty_geometry)
        if id_column is not None:
            ids = _pa.array([], type=self.schema.field(id_column).type)
        else:
            ids = _pa.array([], type=_pa.uint64())

        return _pa.Table.from_arrays([envelope, ids], names=[envelope_name, id_name])

    def _validate_envelope_id_coverage(self, batches):
        """Check that batches cover every dataset row exactly once, in order."""
        fragment_rows = [_fragment_row_count(f) for f in self.get_fragments()]
        fragment_index = 0
        row = 0
        for batch in batches:
            while (
                fragment_index < len(fragment_rows)
                and row == fragment_rows[fragment_index]
            ):
                fragment_index += 1
                row = 0
            if (
                fragment_index == len(fragment_rows)
                or batch.fragment_index != fragment_index
                or batch.row_start != row
                or batch.row_end > fragment_rows[fragment_index]
            ):
                raise ValueError(
                    "Batches do not form a gapless, ordered coverage of the "
                    f"dataset: unexpected batch {batch!r}"
                )
            row = batch.row_end

        while (
            fragment_index < len(fragment_rows)
            and row == fragment_rows[fragment_index]
        ):
            fragment_index += 1
            row = 0
        if fragment_index != len(fragment_rows):
            raise ValueError(
                "Batches do not cover the entire dataset: "
                f"stopped at fragment {fragment_index} of {len(fragment_rows)}"
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
            return _pa.table(metadata, names=["_fragment_index"])

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
                metadata + result_arrays, names=["_fragment_index"] + columns
            )

    @staticmethod
    def _index_box_intersects(index, box, columns):
        xmin, ymin, xmax, ymax = box.values()
        expressions = []
        for col in columns:
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

    @property
    def row_group_ids(self):
        """Row group ID within its source file for each fragment.

        The i-th value is the row group ID of ``get_fragments()[i]`` within
        its Parquet file, which completes the provenance carried by
        :class:`EnvelopeIdBatch` (``fragment_index`` + row range) for
        batches computed by :meth:`GeoDataset.iter_envelope_id_batches`.

        >>> import geoarrow.pyarrow.dataset as gads
        >>> import geoarrow.pyarrow as ga
        >>> import pyarrow as pa
        >>> import pyarrow.parquet as pq
        >>> import tempfile
        >>> import os
        >>> with tempfile.TemporaryDirectory() as td:
        ...     table = pa.table([ga.array(["POINT (0 1)", "POINT (2 3)"])], ["geometry"])
        ...     pq.write_table(table, os.path.join(td, "test.parquet"), row_group_size=1)
        ...     dataset = gads.dataset(os.path.join(td, "test.parquet"))
        ...     dataset.row_group_ids
        (0, 1)
        """
        return tuple(self._row_group_ids)

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

    def _build_index(self, geometry_columns, geometry_types, num_threads=None):
        can_use_statistics = [
            type.coord_type == CoordType.SEPARATED for type in self.geometry_types
        ]

        if not self._use_column_statistics or not any(can_use_statistics):
            return super()._build_index(geometry_columns, geometry_types, num_threads)

        # Build a list of columns that will work with column stats
        bbox_stats_cols = []
        for col, use_stats in zip(geometry_columns, can_use_statistics):
            if use_stats:
                bbox_stats_cols.append(col)

        # Compute the column stats
        bbox_stats = self._build_index_using_stats(bbox_stats_cols)
        normal_stats_cols = list(geometry_columns)
        stats_by_name = {}
        for col, stat in zip(bbox_stats_cols, bbox_stats):
            # stat will contain nulls if any statistics were missing:
            if stat.null_count == 0:
                stats_by_name[col] = stat
                normal_stats_cols.remove(col)

        # Compute any remaining statistics
        normal_stats = super()._build_index(
            normal_stats_cols, geometry_types, num_threads
        )
        for col in normal_stats_cols:
            stats_by_name[col] = normal_stats.column(col)

        # Reorder stats to match the order of geometry_columns
        stat_cols = [stats_by_name[col] for col in geometry_columns]
        return _pa.table(
            [normal_stats.column(0)] + stat_cols,
            names=["_fragment_index"] + list(geometry_columns),
        )

    def _build_index_using_stats(self, geometry_columns):
        parquet_fields_before = ParquetRowGroupGeoDataset._count_fields_before(
            self.schema
        )
        parquet_fields_before = {k: v for k, v in parquet_fields_before}
        parquet_fields_before = [
            parquet_fields_before[(col,)] for col in geometry_columns
        ]
        return self._parquet_field_boxes(parquet_fields_before)

    def _parquet_field_boxes(self, parquet_indices):
        boxes = [[]] * len(parquet_indices)
        pq_file = None
        last_row_group = None

        # Note: probably worth parallelizing by file
        for row_group, fragment in zip(self._row_group_ids, self.get_fragments()):
            if pq_file is None or row_group < last_row_group:
                pq_file = _pq.ParquetFile(
                    fragment.path, filesystem=self._parent.filesystem
                )

            metadata = pq_file.metadata.row_group(row_group)
            for i, parquet_index in enumerate(parquet_indices):
                stats_x = metadata.column(parquet_index).statistics
                stats_y = metadata.column(parquet_index + 1).statistics

                if stats_x is None or stats_y is None:
                    boxes[i].append(None)
                else:
                    boxes[i].append(
                        {
                            "xmin": stats_x.min,
                            "ymin": stats_y.min,
                            "xmax": stats_x.max,
                            "ymax": stats_y.max,
                        }
                    )

            last_row_group = row_group

        type_field_names = ["xmin", "ymin", "xmax", "ymax"]
        type_fields = [
            _pa.field(name, _pa.float64(), nullable=False) for name in type_field_names
        ]
        type = _pa.struct(type_fields)
        return [_pa.array(box, type=type) for box in boxes]

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
