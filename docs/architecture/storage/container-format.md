# `.grafeo` Container Format Specification

The `.grafeo` file is the single-file persistence format for Grafeo databases.
This page describes container format v3, written since 0.6.0. A file holds
**images**: an image is the state of one checkpoint, made of the **chunks** of
its typed **sections** and a chained **directory** that lists them. Every
header, directory block and chunk is checksummed, and checkpoints are
copy-on-write: a new image never overwrites a page the active one uses.

Files written by 0.5.x (container v1 and v2) are described under
[Files Written by 0.5.x](#files-written-by-05x).

## File Layout

```text
Offset    Size     Contents
────────────────────────────────────────────────────
0x0000    4 KiB    File header (magic, format version 3, flags, database id)
0x1000    4 KiB    Database header, slot 0
0x2000    4 KiB    Database header, slot 1
0x3000+   pages    Chunks and directory blocks of the images
```

The page size is 4 KiB. The headers take the first three pages (12 KiB); data
starts at page 3 (`0x3000`). Every chunk and directory block starts on a page
boundary, except that a chunk without bytes is stored with offset 0 and length 0
and takes no pages. All integers are little-endian.

---

## File Header (0x0000, 4 KiB)

Written once when the database is created, never modified afterwards.

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | `[u8; 4]` | `magic` | `GRAF` |
| 4 | 4 | `u32` | `format_version` | `3` |
| 8 | 4 | `u32` | `page_size` | Always `4096` |
| 12 | 4 | `u32` | `flags` | Feature flags (see below) |
| 16 | 16 | `u128` | `database_id` | Random id of the database, set at creation |
| 32 | 8 | `u64` | `creation_timestamp_ms` | Unix epoch milliseconds |
| 40 | 32 | `[u8; 32]` | `creator_version` | UTF-8 Grafeo version, zero-padded |
| 72 | 4 | `u32` | `crc` | CRC-32 of bytes 0..72 |
| 76 | 4020 | - | (reserved) | Written as zero, ignored by readers |

**Flags:** bits 0 to 15 are incompatible features, which change how the file
must be read: a reader refuses a file that sets one it does not know. Bits 16
to 31 are compatible features, which a reader that does not know them ignores.
The only feature today is incompatible bit 0: the file is encrypted.

**Validation on open**, in this order: the magic, the CRC, the format version
(must be 3), the page size (must be 4096) and the incompatible flags.

---

## Database Headers (0x1000 and 0x2000, 4 KiB each)

Two alternating slots provide crash safety. A checkpoint writes its header into
the **inactive** slot, so the active one stays intact until the new header is
on disk.

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | `[u8; 4]` | `magic` | `GDBH` |
| 4 | 4 | `u32` | `format_revision` | Which additions to v3 the image may use (see below) |
| 8 | 8 | `u64` | `iteration` | Checkpoint counter, higher = current |
| 16 | 8 | `u64` | `checkpoint_lsn` | WAL position the checkpoint covers (currently always 0) |
| 24 | 8 | `u64` | `epoch` | MVCC epoch at the checkpoint, which an open continues from |
| 32 | 8 | `u64` | `last_transaction_id` | Last committed transaction id |
| 40 | 8 | `u64` | `root.offset` | Offset of the image's first directory block |
| 48 | 4 | `u32` | `root.length` | Length of that block |
| 52 | 4 | `u32` | `root.crc` | CRC-32 of that block |
| 56 | 8 | `u64` | `node_count` | LPG node count |
| 64 | 8 | `u64` | `edge_count` | LPG edge count |
| 72 | 8 | `u64` | `timestamp_ms` | Checkpoint time (Unix epoch ms) |
| 80 | 4 | `u32` | `crc` | CRC-32 of bytes 0..80 |
| 84 | 4012 | - | (reserved) | Written as zero, ignored by readers |

**Active header selection:** a slot whose bytes are all zero was never written.
Any other slot without the magic and a matching CRC is damaged (a torn or
corrupted write). The valid slot with the higher `iteration` is active (slot 0
on a tie). If no slot is valid and one is damaged, the open fails instead of
treating the file as empty: the image the damaged slot pointed at may still be
in the file. A new database gets a valid iteration-0 header in slot 0 that
points at an image without sections.

**Format revision:** the active header's `format_revision` says which additions
to container v3 its image may use. 0.6.0 writes and reads revision 1. A reader
refuses a revision above the highest it knows ("written by a newer version of
Grafeo") and revision 0 (a 0.6.0 development build from before revisions, which
no release reads), before it reads the directory, and leaves the file as it is.
A checkpoint writes the revision of the header it replaces, so a file keeps its
revision: a 0.6.x release that adds a revision gives it to a file only through
an explicit upgrade, and new files keep revision 1 throughout 0.6.x.

---

## Directory

The directory of an image lists every chunk in fixed 48-byte entries. It is
stored in blocks of at most 64 KiB (up to 1,364 entries each); each block names
the next one, so the number of chunks is not limited. The active database
header points at the first block.

### Directory Block

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | `[u8; 4]` | `magic` | `GDIR` |
| 4 | 4 | `u32` | `entry_count` | Entries in this block |
| 8 | 8 | `u64` | `next.offset` | Offset of the next block |
| 16 | 4 | `u32` | `next.length` | Length of the next block; `0` ends the chain |
| 20 | 4 | `u32` | `next.crc` | CRC-32 of the next block |
| 24 | 8 | `u64` | (reserved) | Zero |
| 32 | 48 each | | entries | `entry_count` directory entries |

The CRC of a block is kept by whoever points at it: the database header for the
first block, the previous block for the others. An open checks that every block
pointer is page-aligned and lies in the data area, that the length fits the
entries, the CRC and the magic, and that the chain never visits a block twice.

### Directory Entry (48 bytes)

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 1 | `u8` | `section_type` | Section the chunk belongs to (see below) |
| 1 | 1 | `u8` | `section_version` | Format version of the section's bytes |
| 2 | 1 | `u8` | `chunk_kind` | What the chunk holds; `0` = raw bytes |
| 3 | 1 | `u8` | `codec` | Codec of the chunk's bytes; `0` = none |
| 4 | 4 | `u32` | `graph_id` | Graph of the chunk (`0` when not graph-specific) |
| 8 | 4 | `u32` | `column_id` | Column of the chunk (`0` when not column-specific) |
| 12 | 4 | `u32` | `row_count` | Rows the chunk holds |
| 16 | 8 | `u64` | `row_start` | First row the chunk holds |
| 24 | 8 | `u64` | `offset` | Byte offset of the chunk (page-aligned; 0 for a chunk without bytes) |
| 32 | 8 | `u64` | `length` | Stored length of the chunk |
| 40 | 4 | `u32` | `crc` | CRC-32 of the stored chunk |
| 44 | 1 | `u8` | `flags` | Bit 0: a reader that does not know the section type skips the entry; bit 1: a reader that knows the section type but not the chunk kind skips it (see below) |
| 45 | 1 | `u8` | `namespace` | The numbering `column_id` belongs to (see [Namespaces](#namespaces)) |
| 46 | 2 | - | (reserved) | Written as zero, ignored by readers |

**Unknown entries.** A newer version may add section types and chunk kinds.
The flags of each entry tell an older reader what to do with one it does not
know:

- Bit 0 (`0x01`, section optional): a reader that does not know the section
  type skips the entry.
- Bit 1 (`0x02`, chunk optional): a reader that knows the section type but not
  the chunk kind skips the entry. Bit 0 never covers an unknown kind of a known
  section, so an optional section can still add a required chunk kind.
- An entry of an unknown section type or chunk kind without its bit is
  refused: the error names the type or kind and says it is required.
- Bits 0 to 3 change how an entry is read, so a reader refuses an entry that
  sets one of them it does not know. Bits 4 to 7 do not, and a reader ignores
  them (the same split as the feature flags of the file header).

Every section type and chunk kind of this release is required: its entries
have flags 0.

A skipped chunk is never read, decrypted or handed to a section. Its place is
checked as a known chunk's, and its pages stay in use while its image is
active, so no checkpoint writes over them. A checkpoint writes only the section
types and chunk kinds it knows, so the next one drops the skipped chunks, and
their pages become free once its image is active.

An open also checks that every chunk, a skipped one included, is page-aligned,
lies within the file, and shares no page with another chunk or directory
block.

---

## Section Types

| Value | Name | Version | Description |
|-------|------|---------|-------------|
| 1 | `CATALOG` | 2 | Schema definitions, index definitions and index names, as [records](#catalog-records) |
| 2 | `LPG_STORE` | 3 | Nodes, edges, properties, named graphs |
| 3 | `RDF_STORE` | 3 | RDF triples, named graphs |
| 4 | `COMPACT_STORE` | - | Reserved, never written: the columnar base that `compact()` of 0.5.40 to 0.5.44 wrote to a 0.5.x file, which an open folds into the LPG store |
| 5 | `OVERLAY_DELETIONS` | - | Reserved, never written: the base nodes and edges deleted after that `compact()` (0.5.42 to 0.5.44), as `COMPACT_STORE` |
| 10 | `VECTOR_STORE` | 3 | HNSW topology of each vector index (the embeddings are node properties) |
| 11 | `TEXT_INDEX` | 2 | BM25 document lengths and posting lists |
| 12 | `RDF_RING` | 3 | Term dictionary, wavelet trees and permutations of the RDF ring |
| 20 | `PROPERTY_INDEX` | - | Reserved, never written |

**Type ranges:**

- 1-9: Data sections (authoritative, cannot be rebuilt)
- 10-19: Index sections (derived, can be rebuilt from data)
- 20+: Acceleration structures

**Versions.** The version is the one 0.6.0 writes, in the `section_version`
byte of every directory entry of the section. A reader accepts its section's
version and refuses any other, naming the section and both versions (for a
derived section that means building it from the data, see below). The one
exception is a section stored as one raw chunk (see [Chunks](#chunks)), which
holds 0.5.x bytes and is read by the 0.5.x reader of its section, whatever its
version byte.

**Derived sections** (`VECTOR_STORE`, `TEXT_INDEX` and `RDF_RING`) only
mirror the data. A reader that cannot decode one, because it has another
section version or a stream that does not decode, builds its indexes from the
data instead, with a warning, and the database opens; a derived section whose
chunks cannot be read (a checksum, I/O or decryption failure) fails the open,
as any section's does. So a later 0.6 release can change a derived layout
without making its files unreadable to this one: it writes the new layout
under a new section version, or as a new section type with the optional bit
set, which this release skips (see [unknown entries](#directory-entry-48-bytes))
and builds from the data. Derived data a later 0.6 release adds, such as
persisted statistics, comes as such optional sections as well.

**`PROPERTY_INDEX` is reserved:** no checkpoint writes it. Property indexes are
built from the data when a database opens, from their definitions in the
catalog.

**Sections without data** (indexes, RDF data, overlay deletions) are left out
of the image: if no RDF data exists, there is no `RDF_STORE` chunk. The
`CATALOG` and `LPG_STORE` sections are always written.

---

## Chunks

A section is written as a stream of chunks, which the reader gathers back by
section type. The chunk fields (graph, column, rows, codec) let a section split
its data into many independently addressable chunks. Every section of a
checkpoint writes a metadata chunk (`chunk_kind` 1), first (in `LPG_STORE`
last), and its data in chunks of their own. A file written by 0.5.x holds
every section as one raw chunk (`chunk_kind` 0) of its serialized bytes,
which the section's 0.5.x reader reads.

| Kind | Name | Holds | Written by |
|------|------|-------|------------|
| 0 | `Raw` | A section's bytes, whole | Every section of a 0.5.x file |
| 1 | `Meta` | The section's metadata: a layout byte, the caps it was written with, and its graphs, columns or streams | Every section, once |
| 2 | `Column` | The values of one column over a range of rows | `LPG_STORE` and `RDF_STORE` |
| 3 | `History` | The older versions of one property column's values over a range of rows | `LPG_STORE`, in builds with the `temporal` feature |
| 4 | `Stream` | A piece of a byte stream | The [stream sections](#stream-sections), `CATALOG` included |

A raw or metadata chunk has every other field of its directory entry set to 0.
All chunks of a section carry the same `section_version`, and no two of them
have the same kind, namespace, graph, column and first row (the chunk's
identity): a reader refuses a section that breaks either rule.

### Namespaces

A chunk's `column_id` is a number within its namespace, so ids repeat across
namespaces: a node property and an edge property column can have the same id.
The namespace bytes come in groups with room to grow:

| Byte | Namespace | Columns |
|------|-----------|---------|
| 0 | Section | The section's own numbering: metadata, raw and stream chunks, and every section without node and edge tables |
| 16 | Node structure | The node table's fixed columns |
| 17 | Node properties | The node table's property columns |
| 32 | Edge structure | The edge table's fixed columns |
| 33 | Edge properties | The edge table's property columns |

Reserved and not written yet: 18 node deletes, 19 node label bitmaps, 20 node
versions, 34 edge deletes, 36 edge versions, 48 outgoing adjacency and 49
incoming adjacency; the other bytes are unassigned. Every namespace a file of a
known revision can hold is known to its reader, so an entry with a namespace
byte the reader does not know is refused as damage, whatever its flags.

### Caps

A chunk holds at most **65,536 rows and 1 MiB** (the caps). A row whose values
pass the byte cap gets a chunk of its own, and every piece of a stream but the
last holds exactly 1 MiB. A section's metadata chunk records the caps it was
written with, and its reader checks the chunks against those. The row cap is
also the format's: a reader refuses caps of more than 65,536 rows and a column
chunk of more rows, so what a chunk decodes into stays bounded. The metadata
chunks are not cut: they hold as many names and graphs as the database has.
The catalog's records are a stream, cut wherever the byte cap falls, so a
record can span pieces.

**No 4 GiB limit** ([#392](https://github.com/GrafeoDB/grafeo/issues/392)).
Chunk offsets, chunk lengths and first rows are 64-bit, and the 32-bit counts
and lengths inside a chunk count only within that chunk. So a database, a
section, a graph and a table can pass 4 GiB and 2^32 rows, and a node can
have any number of labels. What stays limited is a single value (lengths and
counts of at most 2^32 - 1, nesting of at most 128 levels, see
[Value Encoding](#value-encoding)), and the RDF ring index, which holds at
most 2^32 - 1 terms and triples (see [Stream Sections](#stream-sections)).
Each schema entry is one catalog record of at most 2 MiB, whose property
types nest at most 32,768 `LIST<...>` levels in all (see
[Catalog Records](#catalog-records)). A checkpoint that meets a value or an
entry past these limits fails, names it and keeps the WAL.

### Column Chunks

A `Column` or `History` chunk holds the values of one column for a range of
rows of a table: only the rows that have a value, in row order, behind a
presence bitmap, so a row without a value costs one bit. Its directory entry
names the graph (`graph_id`), the column (`column_id`), the first row
(`row_start`), the rows from the first to the last it covers (`row_count`, at
most the row cap) and the codec (`codec`). The chunk repeats the codec and the
row count, and a reader refuses a chunk that disagrees with its entry.
Layout, little-endian:

| Field | Size | Meaning |
|-------|------|---------|
| `codec` | `u8` | The codec, equal to the directory entry's |
| `flags` | `u8` | Bit 0: presence bitmap; bit 1: zone map; bit 2: epochs. A reader refuses other bits |
| (reserved) | `u16` | Zero, ignored by readers |
| `row_count` | `u32` | Equal to the directory entry's, at most 65,536 |
| `value_count` | `u32` | At least 1, at most `row_count` |
| presence bitmap | `ceil(row_count / 64)` `u64` words | Only when `value_count` is below `row_count`: bit `r` is set when row `r` has a value; bits past the rows are 0 |
| zone map | see below | Bounds of the values, for the codecs that have them |
| epochs | a `BitPacked` body | One epoch per value, only when one of them is not 0 |
| body | | The values, in the codec's body |

| Codec | Name | Values | Body |
|-------|------|--------|------|
| 1 | `BitPacked` | Non-negative `Int64` | Bits per value `u8`, count `u32`, word count `u32`, `u64` words |
| 2 | `Dict` | `String` | Entry count `u32`, each entry a length `u32` and UTF-8, then a code count `u32` and one `u32` code per value |
| 3 | `Bitmap` | `Bool` | Bit count `u32`, word count `u32`, `u64` words |
| 4 | `Float64` | `Float64` | Count `u32`, then each value's `f64` bits |
| 5 | `RawI64` | `Int64` | Count `u32`, then each `i64` |
| 6 | `Float32Vector` | Vectors of one dimension count, 1 to 65,535 | Dimensions `u16`, component count `u32`, then each component's `f32` bits |
| 7 | `Values` | Any | Count `u32`, then each value in the [value encoding](#value-encoding) |

A writer picks a typed codec (1 to 6) only when every value of the chunk has
its kind, and `Values` otherwise, so every value reads back exactly as it was
written: an `Int64` next to a `Float64` stays an `Int64`.

**Zone maps.** A chunk of the codecs `BitPacked`, `RawI64`, `Float64` and
`Bitmap` carries the minimum and the maximum of its values, two values in the
[value encoding](#value-encoding); NaN is left out (a chunk of NaN only has no
zone map). Every `Dict` chunk carries a string zone map, by the strings' UTF-8
bytes:

| Field | Encoding | Meaning |
|-------|----------|---------|
| minimum | a length `u8`, then that many bytes | The smallest string, cut to at most 16 bytes |
| maximum | a length `u8`, then that many bytes | The largest string; when longer than 16 bytes, its first 16 with the last raised by one |
| flags | `u8` | Bit 0: the minimum was not cut; bit 1: the maximum was not cut. A reader refuses other bits |
| shortest, longest | `u32` each | The lengths in bytes of the shortest and the longest string |

A cut maximum is raised so it stays an upper bound: every string that starts
with its first 16 bytes is below it (UTF-8 holds no `0xFF` byte, so the 16th
byte can always be raised; the result need not be UTF-8). So no string of the
chunk is below the minimum or above the maximum (or at or above it when it was
cut), and each is from the shortest to the longest length. `Float32Vector` and
`Values` chunks have none. A reader refuses a zone map other than the one the
values give, so it can be trusted. See [Zone Maps](zone-maps.md).

**Epochs.** In a build with the `temporal` feature, each property value
carries the epoch it was set at, in its `Column` chunk's epochs. A chunk whose
epochs are all 0 carries none, and a reader refuses epochs that are all 0. The
fixed columns and the `History` chunks never carry epochs: a history value
holds the epochs of its versions itself.

### Value Encoding

The `Values` codec, the zone maps and the history values store each value as
a tag byte followed by its fields, little-endian. Every kind reads back
exactly: floats by their bits (NaN payloads and -0.0 included), times and
zoned datetimes with their offsets, counters with every replica.

| Tag | Kind | Fields |
|-----|------|--------|
| 0 | Null | None |
| 1 | Bool | `u8`, 0 or 1 |
| 2 | Int64 | `i64` |
| 3 | Float64 | `u64` bits |
| 4 | String | Length `u32`, UTF-8 |
| 5 | Bytes | Length `u32`, bytes |
| 6 | Date | `i32` days since 1970-01-01 |
| 7 | Time | `u64` nanoseconds since midnight, `u8` has an offset (0 or 1), `i32` offset seconds (0 without one) |
| 8 | Timestamp | `i64` microseconds since the Unix epoch |
| 9 | ZonedDatetime | `i64` UTC microseconds, `i32` offset seconds |
| 10 | Duration | `i64` months, `i64` days, `i64` nanoseconds |
| 11 | List | Count `u32`, then the values |
| 12 | Map | Count `u32`, then per entry its key (length `u32` and UTF-8) and its value |
| 13 | Vector | Dimensions `u32`, then each component's `f32` bits |
| 14 | Path | Node count `u32`, the nodes, edge count `u32`, the edges |
| 15 | GCounter | Count `u32`, then per entry its replica (length `u32` and UTF-8) and its `u64` count, sorted by replica |
| 16 | OnCounter | The positive entries, then the negative entries, each as in 15 |

A reader checks every length and count against the bytes left before it
allocates, and refuses an unknown tag, a field the kind does not accept (such
as a bool other than 0 or 1, or a time of a full day or more), map keys or
counter replicas that are not strictly increasing, and lists, maps and paths
nested more than 130 levels deep, so each value has exactly one encoding.
Property values nest at most 128 levels: a write of a deeper value fails
(`GRAFEO-V001`), and the two levels above it hold the lists of a history
value.

### Table Sections

`LPG_STORE` and `RDF_STORE` hold their data as tables of `Column` (and
`History`) chunks. A table is cut into row groups of `max_rows` rows: group
`k` holds rows `[k * max_rows, (k + 1) * max_rows)`. Only the groups that hold
a row are written, and the chunks of one group come together. Every order is
defined (ids, names, keys and terms sorted), so the same data gives the same
bytes.

#### `LPG_STORE` (version 3)

Per graph in id order, the section holds its node table and then its edge
table, and it ends with its metadata chunk. Graph 0 is the default graph; a
named graph gets the next graph id when it is created. A row is a node or edge
id. The columns:

| Namespace | Column | Values |
|-----------|--------|--------|
| Node structure | 0 | The node's label ids, ascending, in decimal, joined by `,` (`""` for a node without labels), as a `String` |
| Edge structure | 1 | The source node id, as an `Int64` |
| Edge structure | 2 | The target node id, as an `Int64` |
| Edge structure | 3 | The edge type id, as an `Int64` |
| Node properties or edge properties | The property key's id | A property column of that table |

**Ids are permanent.** A graph's id, and each graph's label, edge type and
property key ids, are given once and never reassigned or reused, so one
checkpoint and the next name the same things by the same ids. Each graph has
three dictionaries: its labels and edge types (an id given when a name is
first used), and its property keys (an id given when a checkpoint first writes
the key's column), one dictionary of keys for the node and the edge table, the
namespace telling their columns apart. A name nothing uses any more keeps its
id; an id that names nothing is a gap. A dropped graph's id is never given
again: a graph created later under its name gets a new one. A load restores
every id as written. Within a row group, a column's
`Column` chunks, and its `History` chunks, each come in row order without
overlapping; a property column has chunks only where it has values; and the
edge columns 1, 2 and 3 come as three chunks of one range in a row. Only the
nodes and edges visible at the checkpoint are written, with their properties.
A property whose value is null does not exist and is not written.

With the `temporal` feature, a `Column` chunk of a property holds its current
values with their epochs, and its older versions go to a `History` chunk
written right before it: a history value is a list of `[epoch, value]` lists,
with `Int64` epochs, ascending. A property removed last (its latest version
null) has no current value: its history holds every version. A `History`
chunk and the `Column` chunk after it cover the same rows, and either can come
alone. Versions of transactions that did not commit are not written. A build
without `temporal` checks the `History` chunks and does not apply them.

The metadata chunk, little-endian:

| Field | Encoding |
|-------|----------|
| layout | `u8`, `1` |
| `max_rows`, `max_bytes` | `u32` each: the caps |
| epoch | `u64`: the store's epoch with `temporal`, 0 without |
| next graph id | `u32`: above every graph id given out |
| graphs | A count `u32`, then per graph (the default graph first, then the named graphs by id, ascending) its id `u32`, its name (a length `u32` and UTF-8, empty for the default graph), next node id `u64`, next edge id `u64`, and its labels, edge types and property keys |
| a dictionary | Its next id `u32` (above every id given out), a count `u32`, then per name its id `u32` and the name (as above), ids ascending |

The layout has no size limit of its own: a reader checks every count and
length against the bytes left, so the chunk holds as many names as the store
has. It comes last because it lists every name the chunks use, read after the
rows: commits wait while a checkpoint writes, but a transaction still open can
create a label, an edge type or a property key meanwhile, and as the
dictionaries only grow, the ones read after the rows hold every id the rows
name. A graph's next node and edge ids are at least one past its last row, so
no id is given out twice across a reopen; the next graph id and each
dictionary's next id do the same for graphs and names. A reader fetches the
metadata chunk first, restores every graph and its dictionaries, then reads
the other chunks in order.

A reader refuses, naming the graph, the column and the rows: a last chunk that
is not the section's only metadata chunk; a chunk of another kind than
`Column` or `History` (`History` only for property columns), of an unknown
graph, outside the node and edge namespaces, of a fixed column its namespace
does not have or a property key its graph does not have, holding no rows or
more than `max_rows`, crossing its row group or reaching past its table's next
id; groups out of order (by graph, then table, then row group), or overlapping
chunks of one column; edge columns that do not come as three chunks of one
range; a property value of a row whose node or edge is not in its group;
labels that are not ascending ids of the graph's labels, negative endpoints,
and edge type ids the graph does not have; epochs on a fixed column or a
`History` chunk; a null property value; history epochs that go back; and in
the metadata, a first graph other than the default graph, graph ids that do
not ascend or reach the next graph id, two named graphs of one name, and per
dictionary ids that do not ascend or reach its next id, or a name listed
twice.

#### `RDF_STORE` (version 3)

The section starts with its metadata chunk, little-endian:

| Field | Encoding |
|-------|----------|
| layout | `u8`, `1` |
| `max_rows`, `max_bytes` | `u32` each: the caps |
| graphs | A count `u32`, then per graph its name (a length `u32` and UTF-8, empty for graph 0) and its number of triples `u64` |

Graph 0 is the default graph, then come the named graphs in name order; a
graph's id is its position, and a named graph with an empty name stays apart
from the default graph. Then, per graph in id order, its triple table: one row
per triple, from row 0, in a fixed order (by subject, predicate and object).
Its columns are 0 (subject), 1 (predicate) and 2 (object), and each value is
the term in N-Triples syntax, as a `String`. Their chunks are therefore `Dict`
chunks, each holding the dictionary of its own terms, and the section has no
term table of its own. The three columns share their chunk boundaries: their
chunks come as three in a row with one range, and every row has a value. A
graph without triples writes no chunk; the metadata lists it, so an open
creates it.

A reader refuses, naming the graph and the rows: a first chunk other than the
metadata chunk, or a second metadata chunk; a chunk of another kind than
`Column`; a graph the metadata does not list, or whose chunks come after those
of a later graph; a range that does not start where the graph's rows before it
end, holds more than `max_rows` rows, crosses a row group or reaches past the
graph's number of triples; a subject chunk not followed by the predicate and
object chunks of its range; epochs, a row without a value, or a value that is
not a string holding an N-Triples term; and a graph whose rows stop short of
its number of triples.

### Stream Sections

The index sections (`VECTOR_STORE`, `TEXT_INDEX`, `RDF_RING`) and `CATALOG`
hold their data as byte streams. Such a section is its metadata chunk, then
the pieces of its streams, stream after stream:

- The metadata chunk (`chunk_kind` 1, every other field 0) is the bincode
  encoding (bincode 2, standard configuration: variable-length little-endian
  integers) of the section's metadata, which starts with a layout byte (`1`;
  a reader refuses another layout) and the byte cap the streams were cut with.
- A stream piece (`chunk_kind` 4) belongs to graph 0; its `column_id` is the
  stream, its `row_start` the piece's byte offset in the stream, and its
  `row_count` and `codec` are 0. Every piece but the last of a stream holds
  exactly the byte cap (1 MiB by default), and an empty stream has no piece.
- A reader refuses a section whose first chunk is not its metadata chunk, a
  chunk of another kind or graph after it, a piece of a stream the metadata
  does not list, a piece that does not start where the pieces before it end,
  a stream that ends before its contents do, and bytes after them. A reader
  that joins a stream into one buffer first checks, from the directory alone,
  that each piece starts after the one before it, by at most the first
  piece's length and the bytes the piece before it is stored in, so the
  buffer it allocates is bounded by the file.

All numbers inside the streams are little-endian.

**`VECTOR_STORE` (version 3).** The metadata lists the HNSW indexes in strictly
increasing key order (`Label:property`), each with its dimensions, its metric
(`0` cosine, `1` Euclidean, `2` dot product, `3` Manhattan), `m` and
`ef_construction`. Stream `i` holds the topology of index `i`:
`[has_entry_point u8][entry_point u64][max_level u32][node_count u64]`, then
`node_count` records `[id u64][level_count u32]`, each level followed by
`[neighbor_count u32]` and that many `[neighbor u64]`, ids strictly
increasing. There is an entry point exactly when there are nodes; it has
`max_level + 1` levels, and every node has 1 to `max_level + 1`. Every
neighbor listed at a level is a node of the stream with that level, and no
node lists itself. Quantized
indexes are not written: an open builds them from the data. A topology is
restored only into an index with the same dimensions and metric.

**`TEXT_INDEX` (version 2).** The metadata lists the index keys in strictly
increasing order. Stream `i` holds the index of key `i`:
`[k1 f64][b f64][total_length u64][doc_count u64][term_count u64]`, then
`doc_count` document lengths `[node u64][length u32]`, then `term_count` posting
lists `[term_length u32][term][count u64]`, each followed by `count` postings
`[node u64][term_frequency u32]`. The stream is canonical: node ids strictly
increase among the document lengths and within each list, terms (UTF-8)
strictly increase, every length, count and term frequency is at least 1, every
posting's node has a document length, the document lengths add up to
`total_length`, and each document's term frequencies add up to its length.
The index's options (k1, b, tokenizer, stop words) are those of its catalog
record: an open restores a stream into an index with the record's tokenizer
and stop words, and builds the index from the data instead when the stream's
k1 or b are not the record's.

**`RDF_RING` (version 3).** The metadata holds the number of triples. Streams
0 to 5 hold the six parts of the ring, each in its packed format: the term
dictionary, the wavelet trees of the subjects, predicates and objects, and the
permutations from SPO to POS and to OSP order. A store without a ring writes no
`RDF_RING` section.

**`CATALOG` (version 2).** The metadata holds the byte cap. Stream 0 holds the
[catalog records](#catalog-records). A catalog without entries is its metadata
chunk alone.

An index section that can be read but does not decode is no error: a warning
is logged and the index is built from the data. A chunk that cannot be read
(a checksum mismatch, an I/O error) fails the open, as in any other section.
A `CATALOG` section that does not decode fails the open.

### Catalog Records

The `CATALOG` section holds the schema as a stream of records, one per entry:
the schemas, the node and edge types, the graph types and the graphs bound to
them, the named constraints, the definitions of every graph's indexes and the
names `CREATE INDEX` gave them, and the stored procedures. Each record is
framed, little-endian:

| Field | Size | Meaning |
|-------|------|---------|
| `kind` | `u8` | What the record holds (below); `0` is never written |
| `flags` | `u8` | Bit 0: required (see below). A reader refuses another of bits 0 to 3 and ignores bits 4 to 7, as for directory entries |
| `length` | `u32` | Length of the payload, at most 2 MiB (2,097,152 bytes) |
| payload | `length` bytes | The kind's record in bincode (bincode 2, standard configuration: variable-length little-endian integers) |

| Kind | Record | Fields, in order |
|------|--------|------------------|
| 1 | Schema | Name |
| 2 | Node type | Name, properties, type constraints, parent types, `KEY` labels |
| 3 | Edge type | Name, properties, type constraints, endpoint pairs, `KEY` labels |
| 4 | Graph type | Name, node types, edge types, whether it is open |
| 5 | Graph type binding | Graph, graph type |
| 6 | Named constraint | Name, label, properties, kind (`0` unique, `1` node key, `2` not null, `3` exists) |
| 7 | Index definition | Graph (none for the default graph), then the index: `0` a property index and its key; `1` a vector index and its label, property, dimensions, metric (`0` cosine, `1` Euclidean, `2` dot product, `3` Manhattan), `m`, `ef_construction` and quantization (`0` none, `1` scalar, `2` binary, `3` product and its number of subvectors); or `2` a text index and its label, property, BM25 `k1` and `b` (`f64` each, little-endian), tokenizer (`0` simple, `1` standard, `2` CJK bigram) and stop words (none, or a list of lowercased strings in increasing order, in place of the tokenizer's own) |
| 8 | Index name | Name, label, property, kind (`0` hash, `1` B-tree, `2` full text) |
| 9 | Procedure | Name, parameters and result columns (each a list of name and type), body |

A property of a node or edge type is its name, its type, whether it may be
null, and its default value (none, or a byte string holding the value in the
[value encoding](#value-encoding), so it reads back exactly). A type
constraint is `0` a primary key or `1` a unique constraint (each a list of
properties), `2` a not-null property, or `3` a check (an optional name and the
expression). An edge type's endpoints are pairs of a source and a target node
type, either of them none for any node type: the cross product of the types
`CONNECTING (...) TO (...)` lists, and no pairs for an edge type that connects
any nodes. A property type is two `u32`: the number of `LIST<...>` levels
around it (at most 128, and at most 32,768 over all the property types of one
record, so 256 properties of the deepest type) and the code of the type inside
them:

| Code | Type | Code | Type |
|------|------|------|------|
| 0 | `STRING` | 8 | `ZONED DATETIME` |
| 1 | `INT64` | 9 | `DURATION` |
| 2 | `FLOAT64` | 10 | `LIST` (of any values) |
| 3 | `BOOLEAN` | 11 | `MAP` |
| 4 | `DATE` | 12 | `BYTES` |
| 5 | `TIME` | 13 | `NODE` |
| 6 | `TIMESTAMP` | 14 | `EDGE` |
| 7 | `LOCAL DATETIME` | 15 | `ANY` |

The records come in the order of their kinds, and the records of one kind in
increasing order of their key, so the same catalog is written to the same
bytes: schemas, types and procedures by name, bindings by graph, constraints
by name, index definitions by graph (the default graph first), then property,
vector and text indexes, each by key or by label and property, and index
names by name, label, property and kind. A binding to a graph type that was
dropped binds nothing and is left out.

**Newer records.** A later version can add kinds. Its writer clears the
required flag of a record an older reader may skip, and an older reader skips
a record of a kind it does not know when the flag is clear and refuses the
catalog when it is set. Every kind of this release is written required. A
kind's fields do not change: a version that needs other fields adds a kind.

A reader refuses, naming the chunk or the record (by its entry, or by its
position and the byte it starts at): a first chunk other than the metadata
chunk, another layout, bytes after the metadata, a chunk other than a piece
of stream 0 of graph 0; kind 0, a flag among bits 0 to 3 other than bit 0, or an unknown
kind with the required flag; a length over 2 MiB (before it reads the
payload, which it reads into a buffer that grows with the bytes present); a
stream that ends inside a record; a payload that does not decode as its
kind's record or has bytes left over, a property type nested more than 128
levels or of an unknown code, property types that nest more than 32,768 levels
in all (each type's levels are counted before they are built); a record that
repeats or comes before the record of its kind before it (only an index name
may repeat); endpoint pairs that are not a cross product; and a record the
catalog refuses, such as a binding to a graph type no record defines. A schema
record also creates the schema's default graph.

A payload of `n` bytes decodes into at most `120 n` bytes of memory plus 1 MiB
for its `LIST<...>` levels, while it decodes and the database converts it,
before the allocator's own overhead: so at most 241 MiB for a payload of
2 MiB. Every list element, and each null of a default value's list, takes at
most 120 bytes per payload byte (three times its size while its list grows
or is copied: a property, 5 bytes at its smallest, takes three times 88). A
level takes no payload byte of its own, as a type's levels are one count, but
32 bytes of memory (a 16-byte box in the record and another in the database's
type), so its memory is bounded per record instead: the cap of 32,768 levels
holds the levels of a record to 1 MiB, as much as a damaged list length may
allocate ahead of its elements in any record. A damaged length inside a
payload allocates no more than that: a string's length counts against a
decode limit of 16 MiB before the string is allocated (which no payload of
2 MiB reaches), a list allocates at most 1 MiB ahead of its elements, and a
default value's counts are checked against the bytes left. A checkpoint that
meets a type past the cap of levels fails, naming it, and `CREATE NODE TYPE`,
`CREATE EDGE TYPE`, `ALTER ... ADD` and inline graph types refuse one.

### Memory

A checkpoint writes one section at a time and holds, besides the database,
about one chunk per column of a table's row group, or one piece of a stream,
plus the 48-byte directory entries of the chunks written so far. An open chunk
holds its rows as decoded values until it is cut, so a column of small values
(such as node labels) fills at the 65,536-row cap rather than the 1 MiB byte
cap: at the default caps a checkpoint of a large store holds about 14 MiB above
the database, bounded by the caps, not by the database. An open reads one chunk
at a time (the three chunks of one range for the columns that come together),
holds what it builds from them, and reads the same 48-byte directory entries.
These sections hold more, in proportion to their data:

| Section | Writing | Reading |
|---------|---------|---------|
| `CATALOG` | A sorted copy of every entry, and the record being written | The record being read (at most 120 bytes per payload byte plus 1 MiB, see [Catalog Records](#catalog-records)), and the named constraints, index definitions and index names until the last record |
| `LPG_STORE` | The sorted ids of the table being written (8 bytes per node or edge) and, without `temporal`, of each of its property columns (8 bytes per value) | |
| `RDF_STORE` | A reference to every triple, sorted (8 bytes per triple) | |
| `VECTOR_STORE` | A reference to every node of an in-memory topology, sorted by id (16 bytes per node) | |
| `TEXT_INDEX` | References to the terms (16 bytes per term), a copy of the document lengths (16 bytes per document) and, for a posting list not held in node order, a sorted copy of it (16 bytes per posting) | Each document's length and the part of it the term frequencies read so far leave uncovered (16 bytes per document) |
| `RDF_RING` | The packed term dictionary, built whole before it is written | Each of the six streams in one buffer, which becomes that part of the ring |

**Locks.** A checkpoint holds a vector or text index's read lock while it
writes that index's stream. Changes to that index (a node's vector or text
inserted, updated or removed) wait until the stream is written, and so do
searches of the index that arrive after a waiting change, as its locks are
fair.

### Encryption

In an encrypted file (flag bit 0), every chunk and every directory block is
encrypted with AES-256-GCM under a key derived for the database's id, with a
random nonce each. The associated data binds a chunk to its section type, chunk
kind, namespace, graph, column and first row
(`grafeo-chunk:{section}:{kind}:{namespace}:{graph}:{column}:{first row}`, in
decimal), and a directory block to its offset, so a chunk cannot pass for
another part of a section (a node property chunk for an edge property chunk of
the same column id, say), nor a directory block for one at another offset. A stored block is the nonce, the ciphertext and the tag: 28
bytes longer than its plaintext. A chunk's `length` and `crc` cover the stored
bytes; a directory block's pointer holds the plaintext length and CRC. The file
header and the database headers are not encrypted. See
[Encryption at Rest](../../getting-started/security.md#encryption-at-rest).

---

## Checkpoint Flow

Every checkpoint writes the whole database as a new image:

```text
Checkpoint:
  1. (Engine) Start a new WAL file
  2. (Engine) Hand every section to the container; commits wait until the
     image is written
  3. Stream each section's chunks, one section at a time, into free pages as
     the section writes them (CRC-32, encrypted if enabled)
  4. Write the directory blocks into free pages
  5. fsync
  6. Write the new database header (iteration + 1, the new root) into the
     inactive slot
  7. fsync
  8. Shorten the file to end at the last page of the new image
  9. (Engine) Mark the WAL: recovery starts at the new WAL file, and earlier
     WAL files are deleted unless an incremental backup still needs them
```

**Free space:** the pages free for a checkpoint are derived from the pages the
active image uses, which an open reads from its directory: every page from
page 3 up to the end of the last used page that the active image does not use.
No free list is stored. Allocation is first fit and appends at the end of the
file when no gap is large enough, so a checkpoint reuses the pages of the image
before the active one. While a checkpoint runs, the file can hold the active
image and the new one.

**Crash safety:** steps 3 to 5 write only pages the active image does not use,
so a crash before the new header is on disk opens the previous image, intact. A
torn header write fails its CRC, so the other slot stays active. A crash after
step 7 opens the new image. If a checkpoint fails at or after its header
write, the new header may or may not have reached the disk: the next checkpoint
spares the pages of both images, and the WAL is kept until a later checkpoint
succeeds.

---

## Spilled Sections

The `.grafeo` file is read with positional reads and is not memory-mapped.
When memory pressure spills a section (or a `ForceDisk` tier override asks for
it), the engine writes the section to a spill file in the spill directory
(`<file>.spill/` by default) and memory-maps that file. An encrypted database
spills nothing, as spill files are not encrypted. See
[Storage Tiers](../memory/storage-tiers.md).

---

## Recovery

```text
Open database:
  1. Read the file header at 0x0000 and validate it (a file written by 0.5.x
     takes the migration path instead, see below)
  2. Read both database header slots, select the active one, and check its
     format revision
  3. Read the directory chain from the active header's root, checking every
     block (and decrypting it in an encrypted file); set optional entries of
     an unknown section type or chunk kind apart, and refuse required ones
  4. Check that every chunk (a skipped one included) lies within the file and
     that no pages overlap
  5. Read each section's chunks, verify their CRC-32 (and decrypt them), and
     load the section into RAM
  6. If a sidecar WAL exists: replay the changes committed since the last
     checkpoint
  7. Database is ready
```

A read-only open takes a shared lock instead and goes through the same steps, the WAL
replay included, but only into memory: it writes nothing, so a torn tail stays until
the next read-write open. With the WAL enabled, that open seals the tail before it
logs anything new; with `wal_enabled` off, it writes the replayed changes to the file
and removes the WAL, the torn tail with it. A build without the `wal` feature cannot
replay: it refuses to open a database whose sidecar WAL holds commits (a non-empty log
file), read-only or not, and leaves the WAL as it is. Likewise a build refuses a file
that holds data only a feature it lacks reads, which it would load without and its next
checkpoint drop: without `triple-store` RDF triples (an `RdfStore` or `RdfRing`
section, or RDF records in the sidecar WAL), without `vector-index` or `text-index` the
definition of such an index in the catalog; a 0.5.x database (also a WAL directory or a
container v1 snapshot) the same way. A compacted base of a 0.5.x file is read by every
build that opens files. It refuses them read-only or
not, and changes nothing: it looks at the sections of the image and at the catalog
before it loads any other section, and at the records of a WAL as it replays them into
memory.

---

## Periodic Checkpoints

When `Config::checkpoint_interval` is set, a background thread periodically
checkpoints the database to the container. This bounds the size of the WAL,
and so the time a reopen spends replaying it.

The timer polls a shutdown flag every 100 ms. On database close, the timer
is stopped before the final checkpoint to prevent races.

---

## File Locking

- **Exclusive lock** on open (read-write mode): prevents concurrent
  writers on the same file.
- **Shared lock** on open (read-only mode): allows multiple concurrent
  readers. A read-write open and read-only opens exclude each other.
- Locks are released on close or drop.
- A new file is built and synced as `<file>.creating` and then renamed, so the
  database path never holds a partial file.
- A migration of a 0.5.x file holds `<file>.migrate.lock` (see below).

---

## Size Estimates

| Component | Size |
|-----------|------|
| Fixed overhead (file header and database headers) | 12 KiB |
| New database | 16 KiB (the headers and one directory block page) |
| Empty database after its first checkpoint | 28 KiB: a page each for the `CATALOG` and `LPG_STORE` metadata chunks and the directory block, and the page of the image before |
| Per chunk | 48 bytes (directory entry) plus padding to a page boundary; 28 bytes more when encrypted |
| Per column chunk | A 12-byte header, a presence bitmap (8 bytes per 64 rows) when some rows have no value, a zone map, and the codec body: for example 1 bit per `Bool`, 8 bytes per `Float64`, the bit width of the largest value per non-negative `Int64`, a 4-byte code per `String` plus each distinct string of the chunk once |
| Per directory block | 32-byte header, up to 1,364 entries (64 KiB) |
| 10K nodes with four properties (a short string, an integer, a float and one of five city names) and 30K edges with one integer property | about 0.6 MB |
| 1M vectors (384-dim, f32) with an HNSW index (default `m` of 16) | about 1.5 GB of embeddings (node properties) plus about 0.2 GB of HNSW topology |
| During a checkpoint | up to the active image plus the new one |

---

## Files Written by 0.5.x

0.5.x wrote container v1 (0.5.21 to 0.5.34: one bincode snapshot after the
headers) and v2 (0.5.35 to 0.5.44: a section directory page at `0x3000` with
32-byte entries, section data from `0x4000`). Both use bincode headers in the
same three 4 KiB pages. Byte 4 tells the formats apart: in a 0.5.x file it is
the varint `0x01` of the bincode format version, in a v3 file the version is
`03 00 00 00`.

0.6 reads these files only to migrate them, or to open them without changes:

- A read-write open migrates the file. Under `<file>.migrate.lock`, it reads
  the old database (its sidecar WAL replayed), writes it as a v3 image to
  `<file>.migrating`, renames the old files to `<file>.pre-0.6`,
  `<file>.pre-0.6.wal`, `<file>.pre-0.6.checkpoint` and `<file>.pre-0.6.spill`
  (the spill directory, which may hold embeddings a database closed while
  spilled has nowhere else), and renames the image to `<file>`. The old files
  are kept byte for byte, and a crash at any step is resolved from the files
  present at the next read-write open.
- A read-only open and `open_in_memory()` read the file in place, with its
  sidecar WAL, and change nothing.

A 0.5.x WAL directory (a directory holding `wal/`, which 0.5.x created by
default for a path without the `.grafeo` extension) is handled the same way:
a read-write open replays its WAL into a v3 image, keeps the whole directory
as `<path>.pre-0.6/` (and its spill directory `<path>.spill/` as
`<path>.pre-0.6.spill/`) and renames the image to `<path>`, so the database
becomes a file at the same path; a read-only open replays it in place. 0.6
creates no WAL directories.

0.7.0 will no longer read v1 and v2 files or WAL directories. See
[Upgrading from 0.5](../../user-guide/persistence/persistent.md#upgrading-from-05)
for what users need to do.

---

## Version History

| Version | Format | Written by | Notes |
|---------|--------|------------|-------|
| v1 | Monolithic blob after the headers | 0.5.21 to 0.5.34 | Single bincode snapshot; read by 0.6 only to migrate |
| v2 | Section directory at `0x3000` | 0.5.35 to 0.5.44 | Independent sections; read by 0.6 only to migrate |
| v3 | Copy-on-write pages, chained directory | 0.6.0 and later | Checksummed chunks of at most 64Ki rows and 1 MiB, 64-bit offsets (no 4 GiB limit), per-chunk encryption, the catalog as typed records; format revision 1 |
