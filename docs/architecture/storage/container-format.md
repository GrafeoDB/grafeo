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
| 4 | 4 | `u32` | (reserved) | Zero |
| 8 | 8 | `u64` | `iteration` | Checkpoint counter, higher = current |
| 16 | 8 | `u64` | `checkpoint_lsn` | WAL position the checkpoint covers (currently always 0) |
| 24 | 8 | `u64` | `epoch` | MVCC epoch at the checkpoint |
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
| 45 | 3 | - | (reserved) | Written as zero, ignored by readers |

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
| 1 | `CATALOG` | 1 | Schema definitions, index definitions, epoch |
| 2 | `LPG_STORE` | 3 | Nodes, edges, properties, named graphs |
| 3 | `RDF_STORE` | 3 | RDF triples, named graphs |
| 4 | `COMPACT_STORE` | 5 | Columnar base of the layered compact store |
| 5 | `OVERLAY_DELETIONS` | 2 | Base entities the compact store's overlay deleted |
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
version and refuses any other, naming the section and both versions. The one
exception is a section stored as one raw chunk (see [Chunks](#chunks)), which
holds 0.5.x bytes and is read by the 0.5.x reader of its section, whatever its
version byte.

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
last), and its data in chunks of their own, except the `CATALOG` section,
which is still written as one raw chunk (`chunk_kind` 0) holding its
serialized bytes. A file written by 0.5.x holds every section as one raw
chunk, which the section's 0.5.x reader reads.

The container has no 4 GiB limit: chunk offsets and lengths are 64-bit. The LPG
section's own block directory currently uses 32-bit offsets, so a single LPG
section is limited to 4 GiB; a checkpoint over that limit fails with an error
that names it and keeps the WAL ([#392](https://github.com/GrafeoDB/grafeo/issues/392)).

### Stream Sections

The index sections (`VECTOR_STORE`, `TEXT_INDEX`, `RDF_RING`), `COMPACT_STORE`
and `OVERLAY_DELETIONS` hold their data as byte streams. Such a section is its
metadata chunk, then the pieces of its streams, stream after stream:

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
  a stream that ends before its contents do, and bytes after them.

All numbers inside the streams are little-endian.

**`VECTOR_STORE` (version 3).** The metadata lists the HNSW indexes in strictly
increasing key order (`Label:property`), each with its dimensions, its metric
(`0` cosine, `1` Euclidean, `2` dot product, `3` Manhattan), `m` and
`ef_construction`. Stream `i` holds the topology of index `i`:
`[has_entry_point u8][entry_point u64][max_level u32][node_count u64]`, then
`node_count` records `[id u64][level_count u32]`, each level followed by
`[neighbor_count u32]` and that many `[neighbor u64]`, ids strictly
increasing. There is an entry point exactly when there are nodes; it has
`max_level + 1` levels, and every node has 1 to `max_level + 1`. Quantized
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
posting's node has a document length, and the document lengths and the term
frequencies each add up to `total_length`.

**`RDF_RING` (version 3).** The metadata holds the number of triples. Streams
0 to 5 hold the six parts of the ring, each in its packed format: the term
dictionary, the wavelet trees of the subjects, predicates and objects, and the
permutations from SPO to POS and to OSP order. A store without a ring writes no
`RDF_RING` section.

**`COMPACT_STORE` (version 5).** Stream 0 holds the compact store's own
encoding (version 4, magic `GCST`): a header, the node tables, the relationship
tables, the id maps, and a CRC-32 of all of it. Its counts and lengths are
32-bit and its names have 16-bit lengths, so a compacted base with a column or
table past those limits (4 GiB, 2^32 rows, or a name over 64 KiB) fails the
checkpoint with an error that names it.

**`OVERLAY_DELETIONS` (version 2).** The metadata holds the number of deleted
base nodes and edges. Stream 0 holds the node ids and stream 1 the edge ids,
each a `u64`, strictly increasing; a stream that holds another number of ids
than the metadata counts is refused.

An index section that can be read but does not decode is no error: a warning
is logged and the index is built from the data. A chunk that cannot be read
(a checksum mismatch, an I/O error) fails the open, as in any other section.

### Memory

A checkpoint writes one section at a time and holds, besides the database,
about one chunk per column of a table's row group, or one piece of a stream,
plus the 48-byte directory entries of the chunks written so far. An open reads
one chunk at a time (the three chunks of one range for the columns that come
together) and holds what it builds from them. These sections hold more, in
proportion to their data:

| Section | Writing | Reading |
|---------|---------|---------|
| `LPG_STORE` | The sorted ids of the table being written (8 bytes per node or edge) and, without `temporal`, of each of its property columns (8 bytes per value) | |
| `RDF_STORE` | A reference to every triple, sorted (8 bytes per triple) | |
| `VECTOR_STORE` | A reference to every node of an in-memory topology, sorted by id | |
| `TEXT_INDEX` | References to the terms (16 bytes per term), a copy of the document lengths (16 bytes per document) and, for a posting list not held in node order, a sorted copy of it (16 bytes per posting) | |
| `RDF_RING` | The packed term dictionary, built whole before it is written | Each of the six streams in one buffer, which becomes that part of the ring |
| `COMPACT_STORE` | Each column and each adjacency, encoded whole before it is written | The stream in one buffer, which becomes the store's column storage |

**Locks.** A checkpoint holds a vector or text index's read lock while it
writes that index's stream. Changes to that index (a node's vector or text
inserted, updated or removed) wait until the stream is written, and so do
searches of the index that arrive after a waiting change, as its locks are
fair.

### Encryption

In an encrypted file (flag bit 0), every chunk and every directory block is
encrypted with AES-256-GCM under a key derived for the database's id, with a
random nonce each. The associated data binds a chunk to its section type, chunk
kind, graph, column and first row, and a directory block to its offset, so a
chunk cannot pass for another part of a section, nor a directory block for one
at another offset. A stored block is the nonce, the ciphertext and the tag: 28
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
  2. Read both database header slots and select the active one
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
file), read-only or not, and leaves the WAL as it is.

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
| After the first checkpoint | A few pages more: the `CATALOG` and `LPG_STORE` sections are always written |
| Per chunk | 48 bytes (directory entry) plus padding to a page boundary; 28 bytes more when encrypted |
| Per directory block | 32-byte header, up to 1,364 entries (64 KiB) |
| Typical 10K-node LPG | about 1 to 5 MB |
| 1M-vector HNSW index (384-dim, f32) | about 1.5 GB |
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
| v3 | Copy-on-write pages, chained directory | 0.6.0 and later | Checksummed chunks, per-chunk encryption |
