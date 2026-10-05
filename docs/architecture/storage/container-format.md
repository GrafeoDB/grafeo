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
| 44 | 4 | `u32` | (reserved) | Zero |

A reader refuses an entry with a section type or chunk kind it does not know.
An open also checks that every chunk is page-aligned, lies within the file, and
shares no page with another chunk or directory block.

---

## Section Types

| Value | Name | Description |
|-------|------|-------------|
| 1 | `CATALOG` | Schema definitions, index metadata, epoch, configuration |
| 2 | `LPG_STORE` | Nodes, edges, properties, named graphs |
| 3 | `RDF_STORE` | RDF triples, named graphs |
| 4 | `COMPACT_STORE` | Columnar base of the layered compact store |
| 5 | `OVERLAY_DELETIONS` | Base entities the compact store's overlay deleted |
| 10 | `VECTOR_STORE` | Embeddings and HNSW topology |
| 11 | `TEXT_INDEX` | BM25 postings and term dictionary |
| 12 | `RDF_RING` | Wavelet trees and dictionary |
| 20 | `PROPERTY_INDEX` | Property hash and btree indexes |

**Type ranges:**

- 1-9: Data sections (authoritative, cannot be rebuilt)
- 10-19: Index sections (derived, can be rebuilt from data)
- 20+: Acceleration structures

**Optional sections without data** (indexes, RDF data, overlay deletions) are
left out of the image: if no RDF data exists, there is no `RDF_STORE` chunk. The
`CATALOG` and `LPG_STORE` sections are always written.

---

## Chunks

A section is written as a stream of chunks, which the reader gathers back by
section type. The chunk fields (graph, column, rows, codec) let a section split
its data into many independently addressable chunks. Currently every section
is written as one raw chunk holding its serialized bytes.

The container has no 4 GiB limit: chunk offsets and lengths are 64-bit. The LPG
section's own block directory currently uses 32-bit offsets, so a single LPG
section is limited to 4 GiB; a checkpoint over that limit fails with an error
that names it and keeps the WAL ([#392](https://github.com/GrafeoDB/grafeo/issues/392)).

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
  2. (Engine) Serialize every section
  3. Write each section's chunks into free pages (CRC-32, encrypted if enabled)
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
     block (and decrypting it in an encrypted file)
  4. Check that every chunk lies within the file and that no pages overlap
  5. Read each section's chunks, verify their CRC-32 (and decrypt them), and
     load the section into RAM
  6. If a sidecar WAL exists: replay the changes committed since the last
     checkpoint
  7. Database is ready
```

A read-only open takes a shared lock instead and goes through the same steps, the WAL
replay included, but only into memory: it writes nothing, so a torn tail stays for
the next read-write open to seal.

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
  `<file>.pre-0.6.wal` and `<file>.pre-0.6.checkpoint`, and renames the image
  to `<file>`. The old files are kept byte for byte, and a crash at any step is
  resolved from the files present at the next read-write open.
- A read-only open and `open_in_memory()` read the file in place, with its
  sidecar WAL, and change nothing.

A 0.5.x WAL directory (a directory holding `wal/`, which 0.5.x created by
default for a path without the `.grafeo` extension) is handled the same way:
a read-write open replays its WAL into a v3 image, keeps the whole directory
as `<path>.pre-0.6/` and renames the image to `<path>`, so the database
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
