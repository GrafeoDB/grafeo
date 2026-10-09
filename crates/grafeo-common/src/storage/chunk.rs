//! Chunk caps, chunk identities and byte streams over chunks.
//!
//! A section streams out as chunks of at most [`ChunkCaps`] rows and bytes.
//! Within a section no two chunks share an identity
//! ([`ChunkMeta::identity`]), which [`ChunkIdentities`] checks. A section whose
//! encoding is a byte stream writes it through [`ChunkStreamWriter`] as
//! [`ChunkKind::Stream`] pieces and reads it back through
//! [`ChunkStreamReader`] or [`read_stream`].

use std::io;

use bytes::Bytes;

use crate::storage::section::{ChunkKind, ChunkMeta, SectionSink, SectionSource, SectionType};
use crate::utils::error::{Error, Result};
use crate::utils::hash::FxHashSet;

/// The most rows and bytes one chunk holds.
///
/// A value larger than `max_bytes` gets a chunk of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkCaps {
    /// Rows per table chunk, at most.
    pub max_rows: u32,
    /// Bytes per chunk, at most (a single larger value excepted).
    pub max_bytes: u32,
}

impl ChunkCaps {
    /// 65,536 rows and 1 MiB per chunk.
    ///
    /// The row cap is also the format's: a reader refuses a chunk of more
    /// rows (a chunk decodes into one value per row, so its row count bounds
    /// what it decodes into) and caps of more rows, and
    /// [`validate`](Self::validate) refuses them to every writer.
    pub const DEFAULT: Self = Self {
        max_rows: 65_536,
        max_bytes: 1 << 20,
    };

    /// [`DEFAULT`](Self::DEFAULT), or the caps
    /// [`with_chunk_caps`](crate::testing::chunk_caps::with_chunk_caps) set
    /// on this thread.
    #[must_use]
    pub fn current() -> Self {
        crate::testing::chunk_caps::overridden().unwrap_or(Self::DEFAULT)
    }

    /// Refuses 0 rows or 0 bytes, and more rows than the format's cap (the
    /// row cap of [`DEFAULT`](Self::DEFAULT)).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidValue`] naming both caps when either is 0,
    /// and naming the row caps when `max_rows` is above the format's.
    pub fn validate(self) -> Result<()> {
        if self.max_rows == 0 || self.max_bytes == 0 {
            return Err(Error::InvalidValue(format!(
                "chunk caps must allow at least one row and one byte per chunk, got max_rows {} \
                 and max_bytes {}",
                self.max_rows, self.max_bytes
            )));
        }
        if self.max_rows > Self::DEFAULT.max_rows {
            return Err(Error::InvalidValue(format!(
                "chunk caps allow at most {} rows per chunk, the format's row cap, got max_rows {}",
                Self::DEFAULT.max_rows,
                self.max_rows
            )));
        }
        Ok(())
    }
}

/// The identities a section has used so far (see [`ChunkMeta::identity`]).
#[derive(Debug, Default)]
pub struct ChunkIdentities {
    seen: FxHashSet<(u8, u8, u32, u32, u64)>,
}

impl ChunkIdentities {
    /// Records the identity of `meta` for `section_type`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] "section {type:?}: two chunks of kind
    /// {kind:?} in namespace {namespace:?} for graph {g}, column {c}, first
    /// row {r}" when the identity was recorded before.
    pub fn insert(&mut self, section_type: SectionType, meta: &ChunkMeta) -> Result<()> {
        if self.seen.insert(meta.identity()) {
            Ok(())
        } else {
            Err(Error::Serialization(format!(
                "section {section_type:?}: two chunks of kind {:?} in namespace {:?} for graph \
                 {}, column {}, first row {}",
                meta.kind, meta.namespace, meta.graph_id, meta.column_id, meta.row_start
            )))
        }
    }
}

/// Cuts a byte stream into [`ChunkKind::Stream`] chunks of at most
/// `caps.max_bytes`.
///
/// Every piece but the last holds exactly `max_bytes`, and an empty stream
/// writes no piece. A full buffer becomes a piece at once, so the writer
/// holds at most one piece. [`finish`](Self::finish) writes the last piece: a
/// writer dropped without it loses that piece, so every path that succeeds
/// must call it. A path that fails may drop the writer: its stream is not
/// complete anyway, and the error is the caller's to return.
#[must_use = "call finish to write the last piece"]
pub struct ChunkStreamWriter<'s> {
    sink: &'s mut dyn SectionSink,
    graph_id: u32,
    stream: u32,
    max_bytes: usize,
    buffer: Vec<u8>,
    offset: u64,
    /// The first error, kept until [`finish`](Self::finish) returns it.
    error: Option<Error>,
}

impl<'s> ChunkStreamWriter<'s> {
    /// A writer of stream `stream` of graph `graph_id` into `sink`.
    ///
    /// Caps that [`ChunkCaps::validate`] refuses make every write fail, and
    /// [`finish`](Self::finish) return the refusal.
    pub fn new(sink: &'s mut dyn SectionSink, graph_id: u32, stream: u32, caps: ChunkCaps) -> Self {
        let max_bytes = caps.validate().and_then(|()| {
            usize::try_from(caps.max_bytes).map_err(|_| {
                Error::InvalidValue(format!(
                    "chunk cap max_bytes {} does not fit this platform",
                    caps.max_bytes
                ))
            })
        });
        let (max_bytes, error) = match max_bytes {
            Ok(max_bytes) => (max_bytes, None),
            Err(error) => (1, Some(error)),
        };
        Self {
            sink,
            graph_id,
            stream,
            max_bytes,
            buffer: Vec::new(),
            offset: 0,
            error,
        }
    }

    /// Writes the last piece and returns the stream's length; returns the
    /// first sink error, if any.
    ///
    /// # Errors
    ///
    /// Returns the first error a write met (a sink error, or caps of zero),
    /// or the sink's error for the last piece.
    pub fn finish(mut self) -> Result<u64> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        if !self.buffer.is_empty() {
            let piece = std::mem::take(&mut self.buffer);
            self.emit(&piece)?;
        }
        Ok(self.offset)
    }

    /// Hands `piece` to the sink at the current offset and advances.
    fn emit(&mut self, piece: &[u8]) -> Result<()> {
        let meta = ChunkMeta::stream_piece(self.graph_id, self.stream, self.offset);
        self.sink.write_chunk(meta, piece)?;
        self.offset += piece.len() as u64;
        Ok(())
    }

    /// Grows the buffer for `additional` bytes, doubling as `Vec` does but
    /// never past one piece.
    fn reserve(&mut self, additional: usize) {
        let needed = self.buffer.len() + additional;
        if needed > self.buffer.capacity() {
            let target = needed
                .max(self.buffer.capacity().saturating_mul(2))
                .min(self.max_bytes);
            self.buffer.reserve_exact(target - self.buffer.len());
        }
    }

    /// Appends `data`, handing every full piece to the sink.
    fn append(&mut self, data: &[u8]) -> Result<()> {
        let mut rest = data;
        while !rest.is_empty() {
            if self.buffer.is_empty() && rest.len() >= self.max_bytes {
                // A whole piece straight from the caller's bytes, without a copy.
                let (piece, after) = rest.split_at(self.max_bytes);
                self.emit(piece)?;
                rest = after;
                continue;
            }
            let take = (self.max_bytes - self.buffer.len()).min(rest.len());
            let (head, after) = rest.split_at(take);
            self.reserve(take);
            self.buffer.extend_from_slice(head);
            rest = after;
            if self.buffer.len() == self.max_bytes {
                let mut piece = std::mem::take(&mut self.buffer);
                let written = self.emit(&piece);
                piece.clear();
                self.buffer = piece;
                written?;
            }
        }
        Ok(())
    }

    /// The kept error as an I/O error.
    fn kept_error(&self) -> Option<io::Error> {
        self.error
            .as_ref()
            .map(|error| io::Error::other(error.to_string()))
    }
}

impl io::Write for ChunkStreamWriter<'_> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if let Some(error) = self.kept_error() {
            return Err(error);
        }
        match self.append(data) {
            Ok(()) => Ok(data.len()),
            Err(error) => {
                let io_error = io::Error::other(error.to_string());
                self.error = Some(error);
                Err(io_error)
            }
        }
    }

    /// Pieces are cut by size alone, so there is nothing to flush before
    /// [`finish`](ChunkStreamWriter::finish).
    fn flush(&mut self) -> io::Result<()> {
        self.kept_error().map_or(Ok(()), Err)
    }
}

/// Reads the [`ChunkKind::Stream`] chunks of one stream in order, fetching
/// one at a time.
///
/// As each piece is fetched, its offset must be the number of bytes read
/// before it, else the read fails naming both. A piece with a codec is
/// refused: stream pieces are stored as they were written.
pub struct ChunkStreamReader<'s> {
    source: &'s dyn SectionSource,
    graph_id: u32,
    stream: u32,
    /// Positions of the stream's pieces in `source.chunks()`, in order.
    indices: Vec<usize>,
    /// The next entry of `indices` to fetch.
    next: usize,
    /// The piece being read.
    current: Bytes,
    /// Bytes of `current` read so far.
    position: usize,
    /// Bytes of the pieces fetched so far: where the next piece must start.
    offset: u64,
}

impl<'s> ChunkStreamReader<'s> {
    /// Collects the pieces of `stream` of `graph_id`; fetches nothing yet.
    #[must_use]
    pub fn new(source: &'s dyn SectionSource, graph_id: u32, stream: u32) -> Self {
        let indices = source
            .chunks()
            .iter()
            .enumerate()
            .filter(|(_, meta)| {
                meta.kind == ChunkKind::Stream
                    && meta.graph_id == graph_id
                    && meta.column_id == stream
            })
            .map(|(index, _)| index)
            .collect();
        Self {
            source,
            graph_id,
            stream,
            indices,
            next: 0,
            current: Bytes::new(),
            position: 0,
            offset: 0,
        }
    }

    /// Whether every byte of the stream has been read.
    #[must_use]
    pub fn is_at_end(&self) -> bool {
        self.position == self.current.len() && self.next == self.indices.len()
    }

    /// The chunk of the piece at `position` of the stream's pieces, refused
    /// when it has a codec.
    fn piece_meta(&self, position: usize) -> Result<(usize, ChunkMeta)> {
        let index = *self.indices.get(position).ok_or_else(|| {
            Error::Internal(format!(
                "stream {} of graph {}: piece {position} is out of range",
                self.stream, self.graph_id
            ))
        })?;
        let meta = self.source.chunks().get(index).copied().ok_or_else(|| {
            Error::Internal(format!(
                "stream {} of graph {}: chunk {index} is out of range",
                self.stream, self.graph_id
            ))
        })?;
        if meta.codec != 0 {
            return Err(Error::Serialization(format!(
                "stream {} of graph {}: the piece at offset {} has codec {}, but stream pieces \
                 have none",
                self.stream, self.graph_id, meta.row_start, meta.codec
            )));
        }
        Ok((index, meta))
    }

    /// Refuses a piece that does not start where the pieces read so far end.
    fn check_offset(&self, meta: &ChunkMeta) -> Result<()> {
        if meta.row_start != self.offset {
            return Err(Error::Serialization(format!(
                "stream {} of graph {}: a piece starts at offset {}, expected {} (the length \
                 of the pieces before it)",
                self.stream, self.graph_id, meta.row_start, self.offset
            )));
        }
        Ok(())
    }

    /// Checks from the directory alone, without fetching, that the stream's
    /// pieces form one run: after the first, which holds `first_length`
    /// bytes, each piece starts after the piece before it (every piece but
    /// the last holds at least one byte), by at most `first_length` bytes (a
    /// writer cuts every piece but the last at one length) and by at most
    /// the bytes the piece before it is stored in
    /// ([`SectionSource::stored_length`]). So where the last piece starts is
    /// bounded by what the pieces before it are stored in, which a file's
    /// container checked lies inside the file.
    fn check_run(&self, first_length: u64) -> Result<()> {
        let mut start = 0u64;
        // The most bytes the piece at `start` can hold.
        let mut holds = first_length;
        for position in 1..self.indices.len() {
            let (index, meta) = self.piece_meta(position)?;
            if meta.row_start <= start {
                return Err(Error::Serialization(format!(
                    "stream {} of graph {}: a piece starts at offset {}, expected more than \
                     {start} (every piece but the last holds at least one byte)",
                    self.stream, self.graph_id, meta.row_start
                )));
            }
            let reach = start.saturating_add(holds);
            if meta.row_start > reach {
                return Err(Error::Serialization(format!(
                    "stream {} of graph {}: a piece starts at offset {}, expected {reach} at \
                     most (the pieces before it hold at most {reach} bytes)",
                    self.stream, self.graph_id, meta.row_start
                )));
            }
            start = meta.row_start;
            holds = first_length.min(self.source.stored_length(index)?);
        }
        Ok(())
    }

    /// Fetches the next piece, or `None` after the last one.
    fn next_piece(&mut self) -> Result<Option<Bytes>> {
        if self.next == self.indices.len() {
            return Ok(None);
        }
        let (index, meta) = self.piece_meta(self.next)?;
        self.check_offset(&meta)?;
        let piece = self.source.fetch(index)?;
        self.next += 1;
        self.offset += piece.len() as u64;
        Ok(Some(piece))
    }
}

impl io::Read for ChunkStreamReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        while self.position == self.current.len() {
            match self.next_piece().map_err(io::Error::other)? {
                Some(piece) => {
                    self.current = piece;
                    self.position = 0;
                }
                None => {
                    // Let go of the last piece once the stream is read.
                    self.current = Bytes::new();
                    self.position = 0;
                    return Ok(0);
                }
            }
        }
        let available = &self.current[self.position..];
        let count = available.len().min(buffer.len());
        buffer[..count].copy_from_slice(&available[..count]);
        self.position += count;
        Ok(count)
    }
}

/// One stream as one buffer, for sections whose in-memory form is their
/// encoded bytes.
///
/// A stream of one piece is that piece's bytes, without a copy; a stream
/// without pieces is empty. A stream of several pieces is joined into a
/// buffer of exactly its length, which the section may keep for as long as
/// it lives.
///
/// # Errors
///
/// Returns an error when a piece does not start where the pieces before it
/// end or has a codec, when the directory does not show the pieces as one
/// run (each after the piece before it, by at most the first piece's length
/// and the bytes the piece before it is stored in), or when a piece cannot
/// be fetched.
pub fn read_stream(source: &dyn SectionSource, graph_id: u32, stream: u32) -> Result<Bytes> {
    let mut reader = ChunkStreamReader::new(source, graph_id, stream);
    let Some(first) = reader.next_piece()? else {
        return Ok(Bytes::new());
    };
    if reader.next == reader.indices.len() {
        return Ok(first);
    }
    Ok(Bytes::from(join_pieces(&mut reader, first)?))
}

/// Joins `first` and the pieces left in `reader` (at least one) into one
/// buffer of exactly the stream's length, allocated once.
///
/// The length is the last piece's offset plus its bytes, so the last piece is
/// fetched right after the first. Its offset is trusted only once the
/// directory shows the pieces as one run ([`ChunkStreamReader::check_run`]),
/// which bounds it by the bytes the pieces before it are stored in: a crafted
/// offset must not request a huge allocation. The other pieces' offsets are
/// checked again as each is fetched, and the last piece must start where
/// they end.
fn join_pieces(reader: &mut ChunkStreamReader<'_>, first: Bytes) -> Result<Vec<u8>> {
    reader.check_run(first.len() as u64)?;
    let last_position = reader.indices.len().saturating_sub(1);
    let (last_index, last_meta) = reader.piece_meta(last_position)?;
    let last = reader.source.fetch(last_index)?;
    let length = last_meta
        .row_start
        .checked_add(last.len() as u64)
        .and_then(|length| usize::try_from(length).ok())
        .ok_or_else(|| {
            Error::Serialization(format!(
                "stream {} of graph {}: {} bytes do not fit this platform",
                reader.stream, reader.graph_id, last_meta.row_start
            ))
        })?;
    let mut joined = Vec::with_capacity(length);
    joined.extend_from_slice(&first);
    drop(first);
    while reader.next < last_position {
        let Some(piece) = reader.next_piece()? else {
            break;
        };
        joined.extend_from_slice(&piece);
    }
    reader.check_offset(&last_meta)?;
    reader.next = reader.indices.len();
    reader.offset += last.len() as u64;
    joined.extend_from_slice(&last);
    Ok(joined)
}

/// Names `section_type` in an error met while reading one of its streams
/// through the [`io::Read`] of a [`ChunkStreamReader`].
///
/// An error of the reader itself (a misplaced piece, a piece with a codec, a
/// failed fetch) comes back as its own variant, with the section named
/// instead of wrapped in [`Error::Io`]. Any other I/O error, such as a stream
/// that ends before the section's encoding does, is corrupt section data:
/// [`Error::Serialization`].
#[must_use]
pub fn stream_error(section_type: SectionType, error: io::Error) -> Error {
    let kind = error.kind();
    let text = error.to_string();
    match error.into_inner().map(|inner| inner.downcast::<Error>()) {
        Some(Ok(inner)) => match *inner {
            Error::Serialization(message) => {
                Error::Serialization(format!("section {section_type:?}: {message}"))
            }
            Error::Internal(message) => {
                Error::Internal(format!("section {section_type:?}: {message}"))
            }
            Error::Io(inner) => Error::Io(io::Error::new(
                inner.kind(),
                format!("section {section_type:?}: {inner}"),
            )),
            other => Error::Serialization(format!("section {section_type:?}: {other}")),
        },
        _ if kind == io::ErrorKind::UnexpectedEof => Error::Serialization(format!(
            "section {section_type:?}: a stream ends before the section's encoding does ({text})"
        )),
        _ => Error::Serialization(format!("section {section_type:?}: {text}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::image::{ImageSource, MemoryImage};
    use crate::storage::section::{ChunkKind, ChunkMeta, SectionSink, SectionType};
    use crate::testing::chunk_caps::with_chunk_caps;

    #[test]
    fn a_stream_is_cut_into_pieces_of_at_most_the_cap() {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::TextIndex, 2).unwrap();
        let caps = ChunkCaps {
            max_rows: 3,
            max_bytes: 4,
        };
        let mut writer = ChunkStreamWriter::new(&mut image, 0, 19, caps);
        std::io::Write::write_all(&mut writer, b"Amster").unwrap();
        std::io::Write::write_all(&mut writer, b"dam").unwrap();
        assert_eq!(writer.finish().unwrap(), 9);
        let section = image.section_source(SectionType::TextIndex).unwrap();
        let pieces: Vec<(ChunkKind, u32, u64)> = section
            .chunks()
            .iter()
            .map(|m| (m.kind, m.column_id, m.row_start))
            .collect();
        assert_eq!(
            pieces,
            [
                (ChunkKind::Stream, 19, 0),
                (ChunkKind::Stream, 19, 4),
                (ChunkKind::Stream, 19, 8)
            ]
        );
        assert_eq!(section.fetch(2).unwrap().len(), 1);
        assert_eq!(&read_stream(&*section, 0, 19).unwrap()[..], b"Amsterdam");
        // A stream of exactly two caps has no empty third piece.
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::TextIndex, 2).unwrap();
        let mut writer = ChunkStreamWriter::new(&mut image, 0, 3, caps);
        std::io::Write::write_all(&mut writer, b"BerlinPa").unwrap();
        writer.finish().unwrap();
        assert_eq!(image.chunk_count(), 2);
        let section = image.section_source(SectionType::TextIndex).unwrap();
        assert_eq!(&read_stream(&*section, 0, 3).unwrap()[..], b"BerlinPa");
    }

    #[test]
    fn an_empty_stream_writes_no_chunk() {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::TextIndex, 2).unwrap();
        assert_eq!(
            ChunkStreamWriter::new(&mut image, 0, 0, ChunkCaps::DEFAULT)
                .finish()
                .unwrap(),
            0
        );
        assert_eq!(image.chunk_count(), 0);
        assert!(
            image.section_source(SectionType::TextIndex).is_none(),
            "a section without chunks is absent, as in a file"
        );
        // A section with chunks of other streams reads stream 0 as empty.
        image
            .write_chunk(ChunkMeta::stream_piece(0, 3, 0), b"Mia")
            .unwrap();
        let section = image.section_source(SectionType::TextIndex).unwrap();
        assert!(read_stream(&*section, 0, 0).unwrap().is_empty());
        assert_eq!(
            &read_stream(&*section, 0, 3).unwrap()[..],
            b"Mia",
            "a stream of one piece is that piece"
        );
    }

    #[test]
    fn a_stream_with_a_missing_piece_is_refused() {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::TextIndex, 2).unwrap();
        image
            .write_chunk(ChunkMeta::stream_piece(0, 0, 0), b"Pari")
            .unwrap();
        image
            .write_chunk(ChunkMeta::stream_piece(0, 0, 8), b"s")
            .unwrap();
        let section = image.section_source(SectionType::TextIndex).unwrap();
        let error = read_stream(&*section, 0, 0).unwrap_err().to_string();
        assert!(
            error.contains("offset 8") && error.contains("expected 4"),
            "{error}"
        );
    }

    #[test]
    fn the_caps_default_to_64ki_rows_and_1_mib_and_tests_can_shrink_them() {
        assert_eq!(
            ChunkCaps::DEFAULT,
            ChunkCaps {
                max_rows: 65_536,
                max_bytes: 1 << 20
            }
        );
        assert_eq!(ChunkCaps::current(), ChunkCaps::DEFAULT);
        let tiny = ChunkCaps {
            max_rows: 3,
            max_bytes: 1024,
        };
        with_chunk_caps(tiny, || {
            assert_eq!(ChunkCaps::current(), tiny);
            let other = std::thread::spawn(ChunkCaps::current).join().unwrap();
            assert_eq!(
                other,
                ChunkCaps::DEFAULT,
                "the override is this thread's only"
            );
        });
        assert_eq!(ChunkCaps::current(), ChunkCaps::DEFAULT, "restored");
        let panicked =
            std::panic::catch_unwind(|| with_chunk_caps(tiny, || -> u8 { panic!("Mia") }));
        assert!(panicked.is_err());
        assert_eq!(
            ChunkCaps::current(),
            ChunkCaps::DEFAULT,
            "restored after a panic"
        );
    }

    #[test]
    fn caps_of_zero_are_refused() {
        assert!(
            ChunkCaps {
                max_rows: 0,
                max_bytes: 88
            }
            .validate()
            .is_err()
        );
        assert!(
            ChunkCaps {
                max_rows: 3,
                max_bytes: 0
            }
            .validate()
            .is_err()
        );
        ChunkCaps {
            max_rows: 1,
            max_bytes: 1,
        }
        .validate()
        .unwrap();
    }

    /// The default row cap is the format's: a reader refuses more rows per
    /// chunk, so no writer may use more.
    #[test]
    fn caps_above_the_default_row_cap_are_refused() {
        let error = ChunkCaps {
            max_rows: ChunkCaps::DEFAULT.max_rows + 1,
            max_bytes: 88,
        }
        .validate()
        .unwrap_err();
        assert!(
            matches!(&error, Error::InvalidValue(message)
                if message.contains("max_rows 65537") && message.contains("65536")),
            "{error:?}"
        );
        ChunkCaps {
            max_rows: ChunkCaps::DEFAULT.max_rows,
            max_bytes: u32::MAX,
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn nested_overrides_restore_the_caps_around_them() {
        let outer = ChunkCaps {
            max_rows: 19,
            max_bytes: 88,
        };
        let inner = ChunkCaps {
            max_rows: 3,
            max_bytes: 19,
        };
        with_chunk_caps(outer, || {
            with_chunk_caps(inner, || assert_eq!(ChunkCaps::current(), inner));
            assert_eq!(ChunkCaps::current(), outer, "the outer override is back");
        });
        assert_eq!(ChunkCaps::current(), ChunkCaps::DEFAULT);
    }

    #[test]
    fn a_writer_with_caps_of_zero_refuses_to_write() {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::TextIndex, 2).unwrap();
        let caps = ChunkCaps {
            max_rows: 3,
            max_bytes: 0,
        };
        let mut writer = ChunkStreamWriter::new(&mut image, 0, 0, caps);
        assert!(std::io::Write::write_all(&mut writer, b"Prague").is_err());
        let error = writer.finish().unwrap_err().to_string();
        assert!(error.contains("max_bytes"), "{error}");
        assert_eq!(
            image.chunk_count(),
            0,
            "no piece, not an endless run of empty ones"
        );
    }

    #[test]
    fn a_sink_error_is_kept_and_returned_by_finish() {
        // No section was begun, so the image refuses every chunk.
        let mut image = MemoryImage::new();
        let caps = ChunkCaps {
            max_rows: 3,
            max_bytes: 4,
        };
        let mut writer = ChunkStreamWriter::new(&mut image, 0, 0, caps);
        let write_error = std::io::Write::write_all(&mut writer, b"Vincent").unwrap_err();
        assert!(
            write_error.to_string().contains("no section"),
            "{write_error}"
        );
        assert!(
            std::io::Write::write_all(&mut writer, b"x").is_err(),
            "the error stays"
        );
        let error = writer.finish().unwrap_err().to_string();
        assert!(error.contains("no section"), "{error}");
    }

    #[test]
    fn the_reader_serves_a_stream_through_read_in_any_buffer_size() {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::TextIndex, 2).unwrap();
        let caps = ChunkCaps {
            max_rows: 3,
            max_bytes: 4,
        };
        let mut writer = ChunkStreamWriter::new(&mut image, 3, 19, caps);
        std::io::Write::write_all(&mut writer, b"Amsterdam").unwrap();
        writer.finish().unwrap();
        // A piece of another graph and of another stream, between and after.
        image
            .write_chunk(ChunkMeta::stream_piece(0, 19, 0), b"Berlin")
            .unwrap();
        image
            .write_chunk(ChunkMeta::stream_piece(3, 88, 0), b"Prague")
            .unwrap();
        let section = image.section_source(SectionType::TextIndex).unwrap();
        for size in [1, 3, 4, 5, 64] {
            let mut reader = ChunkStreamReader::new(&*section, 3, 19);
            assert!(!reader.is_at_end(), "size {size}: nothing read yet");
            let mut read = Vec::new();
            let mut buffer = vec![0u8; size];
            loop {
                let count = std::io::Read::read(&mut reader, &mut buffer).unwrap();
                if count == 0 {
                    break;
                }
                read.extend_from_slice(&buffer[..count]);
            }
            assert_eq!(read, b"Amsterdam", "size {size}");
            assert!(reader.is_at_end(), "size {size}");
        }
    }

    #[test]
    fn the_reader_refuses_a_gap_or_an_overlap_through_read_too() {
        for (offset, case) in [(8, "gap"), (3, "overlap")] {
            let mut image = MemoryImage::new();
            image.begin_section(SectionType::TextIndex, 2).unwrap();
            image
                .write_chunk(ChunkMeta::stream_piece(0, 0, 0), b"Jules")
                .unwrap();
            image
                .write_chunk(ChunkMeta::stream_piece(0, 0, offset), b"Mia")
                .unwrap();
            let section = image.section_source(SectionType::TextIndex).unwrap();
            let mut reader = ChunkStreamReader::new(&*section, 0, 0);
            let mut read = Vec::new();
            let error = std::io::Read::read_to_end(&mut reader, &mut read)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(&format!("offset {offset}")) && error.contains("expected 5"),
                "{case}: {error}"
            );
            assert_eq!(read, b"Jules", "{case}: the bytes before it were served");
        }
    }

    #[test]
    fn the_reader_is_at_end_only_once_the_last_piece_is_read_through() {
        // Ten bytes in pieces of 4, 4 and 2.
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::TextIndex, 2).unwrap();
        let caps = ChunkCaps {
            max_rows: 3,
            max_bytes: 4,
        };
        let mut writer = ChunkStreamWriter::new(&mut image, 0, 0, caps);
        std::io::Write::write_all(&mut writer, b"AlixGusMia").unwrap();
        assert_eq!(writer.finish().unwrap(), 10);
        let section = image.section_source(SectionType::TextIndex).unwrap();
        let mut reader = ChunkStreamReader::new(&*section, 0, 0);
        let mut head = [0u8; 9];
        std::io::Read::read_exact(&mut reader, &mut head).unwrap();
        assert_eq!(&head, b"AlixGusMi");
        assert!(
            !reader.is_at_end(),
            "the last piece is fetched, but one byte of it is unread"
        );
        let mut last = [0u8; 1];
        std::io::Read::read_exact(&mut reader, &mut last).unwrap();
        assert_eq!(&last, b"a");
        assert!(reader.is_at_end(), "without another read");
    }

    #[test]
    fn the_reader_leaves_out_a_metadata_chunk_next_to_its_stream() {
        // The layout of every stream section: a metadata chunk (graph 0, column 0,
        // first row 0), then stream 0 of graph 0.
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::TextIndex, 2).unwrap();
        image.write_chunk(ChunkMeta::meta(), b"Alix").unwrap();
        let caps = ChunkCaps {
            max_rows: 3,
            max_bytes: 4,
        };
        let mut writer = ChunkStreamWriter::new(&mut image, 0, 0, caps);
        std::io::Write::write_all(&mut writer, b"Prague").unwrap();
        writer.finish().unwrap();
        let section = image.section_source(SectionType::TextIndex).unwrap();
        assert_eq!(&read_stream(&*section, 0, 0).unwrap()[..], b"Prague");
        let mut reader = ChunkStreamReader::new(&*section, 0, 0);
        let mut read = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut read).unwrap();
        assert_eq!(read, b"Prague");
    }

    #[test]
    fn the_writer_never_holds_more_than_one_piece() {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::TextIndex, 2).unwrap();
        let caps = ChunkCaps {
            max_rows: 3,
            max_bytes: 88,
        };
        let mut writer = ChunkStreamWriter::new(&mut image, 0, 0, caps);
        let mut written = 0usize;
        for size in [1, 3, 19, 30, 88, 3, 200, 19, 87, 1] {
            std::io::Write::write_all(&mut writer, &vec![b'x'; size]).unwrap();
            written += size;
            assert!(
                writer.buffer.capacity() <= 88,
                "after {written} bytes the buffer has a capacity of {} bytes",
                writer.buffer.capacity()
            );
            assert_eq!(writer.buffer.len(), written % 88, "after {written} bytes");
        }
        assert_eq!(writer.finish().unwrap(), written as u64);
    }

    #[test]
    fn an_encoder_error_between_writes_comes_back_as_an_error() {
        /// Writes some bytes of its stream, then fails a conversion of its own
        /// and returns early, before `finish`.
        fn encode(sink: &mut dyn SectionSink) -> Result<u64> {
            let mut writer = ChunkStreamWriter::new(sink, 0, 0, ChunkCaps::DEFAULT);
            std::io::Write::write_all(&mut writer, b"Alix")?;
            let stops = 300u32;
            let stops = u8::try_from(stops).map_err(|error| {
                Error::InvalidValue(format!("{stops} stops do not fit a byte: {error}"))
            })?;
            std::io::Write::write_all(&mut writer, &[stops])?;
            writer.finish()
        }
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::TextIndex, 2).unwrap();
        let result = encode(&mut image);
        assert!(
            matches!(&result, Err(Error::InvalidValue(message)) if message.contains("300 stops")),
            "the caller gets its own error back: {result:?}"
        );
        assert_eq!(image.chunk_count(), 0, "no piece of the stream was written");
    }

    #[test]
    fn a_joined_stream_holds_no_spare_capacity() {
        // Four full pieces and 1 byte: a buffer grown by doubling would hold 128 bytes for
        // these 65. Small pieces keep the test fast under Miri; the joining does not depend
        // on the piece size.
        let caps = ChunkCaps {
            max_rows: 65_536,
            max_bytes: 16,
        };
        let mut stream = vec![19u8; 4 * 16];
        stream.push(88);
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::CompactStore, 5).unwrap();
        let mut writer = ChunkStreamWriter::new(&mut image, 0, 0, caps);
        std::io::Write::write_all(&mut writer, &stream).unwrap();
        assert_eq!(writer.finish().unwrap(), stream.len() as u64);
        assert_eq!(image.chunk_count(), 5);
        let section = image.section_source(SectionType::CompactStore).unwrap();
        let mut reader = ChunkStreamReader::new(&*section, 0, 0);
        let first = reader.next_piece().unwrap().unwrap();
        let joined = join_pieces(&mut reader, first).unwrap();
        assert_eq!(
            first_difference(&joined, &stream),
            None,
            "the joined pieces are the stream"
        );
        assert_eq!(
            joined.capacity(),
            joined.len(),
            "the buffer read_stream keeps holds the stream and nothing more"
        );
        let read = read_stream(&*section, 0, 0).unwrap();
        assert_eq!(first_difference(&read, &stream), None, "read_stream");
    }

    /// A section source that notes which chunks were fetched, in order.
    struct Fetches<'a> {
        source: &'a dyn SectionSource,
        fetched: std::cell::RefCell<Vec<usize>>,
    }

    impl SectionSource for Fetches<'_> {
        fn chunks(&self) -> &[ChunkMeta] {
            self.source.chunks()
        }

        fn fetch(&self, index: usize) -> Result<Bytes> {
            self.fetched.borrow_mut().push(index);
            self.source.fetch(index)
        }

        fn stored_length(&self, index: usize) -> Result<u64> {
            self.source.stored_length(index)
        }

        fn section_version(&self) -> u8 {
            self.source.section_version()
        }
    }

    /// The joined buffer is sized once, before the pieces are copied: the last
    /// piece is fetched right after the first, and its offset plus its length
    /// is the stream's length. Each piece is fetched once.
    #[test]
    fn a_joined_stream_is_sized_from_its_last_piece_before_it_is_copied() {
        let caps = ChunkCaps {
            max_rows: 3,
            max_bytes: 16,
        };
        let stream: Vec<u8> = (0..=88u8).collect();
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::CompactStore, 5).unwrap();
        let mut writer = ChunkStreamWriter::new(&mut image, 0, 0, caps);
        std::io::Write::write_all(&mut writer, &stream).unwrap();
        writer.finish().unwrap();
        let section = image.section_source(SectionType::CompactStore).unwrap();
        assert_eq!(
            section.chunks().len(),
            6,
            "five pieces of 16 bytes and one of 9"
        );
        let watched = Fetches {
            source: &*section,
            fetched: std::cell::RefCell::new(Vec::new()),
        };
        let read = read_stream(&watched, 0, 0).unwrap();
        assert_eq!(first_difference(&read, &stream), None);
        assert_eq!(*watched.fetched.borrow(), [0, 5, 1, 2, 3, 4]);
    }

    /// The last piece's offset sizes the buffer, so it is trusted only as far
    /// as the pieces before it can reach (each at most as long as the first):
    /// a crafted offset is refused before anything is allocated for it.
    #[test]
    fn a_last_piece_beyond_the_pieces_before_it_is_refused() {
        for (case, last_offset) in [("far beyond", 1u64 << 40), ("one byte beyond", 33)] {
            let mut image = MemoryImage::new();
            image.begin_section(SectionType::CompactStore, 5).unwrap();
            for (offset, bytes) in [
                (0, &[3u8; 16][..]),
                (16, &[19u8; 16][..]),
                (last_offset, b"Gus"),
            ] {
                image
                    .write_chunk(ChunkMeta::stream_piece(0, 0, offset), bytes)
                    .unwrap();
            }
            let section = image.section_source(SectionType::CompactStore).unwrap();
            let error = read_stream(&*section, 0, 0).unwrap_err();
            assert!(
                matches!(error, Error::Serialization(_)),
                "{case}: {error:?}"
            );
            let text = error.to_string();
            assert!(
                text.contains(&format!("offset {last_offset}")) && text.contains("32 bytes"),
                "{case}: {text}"
            );
        }
        // A last piece that starts where the pieces before it end is read.
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::CompactStore, 5).unwrap();
        for (offset, bytes) in [(0, &[3u8; 16][..]), (16, &[19u8; 16][..]), (32, b"Gus")] {
            image
                .write_chunk(ChunkMeta::stream_piece(0, 0, offset), bytes)
                .unwrap();
        }
        let section = image.section_source(SectionType::CompactStore).unwrap();
        assert_eq!(read_stream(&*section, 0, 0).unwrap().len(), 35);
    }

    /// A last piece that does not start where the pieces before it end is
    /// refused, as any misplaced piece is.
    #[test]
    fn a_last_piece_that_does_not_follow_the_others_is_refused() {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::CompactStore, 5).unwrap();
        // The middle piece is short: the pieces before the last end at 20.
        for (offset, bytes) in [(0, &[3u8; 16][..]), (16, &[19u8; 4][..]), (32, b"Mia")] {
            image
                .write_chunk(ChunkMeta::stream_piece(0, 0, offset), bytes)
                .unwrap();
        }
        let section = image.section_source(SectionType::CompactStore).unwrap();
        let text = read_stream(&*section, 0, 0).unwrap_err().to_string();
        assert!(
            text.contains("offset 32") && text.contains("expected 20"),
            "{text}"
        );
    }

    /// A section source serving the chunks it was given, as a crafted file's
    /// directory lists them: without the identity checks of a memory image,
    /// and with a stored length of its own per piece (at least the bytes it
    /// serves, as for an encrypted chunk).
    struct Crafted {
        metas: Vec<ChunkMeta>,
        pieces: Vec<(Bytes, u64)>,
    }

    impl Crafted {
        /// Stream 0 of graph 0 as `pieces`: each its offset, its bytes and its
        /// stored length.
        fn stream(pieces: Vec<(u64, Bytes, u64)>) -> Self {
            let metas = pieces
                .iter()
                .map(|(offset, _, _)| ChunkMeta::stream_piece(0, 0, *offset))
                .collect();
            let pieces = pieces
                .into_iter()
                .map(|(_, bytes, stored)| (bytes, stored))
                .collect();
            Self { metas, pieces }
        }

        /// Stream 0 of graph 0 as `pieces`, each stored in exactly its bytes.
        fn exact(pieces: Vec<(u64, Bytes)>) -> Self {
            Self::stream(
                pieces
                    .into_iter()
                    .map(|(offset, bytes)| {
                        let stored = bytes.len() as u64;
                        (offset, bytes, stored)
                    })
                    .collect(),
            )
        }

        fn piece(&self, index: usize) -> Result<&(Bytes, u64)> {
            self.pieces
                .get(index)
                .ok_or_else(|| Error::Internal(format!("no chunk {index}")))
        }
    }

    impl SectionSource for Crafted {
        fn chunks(&self) -> &[ChunkMeta] {
            &self.metas
        }

        fn fetch(&self, index: usize) -> Result<Bytes> {
            Ok(self.piece(index)?.0.clone())
        }

        fn stored_length(&self, index: usize) -> Result<u64> {
            Ok(self.piece(index)?.1)
        }

        fn section_version(&self) -> u8 {
            5
        }
    }

    /// A directory can list many pieces that hold nothing, at offsets as far
    /// apart as the first piece is long: (pieces - 1) times the first piece's
    /// length, a terabyte here. The pieces' stored lengths bound the stream
    /// before anything is allocated for it; a reader that trusted the offsets
    /// would ask for more memory than this process can have, and abort. The
    /// rule is the same at any size: the test of pieces that do not form one
    /// run checks it on small pieces, under Miri too.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "a 16 MiB piece and 65,537 chunk descriptions take many minutes under Miri"
    )]
    fn empty_pieces_cannot_claim_a_huge_stream() {
        const FIRST: u64 = 1 << 24;
        const EMPTY: u64 = 1 << 16;
        let mut pieces = vec![(0, Bytes::from(vec![0u8; 1 << 24]))];
        pieces.extend((1..EMPTY).map(|k| (k * FIRST, Bytes::new())));
        pieces.push((EMPTY * FIRST, Bytes::from_static(b"Gus")));
        let crafted = Crafted::exact(pieces);
        let error = read_stream(&crafted, 0, 0).unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        let text = error.to_string();
        assert!(
            text.contains("offset 33554432") && text.contains("expected 16777216 at most"),
            "the second empty piece starts past what the one before it holds: {text}"
        );
    }

    /// The directory must show the pieces as one run before anything is
    /// joined: each starts after the piece before it (every piece but the
    /// last holds a byte), at most the first piece's length after it (a
    /// writer cuts every piece but the last at one length) and at most the
    /// bytes the piece before it is stored in.
    #[test]
    fn pieces_that_do_not_form_one_run_are_refused_before_they_are_joined() {
        let piece = Bytes::from_static;
        let cases: Vec<(&str, Vec<(u64, Bytes)>, &str)> = vec![
            (
                "a piece further after an empty one than it is stored in",
                vec![(0, piece(b"Alix")), (4, Bytes::new()), (8, piece(b"Gus"))],
                "offset 8, expected 4 at most",
            ),
            (
                "an empty piece before the last",
                vec![(0, piece(b"Alix")), (4, Bytes::new()), (4, piece(b"Gus"))],
                "offset 4, expected more than 4",
            ),
            (
                "a piece before the one before it",
                vec![(0, piece(b"Alix")), (4, piece(b"Gus")), (2, piece(b"Mia"))],
                "offset 2, expected more than 4",
            ),
            (
                "a piece longer than the first",
                vec![
                    (0, piece(b"Mia")),
                    (3, piece(b"VincentAmsterdam")),
                    (19, piece(b"Gus")),
                ],
                "offset 19, expected 6 at most",
            ),
        ];
        for (case, pieces, expected) in cases {
            let crafted = Crafted::exact(pieces);
            let error = read_stream(&crafted, 0, 0).unwrap_err();
            assert!(
                matches!(error, Error::Serialization(_)),
                "{case}: {error:?}"
            );
            let text = error.to_string();
            assert!(text.contains(expected), "{case}: {text}");
        }
    }

    /// A piece stored in more bytes than it holds (as an encrypted chunk is)
    /// passes the directory's checks; the bytes fetched decide, and the last
    /// piece must start where they end.
    #[test]
    fn a_piece_holding_less_than_it_is_stored_in_is_refused_when_fetched() {
        let crafted = Crafted::stream(vec![
            (0, Bytes::from_static(&[3u8; 16]), 16),
            (16, Bytes::from_static(&[19u8; 4]), 44),
            (32, Bytes::from_static(b"Jules"), 33),
        ]);
        let text = read_stream(&crafted, 0, 0).unwrap_err().to_string();
        assert!(
            text.contains("offset 32") && text.contains("expected 20"),
            "{text}"
        );
        // Stored in more bytes than they hold, pieces that follow each other
        // are read.
        let crafted = Crafted::stream(vec![
            (0, Bytes::from_static(&[3u8; 16]), 44),
            (16, Bytes::from_static(&[19u8; 16]), 44),
            (32, Bytes::from_static(b"Jules"), 33),
        ]);
        let read = read_stream(&crafted, 0, 0).unwrap();
        assert_eq!(read.len(), 37);
        assert_eq!(&read[32..], b"Jules");
    }

    /// Where two byte strings first differ (a length counts), without printing
    /// megabytes when they do.
    fn first_difference(left: &[u8], right: &[u8]) -> Option<usize> {
        left.iter()
            .zip(right)
            .position(|(left, right)| left != right)
            .or_else(|| (left.len() != right.len()).then(|| left.len().min(right.len())))
    }

    #[test]
    fn stream_errors_name_their_section_and_keep_their_kind() {
        // A misplaced piece, met through `io::Read`.
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::TextIndex, 2).unwrap();
        image
            .write_chunk(ChunkMeta::stream_piece(0, 0, 0), b"Jules")
            .unwrap();
        image
            .write_chunk(ChunkMeta::stream_piece(0, 0, 8), b"Mia")
            .unwrap();
        let section = image.section_source(SectionType::TextIndex).unwrap();
        let mut reader = ChunkStreamReader::new(&*section, 0, 0);
        let error = std::io::Read::read_to_end(&mut reader, &mut Vec::new()).unwrap_err();
        let error = stream_error(SectionType::TextIndex, error);
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        let text = error.to_string();
        assert!(
            text.contains("section TextIndex: stream 0 of graph 0")
                && text.contains("offset 8")
                && !text.contains("I/O error"),
            "{text}"
        );
        // A stream that ends before the section's encoding does.
        let mut reader = ChunkStreamReader::new(&*section, 3, 19);
        let error = std::io::Read::read_exact(&mut reader, &mut [0u8; 3]).unwrap_err();
        let error = stream_error(SectionType::VectorStore, error);
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        let text = error.to_string();
        assert!(
            text.contains("section VectorStore") && text.contains("ends"),
            "{text}"
        );
        // The reader's other errors keep their variant.
        let internal = std::io::Error::other(Error::Internal("Alix".to_string()));
        let error = stream_error(SectionType::RdfRing, internal);
        assert!(
            matches!(&error, Error::Internal(message) if message == "section RdfRing: Alix"),
            "{error:?}"
        );
        let denied = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Gus");
        let error = stream_error(
            SectionType::RdfRing,
            std::io::Error::other(Error::Io(denied)),
        );
        assert!(
            matches!(&error, Error::Io(inner)
                if inner.kind() == std::io::ErrorKind::PermissionDenied
                    && inner.to_string() == "section RdfRing: Gus"),
            "{error:?}"
        );
        // A grafeo error of another variant, and an I/O error that is not the
        // reader's: corrupt section data, with the section named.
        let invalid = std::io::Error::other(Error::InvalidValue("Mia".to_string()));
        let error = stream_error(SectionType::RdfRing, invalid);
        assert!(
            matches!(&error, Error::Serialization(message)
                if message.starts_with("section RdfRing: ") && message.contains("Mia")),
            "{error:?}"
        );
        let foreign = std::io::Error::new(std::io::ErrorKind::InvalidData, "Jules");
        let error = stream_error(SectionType::RdfRing, foreign);
        assert!(
            matches!(&error, Error::Serialization(message)
                if message.starts_with("section RdfRing: ") && message.contains("Jules")),
            "{error:?}"
        );
    }

    #[test]
    fn a_stream_piece_with_a_codec_is_refused() {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::TextIndex, 2).unwrap();
        let coded = ChunkMeta {
            codec: 3,
            ..ChunkMeta::stream_piece(0, 0, 0)
        };
        image.write_chunk(coded, b"Gus").unwrap();
        let section = image.section_source(SectionType::TextIndex).unwrap();
        let error = read_stream(&*section, 0, 0).unwrap_err().to_string();
        assert!(error.contains("codec 3"), "{error}");
    }

    #[test]
    fn a_repeated_identity_names_the_section_and_the_chunk() {
        let mut identities = ChunkIdentities::default();
        let column = ChunkMeta::column(3, 19, 88, 3, 7);
        identities.insert(SectionType::RdfStore, &column).unwrap();
        identities
            .insert(SectionType::RdfStore, &ChunkMeta::column(3, 19, 91, 3, 7))
            .unwrap();
        let error = identities
            .insert(SectionType::RdfStore, &ChunkMeta::column(3, 19, 88, 1, 0))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("section RdfStore")
                && error.contains("two chunks of kind Column")
                && error.contains("graph 3")
                && error.contains("column 19")
                && error.contains("first row 88"),
            "{error}"
        );
    }
}
