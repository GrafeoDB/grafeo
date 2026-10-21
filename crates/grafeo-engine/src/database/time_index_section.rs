//! The `TIME_INDEX` section (H2c): the history of a database that keeps
//! one, besides the versions its LPG section holds.
//!
//! It holds the oldest epoch a read may ask for (the history floor) and the
//! time of each commit from the newest at or before the floor on, which map
//! a read at a time to its epoch. A checkpoint writes it only while the
//! database keeps history: a database without history has no such section,
//! and its file is the one it was without the feature.
//!
//! One byte stream (graph 0, stream 0), cut into pieces by the caps:
//!
//! | Bytes | Field |
//! | --- | --- |
//! | 1 | layout, 1 |
//! | 8 | the floor, u64 little endian |
//! | 8 | the count of commits, u64 little endian |
//! | 16 per commit | its epoch and its time in milliseconds since the Unix epoch, u64 little endian each |
//!
//! Epochs ascend and times never go back. The section is required: a reader
//! that does not know it refuses the file (FD8), as it must not answer a
//! read at a time without it.

use std::sync::atomic::{AtomicBool, Ordering};

use grafeo_common::storage::{
    ChunkCaps, ChunkKind, ChunkStreamWriter, Section, SectionSink, SectionSource, SectionType,
    check_version, read_stream,
};
use grafeo_common::utils::error::{Error, Result};

use crate::transaction::HistoryImage;

/// The section version this module writes and reads.
pub(crate) const TIME_INDEX_SECTION_VERSION: u8 = 1;

/// The layout byte of the stream.
const LAYOUT: u8 = 1;

/// The bytes of the stream before the commits.
const HEADER: usize = 1 + 8 + 8;

/// The `TIME_INDEX` section of a history, or the one a load reads into.
pub(crate) struct TimeIndexSection {
    image: HistoryImage,
    dirty: AtomicBool,
}

impl TimeIndexSection {
    /// The section of `image`, to write.
    pub(crate) fn new(image: HistoryImage) -> Self {
        Self {
            image,
            dirty: AtomicBool::new(true),
        }
    }

    /// An empty section to read into.
    #[cfg(any(test, feature = "lpg"))]
    pub(crate) fn empty() -> Self {
        Self::new(HistoryImage::default())
    }

    /// What it read.
    #[cfg(any(test, feature = "lpg"))]
    pub(crate) fn into_image(self) -> HistoryImage {
        self.image
    }
}

fn refuse(what: impl std::fmt::Display) -> Error {
    Error::corruption(format!("TIME_INDEX section: {what}"))
}

/// The image the stream's bytes hold.
fn decode(bytes: &[u8]) -> Result<HistoryImage> {
    let word = |at: usize| -> Result<u64> {
        bytes
            .get(at..at + 8)
            .and_then(|word| <[u8; 8]>::try_from(word).ok())
            .map(u64::from_le_bytes)
            .ok_or_else(|| refuse(format!("the stream ends at byte {}", bytes.len())))
    };
    match bytes.first() {
        Some(&LAYOUT) => {}
        Some(layout) => {
            return Err(Error::Serialization(format!(
                "TIME_INDEX section: layout {layout}, where this version reads {LAYOUT}: \
                 written by another version"
            )));
        }
        None => return Err(refuse("an empty stream")),
    }
    let floor = word(1)?;
    let count = word(9)?;
    let commits = usize::try_from(count).ok();
    let expected = commits
        .and_then(|count| count.checked_mul(16))
        .and_then(|bytes| bytes.checked_add(HEADER));
    let (Some(commits), true) = (commits, expected == Some(bytes.len())) else {
        return Err(refuse(format!(
            "{count} commits in {} bytes, where each takes 16 after a header of {HEADER}",
            bytes.len()
        )));
    };
    let mut times: Vec<(u64, u64)> = Vec::with_capacity(commits);
    for index in 0..commits {
        let at = HEADER + 16 * index;
        let (epoch, millis) = (word(at)?, word(at + 8)?);
        if let Some((last_epoch, last_millis)) = times.last()
            && (epoch <= *last_epoch || millis < *last_millis)
        {
            return Err(refuse(format!(
                "commit {index}: epoch {epoch} at {millis} does not follow epoch {last_epoch} at \
                 {last_millis}"
            )));
        }
        times.push((epoch, millis));
    }
    Ok(HistoryImage { floor, times })
}

impl Section for TimeIndexSection {
    fn section_type(&self) -> SectionType {
        SectionType::TimeIndex
    }

    fn version(&self) -> u8 {
        TIME_INDEX_SECTION_VERSION
    }

    /// The section has no form as one blob: it is written as a stream.
    fn serialize(&self) -> Result<Vec<u8>> {
        Err(Error::Internal(
            "the TIME_INDEX section is written as a stream, not as one blob".to_string(),
        ))
    }

    /// No release wrote the section as one blob.
    fn deserialize(&mut self, _data: &[u8]) -> Result<()> {
        Err(refuse("a raw chunk, which no release writes"))
    }

    fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
        use std::io::Write;

        let mut stream = ChunkStreamWriter::new(sink, 0, 0, ChunkCaps::current());
        let count = u64::try_from(self.image.times.len()).unwrap_or(u64::MAX);
        let mut header = [0_u8; HEADER];
        header[0] = LAYOUT;
        header[1..9].copy_from_slice(&self.image.floor.to_le_bytes());
        header[9..17].copy_from_slice(&count.to_le_bytes());
        // A write error is kept by the stream and returned by `finish`.
        let _ = stream.write_all(&header);
        for (epoch, millis) in &self.image.times {
            let mut entry = [0_u8; 16];
            entry[..8].copy_from_slice(&epoch.to_le_bytes());
            entry[8..].copy_from_slice(&millis.to_le_bytes());
            let _ = stream.write_all(&entry);
        }
        stream.finish().map(|_| ())
    }

    fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
        check_version(SectionType::TimeIndex, source, TIME_INDEX_SECTION_VERSION)?;
        if source.chunks().iter().any(|chunk| {
            chunk.kind != ChunkKind::Stream || chunk.graph_id != 0 || chunk.column_id != 0
        }) {
            return Err(refuse("a chunk other than a piece of its one stream"));
        }
        let bytes = read_stream(source, 0, 0)?;
        self.image = decode(&bytes)?;
        Ok(())
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.image.times.len() * 16
    }
}

/// The history the image of a database holds, `None` without a
/// `TIME_INDEX` section (a database without history).
///
/// # Errors
///
/// The section's errors: a version or layout of another release, or bytes
/// no writer writes.
#[cfg(any(test, feature = "lpg"))]
pub(crate) fn read_history(
    image: &dyn grafeo_common::storage::ImageSource,
) -> Result<Option<HistoryImage>> {
    let Some(source) = image.section_source(SectionType::TimeIndex) else {
        return Ok(None);
    };
    let mut section = TimeIndexSection::empty();
    section.read_from(&*source)?;
    Ok(Some(section.into_image()))
}

#[cfg(test)]
mod tests {
    use grafeo_common::storage::{
        ChunkCaps, ChunkMeta, MemoryImage, Section, SectionSink, SectionType,
    };
    use grafeo_common::utils::error::Error;

    use super::{TIME_INDEX_SECTION_VERSION, TimeIndexSection, read_history};
    use crate::transaction::HistoryImage;

    fn written(image: &HistoryImage) -> MemoryImage {
        let mut memory = MemoryImage::new();
        memory
            .begin_section(SectionType::TimeIndex, TIME_INDEX_SECTION_VERSION)
            .unwrap();
        TimeIndexSection::new(image.clone())
            .write_to(&mut memory)
            .unwrap();
        memory
    }

    /// The floor and every commit time come back, also across pieces cut
    /// by small caps; a database without the section has no history.
    #[test]
    fn the_history_round_trips_across_pieces() {
        let image = HistoryImage {
            floor: 19,
            times: (19..400)
                .map(|epoch| (epoch, 1_700_000_000_000 + epoch * 7))
                .collect(),
        };
        let caps = ChunkCaps {
            max_rows: 4,
            max_bytes: 100,
        };
        let memory = grafeo_common::testing::chunk_caps::with_chunk_caps(caps, || written(&image));
        assert_eq!(read_history(&memory).unwrap(), Some(image));
        assert_eq!(read_history(&MemoryImage::new()).unwrap(), None);
    }

    /// The reader refuses what no writer writes: commits whose epochs go
    /// back, a count the bytes do not hold, another layout, another version.
    #[test]
    fn the_reader_refuses_what_no_writer_writes() {
        let bytes = |floor: u64, times: &[(u64, u64)], count: u64, layout: u8| {
            let mut bytes = vec![layout];
            bytes.extend_from_slice(&floor.to_le_bytes());
            bytes.extend_from_slice(&count.to_le_bytes());
            for (epoch, millis) in times {
                bytes.extend_from_slice(&epoch.to_le_bytes());
                bytes.extend_from_slice(&millis.to_le_bytes());
            }
            bytes
        };
        let image_of = |bytes: Vec<u8>, version: u8| {
            let mut memory = MemoryImage::new();
            memory
                .begin_section(SectionType::TimeIndex, version)
                .unwrap();
            memory
                .write_chunk(ChunkMeta::stream_piece(0, 0, 0), &bytes)
                .unwrap();
            memory
        };
        let back = image_of(bytes(1, &[(3, 30), (2, 40)], 2, 1), 1);
        assert!(matches!(read_history(&back), Err(Error::Corruption(_))));
        let short = image_of(bytes(1, &[(3, 30)], 2, 1), 1);
        assert!(matches!(read_history(&short), Err(Error::Corruption(_))));
        let layout = image_of(bytes(1, &[], 0, 2), 1);
        assert!(matches!(
            read_history(&layout),
            Err(Error::Serialization(_))
        ));
        let version = image_of(bytes(1, &[], 0, 1), 2);
        assert!(matches!(
            read_history(&version),
            Err(Error::Serialization(_))
        ));
        let ok = image_of(bytes(1, &[(3, 30), (4, 30)], 2, 1), 1);
        assert_eq!(
            read_history(&ok).unwrap(),
            Some(HistoryImage {
                floor: 1,
                times: vec![(3, 30), (4, 30)],
            })
        );
    }
}
