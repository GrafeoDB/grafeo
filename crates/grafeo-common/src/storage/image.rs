//! Images: the sections of one checkpoint, served chunk by chunk.
//!
//! An [`ImageSource`] serves the chunks of each section of one image: a
//! checkpoint in a `.grafeo` file (grafeo-storage), the sections of a 0.5.x
//! file, or a copy in memory ([`MemoryImage`]).

use std::cell::RefCell;

use bytes::Bytes;

use crate::storage::chunk::ChunkIdentities;
use crate::storage::section::{ChunkMeta, Section, SectionSink, SectionSource, SectionType};
use crate::utils::error::{Error, Result};

/// The sections of one image: a checkpoint in a file, or a copy in memory.
pub trait ImageSource {
    /// The chunks of `section_type`, or `None` when the image has none.
    fn section_source(&self, section_type: SectionType) -> Option<Box<dyn SectionSource + '_>>;
}

/// An image held in memory, filled section by section through its
/// [`SectionSink`].
///
/// As a checkpoint writer does, it refuses a section begun twice and two
/// chunks with one identity in a section; as a file does, it serves a section
/// only when the section holds chunks.
#[derive(Debug, Default)]
pub struct MemoryImage {
    sections: Vec<MemorySection>,
    /// The section begun last, which receives the chunks written.
    current: Option<usize>,
    /// The identities the current section has used.
    identities: ChunkIdentities,
}

/// The chunks of one section of a [`MemoryImage`].
#[derive(Debug, Clone)]
pub struct MemorySection {
    section_type: SectionType,
    version: u8,
    metas: Vec<ChunkMeta>,
    chunks: Vec<Bytes>,
}

impl MemoryImage {
    /// An image without sections.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts `section_type`, written with `version`: the chunks written next
    /// belong to it.
    ///
    /// The section begun before stops receiving chunks even when this call is
    /// refused: a chunk written after a refused call fails with "no section
    /// begun" instead of landing in the wrong section.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`] when `section_type` was begun already.
    pub fn begin_section(&mut self, section_type: SectionType, version: u8) -> Result<()> {
        self.current = None;
        if self.has_section(section_type) {
            return Err(Error::Internal(format!(
                "section {section_type:?} was begun twice in one memory image"
            )));
        }
        self.sections.push(MemorySection {
            section_type,
            version,
            metas: Vec::new(),
            chunks: Vec::new(),
        });
        self.current = Some(self.sections.len() - 1);
        self.identities = ChunkIdentities::default();
        Ok(())
    }

    /// [`begin_section`](Self::begin_section) and
    /// [`write_to`](Section::write_to) for each section, in order.
    ///
    /// # Errors
    ///
    /// Returns an error when a section type repeats, a section fails to
    /// write, or a section writes two chunks with one identity.
    pub fn from_sections(sections: &[&dyn Section]) -> Result<Self> {
        let mut image = Self::new();
        for section in sections {
            image.begin_section(section.section_type(), section.version())?;
            section.write_to(&mut image)?;
        }
        Ok(image)
    }

    /// 0.5.x section bytes: each section one raw chunk, version 0.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] naming the section when a section
    /// type is listed twice, which only a corrupt 0.5.x file does: neither
    /// copy is chosen over the other.
    pub fn from_raw(sections: Vec<(SectionType, Vec<u8>)>) -> Result<Self> {
        let mut image = Self::new();
        for (section_type, bytes) in sections {
            if image.has_section(section_type) {
                return Err(Error::Serialization(format!(
                    "a 0.5.x file lists section {section_type:?} twice"
                )));
            }
            image.sections.push(MemorySection {
                section_type,
                version: 0,
                metas: vec![ChunkMeta::raw()],
                chunks: vec![Bytes::from(bytes)],
            });
        }
        Ok(image)
    }

    /// Takes the chunks of `section_type` out of the image, or `None` when it
    /// holds none (as [`section_source`](ImageSource::section_source)
    /// answers).
    ///
    /// The image no longer holds the section: its bytes are freed once the
    /// section returned is dropped. Taking a section also ends the section
    /// begun last, so a chunk written afterwards fails with "no section
    /// begun".
    pub fn take_section(&mut self, section_type: SectionType) -> Option<MemorySection> {
        let at = self.sections.iter().position(|section| {
            section.section_type == section_type && !section.metas.is_empty()
        })?;
        self.current = None;
        Some(self.sections.remove(at))
    }

    /// Whether `section_type` was begun or listed already.
    fn has_section(&self, section_type: SectionType) -> bool {
        self.sections
            .iter()
            .any(|section| section.section_type == section_type)
    }

    /// The section types the image holds chunks of, in the order they were
    /// begun.
    #[must_use]
    pub fn section_types(&self) -> Vec<SectionType> {
        self.sections
            .iter()
            .filter(|section| !section.metas.is_empty())
            .map(|section| section.section_type)
            .collect()
    }

    /// The number of chunks of all sections.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.sections
            .iter()
            .map(|section| section.metas.len())
            .sum()
    }
}

impl SectionSink for MemoryImage {
    /// Appends the chunk to the section begun last.
    ///
    /// # Errors
    ///
    /// Returns an error when no section was begun, or when the section has a
    /// chunk with the same identity already.
    fn write_chunk(&mut self, meta: ChunkMeta, bytes: &[u8]) -> Result<()> {
        let Some(section) = self.current.and_then(|at| self.sections.get_mut(at)) else {
            return Err(Error::Internal(format!(
                "no section begun: a chunk of kind {:?} was written to a memory image before \
                 any section",
                meta.kind
            )));
        };
        self.identities.insert(section.section_type, &meta)?;
        section.metas.push(meta);
        section.chunks.push(Bytes::copy_from_slice(bytes));
        Ok(())
    }
}

impl ImageSource for MemoryImage {
    /// The section's chunks, or `None` when it has none: a section without
    /// chunks is absent, as in a file.
    fn section_source(&self, section_type: SectionType) -> Option<Box<dyn SectionSource + '_>> {
        self.sections
            .iter()
            .find(|section| section.section_type == section_type && !section.metas.is_empty())
            .map(|section| Box::new(BorrowedSection(section)) as Box<dyn SectionSource + '_>)
    }
}

impl SectionSource for MemorySection {
    fn chunks(&self) -> &[ChunkMeta] {
        &self.metas
    }

    fn fetch(&self, index: usize) -> Result<Bytes> {
        self.chunks.get(index).cloned().ok_or_else(|| {
            Error::Internal(format!(
                "chunk {index} out of range: section {:?} has {} chunks",
                self.section_type,
                self.chunks.len()
            ))
        })
    }

    fn section_version(&self) -> u8 {
        self.version
    }
}

/// A [`MemoryImage`] that serves each section once, for a load that reads
/// each section once.
///
/// Serving a section moves its chunks into the source returned, so its bytes
/// are freed as soon as the reader drops that source: a load holds the
/// sections it has not read yet, not those it has loaded already. A second
/// request for a section answers `None`, as for a section the image never
/// held.
#[derive(Debug)]
pub struct ServedOnce {
    image: RefCell<MemoryImage>,
}

impl ServedOnce {
    /// Serves the sections of `image`, each once.
    #[must_use]
    pub fn new(image: MemoryImage) -> Self {
        Self {
            image: RefCell::new(image),
        }
    }

    /// The section types not served yet, in the order they were begun.
    #[must_use]
    pub fn section_types(&self) -> Vec<SectionType> {
        self.image.borrow().section_types()
    }
}

impl ImageSource for ServedOnce {
    /// Hands the section out, or `None` when the image has none or served it
    /// already.
    fn section_source(&self, section_type: SectionType) -> Option<Box<dyn SectionSource + '_>> {
        let section = self.image.borrow_mut().take_section(section_type)?;
        Some(Box::new(section))
    }
}

/// A section of a [`MemoryImage`] served by reference, so serving it copies
/// nothing.
struct BorrowedSection<'a>(&'a MemorySection);

impl SectionSource for BorrowedSection<'_> {
    fn chunks(&self) -> &[ChunkMeta] {
        self.0.chunks()
    }

    fn fetch(&self, index: usize) -> Result<Bytes> {
        self.0.fetch(index)
    }

    fn section_version(&self) -> u8 {
        self.0.section_version()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::section::{ChunkMeta, Section, SectionSink, SectionType, legacy_bytes};
    use crate::utils::error::Result;

    /// A test section writing its bytes as one raw chunk (the trait defaults).
    struct Raw(SectionType, Vec<u8>);

    impl Section for Raw {
        fn section_type(&self) -> SectionType {
            self.0
        }
        fn serialize(&self) -> Result<Vec<u8>> {
            Ok(self.1.clone())
        }
        fn deserialize(&mut self, data: &[u8]) -> Result<()> {
            self.1 = data.to_vec();
            Ok(())
        }
        fn is_dirty(&self) -> bool {
            false
        }
        fn mark_clean(&self) {}
        fn memory_usage(&self) -> usize {
            self.1.len()
        }
    }

    #[test]
    fn a_memory_image_serves_what_its_sections_wrote() {
        let catalog = Raw(SectionType::Catalog, b"Alix".to_vec());
        let store = Raw(SectionType::LpgStore, b"Gus".to_vec());
        let image = MemoryImage::from_sections(&[&catalog, &store]).unwrap();
        assert_eq!(
            image.section_types(),
            [SectionType::Catalog, SectionType::LpgStore]
        );
        let section = image.section_source(SectionType::LpgStore).unwrap();
        assert_eq!(section.section_version(), 1);
        assert_eq!(&section.fetch(0).unwrap()[..], b"Gus");
        assert!(image.section_source(SectionType::RdfStore).is_none());
    }

    #[test]
    fn a_repeated_chunk_identity_or_section_is_refused() {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::LpgStore, 3).unwrap();
        image
            .write_chunk(ChunkMeta::column(0, 16, 0, 3, 7), b"Vincent")
            .unwrap();
        let error = image
            .write_chunk(ChunkMeta::column(0, 16, 0, 3, 7), b"Jules")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("two chunks") && error.contains("column 16"),
            "{error}"
        );
        // Another kind: allowed.
        image
            .write_chunk(ChunkMeta::history(0, 16, 0, 3, 7), b"Jules")
            .unwrap();
        assert!(image.begin_section(SectionType::LpgStore, 3).is_err());
        assert!(
            MemoryImage::new()
                .write_chunk(ChunkMeta::meta(), b"x")
                .is_err(),
            "no section begun"
        );
    }

    #[test]
    fn from_raw_gives_each_section_one_raw_chunk_of_version_0() {
        let image =
            MemoryImage::from_raw(vec![(SectionType::Catalog, b"Berlin".to_vec())]).unwrap();
        let section = image.section_source(SectionType::Catalog).unwrap();
        assert_eq!(
            (section.chunks(), section.section_version()),
            (&[ChunkMeta::raw()][..], 0)
        );
        assert_eq!(
            legacy_bytes(&*section).unwrap().as_deref(),
            Some(&b"Berlin"[..])
        );
    }

    #[test]
    fn a_refused_begin_section_leaves_no_section_current() {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::Catalog, 2).unwrap();
        image.write_chunk(ChunkMeta::meta(), b"Alix").unwrap();
        image.begin_section(SectionType::LpgStore, 3).unwrap();
        image.write_chunk(ChunkMeta::meta(), b"Gus").unwrap();
        assert!(image.begin_section(SectionType::Catalog, 2).is_err());
        let error = image
            .write_chunk(ChunkMeta::column(0, 3, 0, 19, 0), b"Vincent")
            .unwrap_err()
            .to_string();
        assert!(error.contains("no section begun"), "{error}");
        assert_eq!(
            image.chunk_count(),
            2,
            "the chunk went into neither the catalog nor the LPG store"
        );
    }

    #[test]
    fn identities_are_checked_per_section() {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::Catalog, 2).unwrap();
        image.write_chunk(ChunkMeta::meta(), b"Alix").unwrap();
        image.begin_section(SectionType::LpgStore, 3).unwrap();
        image.write_chunk(ChunkMeta::meta(), b"Gus").unwrap();
        assert_eq!(image.chunk_count(), 2);
        let catalog = image.section_source(SectionType::Catalog).unwrap();
        assert_eq!(
            (catalog.section_version(), &catalog.fetch(0).unwrap()[..]),
            (2, &b"Alix"[..]),
            "the second section's chunk went into its own section"
        );
        assert!(catalog.fetch(1).is_err(), "the catalog has one chunk");
    }

    #[test]
    fn a_section_that_writes_nothing_is_neither_served_nor_listed() {
        let empty = Raw(SectionType::RdfStore, Vec::new());
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::VectorStore, 3).unwrap();
        image.begin_section(SectionType::Catalog, 2).unwrap();
        image.write_chunk(ChunkMeta::meta(), b"Mia").unwrap();
        assert_eq!(image.section_types(), [SectionType::Catalog]);
        assert!(image.section_source(SectionType::VectorStore).is_none());
        // A raw section of no bytes still writes its one raw chunk.
        let image = MemoryImage::from_sections(&[&empty]).unwrap();
        assert_eq!(image.section_types(), [SectionType::RdfStore]);
    }

    #[test]
    fn a_taken_section_leaves_the_image() {
        let catalog = Raw(SectionType::Catalog, b"Alix".to_vec());
        let store = Raw(SectionType::LpgStore, b"Gus".to_vec());
        let mut image = MemoryImage::from_sections(&[&catalog, &store]).unwrap();
        let taken = image.take_section(SectionType::LpgStore).unwrap();
        assert_eq!(
            (taken.section_version(), &taken.fetch(0).unwrap()[..]),
            (1, &b"Gus"[..])
        );
        assert_eq!(image.section_types(), [SectionType::Catalog]);
        assert!(image.section_source(SectionType::LpgStore).is_none());
        assert!(
            image.take_section(SectionType::LpgStore).is_none(),
            "a section is taken once"
        );
        assert!(image.take_section(SectionType::RdfStore).is_none());

        // Taking ends the section begun last.
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::Catalog, 2).unwrap();
        image.write_chunk(ChunkMeta::meta(), b"Mia").unwrap();
        image.take_section(SectionType::Catalog).unwrap();
        let error = image
            .write_chunk(ChunkMeta::column(0, 3, 0, 19, 0), b"Vincent")
            .unwrap_err()
            .to_string();
        assert!(error.contains("no section begun"), "{error}");
    }

    #[test]
    fn served_once_serves_each_section_once() {
        let image = ServedOnce::new(
            MemoryImage::from_raw(vec![
                (SectionType::Catalog, b"Amsterdam".to_vec()),
                (SectionType::LpgStore, b"Paris".to_vec()),
            ])
            .unwrap(),
        );
        let store = image.section_source(SectionType::LpgStore).unwrap();
        assert_eq!(
            legacy_bytes(&*store).unwrap().as_deref(),
            Some(&b"Paris"[..])
        );
        assert!(
            image.section_source(SectionType::LpgStore).is_none(),
            "a section is served once"
        );
        assert_eq!(
            image.section_types(),
            [SectionType::Catalog],
            "the image no longer holds what it served"
        );
        assert!(image.section_source(SectionType::RdfStore).is_none());
    }

    /// The bytes of a served section live in the source handed out, and go
    /// when the reader drops it.
    #[test]
    fn a_served_section_is_freed_when_its_source_drops() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        /// Chunk bytes that note when they are freed.
        struct Noted(Arc<AtomicBool>);

        impl AsRef<[u8]> for Noted {
            fn as_ref(&self) -> &[u8] {
                b"Jules"
            }
        }

        impl Drop for Noted {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let freed = Arc::new(AtomicBool::new(false));
        let mut image = MemoryImage::new();
        image.sections.push(MemorySection {
            section_type: SectionType::TextIndex,
            version: 1,
            metas: vec![ChunkMeta::raw()],
            chunks: vec![Bytes::from_owner(Noted(Arc::clone(&freed)))],
        });
        let image = ServedOnce::new(image);

        let source = image.section_source(SectionType::TextIndex).unwrap();
        assert_eq!(&source.fetch(0).unwrap()[..], b"Jules");
        assert!(
            !freed.load(Ordering::SeqCst),
            "the source holds the bytes while it is read"
        );
        drop(source);
        assert!(
            freed.load(Ordering::SeqCst),
            "the bytes are freed once the reader drops the source"
        );

        // A plain image keeps them: its sources borrow the section.
        let kept = Arc::new(AtomicBool::new(false));
        let mut image = MemoryImage::new();
        image.sections.push(MemorySection {
            section_type: SectionType::TextIndex,
            version: 1,
            metas: vec![ChunkMeta::raw()],
            chunks: vec![Bytes::from_owner(Noted(Arc::clone(&kept)))],
        });
        drop(image.section_source(SectionType::TextIndex).unwrap());
        assert!(!kept.load(Ordering::SeqCst), "the image still holds them");
        drop(image);
        assert!(kept.load(Ordering::SeqCst));
    }

    #[test]
    fn from_raw_refuses_a_section_listed_twice() {
        let error = MemoryImage::from_raw(vec![
            (SectionType::Catalog, b"Amsterdam".to_vec()),
            (SectionType::LpgStore, b"Paris".to_vec()),
            (SectionType::Catalog, b"Prague".to_vec()),
        ])
        .unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        let error = error.to_string();
        assert!(
            error.contains("Catalog") && error.contains("twice"),
            "{error}"
        );
        let image = MemoryImage::from_raw(vec![
            (SectionType::Catalog, b"Amsterdam".to_vec()),
            (SectionType::LpgStore, b"Paris".to_vec()),
        ])
        .unwrap();
        assert_eq!(
            image.section_types(),
            [SectionType::Catalog, SectionType::LpgStore]
        );
        let store = image.section_source(SectionType::LpgStore).unwrap();
        assert_eq!(
            legacy_bytes(&*store).unwrap().as_deref(),
            Some(&b"Paris"[..])
        );
    }
}
