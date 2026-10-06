//! What the checkpoint image of a database file holds, for tests of what a
//! checkpoint wrote. Include it with `#[path = "common/image.rs"] mod image;`.

use grafeo_common::storage::{ImageSource, SectionType};
use grafeo_common::utils::error::Result;
use grafeo_engine::GrafeoDB;

/// Whether a section of the image `db`'s file holds now (its active header
/// and the sections it names) contains `needle`: the catalog for a
/// constraint, the LPG store or the compacted base for a graph, the RDF store
/// for a triple, an index section for an index. Every section is searched
/// (see [`source_holds`]). Read through the database's own file handle, which
/// on Windows is the only one that can read a locked file.
///
/// # Panics
///
/// If `db` has no database file, or a section cannot be read.
pub fn image_holds(db: &GrafeoDB, needle: &str) -> bool {
    let fm = db.file_manager().expect("a database file");
    fm.read_image(|image| source_holds(image, needle)).unwrap()
}

/// Whether a section of `image` contains `needle`. Every section type is
/// searched, in the order of their type bytes, each as the bytes of all its
/// chunks, one after the other, as it is read: the search holds one section
/// at a time and stops at the first that holds the needle. An empty needle
/// matches nothing.
///
/// # Errors
///
/// Returns the error of reading a section.
pub fn source_holds(image: &dyn ImageSource, needle: &str) -> Result<bool> {
    let needle = needle.as_bytes();
    if needle.is_empty() {
        return Ok(false);
    }
    for section_type in (0..=u8::MAX).filter_map(SectionType::from_u8) {
        let Some(section) = image.section_source(section_type) else {
            continue;
        };
        let mut bytes = Vec::new();
        for index in 0..section.chunks().len() {
            bytes.extend_from_slice(&section.fetch(index)?);
        }
        if bytes.windows(needle.len()).any(|window| window == needle) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use grafeo_common::storage::{ImageSource, MemoryImage, SectionSource, SectionType};

    use super::source_holds;

    /// An image that notes which sections were asked for.
    struct Watched {
        image: MemoryImage,
        asked: RefCell<Vec<SectionType>>,
    }

    impl Watched {
        /// An image whose sections hold `sections`, one raw chunk each.
        fn of(sections: &[(SectionType, &str)]) -> Self {
            Self {
                image: MemoryImage::from_raw(
                    sections
                        .iter()
                        .map(|(section_type, text)| (*section_type, text.as_bytes().to_vec()))
                        .collect(),
                )
                .unwrap(),
                asked: RefCell::new(Vec::new()),
            }
        }
    }

    impl ImageSource for Watched {
        fn section_source(&self, section_type: SectionType) -> Option<Box<dyn SectionSource + '_>> {
            self.asked.borrow_mut().push(section_type);
            self.image.section_source(section_type)
        }
    }

    /// Every section type is searched: the compacted base, the overlay's
    /// deletions and the RDF ring included.
    #[test]
    fn every_section_is_searched() {
        for section_type in [
            SectionType::CompactStore,
            SectionType::OverlayDeletions,
            SectionType::RdfRing,
            SectionType::PropertyIndex,
        ] {
            let image = Watched::of(&[
                (SectionType::Catalog, "schema"),
                (section_type, "vincent in amsterdam"),
            ]);
            assert!(
                source_holds(&image, "amsterdam").unwrap(),
                "{section_type:?} was not searched"
            );
            assert!(!source_holds(&image, "berlin").unwrap());
        }
    }

    /// An empty needle matches nothing, and does not panic.
    #[test]
    fn an_empty_needle_matches_nothing() {
        let image = Watched::of(&[(SectionType::LpgStore, "mia")]);
        assert!(!source_holds(&image, "").unwrap());
    }

    /// The search stops at the first section that holds the needle.
    #[test]
    fn the_search_stops_at_the_first_match() {
        let image = Watched::of(&[
            (SectionType::Catalog, "gus"),
            (SectionType::LpgStore, "jules"),
        ]);
        assert!(source_holds(&image, "gus").unwrap());
        assert_eq!(
            *image.asked.borrow(),
            vec![SectionType::Catalog],
            "sections read after the match"
        );
    }
}
