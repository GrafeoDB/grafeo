//! What the checkpoint image of a database file holds, for tests of what a
//! checkpoint wrote. Include it with `#[path = "common/image.rs"] mod image;`.

use grafeo_common::storage::SectionType;
use grafeo_engine::GrafeoDB;

/// Whether a section of the image `db`'s file holds now (its active header
/// and the sections it names) contains `needle`: the catalog for a
/// constraint, the LPG store for a graph, the RDF store for a triple, an
/// index section for an index. Each section is searched as the bytes of all
/// its chunks, one after the other. Read through the database's own file
/// handle, which on Windows is the only one that can read a locked file.
///
/// # Panics
///
/// If `db` has no database file, or a section cannot be read.
pub fn image_holds(db: &GrafeoDB, needle: &str) -> bool {
    let fm = db.file_manager().expect("a database file");
    fm.read_image(|image| {
        let mut sections = Vec::new();
        for section_type in [
            SectionType::Catalog,
            SectionType::LpgStore,
            SectionType::RdfStore,
            SectionType::VectorStore,
            SectionType::TextIndex,
            SectionType::PropertyIndex,
        ] {
            if let Some(section) = image.section_source(section_type) {
                let mut bytes = Vec::new();
                for index in 0..section.chunks().len() {
                    bytes.extend_from_slice(&section.fetch(index)?);
                }
                sections.push(bytes);
            }
        }
        Ok(sections)
    })
    .unwrap()
    .iter()
    .any(|bytes| {
        bytes
            .windows(needle.len())
            .any(|window| window == needle.as_bytes())
    })
}
