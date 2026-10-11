//! 0.5.x database files built for a test, for what a build does with data it
//! cannot read: a container v2 file from the sections of a released fixture,
//! some left out and others added. Include it with
//! `#[path = "common/legacy_file.rs"] mod legacy_file;`.

use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use grafeo_common::storage::section::{SectionDirectoryEntry, SectionType};
use grafeo_storage::container::SectionDirectory;
use grafeo_storage::container::directory::{DIRECTORY_OFFSET, SECTION_DATA_OFFSET};
use grafeo_storage::file::header::{write_db_header, write_file_header};
use grafeo_storage::file::legacy::{LegacyContents, LegacyFile};
use grafeo_storage::file::{DbHeader, FileHeader};

/// The released fixture `name` written by `version` (see
/// `tests/fixtures/released/`).
pub fn released(version: &str, name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/released")
        .join(version)
        .join(name)
}

/// The sections of the released 0.5.x file `version/closed.grafeo` (a
/// container v2 file), as stored.
fn released_sections(version: &str) -> Vec<(SectionType, Vec<u8>)> {
    let legacy = LegacyFile::open(&released(version, "closed.grafeo"), None).unwrap();
    let contents = legacy.contents().unwrap();
    let LegacyContents::Sections(sections) = contents else {
        panic!("the released {version} file holds sections, got {contents:?}");
    };
    sections
}

/// Writes at `path` a 0.5.x file (container v2) as 0.5.x left one closed
/// cleanly: the sections of the released `version/closed.grafeo` that `keep`
/// keeps, then the `added` sections. An added section may hold anything: an
/// open that refuses a section looks at which sections the file holds before
/// it reads them.
pub fn write_0_5_file(
    path: &Path,
    version: &str,
    keep: impl Fn(SectionType) -> bool,
    added: &[(SectionType, Vec<u8>)],
) {
    let legacy = LegacyFile::open(&released(version, "closed.grafeo"), None).unwrap();
    let mut sections = released_sections(version);
    sections.retain(|(section_type, _)| keep(*section_type));
    sections.extend(added.iter().cloned());

    let mut file = std::fs::File::create(path).unwrap();
    let mut directory = SectionDirectory::new();
    let mut offset = SECTION_DATA_OFFSET;
    for (section_type, data) in &sections {
        let length = u64::try_from(data.len()).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(data).unwrap();
        directory
            .upsert(SectionDirectoryEntry {
                section_type: *section_type,
                version: 1,
                flags: section_type.default_flags(),
                offset,
                length,
                checksum: crc32fast::hash(data),
            })
            .unwrap();
        offset = (offset + length).next_multiple_of(4096);
    }
    file.seek(SeekFrom::Start(DIRECTORY_OFFSET)).unwrap();
    file.write_all(&directory.to_bytes()).unwrap();
    write_file_header(&mut file, &FileHeader::new()).unwrap();
    write_db_header(&mut file, 0, &DbHeader::EMPTY).unwrap();
    write_db_header(
        &mut file,
        1,
        &DbHeader {
            checksum: directory.checksum(),
            ..legacy.header().clone()
        },
    )
    .unwrap();
}
