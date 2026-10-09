//! Container format v3 (0.6.0): fixed little-endian headers, allocator,
//! directory, writer and reader.

pub mod alloc;
pub mod directory;
pub mod header;

pub mod cipher;
pub mod reader;
pub mod writer;

pub use cipher::ChunkCipher;
pub use reader::{ImageReader, SectionChunks};
pub use writer::CheckpointWriter;

#[cfg(test)]
mod tests {
    use std::fs::File;

    use grafeo_common::storage::{ChunkMeta, SectionSink, SectionSource, SectionType};

    use super::alloc::{PageAllocator, PageRun};
    use super::directory::{
        DirectoryEntry, ENTRY_CHUNK_OPTIONAL, ENTRY_SECTION_OPTIONAL, ENTRY_SIZE, encode_blocks,
        encode_blocks_raw,
    };
    use super::header::{BlockRef, PAGE_SIZE};
    use super::{CheckpointWriter, ChunkCipher, ImageReader};

    fn open(dir: &tempfile::TempDir) -> File {
        File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(dir.path().join("x"))
            .unwrap()
    }

    fn sorted(mut runs: Vec<PageRun>) -> Vec<PageRun> {
        runs.sort();
        runs
    }

    /// Writes `entries` as a directory at the end of the file and returns its root.
    fn write_directory(file: &mut File, entries: &[DirectoryEntry]) -> BlockRef {
        use std::io::{Seek, SeekFrom, Write};
        let at = file.metadata().unwrap().len().next_multiple_of(PAGE_SIZE);
        let (root, blocks) = encode_blocks(entries, |_| Ok(at)).unwrap();
        assert_eq!(blocks.len(), 1);
        file.seek(SeekFrom::Start(at)).unwrap();
        file.write_all(&blocks[0].1).unwrap();
        root
    }

    /// Writes encoded entries as a directory at the end of the file and
    /// returns its root.
    fn write_raw_directory(file: &mut File, entries: &[[u8; ENTRY_SIZE]]) -> BlockRef {
        use std::io::{Seek, SeekFrom, Write};
        let at = file.metadata().unwrap().len().next_multiple_of(PAGE_SIZE);
        let (root, blocks) = encode_blocks_raw(entries, |_| Ok(at)).unwrap();
        assert_eq!(blocks.len(), 1);
        file.seek(SeekFrom::Start(at)).unwrap();
        file.write_all(&blocks[0].1).unwrap();
        root
    }

    #[test]
    fn a_chunk_longer_than_the_file_is_refused_when_the_image_opens() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let mut writer =
            CheckpointWriter::new(&mut file, PageAllocator::from_used([]).unwrap(), None);
        writer.begin_section(SectionType::LpgStore, 1).unwrap();
        writer.write_chunk(ChunkMeta::raw(), b"Vincent").unwrap();
        let (good_root, _) = writer.finish().unwrap();
        let mut entry = ImageReader::open(&mut file, good_root, None)
            .unwrap()
            .entries()[0];
        entry.length = 1 << 40;
        let root = write_directory(&mut file, &[entry]);
        let error = ImageReader::open(&mut file, root, None)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("LpgStore") && error.contains(&entry.offset.to_string()),
            "{error}"
        );
        assert!(
            error.contains("lies beyond the end of the file") && !error.contains("  "),
            "one sentence without a run of spaces: {error}"
        );
    }

    #[test]
    fn a_misaligned_or_early_chunk_offset_is_refused_when_the_image_opens() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let mut writer =
            CheckpointWriter::new(&mut file, PageAllocator::from_used([]).unwrap(), None);
        writer.begin_section(SectionType::LpgStore, 1).unwrap();
        writer.write_chunk(ChunkMeta::raw(), &[3u8; 100]).unwrap();
        let (good_root, _) = writer.finish().unwrap();
        let good = ImageReader::open(&mut file, good_root, None)
            .unwrap()
            .entries()[0];
        for offset in [good.offset + 1, 4096] {
            let entry = DirectoryEntry { offset, ..good };
            let root = write_directory(&mut file, &[entry]);
            let error = ImageReader::open(&mut file, root, None)
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("LpgStore") && error.contains(&offset.to_string()),
                "{error}"
            );
        }
    }

    #[test]
    fn an_image_round_trips_its_sections() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let mut writer =
            CheckpointWriter::new(&mut file, PageAllocator::from_used([]).unwrap(), None);
        writer.begin_section(SectionType::Catalog, 2).unwrap();
        writer.write_chunk(ChunkMeta::raw(), b"Alix").unwrap();
        writer.begin_section(SectionType::LpgStore, 1).unwrap();
        writer
            .write_chunk(ChunkMeta::raw(), &vec![19u8; 9000])
            .unwrap(); // three pages
        let (root, runs) = writer.finish().unwrap();
        assert_eq!(
            runs.iter().map(|r| r.count).sum::<u64>(),
            1 + 3 + 1,
            "two chunks and one directory block"
        );

        let reader = ImageReader::open(&mut file, root, None).unwrap();
        let catalog = reader.section(SectionType::Catalog).unwrap();
        assert_eq!(&catalog.fetch(0).unwrap()[..], b"Alix");
        assert_eq!(
            reader
                .section(SectionType::LpgStore)
                .unwrap()
                .fetch(0)
                .unwrap()
                .len(),
            9000
        );
        assert!(reader.section(SectionType::RdfStore).is_none());
    }

    #[test]
    fn a_changed_chunk_byte_fails_its_crc() {
        use std::io::{Seek, SeekFrom, Write};
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let mut writer =
            CheckpointWriter::new(&mut file, PageAllocator::from_used([]).unwrap(), None);
        writer.begin_section(SectionType::LpgStore, 1).unwrap();
        writer
            .write_chunk(ChunkMeta::raw(), &vec![19u8; 9000])
            .unwrap();
        let (root, _) = writer.finish().unwrap();
        let offset = ImageReader::open(&mut file, root, None).unwrap().entries()[0].offset;
        file.seek(SeekFrom::Start(offset + 4100)).unwrap();
        file.write_all(&[88]).unwrap();
        let reader = ImageReader::open(&mut file, root, None).unwrap();
        let error = reader
            .section(SectionType::LpgStore)
            .unwrap()
            .fetch(0)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("LpgStore") && error.contains(&offset.to_string()),
            "{error}"
        );
    }

    #[test]
    fn a_chunk_without_bytes_takes_no_pages_and_reads_back_empty() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let mut writer =
            CheckpointWriter::new(&mut file, PageAllocator::from_used([]).unwrap(), None);
        writer.begin_section(SectionType::Catalog, 1).unwrap();
        writer.write_chunk(ChunkMeta::raw(), b"").unwrap();
        let (root, runs) = writer.finish().unwrap();
        assert_eq!(
            runs.iter().map(|r| r.count).sum::<u64>(),
            1,
            "directory only"
        );
        let reader = ImageReader::open(&mut file, root, None).unwrap();
        assert_eq!(reader.entries()[0].offset, 0);
        assert_eq!(reader.entries()[0].length, 0);
        assert!(
            reader
                .section(SectionType::Catalog)
                .unwrap()
                .fetch(0)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn writing_a_chunk_before_a_section_begins_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let mut writer =
            CheckpointWriter::new(&mut file, PageAllocator::from_used([]).unwrap(), None);
        assert!(writer.write_chunk(ChunkMeta::raw(), b"Mia").is_err());
    }

    #[test]
    fn a_section_type_begins_at_most_once_per_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let mut writer =
            CheckpointWriter::new(&mut file, PageAllocator::from_used([]).unwrap(), None);
        writer.begin_section(SectionType::Catalog, 1).unwrap();
        writer.write_chunk(ChunkMeta::raw(), b"Mia").unwrap();
        writer.begin_section(SectionType::LpgStore, 1).unwrap();
        let error = writer
            .begin_section(SectionType::Catalog, 2)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("Catalog") && error.contains("already"),
            "{error}"
        );
        assert!(
            writer.write_chunk(ChunkMeta::raw(), b"Paris").is_err(),
            "after the refusal no section is current, so the chunk cannot land in LpgStore"
        );
        let (root, _) = writer.finish().unwrap();
        let reader = ImageReader::open(&mut file, root, None).unwrap();
        let catalog = reader.section(SectionType::Catalog).unwrap();
        assert_eq!(catalog.chunks().len(), 1, "only the first Catalog chunk");
        assert_eq!(&catalog.fetch(0).unwrap()[..], b"Mia");
        assert!(reader.section(SectionType::LpgStore).is_none());
    }

    #[test]
    fn fetching_past_the_last_chunk_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let mut writer =
            CheckpointWriter::new(&mut file, PageAllocator::from_used([]).unwrap(), None);
        writer.begin_section(SectionType::Catalog, 1).unwrap();
        writer.write_chunk(ChunkMeta::raw(), b"Gus").unwrap();
        let (root, _) = writer.finish().unwrap();
        let reader = ImageReader::open(&mut file, root, None).unwrap();
        let section = reader.section(SectionType::Catalog).unwrap();
        assert_eq!(section.chunks().len(), 1);
        assert!(section.fetch(1).is_err());
    }

    #[test]
    fn a_section_of_many_chunks_spans_several_directory_blocks() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let mut writer =
            CheckpointWriter::new(&mut file, PageAllocator::from_used([]).unwrap(), None);
        writer.begin_section(SectionType::LpgStore, 1).unwrap();
        for i in 0..1400u32 {
            let meta = ChunkMeta {
                row_start: u64::from(i) * 88,
                row_count: 88,
                ..ChunkMeta::raw()
            };
            writer.write_chunk(meta, &i.to_le_bytes()).unwrap();
        }
        let (root, runs) = writer.finish().unwrap();
        // 1400 chunks of one page each, plus two directory blocks (1364 entries
        // fit one: 16 pages, then one page for the other 36).
        assert_eq!(runs.len(), 1400 + 2);
        assert_eq!(runs.iter().map(|r| r.count).sum::<u64>(), 1400 + 16 + 1);
        let reader = ImageReader::open(&mut file, root, None).unwrap();
        assert_eq!(sorted(reader.used_runs()), sorted(runs));
        let section = reader.section(SectionType::LpgStore).unwrap();
        assert_eq!(section.chunks().len(), 1400);
        for i in 0..1400u32 {
            assert_eq!(&section.fetch(i as usize).unwrap()[..], &i.to_le_bytes());
        }
    }

    /// The plaintext chunks of each section of one image.
    type Image = [(SectionType, Vec<Vec<u8>>)];

    fn write_image(
        file: &mut File,
        pages: PageAllocator,
        cipher: Option<&ChunkCipher>,
        image: &Image,
    ) -> (BlockRef, Vec<PageRun>) {
        let mut writer = CheckpointWriter::new(file, pages, cipher);
        for (section_type, chunks) in image {
            writer.begin_section(*section_type, 1).unwrap();
            // Each chunk of a section has an identity of its own.
            for (row_start, chunk) in (0..).zip(chunks) {
                let meta = ChunkMeta {
                    row_start,
                    ..ChunkMeta::raw()
                };
                writer.write_chunk(meta, chunk).unwrap();
            }
        }
        writer.finish().unwrap()
    }

    fn assert_reads_back(
        file: &mut File,
        root: BlockRef,
        cipher: Option<&ChunkCipher>,
        image: &Image,
        name: &str,
    ) {
        let reader = ImageReader::open(file, root, cipher).unwrap();
        for (section_type, chunks) in image {
            let section = reader.section(*section_type).unwrap();
            assert_eq!(
                section.chunks().len(),
                chunks.len(),
                "{name}: {section_type:?}"
            );
            for (index, chunk) in chunks.iter().enumerate() {
                assert_eq!(
                    &section.fetch(index).unwrap()[..],
                    &chunk[..],
                    "{name}: chunk {index} of {section_type:?}"
                );
            }
        }
    }

    fn overlap(a: PageRun, b: PageRun) -> bool {
        a.first < b.end().unwrap() && b.first < a.end().unwrap()
    }

    /// Writes image A, then image B while A is active, and checks that B never
    /// touches A's pages, reuses the gaps A leaves before growing the file, and
    /// that both images read back.
    fn a_new_image_never_touches_the_active_one(cipher: Option<&ChunkCipher>) {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        // An older image at pages 3..8, which A replaces, so B may reuse its pages.
        let older = [
            (SectionType::Catalog, vec![vec![3u8; 9000]]),
            (SectionType::LpgStore, vec![b"Gus".to_vec()]),
        ];
        let (_, older_runs) = write_image(
            &mut file,
            PageAllocator::from_used([]).unwrap(),
            cipher,
            &older,
        );
        let a = [
            (SectionType::Catalog, vec![b"Alix".to_vec()]),
            (SectionType::LpgStore, vec![vec![19u8; 5000]]),
        ];
        let (a_root, a_runs) = write_image(
            &mut file,
            PageAllocator::from_used(older_runs).unwrap(),
            cipher,
            &a,
        );
        {
            let reader = ImageReader::open(&mut file, a_root, cipher).unwrap();
            assert_eq!(sorted(reader.used_runs()), sorted(a_runs.clone()));
        }
        let a_bytes = std::fs::read(dir.path().join("x")).unwrap();
        let a_end = a_runs.iter().map(|run| run.end().unwrap()).max().unwrap();
        assert_eq!(a_end, 12, "A uses pages 8..12");

        let pages = PageAllocator::from_used(a_runs.clone()).unwrap();
        assert_eq!(
            pages.free_runs(),
            [PageRun { first: 3, count: 5 }],
            "the older image's pages are A's gap"
        );
        // The last chunk (six pages) cannot fit what is left of the gap.
        let b = [
            (SectionType::Catalog, vec![b"Vincent".to_vec()]),
            (
                SectionType::LpgStore,
                vec![vec![88u8; 9000], vec![3u8; 24_000]],
            ),
        ];
        let (b_root, b_runs) = write_image(&mut file, pages, cipher, &b);

        let now = std::fs::read(dir.path().join("x")).unwrap();
        for run in &a_runs {
            let start = usize::try_from(run.offset().unwrap()).unwrap();
            let end = usize::try_from(run.end().unwrap() * PAGE_SIZE)
                .unwrap()
                .min(a_bytes.len());
            assert_eq!(
                now[start..end],
                a_bytes[start..end],
                "A's run {run:?} still holds A's bytes"
            );
        }
        for b_run in &b_runs {
            for a_run in &a_runs {
                assert!(!overlap(*b_run, *a_run), "B {b_run:?} overlaps A {a_run:?}");
            }
        }
        assert_eq!(
            sorted(b_runs.clone()),
            [
                PageRun { first: 3, count: 1 },
                PageRun { first: 4, count: 3 },
                PageRun { first: 7, count: 1 },
                PageRun {
                    first: 12,
                    count: 6
                },
            ],
            "B fills A's gap (pages 3..8) first and grows the file only for the chunk that does not fit"
        );

        assert_reads_back(&mut file, a_root, cipher, &a, "image A");
        assert_reads_back(&mut file, b_root, cipher, &b, "image B");
    }

    #[test]
    fn a_new_plain_image_never_touches_the_active_one() {
        a_new_image_never_touches_the_active_one(None);
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn a_new_encrypted_image_never_touches_the_active_one() {
        let alix = grafeo_common::encryption::PageEncryptor::new(&[3u8; 32]);
        a_new_image_never_touches_the_active_one(Some(&alix));
    }

    #[test]
    fn a_file_truncated_inside_its_last_directory_block_fails_naming_the_block() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let mut writer =
            CheckpointWriter::new(&mut file, PageAllocator::from_used([]).unwrap(), None);
        writer.begin_section(SectionType::Catalog, 1).unwrap();
        writer.write_chunk(ChunkMeta::raw(), b"Amsterdam").unwrap();
        let (root, _) = writer.finish().unwrap();
        assert_eq!(
            file.metadata().unwrap().len(),
            root.offset + u64::from(root.length),
            "the directory block ends the file"
        );
        file.set_len(root.offset + u64::from(root.length) / 2)
            .unwrap();
        let error = ImageReader::open(&mut file, root, None)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(&format!("directory block at offset {}", root.offset)),
            "{error}"
        );
    }

    #[test]
    fn a_directory_whose_runs_overlap_is_refused_when_the_image_opens() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let mut writer =
            CheckpointWriter::new(&mut file, PageAllocator::from_used([]).unwrap(), None);
        writer.begin_section(SectionType::Catalog, 1).unwrap();
        writer.write_chunk(ChunkMeta::raw(), &[3u8; 5000]).unwrap();
        writer.begin_section(SectionType::LpgStore, 1).unwrap();
        writer.write_chunk(ChunkMeta::raw(), b"Jules").unwrap();
        let (good_root, _) = writer.finish().unwrap();
        let good = ImageReader::open(&mut file, good_root, None)
            .unwrap()
            .entries()
            .to_vec();

        // The second chunk on the page the next directory block goes to (the
        // first loop pass writes it there).
        let mut into_directory = good.clone();
        into_directory[1].offset = file.metadata().unwrap().len().next_multiple_of(PAGE_SIZE);
        into_directory[1].length = 1;
        // The second chunk inside the first chunk's second page.
        let mut into_chunk = good.clone();
        into_chunk[1].offset = good[0].offset + PAGE_SIZE;
        for entries in [into_directory, into_chunk] {
            let root = write_directory(&mut file, &entries);
            let error = ImageReader::open(&mut file, root, None)
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(error.contains("overlaps"), "{error}");
        }
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn an_encrypted_image_needs_its_key() {
        use grafeo_common::encryption::PageEncryptor;
        let alix = PageEncryptor::new(&[3u8; 32]);
        let gus = PageEncryptor::new(&[19u8; 32]);
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let mut writer = CheckpointWriter::new(
            &mut file,
            PageAllocator::from_used([]).unwrap(),
            Some(&alix),
        );
        writer.begin_section(SectionType::Catalog, 1).unwrap();
        writer.write_chunk(ChunkMeta::raw(), b"Alix").unwrap();
        let (root, _) = writer.finish().unwrap();
        let reader = ImageReader::open(&mut file, root, Some(&alix)).unwrap();
        assert_eq!(
            &reader
                .section(SectionType::Catalog)
                .unwrap()
                .fetch(0)
                .unwrap()[..],
            b"Alix"
        );
        let stored = reader.entries()[0];
        assert_eq!(stored.length, 4 + 28, "stored bytes carry nonce and tag");
        drop(reader);
        let wrong = ImageReader::open(&mut file, root, Some(&gus))
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(wrong.contains("decrypt"), "{wrong}");
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn an_encrypted_directory_that_ends_on_a_page_boundary_reports_its_extra_page() {
        use grafeo_common::encryption::PageEncryptor;
        let alix = PageEncryptor::new(&[3u8; 32]);
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let mut writer = CheckpointWriter::new(
            &mut file,
            PageAllocator::from_used([]).unwrap(),
            Some(&alix),
        );
        writer.begin_section(SectionType::LpgStore, 1).unwrap();
        // 170 entries: a 8192 byte block, stored as 8220 bytes over three pages.
        for i in 0..170u32 {
            let meta = ChunkMeta {
                row_start: u64::from(i),
                ..ChunkMeta::raw()
            };
            writer.write_chunk(meta, &i.to_le_bytes()).unwrap();
        }
        let (root, runs) = writer.finish().unwrap();
        let reader = ImageReader::open(&mut file, root, Some(&alix)).unwrap();
        assert_eq!(sorted(reader.used_runs()), sorted(runs));
    }

    /// An image with a section of a newer version (type 250, optional) and a
    /// chunk of a newer kind in a known section (LpgStore, kind 250,
    /// optional), next to two known chunks.
    fn image_with_foreign_entries(
        file: &mut File,
        cipher: Option<&ChunkCipher>,
        section_flags: u8,
        kind_flags: u8,
    ) -> BlockRef {
        let mut writer = CheckpointWriter::new(file, PageAllocator::from_used([]).unwrap(), cipher);
        writer.begin_section(SectionType::Catalog, 2).unwrap();
        writer.write_chunk(ChunkMeta::meta(), b"Alix").unwrap();
        writer
            .write_foreign_chunk(250, 0, section_flags, &vec![19u8; 9000])
            .unwrap();
        writer.begin_section(SectionType::LpgStore, 3).unwrap();
        writer.write_chunk(ChunkMeta::meta(), b"Gus").unwrap();
        writer
            .write_foreign_chunk(SectionType::LpgStore.to_u8(), 250, kind_flags, b"Vincent")
            .unwrap();
        writer.finish().unwrap().0
    }

    #[test]
    fn an_image_with_unknown_optional_entries_opens_and_spares_their_pages() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let root = image_with_foreign_entries(
            &mut file,
            None,
            ENTRY_SECTION_OPTIONAL,
            ENTRY_CHUNK_OPTIONAL,
        );
        let reader = ImageReader::open(&mut file, root, None).unwrap();
        assert_eq!(
            &reader
                .section(SectionType::Catalog)
                .unwrap()
                .fetch(0)
                .unwrap()[..],
            b"Alix"
        );
        assert_eq!(
            reader.section(SectionType::LpgStore).unwrap().chunks(),
            [ChunkMeta::meta()],
            "the section reader never sees the skipped chunk"
        );
        let skipped: Vec<(u8, u8)> = reader
            .skipped()
            .iter()
            .map(|e| (e.section_type, e.kind))
            .collect();
        assert_eq!(skipped, [(250, 0), (2, 250)]);
        let pages: u64 = reader.used_runs().iter().map(|run| run.count).sum();
        assert_eq!(
            pages,
            1 + 3 + 1 + 1 + 1,
            "two known chunks, the two skipped ones and the directory block"
        );
        let mut next = PageAllocator::from_used(reader.used_runs()).unwrap();
        assert!(
            next.free_runs().is_empty(),
            "the skipped chunks leave no free page in the image"
        );
        // One page: it would fit the gap a skipped chunk left out of the
        // used pages would leave.
        let run = next.allocate(1).unwrap();
        assert!(
            reader.skipped().iter().all(|e| {
                let used = e.run();
                used.first + used.count <= run.first || run.first + run.count <= used.first
            }),
            "a checkpoint never writes over a skipped chunk while its image is active"
        );
        assert!(
            reader.entries().iter().all(|entry| entry.flags == 0),
            "the writer of this release marks every entry required"
        );
    }

    #[test]
    fn an_unknown_required_entry_is_refused_naming_it() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let root = image_with_foreign_entries(&mut file, None, 0, ENTRY_CHUNK_OPTIONAL);
        let error = ImageReader::open(&mut file, root, None)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("section type 250") && error.contains("required"),
            "{error}"
        );
        let mut file = open(&dir);
        let root = image_with_foreign_entries(&mut file, None, ENTRY_SECTION_OPTIONAL, 0);
        let error = ImageReader::open(&mut file, root, None)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("chunk kind 250") && error.contains("LpgStore"),
            "{error}"
        );
    }

    /// A skipped chunk is held to the placement rules of a known one: page
    /// aligned, in the data area, inside the file and on pages of its own.
    #[test]
    fn a_misplaced_skipped_chunk_is_refused_when_the_image_opens() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let root = image_with_foreign_entries(
            &mut file,
            None,
            ENTRY_SECTION_OPTIONAL,
            ENTRY_CHUNK_OPTIONAL,
        );
        let (known, foreign) = {
            let reader = ImageReader::open(&mut file, root, None).unwrap();
            (reader.entries().to_vec(), reader.skipped()[0])
        };
        let file_length = file.metadata().unwrap().len();
        for (offset, length, expected) in [
            (foreign.offset + 1, foreign.length, "misplaced"),
            (PAGE_SIZE, foreign.length, "misplaced"),
            (foreign.offset, 1 << 40, "beyond the end of the file"),
            (known[0].offset, foreign.length, "overlaps"),
        ] {
            let mut raw: Vec<[u8; ENTRY_SIZE]> = known
                .iter()
                .map(|entry| {
                    let mut bytes = [0u8; ENTRY_SIZE];
                    entry.encode(&mut bytes);
                    bytes
                })
                .collect();
            let mut moved = [0u8; ENTRY_SIZE];
            moved[0] = foreign.section_type;
            moved[2] = foreign.kind;
            moved[24..32].copy_from_slice(&offset.to_le_bytes());
            moved[32..40].copy_from_slice(&length.to_le_bytes());
            moved[44] = foreign.flags;
            raw.push(moved);
            let root = write_raw_directory(&mut file, &raw);
            let error = ImageReader::open(&mut file, root, None)
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(expected),
                "skipped chunk at offset {offset}, length {length} (file {file_length} bytes): \
                 {error}"
            );
        }
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn the_optional_bits_survive_an_encrypted_image() {
        use grafeo_common::encryption::PageEncryptor;
        let alix = PageEncryptor::new(&[3u8; 32]);
        let gus = PageEncryptor::new(&[19u8; 32]);
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let root = image_with_foreign_entries(
            &mut file,
            Some(&alix),
            ENTRY_SECTION_OPTIONAL,
            ENTRY_CHUNK_OPTIONAL,
        );
        let reader = ImageReader::open(&mut file, root, Some(&alix)).unwrap();
        // The bits come back from the decrypted directory block: they are part
        // of the entry, which the block's authentication covers, not of the
        // chunk's payload.
        assert_eq!(
            reader.skipped().iter().map(|e| e.flags).collect::<Vec<_>>(),
            [ENTRY_SECTION_OPTIONAL, ENTRY_CHUNK_OPTIONAL]
        );
        assert_eq!(
            &reader
                .section(SectionType::Catalog)
                .unwrap()
                .fetch(0)
                .unwrap()[..],
            b"Alix"
        );
        drop(reader);
        let mut file = open(&dir);
        let root = image_with_foreign_entries(&mut file, Some(&alix), 0, ENTRY_CHUNK_OPTIONAL);
        let error = ImageReader::open(&mut file, root, Some(&alix))
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("section type 250"), "{error}");
        let error = ImageReader::open(&mut file, root, Some(&gus))
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("decrypt"), "{error}");
    }

    /// A foreign chunk of an encrypted image is encrypted as a newer version
    /// would encrypt it: with the associated data of its on-disk numbers.
    #[cfg(feature = "encryption")]
    #[test]
    fn a_foreign_chunk_is_encrypted_with_the_associated_data_of_its_numbers() {
        use grafeo_common::encryption::PageEncryptor;

        use super::cipher::chunk_aad_parts;

        let alix = PageEncryptor::new(&[3u8; 32]);
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let root = image_with_foreign_entries(
            &mut file,
            Some(&alix),
            ENTRY_SECTION_OPTIONAL,
            ENTRY_CHUNK_OPTIONAL,
        );
        let foreign = ImageReader::open(&mut file, root, Some(&alix))
            .unwrap()
            .skipped()[1];
        let bytes = std::fs::read(dir.path().join("x")).unwrap();
        let start = usize::try_from(foreign.offset).unwrap();
        let stored = &bytes[start..start + usize::try_from(foreign.length).unwrap()];
        assert_eq!(
            alix.decrypt(stored, &chunk_aad_parts(2, 250, 0, 0, 0, 0))
                .unwrap(),
            b"Vincent"
        );
    }

    /// The writer binds a chunk to its section type, kind, namespace, graph,
    /// column and first row in exactly the bytes every encrypted image uses:
    /// the stored chunk decrypts with the literal associated data.
    #[cfg(feature = "encryption")]
    #[test]
    fn the_writer_encrypts_a_chunk_with_the_associated_data_of_its_place() {
        use grafeo_common::encryption::PageEncryptor;

        let alix = PageEncryptor::new(&[3u8; 32]);
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let meta = ChunkMeta::column(3, 19, 88, 2, 0);
        let mut writer = CheckpointWriter::new(
            &mut file,
            PageAllocator::from_used([]).unwrap(),
            Some(&alix),
        );
        writer.begin_section(SectionType::LpgStore, 3).unwrap();
        writer.write_chunk(meta, b"Prague").unwrap();
        let (root, _) = writer.finish().unwrap();
        let reader = ImageReader::open(&mut file, root, Some(&alix)).unwrap();
        let section = reader.section(SectionType::LpgStore).unwrap();
        assert_eq!(section.chunks(), [meta]);
        assert_eq!(&section.fetch(0).unwrap()[..], b"Prague", "it reads back");
        let entry = reader.entries()[0];
        drop(reader);
        let bytes = std::fs::read(dir.path().join("x")).unwrap();
        let start = usize::try_from(entry.offset).unwrap();
        let stored = &bytes[start..start + usize::try_from(entry.length).unwrap()];
        assert_eq!(
            alix.decrypt(stored, b"grafeo-chunk:2:2:0:3:19:88").unwrap(),
            b"Prague",
            "section type 2, kind 2 (column), namespace 0, graph 3, column 19, first row 88"
        );
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn the_same_plaintext_at_the_same_offset_is_stored_differently_each_time() {
        use grafeo_common::encryption::PageEncryptor;
        let alix = PageEncryptor::new(&[3u8; 32]);
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let mut stored = Vec::new();
        for _ in 0..2 {
            let mut writer = CheckpointWriter::new(
                &mut file,
                PageAllocator::from_used([]).unwrap(),
                Some(&alix),
            );
            writer.begin_section(SectionType::Catalog, 1).unwrap();
            writer.write_chunk(ChunkMeta::raw(), b"Berlin").unwrap();
            let (root, _) = writer.finish().unwrap();
            let reader = ImageReader::open(&mut file, root, Some(&alix)).unwrap();
            let entry = reader.entries()[0];
            drop(reader);
            stored.push((entry.offset, entry.crc));
        }
        assert_eq!(stored[0].0, stored[1].0, "same offset");
        assert_ne!(stored[0].1, stored[1].1, "different stored bytes");
    }
}
