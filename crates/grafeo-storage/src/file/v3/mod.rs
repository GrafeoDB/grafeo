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
    use super::directory::{DirectoryEntry, encode_blocks};
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
            for chunk in chunks {
                writer.write_chunk(ChunkMeta::raw(), chunk).unwrap();
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
        a.first < b.end() && b.first < a.end()
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
        let a_end = a_runs.iter().map(|run| run.end()).max().unwrap();
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
            let start = usize::try_from(run.offset()).unwrap();
            let end = usize::try_from(run.end() * PAGE_SIZE)
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
            writer
                .write_chunk(ChunkMeta::raw(), &i.to_le_bytes())
                .unwrap();
        }
        let (root, runs) = writer.finish().unwrap();
        let reader = ImageReader::open(&mut file, root, Some(&alix)).unwrap();
        assert_eq!(sorted(reader.used_runs()), sorted(runs));
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
