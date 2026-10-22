# LPG sections of earlier releases

The readers of the LPG sections no build writes any more read these: the chunks of version 3 (0.6.0-beta.1, read by
`graph/lpg/chunked.rs`) and the block layout of 0.5.x (one raw chunk, read by `graph/lpg/section.rs` and
`graph/lpg/block.rs`). Both load into the row-group store, so the tests compare what a file holds with a row-group
store built by the same writes (the builders in the tests that read each fixture).

They were written by `LpgStore`'s section writers on commit `06d3f24b` (release/0.6.0, before 0.6.0 deleted
`LpgStore`); they cannot be written again. The `history*` fixtures were written by a build with `temporal`, which
gave each value its epoch and wrote the older versions as `History` chunks.

A `.chunks` file is one section of version 3 as a list of chunks, little-endian: `LPG3`, a chunk count (`u32`), then
per chunk its kind, namespace and codec (a byte each), graph (`u32`), column (`u32`), first row (`u64`), row count
(`u32`), the length of its bytes (`u32`) and the bytes. `graph::load::testing::version_3_fixture` reads one into a
memory image. A `.bin` file is the bytes of a 0.5.x section (its one raw chunk).

| File | Graph | Caps |
| --- | --- | --- |
| `kinds.chunks`, `kinds_caps_4_512.chunks`, `kinds_caps_1_64.chunks` | every value kind, nodes of 0, 1 and 3 labels, two edge types, named graphs `trips`, `""` and `travel/__default__` (empty) | default, 4 rows and 512 bytes, 1 row and 64 bytes |
| `beta_caps_2_64.chunks` | people with a deleted node (an id gap), a value set twice, a named graph `trips` | 2 rows and 64 bytes |
| `beta_dangling_edge.chunks` | the same with an edge to node 88, which the graph does not hold | default |
| `beta_0_5.bin` | the same people, in the 0.5.x block layout | |
| `sample_0_5.bin` | the value kinds the block layout keeps, a named graph | |
| `unused_names.chunks` | labels and edge types no row uses any more, in the default graph and in `trips` | default |
| `dropped_graph.chunks` | graphs `trips` (1) and `archive` (3), `travel` (2) dropped | default |
| `sparse_caps_4.chunks`, `far_ids_caps_4.chunks` | ids with gaps across row groups of 4; one node at id 1000 | 4 rows |
| `retired_ids.chunks` | deleted nodes 3 and 4 and edge 1 | default |
| `history.chunks`, `history_caps_1_64.chunks` | values with older versions at epochs 3, 19 and 88, a value removed last | default, 1 row and 64 bytes |
