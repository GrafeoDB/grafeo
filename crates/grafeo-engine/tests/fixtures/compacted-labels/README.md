# A compacted database with nodes of several labels

`tests/compacted_file_labels.rs` opens `0.5.44/labels.grafeo`, written by `write_fixture.py` with the released
`grafeo==0.5.44` wheel (the command is in the script). Up to 0.5.44, `compact()` stored a node with several labels
under one label, its labels joined with `|` (#595), and a write after it that matched such a node without a label
copied it to the overlay with that one label. The file holds both: nodes of several labels in the compacted base,
and overlay copies of two of them (Alix, given a property and a label, and Mia, the end of a new edge). The tests
describe what it holds. It goes with the 0.5.x readers in 0.7.0.
