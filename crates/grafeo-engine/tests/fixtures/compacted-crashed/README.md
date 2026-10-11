# A compacted database whose process exited without close()

`tests/compacted_file_in_every_build.rs` opens `0.5.44/crashed.grafeo` and its sidecar WAL `crashed.grafeo.wal/`,
written by `write_fixture.py` with the released `grafeo==0.5.44` wheel (the command is in the script). The file holds a
compacted base of five people and three `KNOWS` edges, checkpointed after `compact()`. The WAL holds the direct calls
made after it: Alix's city changed, Gus given the label `Employee`, Vincent's city removed, the edge from Gus to Vincent
and Butch deleted, and Jules created with an edge to Mia. The query after it (Django) is in neither: 0.5.44 did not log
queries after `compact()` (#558).

0.5.44 itself replayed that WAL before it wired the base, so its reopen kept Jules and his edge but lost every change
to a base node or edge; 0.6 folds the base first. The tests describe what the file holds. It goes with the 0.5.x
readers in 0.7.0.
