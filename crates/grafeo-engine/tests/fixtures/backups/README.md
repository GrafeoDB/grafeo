# A backup chain under a bincode manifest

`tests/backup_restore.rs` restores these. Up to 0.5.44 the backup manifest, `backup_manifest.json`, held bincode,
despite its name; 0.6 writes it as JSON and still reads the bincode manifest, so a backup directory of 0.5.x restores
and takes new backups. Each directory holds one backup directory, written by `write_fixture.py` with the released
wheel (the command is in the script):

- `0.5.44/`: written by the `grafeo==0.5.44` wheel (a bincode manifest and a 0.5.x full backup).

The manifest is kept as `backup_manifest.bincode`, so the end-of-file and whitespace hooks, which take a `.json` file
for text, leave its bytes alone; the tests copy it back as `backup_manifest.json`.

The source database went through these steps, each followed by its backup into the chain:

1. Alix (Amsterdam) and Gus (Berlin), and Alix `KNOWS` Gus (since 2019); a full backup, `backup_full_0000.grafeo`,
   at epoch 3.
2. Vincent (Paris); an incremental backup, `backup_incr_0001.wal`, of epoch 4.
3. Mia (Prague) at epoch 5, Mia `KNOWS` Alix (since 1988) at 6, and Gus's city set to Barcelona at 7; an incremental
   backup, `backup_incr_0002.wal`, of epochs 5 to 7.

So a restore to epoch 3 finds Alix and Gus, to epoch 4 also Vincent, and to epoch 7 all four, Gus in Barcelona and
both edges. Regenerate it only to add content; it goes with the 0.5.x readers in 0.7.0.
