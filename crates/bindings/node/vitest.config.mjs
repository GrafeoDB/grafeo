import { defineConfig } from 'vitest/config'

// Many tests create, close, back up, restore and reopen database files, and
// each of those syncs to disk. On one Windows CI runner every such test ran 4
// to 8 times slower than on another runner of the same job (index.spec 16.1 s
// against 2.7 s, a backup test 5.4 s against 0.47 s) while the in-memory tests
// took the same time, so a test past vitest's 5 s default failed for the disk
// alone. The tests themselves take well under a second: a timeout of 30 s
// still catches a test that hangs.
export default defineConfig({
  test: {
    testTimeout: 30_000,
  },
})
