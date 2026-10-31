// Persona: a first-time user copies the README's Node.js example verbatim.
//
// The example of the Node.js README and the project README run as written
// with plain Node (not vitest, which rewrites imports), in a new project that
// depends on @grafeo-db/js, and print the rows their comments promise.

import { execFileSync } from 'node:child_process'
import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, unlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, it, expect, afterAll } from 'vitest'

const here = dirname(fileURLToPath(import.meta.url))
const packageDir = join(here, '..')
const root = join(packageDir, '..', '..', '..')

const project = mkdtempSync(join(tmpdir(), 'grafeo-first-time-'))
mkdirSync(join(project, 'node_modules', '@grafeo-db'), { recursive: true })
const link = join(project, 'node_modules', '@grafeo-db', 'js')
symlinkSync(packageDir, link, 'junction')
afterAll(() => {
  // The link first, so removing the project never reaches the package.
  unlinkSync(link)
  rmSync(project, { recursive: true, force: true })
})

function firstExample(readme) {
  const text = readFileSync(readme, 'utf8')
  // `\r?\n`: a Windows checkout ends the README's lines with CRLF.
  const blocks = [...text.matchAll(/```(?:javascript|js|typescript|ts)\r?\n([\s\S]*?)```/g)].map((m) => m[1])
  return blocks.find((block) => block.includes('GrafeoDB.create'))
}

function runAs(file, code) {
  writeFileSync(join(project, file), code)
  return execFileSync(process.execPath, [file], { cwd: project, encoding: 'utf8', stdio: 'pipe' })
}

describe('persona first-time user (Node.js)', () => {
  it("runs the Node.js README's example as an ES module", () => {
    const code = firstExample(join(packageDir, 'README.md'))
    const out = runAs('readme.mjs', code)
    expect(out).toContain("'p.name': 'Alix'")
    expect(out).toContain("'p.name': 'Gus'")
  })

  it("runs the project README's Node.js example as written", () => {
    const code = firstExample(join(root, 'README.md'))
    // A file named as the package would name it: index.js.
    const out = runAs('index.js', code)
    expect(out).toContain("'friend.name': 'Gus'")
  })
})
