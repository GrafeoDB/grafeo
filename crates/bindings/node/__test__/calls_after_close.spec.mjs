// Every call on a closed database fails.
//
// `close()` releases what the database holds. Every later call on it fails
// with `GRAFEO-T007` (a thrown error, or a rejected promise for an async
// method), reads, getters and `memoryUsage()` included, and so does every
// call on what it handed out before the close: sessions, transactions,
// result streams, graph handles and time-travel views. Only `close()`
// answers, as a no-op. Before 0.6.4 reads went on after `close()`, and an
// in-memory database kept taking writes.

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, it, expect, afterEach } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const dirs = []

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    try { fs.rmSync(dir, { recursive: true, force: true }) } catch { /* ignore */ }
  }
})

function tempDir() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'grafeo-closed-'))
  dirs.push(dir)
  return dir
}

function openDatabase(kind, dir) {
  const db = kind === 'persistent' ? GrafeoDB.create(path.join(dir, 'amsterdam.grafeo')) : GrafeoDB.create()
  const alix = db.createNode(['Person'], { name: 'Alix' })
  const gus = db.createNode(['Person'], { name: 'Gus' })
  db.createEdge(alix.id, gus.id, 'KNOWS')
  return db
}

/** The public methods and getters of a class, as its prototype holds them. */
function publicNames(cls) {
  return new Set(Object.getOwnPropertyNames(cls.prototype).filter((name) => name !== 'constructor'))
}

/** The rows whose call did not fail with `GRAFEO-T007`, by name. */
async function failures(calls, available) {
  const missed = []
  for (const [name, call] of Object.entries(calls).sort(([a], [b]) => a.localeCompare(b))) {
    if (!available.has(name)) continue
    try {
      await call()
      missed.push(`${name}: answered`)
    } catch (error) {
      if (error?.errorCode !== 'GRAFEO-T007') {
        missed.push(`${name}: ${error?.errorCode ?? ''} ${error?.message ?? error}`)
      }
    }
  }
  return missed
}

function databaseCalls(db, dir) {
  const vector = [0.1, 0.2, 0.3, 0.4]
  const query = 'MATCH (p:Person) RETURN p.name'
  return {
    currentEpoch: () => db.currentEpoch(),
    asOf: () => db.asOf(0n),
    executeAsOf: () => db.executeAsOf(query, 0n),
    epochAsOf: () => db.epochAsOf(0n),
    execute: () => db.execute(query),
    executeStream: () => db.executeStream(query),
    executeCypher: () => db.executeCypher(query),
    executeSql: () => db.executeSql('SELECT * FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))'),
    executeGremlin: () => db.executeGremlin('g.V()'),
    executeGraphql: () => db.executeGraphql('{ Person { name } }'),
    executeSparql: () => db.executeSparql('SELECT ?s WHERE { ?s ?p ?o }'),
    executeLanguage: () => db.executeLanguage('gql', query),
    createNode: () => db.createNode(['Person'], { name: 'Vincent' }),
    createEdge: () => db.createEdge(0, 1, 'KNOWS'),
    getNode: () => db.getNode(0),
    getNodeEdges: () => db.getNodeEdges(0),
    getEdge: () => db.getEdge(0),
    deleteNode: () => db.deleteNode(0),
    detachDeleteNode: () => db.detachDeleteNode(0),
    deleteEdge: () => db.deleteEdge(0),
    setNodeProperty: () => db.setNodeProperty(0, 'city', 'Amsterdam'),
    setEdgeProperty: () => db.setEdgeProperty(0, 'since', 2020),
    nodeCount: () => db.nodeCount(),
    edgeCount: () => db.edgeCount(),
    beginTransaction: () => db.beginTransaction(),
    session: () => db.session(),
    createVectorIndex: () => db.createVectorIndex('Item', 'embedding', 4),
    batchCreateNodes: () => db.batchCreateNodes('Item', 'embedding', [vector]),
    removeNodeProperty: () => db.removeNodeProperty(0, 'name'),
    removeEdgeProperty: () => db.removeEdgeProperty(0, 'since'),
    addNodeLabel: () => db.addNodeLabel(0, 'Director'),
    removeNodeLabel: () => db.removeNodeLabel(0, 'Person'),
    getNodeLabels: () => db.getNodeLabels(0),
    memoryUsage: () => db.memoryUsage(),
    openWarnings: () => db.openWarnings(),
    info: () => db.info(),
    schema: () => db.schema(),
    version: () => db.version(),
    clearPlanCache: () => db.clearPlanCache(),
    setCheckpointTriggers: () => db.setCheckpointTriggers({ intervalMs: 60000 }),
    checkpointStatus: () => db.checkpointStatus(),
    walCheckpoint: () => db.walCheckpoint(),
    setSchema: () => db.setSchema('berlin'),
    resetSchema: () => db.resetSchema(),
    currentSchema: () => db.currentSchema(),
    createProjection: () => db.createProjection('people', ['Person']),
    dropProjection: () => db.dropProjection('people'),
    listProjections: () => db.listProjections(),
    vectorSearch: () => db.vectorSearch('Item', 'embedding', vector, 1),
    dropVectorIndex: () => db.dropVectorIndex('Item', 'embedding'),
    rebuildVectorIndex: () => db.rebuildVectorIndex('Item', 'embedding'),
    batchVectorSearch: () => db.batchVectorSearch('Item', 'embedding', [vector], 1),
    mmrSearch: () => db.mmrSearch('Item', 'embedding', vector, 1),
    save: () => db.save(path.join(dir, 'copy.grafeo')),
    backupFull: () => db.backupFull(path.join(dir, 'backups')),
    backupIncremental: () => db.backupIncremental(path.join(dir, 'backups')),
    releaseBackupCursor: () => db.releaseBackupCursor(),
    createTextIndex: () => db.createTextIndex('Person', 'name'),
    dropTextIndex: () => db.dropTextIndex('Person', 'name'),
    rebuildTextIndex: () => db.rebuildTextIndex('Person', 'name'),
    textSearch: () => db.textSearch('Person', 'name', 'Alix', 1),
    hybridSearch: () => db.hybridSearch('Item', 'name', 'embedding', 'Alix', 1),
    registerEmbeddingModel: () =>
      db.registerEmbeddingModel('model', path.join(dir, 'model.onnx'), path.join(dir, 'tokenizer.json')),
    embedText: () => db.embedText('model', ['Amsterdam']),
    vectorSearchText: () => db.vectorSearchText('Item', 'embedding', 'model', 'Amsterdam', 1),
    compact: () => db.compact(),
    enableCdc: () => db.enableCdc(),
    disableCdc: () => db.disableCdc(),
    isCdcEnabled: () => db.isCdcEnabled,
    nodeHistory: () => db.nodeHistory(0),
    edgeHistory: () => db.edgeHistory(0),
    nodeHistorySince: () => db.nodeHistorySince(0, 0),
    changesBetween: () => db.changesBetween(0, 1),
    importCsv: () => db.importCsv(path.join(dir, 'rows.csv')),
    importJsonl: () => db.importJsonl(path.join(dir, 'rows.jsonl')),
    batchCreateNodesWithProps: () => db.batchCreateNodesWithProps('Person', [{ name: 'Mia' }]),
    batchCreateEdges: () => db.batchCreateEdges([{ src: 0, dst: 1, type: 'KNOWS' }]),
    graph: () => db.graph('berlin'),
    upsertNodes: () => db.upsertNodes(['Person'], [{ id: 1, name: 'Jules' }]),
    upsertEdges: () => db.upsertEdges('KNOWS', [{ id: 1, src: 1, dst: 2 }]),
  }
}

function graphHandleCalls(graph) {
  const query = 'MATCH (p:Person) RETURN p.name'
  return {
    name: () => graph.name,
    execute: () => graph.execute(query),
    executeLanguage: () => graph.executeLanguage('gql', query),
    beginTransaction: () => graph.beginTransaction(),
    createNode: () => graph.createNode(['Person'], { name: 'Butch' }),
    createEdge: () => graph.createEdge(0, 1, 'KNOWS'),
    getNode: () => graph.getNode(0),
    getNodeEdges: () => graph.getNodeEdges(0),
    getEdge: () => graph.getEdge(0),
    setNodeProperty: () => graph.setNodeProperty(0, 'city', 'Berlin'),
    setEdgeProperty: () => graph.setEdgeProperty(0, 'since', 2020),
    removeNodeProperty: () => graph.removeNodeProperty(0, 'name'),
    removeEdgeProperty: () => graph.removeEdgeProperty(0, 'since'),
    addNodeLabel: () => graph.addNodeLabel(0, 'Director'),
    removeNodeLabel: () => graph.removeNodeLabel(0, 'Person'),
    deleteNode: () => graph.deleteNode(0),
    detachDeleteNode: () => graph.detachDeleteNode(0),
    deleteEdge: () => graph.deleteEdge(0),
    batchCreateNodesWithProps: () => graph.batchCreateNodesWithProps('Person', [{ name: 'Mia' }]),
    batchCreateEdges: () => graph.batchCreateEdges([{ src: 0, dst: 1, type: 'KNOWS' }]),
    upsertNodes: () => graph.upsertNodes(['Person'], [{ id: 1, name: 'Hans' }]),
    upsertEdges: () => graph.upsertEdges('KNOWS', [{ id: 1, src: 1, dst: 2 }]),
  }
}

describe('a closed database', () => {
  for (const kind of ['persistent', 'in-memory']) {
    it(`fails every method with GRAFEO-T007 (${kind})`, async () => {
      const dir = tempDir()
      const db = openDatabase(kind, dir)
      db.close()

      const calls = databaseCalls(db, dir)
      const available = publicNames(GrafeoDB)
      const unlisted = [...available].filter((name) => !(name in calls) && name !== 'close')
      expect(unlisted, 'public methods without a row in this table').toEqual([])
      expect(await failures(calls, available)).toEqual([])
    })
  }

  it('names its code and what to do', async () => {
    const db = openDatabase('persistent', tempDir())
    db.close()

    const error = await db.execute('MATCH (p:Person) RETURN p.name').catch((caught) => caught)
    expect(error).toBeInstanceOf(Error)
    expect(error.errorCode).toBe('GRAFEO-T007')
    expect(error.isRetryable).toBe(false)
    expect(error.message).toMatch(/the database is closed: open it again/)
  })

  it('closes twice without an error', () => {
    const db = openDatabase('persistent', tempDir())
    db.close()
    expect(() => db.close()).not.toThrow()
  })

  it('wrote nothing a call after its close tried', async () => {
    const dir = tempDir()
    const dbPath = path.join(dir, 'prague.grafeo')
    const db = GrafeoDB.create(dbPath)
    db.createNode(['Person'], { name: 'Alix' })
    db.close()

    await expect(db.execute("INSERT (:Person {name: 'Gus'})")).rejects.toThrow()
    expect(() => db.createNode(['Person'], { name: 'Vincent' })).toThrow()
    expect(() => db.save(path.join(dir, 'copy.grafeo'))).toThrow()
    expect(fs.existsSync(path.join(dir, 'copy.grafeo'))).toBe(false)

    const reopened = GrafeoDB.open(dbPath)
    expect(reopened.nodeCount()).toBe(1)
    reopened.close()
  })
})

describe('what a closed database handed out', () => {
  it('fails every method of a graph handle held across close()', async () => {
    const dir = tempDir()
    const db = openDatabase('persistent', dir)
    await db.execute('CREATE GRAPH berlin')
    const graph = db.graph('berlin')
    db.close()

    const calls = graphHandleCalls(graph)
    const available = publicNames(graph.constructor)
    // `toString` answers, so a closed handle still prints.
    const unlisted = [...available].filter((name) => !(name in calls) && name !== 'toString')
    expect(unlisted, 'public methods without a row in this table').toEqual([])
    expect(await failures(calls, available)).toEqual([])
  })

  it('fails every method of the sessions, transactions, streams and views obtained before close()', async () => {
    const dir = tempDir()
    const dbPath = path.join(dir, 'amsterdam.grafeo')
    const db = openDatabase('persistent', dir)
    const session = db.session()
    const transaction = db.beginTransaction()
    await transaction.execute("INSERT (:Person {name: 'Mia'})")
    const view = db.asOf(db.currentEpoch())
    const stream = await db.executeStream('MATCH (p:Person) RETURN p.name')
    db.close()

    const query = 'MATCH (p:Person) RETURN p.name'
    const sql = 'SELECT * FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))'
    const kinds = {
      Session: [session, {
        execute: () => session.execute(query),
        executeLanguage: () => session.executeLanguage('gql', query),
        executeStream: () => session.executeStream(query),
        beginTransaction: () => session.beginTransaction(),
        executeCypher: () => session.executeCypher(query),
        executeSql: () => session.executeSql(sql),
        executeGremlin: () => session.executeGremlin('g.V()'),
        executeGraphql: () => session.executeGraphql('{ Person { name } }'),
        executeSparql: () => session.executeSparql('SELECT ?s WHERE { ?s ?p ?o }'),
      }],
      Transaction: [transaction, {
        execute: () => transaction.execute(query),
        isActive: () => transaction.isActive,
        executeCypher: () => transaction.executeCypher(query),
        executeSql: () => transaction.executeSql(sql),
        executeGremlin: () => transaction.executeGremlin('g.V()'),
        executeGraphql: () => transaction.executeGraphql('{ Person { name } }'),
        executeSparql: () => transaction.executeSparql('SELECT ?s WHERE { ?s ?p ?o }'),
        commit: () => transaction.commit(),
        rollback: () => transaction.rollback(),
      }],
      AsOfView: [view, {
        epoch: () => view.epoch,
        execute: () => view.execute(query),
        getNode: () => view.getNode(0),
        getEdge: () => view.getEdge(0),
        getNodeProperty: () => view.getNodeProperty(0, 'name'),
      }],
      ResultStream: [stream, {
        columns: () => stream.columns,
        warnings: () => stream.warnings,
        next: () => stream.next(),
      }],
    }
    for (const [kind, [object, calls]] of Object.entries(kinds)) {
      // Each one's own close() answers, as a no-op.
      const available = publicNames(object.constructor)
      const unlisted = [...available].filter((name) => !(name in calls) && name !== 'close')
      expect(unlisted, `${kind}: public methods without a row in this table`).toEqual([])
      expect(await failures(calls, available), kind).toEqual([])
    }

    const reopened = GrafeoDB.open(dbPath)
    expect(reopened.nodeCount(), 'the transaction open across close() left nothing').toBe(2)
    reopened.close()
  })
})
