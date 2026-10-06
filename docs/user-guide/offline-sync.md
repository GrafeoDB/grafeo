---
title: Offline-First Sync
description: Sync a local Grafeo database with a remote grafeo-server instance using the pull/push changefeed protocol.
tags:
  - sync
  - offline-first
  - cdc
  - dart
  - flutter
  - wasm
---

# Offline-First Sync

!!! warning "Feature flag requirements"
    Offline sync depends on several feature flags:

    - **grafeo (engine):** the `cdc` feature must be enabled for change tracking. It is included in the `ai` and `embedded` profiles.
    - **grafeo-server:** the `replication` feature must be enabled, and the server must be started with `--replication-mode primary` for the primary instance (or `--replication-mode replica` for replicas).
    - **grafeo-server:** the `sync` feature must be enabled for the `POST /db/{name}/sync` endpoint.

    The default grafeo-server build includes all of these. Custom builds must opt in explicitly.

Grafeo supports offline-first applications: a local instance (embedded in a Dart/Flutter app or
running as WASM in the browser) can accumulate changes while disconnected, then sync bidirectionally
with a remote **grafeo-server** when connectivity is restored.

The sync protocol is built on two HTTP endpoints provided by grafeo-server:

| Endpoint | Direction | Purpose |
|----------|-----------|---------|
| `GET /db/{name}/changes?since={epoch}` | Server to client | Pull change events since a known epoch |
| `POST /db/{name}/sync` | Client to server | Push local changes, receive conflicts and ID mappings |

## How it Works

```
Client (Dart / WASM)          grafeo-server
       |                            |
       |   GET /changes?since=0     |
       |<---------------------------|  (initial hydration)
       |                            |
       | ... offline, makes changes |
       |                            |
       |   POST /sync               |
       |--------------------------->|  (push local changes)
       |<---------------------------| (applied, skipped, conflicts, id_mappings)
       |                            |
       |   GET /changes?since=N     |
       |<---------------------------|  (pull any server-side changes)
```

**Epoch tracking:** every change event carries a `server_epoch` value. Store the epoch returned
by the last pull. Pass it as `since` on your next poll. If `changes.len() == limit`, there may
be more events available: poll again using the epoch of the last event you received.

## Wire Protocol

### Pull: `GET /db/{name}/changes`

**Query parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `since` | `0` | Return events where epoch >= this value. Pass `0` for full history. |
| `limit` | `1000` | Maximum events per response (max `10000`). |

**Response:**

```json
{
  "server_epoch": 42,
  "changes": [
    {
      "id": 1,
      "entity_type": "node",
      "kind": "create",
      "epoch": 1,
      "timestamp": 1720000000000,
      "labels": ["Person"],
      "after": { "name": { "String": "Alix" } }
    },
    {
      "id": 2,
      "entity_type": "edge",
      "kind": "create",
      "epoch": 2,
      "timestamp": 1720000001000,
      "edge_type": "KNOWS",
      "src_id": 1,
      "dst_id": 3
    }
  ]
}
```

`entity_type` is `"node"`, `"edge"`, or `"triple"` (RDF). `kind` is `"create"`, `"update"`, or
`"delete"`. Property values use grafeo's tagged JSON encoding: `{"String": "Alix"}`,
`{"Int64": 42}`, `{"Float64": 3.14}`, `{"Bool": true}`.

### Push: `POST /db/{name}/sync`

**Request body:**

```json
{
  "client_id": "device-abc123",
  "last_seen_epoch": 42,
  "changes": [
    {
      "kind": "create",
      "entity_type": "node",
      "labels": ["Person"],
      "after": { "name": { "String": "Gus" } },
      "timestamp": 1720000010000
    },
    {
      "kind": "update",
      "entity_type": "node",
      "id": 5,
      "after": { "status": { "String": "active" } },
      "timestamp": 1720000010500
    },
    {
      "kind": "delete",
      "entity_type": "node",
      "id": 7,
      "timestamp": 1720000011000
    }
  ]
}
```

- **Creates**: omit `id`; the server assigns a new ID and returns it in `id_mappings`.
- **Updates**: include `id` (server-side). The `after` object is a property delta (only changed keys).
- **Deletes**: include `id`. The `timestamp` is used for last-write-wins conflict detection.

**Response:**

```json
{
  "server_epoch": 45,
  "applied": 2,
  "skipped": 1,
  "conflicts": [
    {
      "request_index": 1,
      "reason": "server_newer"
    }
  ],
  "id_mappings": [
    {
      "request_index": 0,
      "server_id": 11
    }
  ]
}
```

`id_mappings` maps each create request (by its zero-based index in `changes`) to the
server-assigned entity ID. Use these to update your local ID→server-ID table.

## Timestamps and Ordering

### Hybrid Logical Clock (HLC)

*Since 0.5.32.* CDC events use a Hybrid Logical Clock timestamp instead of raw wall-clock time.
Each `timestamp` field is a 64-bit value packing physical milliseconds (48 bits) and a logical
counter (16 bits):

```text
┌──────────────────────────────────┬──────────┐
│  physical ms (48 bits)           │ logical  │
│  milliseconds since Unix epoch   │ (16 bits)│
└──────────────────────────────────┴──────────┘
```

The logical counter increments when multiple events occur within the same millisecond, guaranteeing
**strict monotonic ordering** even under clock skew. This means:

- Events from the same node are always totally ordered
- `timestamp_a < timestamp_b` implies event A happened before event B (on the same node)
- Cross-node ordering uses the physical component for last-write-wins resolution

For display, extract the physical component: `physical_ms = timestamp >> 16`.

### Session-driven CDC

*Since 0.5.32.* Mutations made through query sessions (`INSERT`, `SET`, `DELETE` via GQL, Cypher,
or any supported language) now generate CDC events. Previously, only direct API mutations
(`create_node`, `set_node_property`) were tracked.

CDC events are buffered during a transaction and flushed atomically on commit. If a transaction
rolls back, its CDC events are discarded. This guarantees that the change feed reflects only
committed state.

### Epoch monotonicity

The `epoch` field in change events is strictly monotonic: `changes_between(from, to)` returns
events with no gaps, no duplicates, and strictly increasing epoch values. This is enforced by
stress tests with 5 concurrent writers.

### Atomic sync apply

*Since 0.5.32.* When a replica applies incoming changes via `POST /db/{name}/sync`, all mutations
are wrapped in a single session transaction. Readers on the replica never see a partially applied
batch: either all changes become visible (on commit) or none (on rollback or crash).

### Persistent replica epoch

*Since 0.5.32.* Replicas that run with `--data-dir` persist the last-applied epoch per database to
`{data_dir}/.replica-epochs`. After a restart, the replica resumes pulling from where it left off
instead of re-fetching from epoch 0.

### CDC auto-activation

*Since 0.5.32.* CDC is opt-in per database in the engine. The server automatically enables CDC on
all databases when running as a replication primary (`--replication-mode primary`), so that all
mutations (both direct API and session-driven) produce change events for replicas to consume.

## Conflict Resolution

The server uses **last-write-wins (LWW)**: HLC timestamps are compared. If the server has a CDC
record for the target entity with a `timestamp` strictly greater than the client's
`change.timestamp`, the server's version wins and the client change is skipped. Skipped changes
appear in `conflicts`.

Create operations are never conflicted: the server always assigns a new ID.

## ID Mapping Workflow

When your local app creates a node offline, it uses a temporary local ID. After syncing, you
receive a `server_id`. You must update all local references (including edges that reference this
node as `src_id`/`dst_id`) before pushing those edges.

Recommended order for a batch sync:

1. Push all node creates first, collect `id_mappings`.
2. Remap `src_id`/`dst_id` in pending edge creates using the mapping table.
3. Push edge creates with corrected IDs.
4. Push updates and deletes.

## Dart / Flutter Example

This example uses `grafeo_dart` (FFI bindings) for the local graph and the `http` package for
sync calls. Store `last_epoch` in `SharedPreferences` so syncing survives app restarts.

```dart
import 'package:grafeo_dart/grafeo_dart.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

const serverUrl = 'http://your-server:7474';
const dbName    = 'default';
const clientId  = 'my-device-id';  // unique per installation

class GrafeoSyncManager {
  final GrafeoDB localDb;

  GrafeoSyncManager(this.localDb);

  Future<int> _loadLastEpoch() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getInt('grafeo_last_epoch') ?? 0;
  }

  Future<void> _saveLastEpoch(int epoch) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt('grafeo_last_epoch', epoch);
  }

  /// Pull server changes and apply them to the local database.
  Future<void> pull() async {
    final lastEpoch = await _loadLastEpoch();
    final uri = Uri.parse(
      '$serverUrl/db/$dbName/changes?since=$lastEpoch&limit=1000',
    );

    final resp = await http.get(uri);
    if (resp.statusCode != 200) throw Exception('pull failed: ${resp.body}');

    final body = jsonDecode(resp.body) as Map<String, dynamic>;
    final changes = body['changes'] as List<dynamic>;

    for (final event in changes) {
      _applyChangeLocally(event as Map<String, dynamic>);
    }

    await _saveLastEpoch(body['server_epoch'] as int);
  }

  void _applyChangeLocally(Map<String, dynamic> event) {
    final kind       = event['kind'] as String;
    final entityType = event['entity_type'] as String;

    if (entityType == 'node') {
      if (kind == 'create') {
        // Simplified: create node with labels, ignore id (server-assigned)
        final labels = (event['labels'] as List?)?.cast<String>() ?? [];
        localDb.createNode(labels);
      } else if (kind == 'delete') {
        localDb.deleteNode(event['id'] as int);
      }
      // updates: apply property delta from event['after']
    }
  }

  /// Push local pending changes to the server.
  ///
  /// [pendingChanges] is the list of local mutations to replay.
  Future<Map<String, int>> push(List<Map<String, dynamic>> pendingChanges) async {
    final lastEpoch = await _loadLastEpoch();
    final body = jsonEncode({
      'client_id':       clientId,
      'last_seen_epoch': lastEpoch,
      'changes':         pendingChanges,
    });

    final resp = await http.post(
      Uri.parse('$serverUrl/db/$dbName/sync'),
      headers: {'Content-Type': 'application/json'},
      body: body,
    );
    if (resp.statusCode != 200) throw Exception('push failed: ${resp.body}');

    final result = jsonDecode(resp.body) as Map<String, dynamic>;

    // Build local-id to server-id map from id_mappings
    final mappings = <int, int>{};
    for (final m in result['id_mappings'] as List) {
      final mi = m as Map<String, dynamic>;
      mappings[mi['request_index'] as int] = mi['server_id'] as int;
    }

    await _saveLastEpoch(result['server_epoch'] as int);
    return mappings;
  }
}

// Usage in a widget / lifecycle hook
void onAppResumed(GrafeoSyncManager sync) async {
  try {
    await sync.pull();
    // await sync.push(localPendingChanges);
  } catch (e) {
    // Offline: ignore, retry on next resume
  }
}
```

### Tracking Local Changes in Dart

The simplest approach is to maintain a pending-changes queue alongside your local graph:

```dart
final List<Map<String, dynamic>> _pendingChanges = [];

void createPersonOffline(String name) {
  final tempId = localDb.createNode(['Person']);
  localDb.setNodeProperty(tempId, 'name', name);

  _pendingChanges.add({
    'kind':        'create',
    'entity_type': 'node',
    'labels':      ['Person'],
    'after':       {'name': {'String': name}},
    'timestamp':   DateTime.now().millisecondsSinceEpoch,
    '_temp_id':    tempId,  // not sent to server; used for local remapping
  });
}
```

After a successful push, use the returned `id_mappings` to remap local edges before pushing them.

## WASM / Browser Example

This example uses `@grafeo-db/wasm` for an in-memory local graph in the browser and the
`fetch` API for sync calls. `localStorage` stores the last epoch.

```typescript
import init, { GrafeoDB } from '@grafeo-db/wasm';

const SERVER_URL = 'http://your-server:7474';
const DB_NAME    = 'default';
const CLIENT_ID  = crypto.randomUUID(); // store in localStorage for persistence

await init();
const localDb = new GrafeoDB();

// --- Pull ---
async function pull(): Promise<void> {
  const lastEpoch = parseInt(localStorage.getItem('grafeo_last_epoch') ?? '0', 10);

  const resp = await fetch(
    `${SERVER_URL}/db/${DB_NAME}/changes?since=${lastEpoch}&limit=1000`,
  );
  if (!resp.ok) throw new Error(`pull failed: ${resp.status}`);

  const body = await resp.json();

  for (const event of body.changes) {
    applyChangeLocally(event);
  }

  localStorage.setItem('grafeo_last_epoch', String(body.server_epoch));
}

function applyChangeLocally(event: any): void {
  if (event.entity_type === 'node') {
    if (event.kind === 'create') {
      // localDb.createNode(event.labels ?? []);
    } else if (event.kind === 'delete') {
      // localDb.deleteNode(BigInt(event.id));
    }
  }
}

// --- Push ---
interface SyncChange {
  kind: string;
  entity_type: string;
  id?: number;
  labels?: string[];
  edge_type?: string;
  src_id?: number;
  dst_id?: number;
  after?: Record<string, any>;
  timestamp: number;
}

async function push(pendingChanges: SyncChange[]): Promise<Map<number, number>> {
  const lastEpoch = parseInt(localStorage.getItem('grafeo_last_epoch') ?? '0', 10);

  const resp = await fetch(`${SERVER_URL}/db/${DB_NAME}/sync`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      client_id:       CLIENT_ID,
      last_seen_epoch: lastEpoch,
      changes:         pendingChanges,
    }),
  });
  if (!resp.ok) throw new Error(`push failed: ${resp.status}`);

  const result = await resp.json();
  localStorage.setItem('grafeo_last_epoch', String(result.server_epoch));

  // Map request_index -> server_id
  const idMap = new Map<number, number>();
  for (const m of result.id_mappings) {
    idMap.set(m.request_index, m.server_id);
  }

  if (result.conflicts.length > 0) {
    console.warn('Sync conflicts:', result.conflicts);
  }

  return idMap;
}

// --- Service Worker: buffer pushes while offline ---
// In your service worker (sw.ts), intercept failed sync requests and
// store them in IndexedDB. Replay on the 'sync' event when back online:
//
// self.addEventListener('sync', (event: SyncEvent) => {
//   if (event.tag === 'grafeo-sync') {
//     event.waitUntil(replayPendingPushes());
//   }
// });
```

### Using the Background Sync API

Register a background sync so the browser retries the push automatically:

```typescript
async function scheduleSync(): Promise<void> {
  if ('serviceWorker' in navigator && 'sync' in ServiceWorkerRegistration.prototype) {
    const reg = await navigator.serviceWorker.ready;
    await reg.sync.register('grafeo-sync');
  } else {
    // Fallback: try immediately, ignore failure
    try { await push(getPendingChanges()); } catch { /* offline */ }
  }
}
```

## Handling Conflicts

Inspect `response.conflicts` after every push:

```typescript
for (const conflict of result.conflicts) {
  const change = pendingChanges[conflict.request_index];

  if (conflict.reason === 'server_newer') {
    // Server has a more recent version: discard local change or surface to the user
    console.log(`Discarding stale update for entity ${change.id}`);
  } else {
    // Structural error (missing field, unknown entity type, etc.)
    console.error(`Sync error at index ${conflict.request_index}: ${conflict.reason}`);
  }
}
```

Common `reason` values:

| Reason | Meaning |
|--------|---------|
| `server_newer` | Server has a CDC timestamp > client's `timestamp`; client change skipped |
| `update_missing_id` | Update request did not include `id` |
| `update_missing_after` | Update request had no `after` properties |
| `delete_missing_id` | Delete request did not include `id` |
| `edge_create_missing_src_dst_or_type` | Edge create missing `src_id`, `dst_id`, or `edge_type` |

## Further Reading

- [grafeo-server ecosystem page](../ecosystem/grafeo-server.md)
- [Change Data Capture](observability.md) (CDC feature flag)
- [Temporal queries](temporal.md) (time-travel reads using epochs)
