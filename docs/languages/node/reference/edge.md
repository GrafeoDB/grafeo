---
title: JsEdge
description: Edge class reference for Node.js.
tags:
  - api
  - nodejs
---

# JsEdge

Represents a graph edge (relationship) returned from queries or `createEdge()`.

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `number` | Internal edge ID |
| `edgeType` | `string` | Edge type (relationship type) |
| `sourceId` | `number` | Source node ID |
| `targetId` | `number` | Target node ID |

## Methods

### get()

Get a property value by key. Returns `undefined` if the property doesn't exist.

```typescript
get(key: string): any
```

```typescript
const edge = db.getEdge(0);
console.log(edge.get('since'));  // 2024
```

### properties()

Get all properties as a plain object.

```typescript
properties(): object
```

```typescript
const props = edge.properties();
console.log(props);  // { since: 2024, weight: 0.8 }
```

### toString()

String representation of the edge.

```typescript
toString(): string
```

```typescript
console.log(edge.toString());  // '()-[:KNOWS]->() (id=0)'
```

## Example

```typescript
// Create an edge
const edge = db.createEdge(0, 1, 'KNOWS', { since: 2024 });
console.log(edge.edgeType);   // 'KNOWS'
console.log(edge.sourceId);   // 0
console.log(edge.targetId);   // 1

// From query results
const result = await db.execute('MATCH ()-[r:KNOWS]->() RETURN r');
for (const edge of result.edges()) {
  console.log(`${edge.sourceId} -[:${edge.edgeType}]-> ${edge.targetId}`);
}

// Property management
db.setEdgeProperty(edge.id, 'weight', 0.8);
db.removeEdgeProperty(edge.id, 'weight');
```
