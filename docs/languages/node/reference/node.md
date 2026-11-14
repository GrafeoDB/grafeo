---
title: JsNode
description: Node class reference for Node.js.
tags:
  - api
  - nodejs
---

# JsNode

Represents a graph node returned from queries or `createNode()`.

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `number` | Internal node ID |
| `labels` | `string[]` | Node labels |

## Methods

### get()

Get a property value by key. Returns `undefined` if the property doesn't exist.

```typescript
get(key: string): any
```

```typescript
const node = db.getNode(0);
console.log(node.get('name'));  // 'Alix'
console.log(node.get('missing'));  // undefined
```

### properties()

Get all properties as a plain object.

```typescript
properties(): object
```

```typescript
const props = node.properties();
console.log(props);  // { name: 'Alix', age: 30 }
```

### hasLabel()

Check if the node has a specific label.

```typescript
hasLabel(label: string): boolean
```

```typescript
node.hasLabel('Person');  // true
node.hasLabel('Animal');  // false
```

### toString()

String representation of the node.

```typescript
toString(): string
```

```typescript
console.log(node.toString());  // '(:Person {id: 0})'
```

## Example

```typescript
// From query results
const result = await db.execute('MATCH (p:Person) RETURN p');
for (const node of result.nodes()) {
  console.log(`${node.id}: ${node.labels.join(':')} - ${node.get('name')}`);
}

// Direct creation
const node = db.createNode(['Person', 'Employee'], { name: 'Alix', age: 30 });
console.log(node.id);
console.log(node.labels);
console.log(node.properties());

// Label management
db.addNodeLabel(node.id, 'Manager');
db.removeNodeLabel(node.id, 'Employee');
console.log(db.getNodeLabels(node.id));  // ['Person', 'Manager']
```
