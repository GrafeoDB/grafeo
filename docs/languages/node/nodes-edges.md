---
title: Nodes & Edges
description: CRUD operations on nodes and edges in Node.js.
tags:
  - node
  - nodes
  - edges
  - crud
---

# Nodes & Edges

## Creating Nodes

```typescript
// Create with labels and properties
const node = db.createNode(['Person'], { name: 'Alix', age: 30 });
console.log(node.id);         // 0
console.log(node.labels);     // ['Person']
console.log(node.get('name')); // 'Alix'
```

Properties can be `null`, to create a node with only labels:

```typescript
const node = db.createNode(['Tag']);
```

## Reading Nodes

```typescript
const node = db.getNode(0);
if (node) {
  console.log(node.id);
  console.log(node.labels);
  console.log(node.get('name'));
  console.log(node.properties()); // all properties as object
  console.log(node.hasLabel('Person')); // true
}
```

## Updating Node Properties

```typescript
db.setNodeProperty(0, 'age', 31);
db.removeNodeProperty(0, 'age'); // returns true if existed
```

## Labels

```typescript
db.addNodeLabel(0, 'Employee');    // returns true if added
db.removeNodeLabel(0, 'Employee'); // returns true if removed
db.getNodeLabels(0);               // ['Person'] or null if not found
```

## Deleting Nodes

```typescript
db.deleteNode(0); // returns true if the node existed
```

To delete a node and all its edges, use a query:

```typescript
await db.execute('MATCH (p:Person {name: "Alix"}) DETACH DELETE p');
```

## Creating Edges

```typescript
const edge = db.createEdge(0, 1, 'KNOWS', { since: 2024 });
console.log(edge.id);        // 0
console.log(edge.edgeType);  // 'KNOWS'
console.log(edge.sourceId);  // 0
console.log(edge.targetId);  // 1
```

## Reading Edges

```typescript
const edge = db.getEdge(0);
if (edge) {
  console.log(edge.edgeType);
  console.log(edge.get('since'));
  console.log(edge.properties());
}
```

## Updating Edge Properties

```typescript
db.setEdgeProperty(0, 'since', 2025);
db.removeEdgeProperty(0, 'since');
```

## Deleting Edges

```typescript
db.deleteEdge(0); // returns true if existed
```

## String Representation

Both `JsNode` and `JsEdge` have a `toString()` method for debugging:

```typescript
console.log(node.toString()); // (:Person {id: 0})
console.log(edge.toString()); // ()-[:KNOWS]->() (id=0)
```
