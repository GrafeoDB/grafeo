package grafeo

/*
#include "grafeo.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// Graph works in one named graph of a database without switching the database's current
// graph. Each call looks the graph up again: once it is dropped, every call fails.
// Close frees it.
type Graph struct {
	handle *C.GrafeoGraph
}

// Graph returns a handle on the named graph of the current schema; an error when it does
// not exist.
func (db *Database) Graph(name string) (*Graph, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	runtime.LockOSThread()
	handle := C.grafeo_graph(db.handle, cName)
	if handle == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	g := &Graph{handle: handle}
	runtime.SetFinalizer(g, (*Graph).Close)
	return g, nil
}

// Close frees the handle; calling it again does nothing.
func (g *Graph) Close() {
	if g.handle != nil {
		C.grafeo_free_graph(g.handle)
		g.handle = nil
	}
	runtime.SetFinalizer(g, nil)
}

func (g *Graph) result(call func() *C.GrafeoResult) (*QueryResult, error) {
	runtime.LockOSThread()
	r := call()
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// Execute runs a GQL query in the graph.
func (g *Graph) Execute(query string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	return g.result(func() *C.GrafeoResult { return C.grafeo_graph_execute(g.handle, cQuery) })
}

// ExecuteParams runs a GQL query with parameters in the graph.
func (g *Graph) ExecuteParams(query string, params map[string]any) (*QueryResult, error) {
	return g.ExecuteLanguage("gql", query, params)
}

// ExecuteLanguage runs a query in a query language ("gql", "cypher", ...) with parameters
// (nil for none) in the graph.
func (g *Graph) ExecuteLanguage(language, query string, params map[string]any) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	cLanguage := C.CString(language)
	defer C.free(unsafe.Pointer(cLanguage))
	var cParams *C.char
	if params != nil {
		var err error
		if cParams, err = cJSON(params); err != nil {
			return nil, err
		}
		defer freeC(cParams)
	}
	return g.result(func() *C.GrafeoResult {
		return C.grafeo_graph_execute_language(g.handle, cLanguage, cQuery, cParams)
	})
}

func (g *Graph) created(call func() C.uint64_t) (uint64, error) {
	runtime.LockOSThread()
	id := uint64(call())
	if id == ^uint64(0) {
		err := lastError()
		runtime.UnlockOSThread()
		return 0, err
	}
	runtime.UnlockOSThread()
	return id, nil
}

func (g *Graph) flag(call func() C.int32_t) (bool, error) {
	runtime.LockOSThread()
	result := int(call())
	if result < 0 {
		err := lastError()
		runtime.UnlockOSThread()
		return false, err
	}
	runtime.UnlockOSThread()
	return result == 1, nil
}

// CreateNode creates a node with labels and optional properties in the graph.
func (g *Graph) CreateNode(labels []string, properties map[string]any) (*Node, error) {
	if labels == nil {
		labels = []string{}
	}
	cLabels, err := cJSON(labels)
	if err != nil {
		return nil, err
	}
	defer freeC(cLabels)
	var cProps *C.char
	if properties != nil {
		if cProps, err = cJSON(properties); err != nil {
			return nil, err
		}
		defer freeC(cProps)
	}
	id, err := g.created(func() C.uint64_t { return C.grafeo_graph_create_node(g.handle, cLabels, cProps) })
	if err != nil {
		return nil, err
	}
	if properties == nil {
		properties = make(map[string]any)
	}
	return &Node{ID: id, Labels: labels, Properties: properties}, nil
}

// CreateEdge creates an edge with optional properties in the graph.
func (g *Graph) CreateEdge(sourceID, targetID uint64, edgeType string, properties map[string]any) (*Edge, error) {
	cType := C.CString(edgeType)
	defer C.free(unsafe.Pointer(cType))
	var cProps *C.char
	if properties != nil {
		var err error
		if cProps, err = cJSON(properties); err != nil {
			return nil, err
		}
		defer freeC(cProps)
	}
	id, err := g.created(func() C.uint64_t {
		return C.grafeo_graph_create_edge(g.handle, C.uint64_t(sourceID), C.uint64_t(targetID), cType, cProps)
	})
	if err != nil {
		return nil, err
	}
	if properties == nil {
		properties = make(map[string]any)
	}
	return &Edge{ID: id, SourceID: sourceID, TargetID: targetID, Type: edgeType, Properties: properties}, nil
}

// GetNode returns a node of the graph: ErrNotFound when it does not exist.
func (g *Graph) GetNode(id uint64) (*Node, error) {
	var cNode *C.GrafeoNode
	if err := lockAndCheckStatus(func() C.GrafeoStatus {
		return C.grafeo_graph_get_node(g.handle, C.uint64_t(id), &cNode)
	}); err != nil {
		return nil, err
	}
	if cNode == nil {
		return nil, notFound("node", id)
	}
	return nodeFromC(cNode), nil
}

// GetEdge returns an edge of the graph: ErrNotFound when it does not exist.
func (g *Graph) GetEdge(id uint64) (*Edge, error) {
	var cEdge *C.GrafeoEdge
	if err := lockAndCheckStatus(func() C.GrafeoStatus {
		return C.grafeo_graph_get_edge(g.handle, C.uint64_t(id), &cEdge)
	}); err != nil {
		return nil, err
	}
	if cEdge == nil {
		return nil, notFound("edge", id)
	}
	return edgeFromC(cEdge), nil
}

func (g *Graph) setProperty(key string, value any, call func(cKey, cValue *C.char) C.GrafeoStatus) error {
	data, err := marshalJSON(value)
	if err != nil {
		return err
	}
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	cValue := C.CString(string(data))
	defer C.free(unsafe.Pointer(cValue))
	return lockAndCheckStatus(func() C.GrafeoStatus { return call(cKey, cValue) })
}

// SetNodeProperty sets a property of a node of the graph.
func (g *Graph) SetNodeProperty(id uint64, key string, value any) error {
	return g.setProperty(key, value, func(cKey, cValue *C.char) C.GrafeoStatus {
		return C.grafeo_graph_set_node_property(g.handle, C.uint64_t(id), cKey, cValue)
	})
}

// SetEdgeProperty sets a property of an edge of the graph.
func (g *Graph) SetEdgeProperty(id uint64, key string, value any) error {
	return g.setProperty(key, value, func(cKey, cValue *C.char) C.GrafeoStatus {
		return C.grafeo_graph_set_edge_property(g.handle, C.uint64_t(id), cKey, cValue)
	})
}

func (g *Graph) named(name string, call func(cName *C.char) C.int32_t) (bool, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return g.flag(func() C.int32_t { return call(cName) })
}

// RemoveNodeProperty removes a property of a node of the graph; whether it was there.
func (g *Graph) RemoveNodeProperty(id uint64, key string) (bool, error) {
	return g.named(key, func(cKey *C.char) C.int32_t {
		return C.grafeo_graph_remove_node_property(g.handle, C.uint64_t(id), cKey)
	})
}

// RemoveEdgeProperty removes a property of an edge of the graph; whether it was there.
func (g *Graph) RemoveEdgeProperty(id uint64, key string) (bool, error) {
	return g.named(key, func(cKey *C.char) C.int32_t {
		return C.grafeo_graph_remove_edge_property(g.handle, C.uint64_t(id), cKey)
	})
}

// AddNodeLabel adds a label to a node of the graph; false if it had it.
func (g *Graph) AddNodeLabel(id uint64, label string) (bool, error) {
	return g.named(label, func(cLabel *C.char) C.int32_t {
		return C.grafeo_graph_add_node_label(g.handle, C.uint64_t(id), cLabel)
	})
}

// RemoveNodeLabel removes a label from a node of the graph; whether it had it.
func (g *Graph) RemoveNodeLabel(id uint64, label string) (bool, error) {
	return g.named(label, func(cLabel *C.char) C.int32_t {
		return C.grafeo_graph_remove_node_label(g.handle, C.uint64_t(id), cLabel)
	})
}

// DeleteNode deletes a node of the graph (one with edges fails); whether it existed.
func (g *Graph) DeleteNode(id uint64) (bool, error) {
	return g.flag(func() C.int32_t { return C.grafeo_graph_delete_node(g.handle, C.uint64_t(id)) })
}

// DetachDeleteNode deletes a node of the graph with all its edges; whether it existed.
func (g *Graph) DetachDeleteNode(id uint64) (bool, error) {
	return g.flag(func() C.int32_t { return C.grafeo_graph_detach_delete_node(g.handle, C.uint64_t(id)) })
}

// DeleteEdge deletes an edge of the graph; whether it existed.
func (g *Graph) DeleteEdge(id uint64) (bool, error) {
	return g.flag(func() C.int32_t { return C.grafeo_graph_delete_edge(g.handle, C.uint64_t(id)) })
}

// BatchCreateNodesWithProps is Database.BatchCreateNodesWithProps in the graph.
func (g *Graph) BatchCreateNodesWithProps(labels []string, rows []map[string]any) ([]uint64, error) {
	cLabels, cRows, err := nodeRows(labels, rows)
	if err != nil {
		return nil, err
	}
	defer freeC(cLabels)
	defer freeC(cRows)
	return idsCall(func(ids **C.uint64_t, count *C.size_t) C.GrafeoStatus {
		return C.grafeo_graph_batch_create_nodes_with_props(g.handle, cLabels, cRows, ids, count)
	})
}

// BatchCreateEdges is Database.BatchCreateEdges in the graph.
func (g *Graph) BatchCreateEdges(edges []BatchEdge) ([]uint64, error) {
	if edges == nil {
		edges = []BatchEdge{}
	}
	cEdges, err := cJSON(edges)
	if err != nil {
		return nil, err
	}
	defer freeC(cEdges)
	return idsCall(func(ids **C.uint64_t, count *C.size_t) C.GrafeoStatus {
		return C.grafeo_graph_batch_create_edges(g.handle, cEdges, ids, count)
	})
}

// UpsertNodes is Database.UpsertNodes in the graph.
func (g *Graph) UpsertNodes(labels []string, rows []map[string]any, options *UpsertNodesOptions) (*UpsertSummary, error) {
	return upsertNodes(labels, rows, options, func(cLabels, cRows, cOptions *C.char, out **C.char) C.GrafeoStatus {
		return C.grafeo_graph_upsert_nodes(g.handle, cLabels, cRows, cOptions, out)
	})
}

// UpsertEdges is Database.UpsertEdges in the graph.
func (g *Graph) UpsertEdges(edgeType string, rows []map[string]any, options *UpsertEdgesOptions) (*UpsertSummary, error) {
	return upsertEdges(edgeType, rows, options, func(cType, cRows, cOptions *C.char, out **C.char) C.GrafeoStatus {
		return C.grafeo_graph_upsert_edges(g.handle, cType, cRows, cOptions, out)
	})
}
