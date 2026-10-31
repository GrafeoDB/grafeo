package grafeo

/*
#include "grafeo.h"
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"runtime"
	"strings"
	"unsafe"
)

// WriteCounters tells what a statement wrote; all zero for a read.
type WriteCounters struct {
	NodesCreated uint64
	NodesDeleted uint64
	EdgesCreated uint64
	// EdgesDeleted also counts the edges DETACH DELETE removed.
	EdgesDeleted uint64
	// PropertiesSet counts values written or removed, also those of created entities.
	PropertiesSet uint64
	LabelsAdded   uint64
	LabelsRemoved uint64
}

func countersOf(r *C.GrafeoResult) WriteCounters {
	return WriteCounters{
		NodesCreated:  uint64(C.grafeo_result_nodes_created(r)),
		NodesDeleted:  uint64(C.grafeo_result_nodes_deleted(r)),
		EdgesCreated:  uint64(C.grafeo_result_edges_created(r)),
		EdgesDeleted:  uint64(C.grafeo_result_edges_deleted(r)),
		PropertiesSet: uint64(C.grafeo_result_properties_set(r)),
		LabelsAdded:   uint64(C.grafeo_result_labels_added(r)),
		LabelsRemoved: uint64(C.grafeo_result_labels_removed(r)),
	}
}

// BatchEdge is one edge of BatchCreateEdges.
type BatchEdge struct {
	Src        uint64         `json:"src"`
	Dst        uint64         `json:"dst"`
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties,omitempty"`
}

// ChangeEvent is one recorded change of a node or edge (change data capture).
type ChangeEvent struct {
	EntityID   uint64 `json:"entity_id"`
	EntityType string `json:"entity_type"` // "node" or "edge"
	// Graph is nil for the default graph.
	Graph *string `json:"graph"`
	Kind  string  `json:"kind"` // "create", "update" or "delete"
	Epoch uint64  `json:"epoch"`
	// Timestamp is a hybrid logical clock value.
	Timestamp    uint64         `json:"timestamp"`
	Before       map[string]any `json:"before"`
	After        map[string]any `json:"after"`
	Labels       []string       `json:"labels"`
	BeforeLabels []string       `json:"before_labels"`
	EdgeType     *string        `json:"edge_type"`
	SrcID        *uint64        `json:"src_id"`
	DstID        *uint64        `json:"dst_id"`
	// DeletedWith is, on an edge a detach delete removed, the node it was removed with.
	DeletedWith *uint64 `json:"deleted_with"`
}

// UpsertSummary tells what an upsert did with its rows.
type UpsertSummary struct {
	Created int `json:"created"`
	Updated int `json:"updated"`
	// Skipped counts rows without their key, and edge rows whose endpoint matches no node or several.
	Skipped     int   `json:"skipped"`
	SkippedRows []int `json:"skipped_rows"`
}

// UpsertNodesOptions configures UpsertNodes; nil means the defaults.
type UpsertNodesOptions struct {
	// Key is the property that identifies a node (default "id").
	Key string `json:"key,omitempty"`
	// Replace makes a node's properties exactly the row's (default: merged in).
	Replace bool `json:"replace,omitempty"`
}

// UpsertEdgesOptions configures UpsertEdges; nil means the defaults.
type UpsertEdgesOptions struct {
	Key            string   `json:"key,omitempty"`
	EndpointKey    string   `json:"endpoint_key,omitempty"`
	EndpointLabels []string `json:"endpoint_labels,omitempty"`
	SrcField       string   `json:"src_field,omitempty"`
	DstField       string   `json:"dst_field,omitempty"`
	Replace        bool     `json:"replace,omitempty"`
}

// cJSON marshals v into a C string the caller frees; NULL for a nil v.
func cJSON(v any) (*C.char, error) {
	if v == nil {
		return nil, nil
	}
	data, err := marshalJSON(v)
	if err != nil {
		return nil, err
	}
	return C.CString(string(data)), nil
}

// freeC frees a C string made by cJSON (NULL is fine).
func freeC(s *C.char) {
	if s != nil {
		C.free(unsafe.Pointer(s))
	}
}

// takeIDs copies and frees an id array a batch call handed out.
func takeIDs(ids *C.uint64_t, count C.size_t) []uint64 {
	if count == 0 {
		return []uint64{}
	}
	defer C.grafeo_free_node_ids(ids, count)
	return append([]uint64(nil), unsafe.Slice((*uint64)(unsafe.Pointer(ids)), int(count))...)
}

// takeJSON decodes and frees a JSON string a call handed out.
func takeJSON(s *C.char, into any) error {
	defer C.grafeo_free_string(s)
	dec := json.NewDecoder(strings.NewReader(C.GoString(s)))
	return dec.Decode(into)
}

// idsCall runs a batch call that hands out ids, on one OS thread.
func idsCall(call func(ids **C.uint64_t, count *C.size_t) C.GrafeoStatus) ([]uint64, error) {
	var ids *C.uint64_t
	var count C.size_t
	runtime.LockOSThread()
	status := call(&ids, &count)
	if status != C.GRAFEO_OK {
		err := statusToError(status)
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	return takeIDs(ids, count), nil
}

// jsonCall runs a call that hands out a JSON string through an out pointer.
func jsonCall(into any, call func(out **C.char) C.GrafeoStatus) error {
	var out *C.char
	runtime.LockOSThread()
	status := call(&out)
	if status != C.GRAFEO_OK {
		err := statusToError(status)
		runtime.UnlockOSThread()
		return err
	}
	runtime.UnlockOSThread()
	return takeJSON(out, into)
}

// eventsCall runs a CDC call that returns a JSON string, NULL on error.
func eventsCall(call func() *C.char) ([]ChangeEvent, error) {
	runtime.LockOSThread()
	out := call()
	if out == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	var events []ChangeEvent
	if err := takeJSON(out, &events); err != nil {
		return nil, err
	}
	return events, nil
}

// nodeRows marshals the labels and rows of a batch node create or an upsert.
func nodeRows(labels []string, rows []map[string]any) (*C.char, *C.char, error) {
	if labels == nil {
		labels = []string{}
	}
	if rows == nil {
		rows = []map[string]any{}
	}
	cLabels, err := cJSON(labels)
	if err != nil {
		return nil, nil, err
	}
	cRows, err := cJSON(rows)
	if err != nil {
		freeC(cLabels)
		return nil, nil, err
	}
	return cLabels, cRows, nil
}

// BatchCreateNodesWithProps creates one node with labels per property map of rows, in one
// transaction (a row the schema refuses creates none). Returns the node ids in row order.
func (db *Database) BatchCreateNodesWithProps(labels []string, rows []map[string]any) ([]uint64, error) {
	cLabels, cRows, err := nodeRows(labels, rows)
	if err != nil {
		return nil, err
	}
	defer freeC(cLabels)
	defer freeC(cRows)
	return idsCall(func(ids **C.uint64_t, count *C.size_t) C.GrafeoStatus {
		return C.grafeo_batch_create_nodes_with_props(db.handle, cLabels, cRows, ids, count)
	})
}

// BatchCreateEdges creates the edges in one transaction (an edge naming a node that does not
// exist creates none). Returns the edge ids in input order.
func (db *Database) BatchCreateEdges(edges []BatchEdge) ([]uint64, error) {
	if edges == nil {
		edges = []BatchEdge{}
	}
	cEdges, err := cJSON(edges)
	if err != nil {
		return nil, err
	}
	defer freeC(cEdges)
	return idsCall(func(ids **C.uint64_t, count *C.size_t) C.GrafeoStatus {
		return C.grafeo_batch_create_edges(db.handle, cEdges, ids, count)
	})
}

// BatchVectorSearch finds the k nearest neighbours of each query vector, nearest first. ef 0
// uses the index's default; filters (nil for none) are property equalities the results match.
func (db *Database) BatchVectorSearch(label, property string, queries [][]float32, k, ef int, filters map[string]any) ([][]VectorResult, error) {
	dims := 0
	if len(queries) > 0 {
		dims = len(queries[0])
	}
	flat := make([]float32, 0, len(queries)*dims)
	for _, q := range queries {
		flat = append(flat, q...)
	}
	cLabel := C.CString(label)
	defer C.free(unsafe.Pointer(cLabel))
	cProp := C.CString(property)
	defer C.free(unsafe.Pointer(cProp))
	var cFilters *C.char
	if filters != nil {
		var err error
		if cFilters, err = cJSON(filters); err != nil {
			return nil, err
		}
		defer freeC(cFilters)
	}
	var ptr *C.float
	if len(flat) > 0 {
		ptr = (*C.float)(unsafe.Pointer(&flat[0]))
	}
	var raw [][]struct {
		ID       uint64  `json:"id"`
		Distance float32 `json:"distance"`
	}
	err := jsonCall(&raw, func(out **C.char) C.GrafeoStatus {
		return C.grafeo_batch_vector_search(db.handle, cLabel, cProp, ptr, C.size_t(len(queries)), C.size_t(dims), C.size_t(k), C.size_t(ef), cFilters, out)
	})
	if err != nil {
		return nil, err
	}
	results := make([][]VectorResult, len(raw))
	for i, hits := range raw {
		results[i] = make([]VectorResult, len(hits))
		for j, hit := range hits {
			results[i][j] = VectorResult{NodeID: hit.ID, Distance: hit.Distance}
		}
	}
	return results, nil
}

// SetCDCEnabled turns change data capture on or off for new sessions.
func (db *Database) SetCDCEnabled(enabled bool) {
	C.grafeo_set_cdc_enabled(db.handle, C.bool(enabled))
}

// IsCDCEnabled tells whether change data capture is on for new sessions.
func (db *Database) IsCDCEnabled() bool {
	return bool(C.grafeo_is_cdc_enabled(db.handle))
}

// NodeHistory returns every recorded change of a node, oldest first (requires change data capture).
func (db *Database) NodeHistory(id uint64) ([]ChangeEvent, error) {
	return eventsCall(func() *C.char { return C.grafeo_node_history(db.handle, C.uint64_t(id)) })
}

// EdgeHistory returns every recorded change of an edge, oldest first.
func (db *Database) EdgeHistory(id uint64) ([]ChangeEvent, error) {
	return eventsCall(func() *C.char { return C.grafeo_edge_history(db.handle, C.uint64_t(id)) })
}

// NodeHistorySince returns the changes of a node at or after an epoch.
func (db *Database) NodeHistorySince(id, sinceEpoch uint64) ([]ChangeEvent, error) {
	return eventsCall(func() *C.char {
		return C.grafeo_node_history_since(db.handle, C.uint64_t(id), C.uint64_t(sinceEpoch))
	})
}

// ChangesBetween returns every change in the epochs from start to end (both included), in commit order.
func (db *Database) ChangesBetween(startEpoch, endEpoch uint64) ([]ChangeEvent, error) {
	return eventsCall(func() *C.char {
		return C.grafeo_changes_between(db.handle, C.uint64_t(startEpoch), C.uint64_t(endEpoch))
	})
}

// upsertArgs marshals the rows and options of an upsert.
func upsertArgs(rows []map[string]any, options any) (*C.char, *C.char, error) {
	if rows == nil {
		rows = []map[string]any{}
	}
	cRows, err := cJSON(rows)
	if err != nil {
		return nil, nil, err
	}
	cOptions, err := cJSON(options)
	if err != nil {
		freeC(cRows)
		return nil, nil, err
	}
	return cRows, cOptions, nil
}

// UpsertNodes creates or updates one node per row, matched by the key and all labels, in one
// statement. A row without the key is skipped.
func (db *Database) UpsertNodes(labels []string, rows []map[string]any, options *UpsertNodesOptions) (*UpsertSummary, error) {
	return upsertNodes(labels, rows, options, func(cLabels, cRows, cOptions *C.char, out **C.char) C.GrafeoStatus {
		return C.grafeo_upsert_nodes(db.handle, cLabels, cRows, cOptions, out)
	})
}

// UpsertEdges creates or updates one edge of edgeType per row between existing nodes, in one
// statement. Each row names its endpoints in the source and target fields ("src" and "dst").
func (db *Database) UpsertEdges(edgeType string, rows []map[string]any, options *UpsertEdgesOptions) (*UpsertSummary, error) {
	return upsertEdges(edgeType, rows, options, func(cType, cRows, cOptions *C.char, out **C.char) C.GrafeoStatus {
		return C.grafeo_upsert_edges(db.handle, cType, cRows, cOptions, out)
	})
}

func upsertNodes(labels []string, rows []map[string]any, options *UpsertNodesOptions, call func(cLabels, cRows, cOptions *C.char, out **C.char) C.GrafeoStatus) (*UpsertSummary, error) {
	if labels == nil {
		labels = []string{}
	}
	cLabels, err := cJSON(labels)
	if err != nil {
		return nil, err
	}
	defer freeC(cLabels)
	var opts any
	if options != nil {
		opts = options
	}
	cRows, cOptions, err := upsertArgs(rows, opts)
	if err != nil {
		return nil, err
	}
	defer freeC(cRows)
	defer freeC(cOptions)
	var summary UpsertSummary
	if err := jsonCall(&summary, func(out **C.char) C.GrafeoStatus { return call(cLabels, cRows, cOptions, out) }); err != nil {
		return nil, err
	}
	return &summary, nil
}

func upsertEdges(edgeType string, rows []map[string]any, options *UpsertEdgesOptions, call func(cType, cRows, cOptions *C.char, out **C.char) C.GrafeoStatus) (*UpsertSummary, error) {
	cType := C.CString(edgeType)
	defer C.free(unsafe.Pointer(cType))
	var opts any
	if options != nil {
		opts = options
	}
	cRows, cOptions, err := upsertArgs(rows, opts)
	if err != nil {
		return nil, err
	}
	defer freeC(cRows)
	defer freeC(cOptions)
	var summary UpsertSummary
	if err := jsonCall(&summary, func(out **C.char) C.GrafeoStatus { return call(cType, cRows, cOptions, out) }); err != nil {
		return nil, err
	}
	return &summary, nil
}
