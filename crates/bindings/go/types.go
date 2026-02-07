package grafeo

/*
#include "grafeo.h"
*/
import "C"
import (
	"encoding/json"
)

// QueryResult holds the result of a query execution.
type QueryResult struct {
	Columns         []string
	Rows            []Row
	ExecutionTimeMs float64
	RowsScanned     uint64
}

// Row is a single result row, mapping column names to values.
type Row map[string]any

// Node represents a graph node with labels and properties.
type Node struct {
	ID         uint64
	Labels     []string
	Properties map[string]any
}

// Edge represents a graph edge between two nodes.
type Edge struct {
	ID         uint64
	SourceID   uint64
	TargetID   uint64
	Type       string
	Properties map[string]any
}

// IsolationLevel controls transaction isolation.
type IsolationLevel int32

const (
	// ReadCommitted sees only committed data but may see different versions
	// within a transaction.
	ReadCommitted IsolationLevel = 0

	// SnapshotIsolation (default) sees a consistent snapshot as of transaction start.
	SnapshotIsolation IsolationLevel = 1

	// Serializable provides full serializability via SSI conflict detection.
	Serializable IsolationLevel = 2
)

// VectorResult holds a single nearest-neighbor search result.
type VectorResult struct {
	NodeID   uint64
	Distance float32
}

// VectorIndexOption configures vector index creation.
type VectorIndexOption func(*vectorIndexConfig)

type vectorIndexConfig struct {
	dimensions     int32
	metric         string
	m              int32
	efConstruction int32
}

// WithDimensions sets the vector dimensions.
func WithDimensions(d int) VectorIndexOption {
	return func(c *vectorIndexConfig) { c.dimensions = int32(d) }
}

// WithMetric sets the distance metric ("cosine", "euclidean", "dot_product", "manhattan").
func WithMetric(m string) VectorIndexOption {
	return func(c *vectorIndexConfig) { c.metric = m }
}

// WithM sets the HNSW M parameter (max connections per node).
func WithM(m int) VectorIndexOption {
	return func(c *vectorIndexConfig) { c.m = int32(m) }
}

// WithEfConstruction sets the HNSW ef_construction parameter.
func WithEfConstruction(ef int) VectorIndexOption {
	return func(c *vectorIndexConfig) { c.efConstruction = int32(ef) }
}

// SearchOption configures vector search.
type SearchOption func(*searchConfig)

type searchConfig struct {
	ef int32
}

// WithEf sets the search ef parameter for recall/speed tradeoff.
func WithEf(ef int) SearchOption {
	return func(c *searchConfig) { c.ef = int32(ef) }
}

// parseResult converts a C GrafeoResult into a Go QueryResult.
func parseResult(r *C.GrafeoResult) (*QueryResult, error) {
	jsonPtr := C.grafeo_result_json(r)
	if jsonPtr == nil {
		return &QueryResult{}, nil
	}
	jsonStr := C.GoString(jsonPtr)

	var rawRows []map[string]any
	if err := json.Unmarshal([]byte(jsonStr), &rawRows); err != nil {
		return nil, err
	}

	// Extract column names from first row if available.
	var columns []string
	if len(rawRows) > 0 {
		for k := range rawRows[0] {
			columns = append(columns, k)
		}
	}

	rows := make([]Row, len(rawRows))
	for i, raw := range rawRows {
		rows[i] = Row(raw)
	}

	return &QueryResult{
		Columns:         columns,
		Rows:            rows,
		ExecutionTimeMs: float64(C.grafeo_result_execution_time_ms(r)),
		RowsScanned:     uint64(C.grafeo_result_rows_scanned(r)),
	}, nil
}
