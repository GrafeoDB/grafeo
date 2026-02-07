package grafeo

/*
#include "grafeo.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// CreateVectorIndex creates an HNSW similarity index on a vector property.
func (db *Database) CreateVectorIndex(label, property string, opts ...VectorIndexOption) error {
	cfg := vectorIndexConfig{dimensions: -1, m: -1, efConstruction: -1}
	for _, opt := range opts {
		opt(&cfg)
	}

	cLabel := C.CString(label)
	defer C.free(unsafe.Pointer(cLabel))
	cProp := C.CString(property)
	defer C.free(unsafe.Pointer(cProp))

	var cMetric *C.char
	if cfg.metric != "" {
		cMetric = C.CString(cfg.metric)
		defer C.free(unsafe.Pointer(cMetric))
	}

	return statusToError(C.grafeo_create_vector_index(
		db.handle, cLabel, cProp,
		C.int32_t(cfg.dimensions), cMetric,
		C.int32_t(cfg.m), C.int32_t(cfg.efConstruction),
	))
}

// VectorSearch finds the k nearest neighbors of a query vector.
func (db *Database) VectorSearch(label, property string, query []float32, k int, opts ...SearchOption) ([]VectorResult, error) {
	cfg := searchConfig{ef: -1}
	for _, opt := range opts {
		opt(&cfg)
	}

	cLabel := C.CString(label)
	defer C.free(unsafe.Pointer(cLabel))
	cProp := C.CString(property)
	defer C.free(unsafe.Pointer(cProp))

	var outIDs *C.uint64_t
	var outDists *C.float
	var outCount C.size_t

	status := C.grafeo_vector_search(
		db.handle, cLabel, cProp,
		(*C.float)(unsafe.Pointer(&query[0])), C.size_t(len(query)),
		C.size_t(k), C.int32_t(cfg.ef),
		&outIDs, &outDists, &outCount,
	)
	if status != C.GRAFEO_OK {
		return nil, statusToError(status)
	}

	count := int(outCount)
	if count == 0 {
		return nil, nil
	}
	defer C.grafeo_free_vector_results(outIDs, outDists, outCount)

	results := make([]VectorResult, count)
	ids := unsafe.Slice((*uint64)(unsafe.Pointer(outIDs)), count)
	dists := unsafe.Slice((*float32)(unsafe.Pointer(outDists)), count)
	for i := range count {
		results[i] = VectorResult{NodeID: ids[i], Distance: dists[i]}
	}
	return results, nil
}

// BatchCreateNodes bulk-inserts nodes with vector properties.
// Returns the IDs of the created nodes.
func (db *Database) BatchCreateNodes(label, property string, vectors [][]float32) ([]uint64, error) {
	if len(vectors) == 0 {
		return nil, nil
	}
	dims := len(vectors[0])

	// Flatten vectors into a contiguous array.
	flat := make([]float32, 0, len(vectors)*dims)
	for _, v := range vectors {
		flat = append(flat, v...)
	}

	cLabel := C.CString(label)
	defer C.free(unsafe.Pointer(cLabel))
	cProp := C.CString(property)
	defer C.free(unsafe.Pointer(cProp))

	var outIDs *C.uint64_t
	status := C.grafeo_batch_create_nodes(
		db.handle, cLabel, cProp,
		(*C.float)(unsafe.Pointer(&flat[0])),
		C.size_t(len(vectors)), C.size_t(dims),
		&outIDs,
	)
	if status != C.GRAFEO_OK {
		return nil, statusToError(status)
	}
	defer C.grafeo_free_node_ids(outIDs, C.size_t(len(vectors)))

	ids := make([]uint64, len(vectors))
	raw := unsafe.Slice((*uint64)(unsafe.Pointer(outIDs)), len(vectors))
	copy(ids, raw)
	return ids, nil
}
