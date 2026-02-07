package grafeo

/*
#include "grafeo.h"
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"unsafe"
)

// CreateEdge creates an edge between two nodes.
func (db *Database) CreateEdge(sourceID, targetID uint64, edgeType string, properties map[string]any) (*Edge, error) {
	cType := C.CString(edgeType)
	defer C.free(unsafe.Pointer(cType))

	var cProps *C.char
	if properties != nil {
		propsJSON, err := json.Marshal(properties)
		if err != nil {
			return nil, err
		}
		cProps = C.CString(string(propsJSON))
		defer C.free(unsafe.Pointer(cProps))
	}

	id := uint64(C.grafeo_create_edge(db.handle, C.uint64_t(sourceID), C.uint64_t(targetID), cType, cProps))
	if id == ^uint64(0) {
		return nil, lastError()
	}

	if properties == nil {
		properties = make(map[string]any)
	}
	return &Edge{
		ID:         id,
		SourceID:   sourceID,
		TargetID:   targetID,
		Type:       edgeType,
		Properties: properties,
	}, nil
}

// GetEdge retrieves an edge by ID. Returns nil if not found.
func (db *Database) GetEdge(id uint64) (*Edge, error) {
	var cEdge *C.GrafeoEdge
	status := C.grafeo_get_edge(db.handle, C.uint64_t(id), &cEdge)
	if status != C.GRAFEO_OK {
		return nil, statusToError(status)
	}
	defer C.grafeo_free_edge(cEdge)

	edgeID := uint64(C.grafeo_edge_id(cEdge))
	srcID := uint64(C.grafeo_edge_source_id(cEdge))
	dstID := uint64(C.grafeo_edge_target_id(cEdge))
	edgeType := C.GoString(C.grafeo_edge_type(cEdge))

	var props map[string]any
	propsPtr := C.grafeo_edge_properties_json(cEdge)
	if propsPtr != nil {
		_ = json.Unmarshal([]byte(C.GoString(propsPtr)), &props)
	}
	if props == nil {
		props = make(map[string]any)
	}

	return &Edge{
		ID:         edgeID,
		SourceID:   srcID,
		TargetID:   dstID,
		Type:       edgeType,
		Properties: props,
	}, nil
}

// DeleteEdge deletes an edge by ID. Returns true if the edge existed.
func (db *Database) DeleteEdge(id uint64) (bool, error) {
	result := int(C.grafeo_delete_edge(db.handle, C.uint64_t(id)))
	if result < 0 {
		return false, lastError()
	}
	return result == 1, nil
}

// SetEdgeProperty sets a property on an edge.
func (db *Database) SetEdgeProperty(id uint64, key string, value any) error {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	valueJSON, err := json.Marshal(value)
	if err != nil {
		return err
	}
	cValue := C.CString(string(valueJSON))
	defer C.free(unsafe.Pointer(cValue))
	return statusToError(C.grafeo_set_edge_property(db.handle, C.uint64_t(id), cKey, cValue))
}

// RemoveEdgeProperty removes a property from an edge.
func (db *Database) RemoveEdgeProperty(id uint64, key string) (bool, error) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	result := int(C.grafeo_remove_edge_property(db.handle, C.uint64_t(id), cKey))
	if result < 0 {
		return false, lastError()
	}
	return result == 1, nil
}
