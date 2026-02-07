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

// CreateNode creates a node with the given labels and optional properties.
func (db *Database) CreateNode(labels []string, properties map[string]any) (*Node, error) {
	labelsJSON, err := json.Marshal(labels)
	if err != nil {
		return nil, err
	}
	cLabels := C.CString(string(labelsJSON))
	defer C.free(unsafe.Pointer(cLabels))

	var cProps *C.char
	if properties != nil {
		propsJSON, err := json.Marshal(properties)
		if err != nil {
			return nil, err
		}
		cProps = C.CString(string(propsJSON))
		defer C.free(unsafe.Pointer(cProps))
	}

	id := uint64(C.grafeo_create_node(db.handle, cLabels, cProps))
	if id == ^uint64(0) { // UINT64_MAX
		return nil, lastError()
	}

	if properties == nil {
		properties = make(map[string]any)
	}
	return &Node{ID: id, Labels: labels, Properties: properties}, nil
}

// GetNode retrieves a node by ID. Returns nil if not found.
func (db *Database) GetNode(id uint64) (*Node, error) {
	var cNode *C.GrafeoNode
	status := C.grafeo_get_node(db.handle, C.uint64_t(id), &cNode)
	if status != C.GRAFEO_OK {
		return nil, statusToError(status)
	}
	defer C.grafeo_free_node(cNode)

	nodeID := uint64(C.grafeo_node_id(cNode))

	var labels []string
	labelsPtr := C.grafeo_node_labels_json(cNode)
	if labelsPtr != nil {
		_ = json.Unmarshal([]byte(C.GoString(labelsPtr)), &labels)
	}

	var props map[string]any
	propsPtr := C.grafeo_node_properties_json(cNode)
	if propsPtr != nil {
		_ = json.Unmarshal([]byte(C.GoString(propsPtr)), &props)
	}
	if props == nil {
		props = make(map[string]any)
	}

	return &Node{ID: nodeID, Labels: labels, Properties: props}, nil
}

// DeleteNode deletes a node by ID. Returns true if the node existed.
func (db *Database) DeleteNode(id uint64) (bool, error) {
	result := int(C.grafeo_delete_node(db.handle, C.uint64_t(id)))
	if result < 0 {
		return false, lastError()
	}
	return result == 1, nil
}

// SetNodeProperty sets a property on a node.
func (db *Database) SetNodeProperty(id uint64, key string, value any) error {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	valueJSON, err := json.Marshal(value)
	if err != nil {
		return err
	}
	cValue := C.CString(string(valueJSON))
	defer C.free(unsafe.Pointer(cValue))
	return statusToError(C.grafeo_set_node_property(db.handle, C.uint64_t(id), cKey, cValue))
}

// RemoveNodeProperty removes a property from a node.
func (db *Database) RemoveNodeProperty(id uint64, key string) (bool, error) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	result := int(C.grafeo_remove_node_property(db.handle, C.uint64_t(id), cKey))
	if result < 0 {
		return false, lastError()
	}
	return result == 1, nil
}

// AddNodeLabel adds a label to a node. Returns true if the label was newly added.
func (db *Database) AddNodeLabel(id uint64, label string) (bool, error) {
	cLabel := C.CString(label)
	defer C.free(unsafe.Pointer(cLabel))
	result := int(C.grafeo_add_node_label(db.handle, C.uint64_t(id), cLabel))
	if result < 0 {
		return false, lastError()
	}
	return result == 1, nil
}

// RemoveNodeLabel removes a label from a node. Returns true if the label was present.
func (db *Database) RemoveNodeLabel(id uint64, label string) (bool, error) {
	cLabel := C.CString(label)
	defer C.free(unsafe.Pointer(cLabel))
	result := int(C.grafeo_remove_node_label(db.handle, C.uint64_t(id), cLabel))
	if result < 0 {
		return false, lastError()
	}
	return result == 1, nil
}

// GetNodeLabels returns all labels for a node.
func (db *Database) GetNodeLabels(id uint64) ([]string, error) {
	cLabels := C.grafeo_get_node_labels(db.handle, C.uint64_t(id))
	if cLabels == nil {
		return nil, nil
	}
	defer C.grafeo_free_string(cLabels)
	var labels []string
	_ = json.Unmarshal([]byte(C.GoString(cLabels)), &labels)
	return labels, nil
}
