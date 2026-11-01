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

// CreatePropertyIndex creates a property index for fast lookups.
func (db *Database) CreatePropertyIndex(property string) error {
	cProp := C.CString(property)
	defer C.free(unsafe.Pointer(cProp))
	return lockAndCheckStatus(func() C.GrafeoStatus {
		return C.grafeo_create_property_index(db.handle, cProp)
	})
}

// DropPropertyIndex drops a property index. Returns true if it existed.
func (db *Database) DropPropertyIndex(property string) (bool, error) {
	cProp := C.CString(property)
	defer C.free(unsafe.Pointer(cProp))
	runtime.LockOSThread()
	result := int(C.grafeo_drop_property_index(db.handle, cProp))
	if result < 0 {
		err := lastError()
		runtime.UnlockOSThread()
		return false, err
	}
	runtime.UnlockOSThread()
	return result == 1, nil
}

// HasPropertyIndex checks whether a property index exists.
func (db *Database) HasPropertyIndex(property string) bool {
	cProp := C.CString(property)
	defer C.free(unsafe.Pointer(cProp))
	runtime.LockOSThread()
	result := int(C.grafeo_has_property_index(db.handle, cProp)) == 1
	runtime.UnlockOSThread()
	return result
}

// FindNodesByProperty finds nodes with a matching property value.
func (db *Database) FindNodesByProperty(property string, value any) ([]uint64, error) {
	cProp := C.CString(property)
	defer C.free(unsafe.Pointer(cProp))

	valueJSON, err := marshalJSON(value)
	if err != nil {
		return nil, err
	}
	cValue := C.CString(string(valueJSON))
	defer C.free(unsafe.Pointer(cValue))

	var outIDs *C.uint64_t
	var outCount C.size_t

	runtime.LockOSThread()
	status := C.grafeo_find_nodes_by_property(db.handle, cProp, cValue, &outIDs, &outCount)
	if status != C.GRAFEO_OK {
		err := statusToError(status)
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()

	count := int(outCount)
	if count == 0 {
		return nil, nil
	}
	defer C.grafeo_free_node_ids(outIDs, outCount)

	ids := make([]uint64, count)
	raw := unsafe.Slice((*uint64)(unsafe.Pointer(outIDs)), count)
	copy(ids, raw)
	return ids, nil
}
