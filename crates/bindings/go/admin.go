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

// DatabaseInfo holds high-level database information.
type DatabaseInfo struct {
	NodeCount    int    `json:"node_count"`
	EdgeCount    int    `json:"edge_count"`
	IsPersistent bool   `json:"is_persistent"`
	Path         string `json:"path"`
	WalEnabled   bool   `json:"wal_enabled"`
	Version      string `json:"version"`
}

// Info returns high-level database information.
func (db *Database) Info() (*DatabaseInfo, error) {
	cInfo := C.grafeo_info(db.handle)
	if cInfo == nil {
		return nil, lastError()
	}
	defer C.grafeo_free_string(cInfo)

	var info DatabaseInfo
	if err := json.Unmarshal([]byte(C.GoString(cInfo)), &info); err != nil {
		return nil, err
	}
	return &info, nil
}

// Save persists the database to the given path.
func (db *Database) Save(path string) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	return statusToError(C.grafeo_save(db.handle, cPath))
}

// WalCheckpoint triggers a WAL checkpoint.
func (db *Database) WalCheckpoint() error {
	return statusToError(C.grafeo_wal_checkpoint(db.handle))
}
