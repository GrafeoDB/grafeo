package grafeo

/*
#include "grafeo.h"
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"runtime"
	"time"
	"unsafe"
)

// DatabaseInfo holds high-level database information.
type DatabaseInfo struct {
	NodeCount    int      `json:"node_count"`
	EdgeCount    int      `json:"edge_count"`
	IsPersistent bool     `json:"is_persistent"`
	Path         string   `json:"path"`
	WalEnabled   bool     `json:"wal_enabled"`
	Version      string   `json:"version"`
	Features     []string `json:"features"`
}

// Info returns high-level database information.
func (db *Database) Info() (*DatabaseInfo, error) {
	runtime.LockOSThread()
	cInfo := C.grafeo_info(db.handle)
	if cInfo == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
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
	return lockAndCheckStatus(func() C.GrafeoStatus {
		return C.grafeo_save(db.handle, cPath)
	})
}

// CheckpointStatus tells how the automatic checkpoints fare.
type CheckpointStatus struct {
	// IntervalMs is the interval trigger in milliseconds, nil when off.
	IntervalMs *uint64 `json:"interval_ms"`
	// WalBytes is the WAL size trigger in bytes, nil when off.
	WalBytes *uint64 `json:"wal_bytes"`
	// LastCheckpoint is when the last checkpoint of any kind finished (Unix seconds).
	LastCheckpoint *uint64 `json:"last_checkpoint"`
	// Failures counts the automatic checkpoints that failed since the last that succeeded.
	Failures uint64 `json:"failures"`
	// LastError is the error of the last failed automatic checkpoint.
	LastError *string `json:"last_error"`
	// LastFailure is when it failed (Unix seconds).
	LastFailure *uint64 `json:"last_failure"`
	// Stopped tells that no automatic checkpoint can succeed until a reopen.
	Stopped bool `json:"stopped"`
}

// SetCheckpointTriggers sets when the database checkpoints by itself: every
// interval, and once the WAL holds walBytes bytes since the last checkpoint.
// Zero turns a trigger off; both zero turn automatic checkpoints off (the
// default). For this open only.
func (db *Database) SetCheckpointTriggers(interval time.Duration, walBytes uint64) error {
	return lockAndCheckStatus(func() C.GrafeoStatus {
		return C.grafeo_set_checkpoint_triggers(db.handle, C.uint64_t(interval.Milliseconds()), C.uint64_t(walBytes))
	})
}

// CheckpointStatus returns how the automatic checkpoints fare.
func (db *Database) CheckpointStatus() (*CheckpointStatus, error) {
	runtime.LockOSThread()
	cStatus := C.grafeo_checkpoint_status(db.handle)
	if cStatus == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_string(cStatus)

	var status CheckpointStatus
	if err := json.Unmarshal([]byte(C.GoString(cStatus)), &status); err != nil {
		return nil, err
	}
	return &status, nil
}

// WalCheckpoint triggers a WAL checkpoint.
func (db *Database) WalCheckpoint() error {
	return lockAndCheckStatus(func() C.GrafeoStatus {
		return C.grafeo_wal_checkpoint(db.handle)
	})
}
