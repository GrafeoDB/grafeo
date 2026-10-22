package grafeo

/*
#include "grafeo.h"
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
)

// ErrDatabase is the base error for all Grafeo database errors.
var ErrDatabase = errors.New("grafeo")

// lastError reads the thread-local error from the C layer.
// Must be called on the same OS thread as the C call that set the error.
func lastError() error {
	msg := C.grafeo_last_error()
	if msg == nil {
		return fmt.Errorf("%w: unknown error", ErrDatabase)
	}
	return fmt.Errorf("%w: %s", ErrDatabase, C.GoString(msg))
}

// statusToError converts a GrafeoStatus to a Go error (nil on success).
// Must be called on the same OS thread as the C call that produced the status.
func statusToError(status C.GrafeoStatus) error {
	if status == C.GRAFEO_OK {
		return nil
	}
	return lastError()
}

// lockAndCheckStatus pins the goroutine to an OS thread, calls fn,
// and reads any error from the thread-local. This ensures the C call
// and error retrieval happen on the same OS thread.
func lockAndCheckStatus(fn func() C.GrafeoStatus) error {
	runtime.LockOSThread()
	status := fn()
	err := statusToError(status)
	runtime.UnlockOSThread()
	return err
}
