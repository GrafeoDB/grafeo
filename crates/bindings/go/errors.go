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

// Error is a Grafeo database error: its message, which starts with its
// Grafeo code (such as GRAFEO-Q006), and its GQLSTATUS, the five-character
// ISO GQL status of the condition ("22012" for a division by zero, "42001"
// for a syntax error). It matches ErrDatabase with errors.Is; get it with
// errors.As:
//
//	var grafeoErr *grafeo.Error
//	if errors.As(err, &grafeoErr) && grafeoErr.GQLStatus == "22012" { ... }
type Error struct {
	Message   string
	GQLStatus string
}

// Error returns the message, as "grafeo: <message>".
func (e *Error) Error() string {
	return fmt.Sprintf("%s: %s", ErrDatabase, e.Message)
}

// Unwrap returns ErrDatabase, so errors.Is(err, ErrDatabase) holds.
func (e *Error) Unwrap() error {
	return ErrDatabase
}

// lastError reads the thread-local error from the C layer.
// Must be called on the same OS thread as the C call that set the error.
func lastError() error {
	msg := C.grafeo_last_error()
	if msg == nil {
		return &Error{Message: "unknown error"}
	}
	err := &Error{Message: C.GoString(msg)}
	if status := C.grafeo_last_error_gql_status(); status != nil {
		err.GQLStatus = C.GoString(status)
	}
	return err
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
