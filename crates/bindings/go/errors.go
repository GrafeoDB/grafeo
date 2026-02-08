package grafeo

/*
#include "grafeo.h"
*/
import "C"
import (
	"errors"
	"fmt"
)

// ErrDatabase is the base error for all Grafeo database errors.
var ErrDatabase = errors.New("grafeo")

// lastError reads the thread-local error from the C layer.
func lastError() error {
	msg := C.grafeo_last_error()
	if msg == nil {
		return fmt.Errorf("%w: unknown error", ErrDatabase)
	}
	return fmt.Errorf("%w: %s", ErrDatabase, C.GoString(msg))
}

// statusToError converts a GrafeoStatus to a Go error (nil on success).
func statusToError(status C.GrafeoStatus) error {
	if status == C.GRAFEO_OK {
		return nil
	}
	return lastError()
}
