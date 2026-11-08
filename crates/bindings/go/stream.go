package grafeo

/*
#include "grafeo.h"
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"errors"
	"runtime"
	"strings"
	"unsafe"
)

// ResultStream is a lazy cursor over the rows of a read-only GQL query.
//
// Rows are pulled from the Rust engine on demand via Next, so memory stays
// bounded regardless of the total result size. ResultStream rejects mutating
// queries (INSERT / DELETE / SET), EXPLAIN / PROFILE, session/schema
// commands, and queries that compile to push-based pipelines (ORDER BY,
// aggregate, DISTINCT). Use Execute for those.
//
// Always call Close when done to release the underlying Rust iterator. The
// zero value is not useable; obtain a ResultStream from Database.ExecuteStream.
//
// Experimental (0.5.40+): API may change before Beta.
type ResultStream struct {
	handle  *C.GrafeoStream
	columns []string
	// warnings is what the stream had when it was closed.
	warnings []QueryWarning
}

// ErrStreamClosed is returned by Next after Close has been called.
var ErrStreamClosed = errors.New("grafeo: stream is closed")

// ExecuteStream opens a lazy cursor over a read-only GQL query.
func (db *Database) ExecuteStream(query string) (*ResultStream, error) {
	defer runtime.KeepAlive(db)
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	runtime.LockOSThread()
	h := C.grafeo_stream_open(db.handle, cQuery)
	if h == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()

	cols, err := readStreamColumns(h)
	if err != nil {
		C.grafeo_stream_free(h)
		return nil, err
	}

	stream := &ResultStream{handle: h, columns: cols}
	runtime.SetFinalizer(stream, (*ResultStream).finalize)
	return stream, nil
}

// Columns returns the column names in the order they appear in each row map.
func (s *ResultStream) Columns() []string {
	return s.columns
}

// Warnings returns the warnings the statement raised so far, ordered by
// code: its translation's from the open, its run's as the rows that raise
// them are pulled, all of them once Next returned (nil, nil). After Close it
// returns what the stream had when it was closed.
//
// Experimental (0.6.2+).
func (s *ResultStream) Warnings() []QueryWarning {
	defer runtime.KeepAlive(s)
	if s.handle == nil {
		return s.warnings
	}
	count := int(C.grafeo_stream_warning_count(s.handle))
	warnings := make([]QueryWarning, 0, count)
	for i := 0; i < count; i++ {
		index := C.size_t(i)
		warnings = append(warnings, QueryWarning{
			Code:    C.GoString(C.grafeo_stream_warning_code(s.handle, index)),
			Kind:    C.GoString(C.grafeo_stream_warning_kind(s.handle, index)),
			Message: C.GoString(C.grafeo_stream_warning_message(s.handle, index)),
		})
	}
	return warnings
}

// Next pulls the next row as a map keyed by column name. Returns (nil, nil)
// when the stream is exhausted.
func (s *ResultStream) Next() (map[string]any, error) {
	defer runtime.KeepAlive(s)
	if s.handle == nil {
		return nil, ErrStreamClosed
	}

	var outPtr *C.char
	runtime.LockOSThread()
	status := C.grafeo_stream_next_row_json(s.handle, &outPtr)
	if status != C.GRAFEO_OK {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()

	if outPtr == nil {
		return nil, nil // exhausted
	}
	defer C.grafeo_free_string(outPtr)

	// Numbers as json.Number, as Execute gives them: a float64 loses
	// integers above 2^53
	var row map[string]any
	dec := json.NewDecoder(strings.NewReader(C.GoString(outPtr)))
	dec.UseNumber()
	if err := dec.Decode(&row); err != nil {
		return nil, err
	}
	return decodeFields(row), nil
}

// Rows returns a channel-free iterator for a classic Go for-loop:
//
//	for {
//	    row, err := stream.Next()
//	    if err != nil { return err }
//	    if row == nil { break }
//	    // process row
//	}
//
// For convenience, Collect drains the remaining rows into a slice.
func (s *ResultStream) Collect() ([]map[string]any, error) {
	var rows []map[string]any
	for {
		row, err := s.Next()
		if err != nil {
			return rows, err
		}
		if row == nil {
			return rows, nil
		}
		rows = append(rows, row)
	}
}

// Close releases the underlying Rust iterator. Safe to call multiple times.
func (s *ResultStream) Close() {
	defer runtime.KeepAlive(s)
	if s.handle != nil {
		s.warnings = s.Warnings()
		C.grafeo_stream_free(s.handle)
		s.handle = nil
		runtime.SetFinalizer(s, nil)
	}
}

// finalize is the Go finalizer for leak prevention.
func (s *ResultStream) finalize() {
	defer runtime.KeepAlive(s)
	if s.handle != nil {
		C.grafeo_stream_free(s.handle)
		s.handle = nil
	}
}

// readStreamColumns fetches the JSON column-names array and decodes it.
func readStreamColumns(h *C.GrafeoStream) ([]string, error) {
	runtime.LockOSThread()
	ptr := C.grafeo_stream_columns_json(h)
	if ptr == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_string(ptr)

	raw := C.GoString(ptr)
	var cols []string
	if err := json.Unmarshal([]byte(raw), &cols); err != nil {
		return nil, err
	}
	return cols, nil
}
