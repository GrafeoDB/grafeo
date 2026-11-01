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
}

// ErrStreamClosed is returned by Next after Close has been called.
var ErrStreamClosed = errors.New("grafeo: stream is closed")

// ExecuteStream opens a lazy cursor over a read-only GQL query.
func (db *Database) ExecuteStream(query string) (*ResultStream, error) {
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

// Next pulls the next row as a map keyed by column name. Returns (nil, nil)
// when the stream is exhausted.
func (s *ResultStream) Next() (map[string]any, error) {
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

	jsonStr := C.GoString(outPtr)
	var row map[string]any
	if err := json.Unmarshal([]byte(jsonStr), &row); err != nil {
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
	if s.handle != nil {
		C.grafeo_stream_free(s.handle)
		s.handle = nil
		runtime.SetFinalizer(s, nil)
	}
}

// finalize is the Go finalizer for leak prevention.
func (s *ResultStream) finalize() {
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
