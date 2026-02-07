// Package grafeo provides Go bindings for the Grafeo graph database.
//
// It uses CGO to link against the grafeo-c shared library, which provides
// a C-compatible FFI layer on top of the Rust engine.
//
// Quick start:
//
//	db, err := grafeo.OpenInMemory()
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer db.Close()
//
//	db.Execute(`CREATE (:Person {name: 'Alice', age: 30})`)
//	result, _ := db.Execute(`MATCH (p:Person) RETURN p.name`)
package grafeo

/*
#cgo LDFLAGS: -lgrafeo_c
#cgo linux LDFLAGS: -lm -ldl -lpthread
#cgo darwin LDFLAGS: -lm -ldl -lpthread -framework Security
#cgo windows LDFLAGS: -lws2_32 -lbcrypt -lntdll -luserenv

#include "grafeo.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// Database is the primary handle to a Grafeo graph database.
// It is safe for concurrent use from multiple goroutines.
type Database struct {
	handle *C.GrafeoDatabase
}

// OpenInMemory creates a new in-memory database.
func OpenInMemory() (*Database, error) {
	h := C.grafeo_open_memory()
	if h == nil {
		return nil, lastError()
	}
	db := &Database{handle: h}
	runtime.SetFinalizer(db, (*Database).free)
	return db, nil
}

// Open opens or creates a persistent database at the given path.
func Open(path string) (*Database, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	h := C.grafeo_open(cPath)
	if h == nil {
		return nil, lastError()
	}
	db := &Database{handle: h}
	runtime.SetFinalizer(db, (*Database).free)
	return db, nil
}

// Close flushes any pending writes and releases the database handle.
func (db *Database) Close() error {
	if db.handle == nil {
		return nil
	}
	status := C.grafeo_close(db.handle)
	C.grafeo_free_database(db.handle)
	db.handle = nil
	runtime.SetFinalizer(db, nil)
	return statusToError(status)
}

// free is called by the Go runtime finalizer for leak prevention.
func (db *Database) free() {
	if db.handle != nil {
		C.grafeo_close(db.handle)
		C.grafeo_free_database(db.handle)
		db.handle = nil
	}
}

// Execute runs a GQL query and returns the results.
func (db *Database) Execute(query string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	r := C.grafeo_execute(db.handle, cQuery)
	if r == nil {
		return nil, lastError()
	}
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteWithParams runs a GQL query with parameters encoded as a JSON object.
func (db *Database) ExecuteWithParams(query string, paramsJSON string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	cParams := C.CString(paramsJSON)
	defer C.free(unsafe.Pointer(cParams))
	r := C.grafeo_execute_with_params(db.handle, cQuery, cParams)
	if r == nil {
		return nil, lastError()
	}
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteCypher runs a Cypher query (requires cypher feature at compile time).
func (db *Database) ExecuteCypher(query string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	r := C.grafeo_execute_cypher(db.handle, cQuery)
	if r == nil {
		return nil, lastError()
	}
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteGremlin runs a Gremlin query (requires gremlin feature at compile time).
func (db *Database) ExecuteGremlin(query string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	r := C.grafeo_execute_gremlin(db.handle, cQuery)
	if r == nil {
		return nil, lastError()
	}
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteGraphQL runs a GraphQL query (requires graphql feature at compile time).
func (db *Database) ExecuteGraphQL(query string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	r := C.grafeo_execute_graphql(db.handle, cQuery)
	if r == nil {
		return nil, lastError()
	}
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteSPARQL runs a SPARQL query (requires sparql feature at compile time).
func (db *Database) ExecuteSPARQL(query string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	r := C.grafeo_execute_sparql(db.handle, cQuery)
	if r == nil {
		return nil, lastError()
	}
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// NodeCount returns the number of nodes in the database.
func (db *Database) NodeCount() int {
	return int(C.grafeo_node_count(db.handle))
}

// EdgeCount returns the number of edges in the database.
func (db *Database) EdgeCount() int {
	return int(C.grafeo_edge_count(db.handle))
}

// Version returns the Grafeo library version.
func Version() string {
	return C.GoString(C.grafeo_version())
}
