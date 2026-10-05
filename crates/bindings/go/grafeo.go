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
//	db.Execute(`CREATE (:Person {name: 'Alix', age: 30})`)
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
	"encoding/json"
	"fmt"
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
	runtime.LockOSThread()
	h := C.grafeo_open_memory()
	if h == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	db := &Database{handle: h}
	runtime.SetFinalizer(db, (*Database).free)
	return db, nil
}

// Open opens or creates a persistent database at the given path.
func Open(path string) (*Database, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	runtime.LockOSThread()
	h := C.grafeo_open(cPath)
	if h == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	db := &Database{handle: h}
	runtime.SetFinalizer(db, (*Database).free)
	return db, nil
}

// OpenSingleFile opens or creates a persistent database at the given path,
// as Open does: since 0.6 every database is a single file, whatever the
// extension of its path.
//
// Deprecated: use Open, which does the same since 0.6.0. OpenSingleFile is
// removed in 0.7.0.
func OpenSingleFile(path string) (*Database, error) {
	return Open(path)
}

// Close flushes any pending writes and releases the database handle.
func (db *Database) Close() error {
	if db.handle == nil {
		return nil
	}
	runtime.LockOSThread()
	status := C.grafeo_close(db.handle)
	err := statusToError(status)
	runtime.UnlockOSThread()
	C.grafeo_free_database(db.handle)
	db.handle = nil
	runtime.SetFinalizer(db, nil)
	return err
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
	runtime.LockOSThread()
	r := C.grafeo_execute(db.handle, cQuery)
	var err error
	if r == nil {
		err = lastError()
	}
	runtime.UnlockOSThread()
	if err != nil {
		return nil, err
	}
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteParams runs a GQL query with parameters as a Go map.
// The map is marshaled to JSON internally.
func (db *Database) ExecuteParams(query string, params map[string]any) (*QueryResult, error) {
	data, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("%w: failed to marshal params: %v", ErrDatabase, err)
	}
	return db.ExecuteWithParams(query, string(data))
}

// ExecuteWithParams runs a GQL query with parameters encoded as a JSON object.
func (db *Database) ExecuteWithParams(query string, paramsJSON string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	cParams := C.CString(paramsJSON)
	defer C.free(unsafe.Pointer(cParams))
	runtime.LockOSThread()
	r := C.grafeo_execute_with_params(db.handle, cQuery, cParams)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteCypher runs a Cypher query (requires cypher feature at compile time).
func (db *Database) ExecuteCypher(query string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	runtime.LockOSThread()
	r := C.grafeo_execute_cypher(db.handle, cQuery)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteGremlin runs a Gremlin query (requires gremlin feature at compile time).
func (db *Database) ExecuteGremlin(query string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	runtime.LockOSThread()
	r := C.grafeo_execute_gremlin(db.handle, cQuery)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteGraphQL runs a GraphQL query (requires graphql feature at compile time).
func (db *Database) ExecuteGraphQL(query string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	runtime.LockOSThread()
	r := C.grafeo_execute_graphql(db.handle, cQuery)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteSPARQL runs a SPARQL query (requires sparql feature at compile time).
func (db *Database) ExecuteSPARQL(query string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	runtime.LockOSThread()
	r := C.grafeo_execute_sparql(db.handle, cQuery)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteSQL runs a SQL/PGQ query (requires sql-pgq feature at compile time).
func (db *Database) ExecuteSQL(query string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	runtime.LockOSThread()
	r := C.grafeo_execute_sql(db.handle, cQuery)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteCypherWithParams runs a Cypher query with JSON-encoded parameters.
func (db *Database) ExecuteCypherWithParams(query, paramsJSON string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	cParams := C.CString(paramsJSON)
	defer C.free(unsafe.Pointer(cParams))
	runtime.LockOSThread()
	r := C.grafeo_execute_cypher_with_params(db.handle, cQuery, cParams)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteGremlinWithParams runs a Gremlin query with JSON-encoded parameters.
func (db *Database) ExecuteGremlinWithParams(query, paramsJSON string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	cParams := C.CString(paramsJSON)
	defer C.free(unsafe.Pointer(cParams))
	runtime.LockOSThread()
	r := C.grafeo_execute_gremlin_with_params(db.handle, cQuery, cParams)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteGraphQLWithParams runs a GraphQL query with JSON-encoded parameters.
func (db *Database) ExecuteGraphQLWithParams(query, paramsJSON string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	cParams := C.CString(paramsJSON)
	defer C.free(unsafe.Pointer(cParams))
	runtime.LockOSThread()
	r := C.grafeo_execute_graphql_with_params(db.handle, cQuery, cParams)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteSPARQLWithParams runs a SPARQL query with JSON-encoded parameters.
func (db *Database) ExecuteSPARQLWithParams(query, paramsJSON string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	cParams := C.CString(paramsJSON)
	defer C.free(unsafe.Pointer(cParams))
	runtime.LockOSThread()
	r := C.grafeo_execute_sparql_with_params(db.handle, cQuery, cParams)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteSQLWithParams runs a SQL/PGQ query with JSON-encoded parameters.
func (db *Database) ExecuteSQLWithParams(query, paramsJSON string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	cParams := C.CString(paramsJSON)
	defer C.free(unsafe.Pointer(cParams))
	runtime.LockOSThread()
	r := C.grafeo_execute_sql_with_params(db.handle, cQuery, cParams)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteLanguage runs a query in the given language with optional JSON-encoded
// parameters. language is one of: "gql", "cypher", "gremlin", "graphql",
// "sparql", "sql". Pass "" for paramsJSON if no parameters are needed.
func (db *Database) ExecuteLanguage(language, query, paramsJSON string) (*QueryResult, error) {
	cLang := C.CString(language)
	defer C.free(unsafe.Pointer(cLang))
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	var cParams *C.char
	if paramsJSON != "" {
		cParams = C.CString(paramsJSON)
		defer C.free(unsafe.Pointer(cParams))
	}
	runtime.LockOSThread()
	r := C.grafeo_execute_language(db.handle, cLang, cQuery, cParams)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// DropVectorIndex drops a vector index for the given label and property.
// Returns true if the index existed and was removed.
func (db *Database) DropVectorIndex(label, property string) bool {
	cLabel := C.CString(label)
	defer C.free(unsafe.Pointer(cLabel))
	cProp := C.CString(property)
	defer C.free(unsafe.Pointer(cProp))
	return C.grafeo_drop_vector_index(db.handle, cLabel, cProp) != 0
}

// RebuildVectorIndex drops and recreates a vector index, rescanning all
// matching nodes. Preserves the original index configuration.
func (db *Database) RebuildVectorIndex(label, property string) error {
	cLabel := C.CString(label)
	defer C.free(unsafe.Pointer(cLabel))
	cProp := C.CString(property)
	defer C.free(unsafe.Pointer(cProp))
	return lockAndCheckStatus(func() C.GrafeoStatus {
		return C.grafeo_rebuild_vector_index(db.handle, cLabel, cProp)
	})
}

// MmrSearch finds diverse nearest neighbors using Maximal Marginal Relevance.
// fetchK is the number of HNSW candidates (use -1 for default 4*k).
// lambda controls relevance vs diversity (0=diverse, 1=relevant; use -1 for default 0.5).
// ef is the HNSW beam width (use -1 for default).
func (db *Database) MmrSearch(label, property string, query []float32, k int, fetchK int, lambda float32, ef int) ([]VectorResult, error) {
	cLabel := C.CString(label)
	defer C.free(unsafe.Pointer(cLabel))
	cProp := C.CString(property)
	defer C.free(unsafe.Pointer(cProp))

	var outIDs *C.uint64_t
	var outDists *C.float
	var outCount C.size_t

	runtime.LockOSThread()
	status := C.grafeo_mmr_search(
		db.handle, cLabel, cProp,
		(*C.float)(unsafe.Pointer(&query[0])), C.size_t(len(query)),
		C.size_t(k), C.int32_t(fetchK), C.float(lambda), C.int32_t(ef),
		&outIDs, &outDists, &outCount,
	)
	if status != C.GRAFEO_OK {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	count := int(outCount)
	if count == 0 {
		return nil, nil
	}
	defer C.grafeo_free_vector_results(outIDs, outDists, outCount)

	results := make([]VectorResult, count)
	ids := unsafe.Slice((*uint64)(unsafe.Pointer(outIDs)), count)
	dists := unsafe.Slice((*float32)(unsafe.Pointer(outDists)), count)
	for i := range count {
		results[i] = VectorResult{NodeID: ids[i], Distance: dists[i]}
	}
	return results, nil
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
