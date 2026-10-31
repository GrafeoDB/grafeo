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

// Transaction represents a database transaction with explicit commit/rollback.
// If neither Commit nor Rollback is called, the transaction is automatically
// rolled back when garbage collected.
type Transaction struct {
	handle *C.GrafeoTransaction
}

// BeginTransaction starts a new transaction with default isolation (snapshot).
func (db *Database) BeginTransaction() (*Transaction, error) {
	runtime.LockOSThread()
	h := C.grafeo_begin_transaction(db.handle)
	if h == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	tx := &Transaction{handle: h}
	runtime.SetFinalizer(tx, (*Transaction).free)
	return tx, nil
}

// BeginTransactionWith starts a transaction with a specific isolation level.
func (db *Database) BeginTransactionWith(level IsolationLevel) (*Transaction, error) {
	runtime.LockOSThread()
	h := C.grafeo_begin_transaction_with_isolation(db.handle, C.GrafeoIsolationLevel(level))
	if h == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	tx := &Transaction{handle: h}
	runtime.SetFinalizer(tx, (*Transaction).free)
	return tx, nil
}

// Execute runs a query within this transaction.
func (tx *Transaction) Execute(query string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	runtime.LockOSThread()
	r := C.grafeo_transaction_execute(tx.handle, cQuery)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteWithParams runs a query with JSON parameters within this transaction.
func (tx *Transaction) ExecuteWithParams(query string, paramsJSON string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	cParams := C.CString(paramsJSON)
	defer C.free(unsafe.Pointer(cParams))
	runtime.LockOSThread()
	r := C.grafeo_transaction_execute_with_params(tx.handle, cQuery, cParams)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// ExecuteLanguage runs a query in the given language within this transaction.
// language is one of: "gql", "cypher", "gremlin", "graphql", "sparql", "sql".
// Pass "" for paramsJSON if no parameters are needed.
func (tx *Transaction) ExecuteLanguage(language, query, paramsJSON string) (*QueryResult, error) {
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
	r := C.grafeo_transaction_execute_language(tx.handle, cLang, cQuery, cParams)
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}

// Commit commits the transaction. A commit that fails (a write conflict,
// GQLSTATUS 40000) ends the transaction all the same.
func (tx *Transaction) Commit() error {
	runtime.LockOSThread()
	err := statusToError(C.grafeo_commit(tx.handle))
	runtime.UnlockOSThread()
	if err == nil {
		// Prevent double-free on GC.
		runtime.SetFinalizer(tx, nil)
	}
	return err
}

// Rollback aborts the transaction. After a Commit that failed it does
// nothing and returns nil: a commit that fails ends the transaction, so a
// deferred or error-path Rollback does not hide the commit's error.
func (tx *Transaction) Rollback() error {
	runtime.LockOSThread()
	err := statusToError(C.grafeo_rollback(tx.handle))
	runtime.UnlockOSThread()
	if err == nil {
		runtime.SetFinalizer(tx, nil)
	}
	return err
}

// free is the GC finalizer — auto-rollback + free if user forgot.
func (tx *Transaction) free() {
	if tx.handle != nil {
		C.grafeo_free_transaction(tx.handle)
		tx.handle = nil
	}
}
