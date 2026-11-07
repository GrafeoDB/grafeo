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
// rolled back when garbage collected. Commit and Rollback free its C
// transaction once it has ended. A Transaction is not safe for concurrent
// use: call its methods from one goroutine at a time.
type Transaction struct {
	handle *C.GrafeoTransaction
	// ended is how the transaction ended, once its handle is freed.
	ended transactionEnd
}

// transactionEnd is how a transaction ended.
type transactionEnd int

const (
	transactionOpen transactionEnd = iota
	transactionCommitted
	transactionRolledBack
	// transactionCommitFailed: a commit was tried and failed, which ended
	// the transaction without its changes.
	transactionCommitFailed
)

// endedError is the error of a call on a transaction that has ended: the
// message and GQLSTATUS the C layer gives (a data exception, 22000).
func endedError(message string) error {
	return &Error{Message: message, GQLStatus: "22000"}
}

// BeginTransaction starts a new transaction with default isolation (snapshot).
func (db *Database) BeginTransaction() (*Transaction, error) {
	defer runtime.KeepAlive(db)
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
	defer runtime.KeepAlive(db)
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
	defer runtime.KeepAlive(tx)
	if tx.handle == nil {
		return nil, endedError("Transaction is no longer active")
	}
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
	defer runtime.KeepAlive(tx)
	if tx.handle == nil {
		return nil, endedError("Transaction is no longer active")
	}
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
	defer runtime.KeepAlive(tx)
	if tx.handle == nil {
		return nil, endedError("Transaction is no longer active")
	}
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
// GQLSTATUS 40000) ends the transaction all the same. Either way the C
// transaction is freed: a later call gives the error the C layer gives.
func (tx *Transaction) Commit() error {
	defer runtime.KeepAlive(tx)
	switch tx.ended {
	case transactionCommitted:
		return endedError("Transaction already committed")
	case transactionRolledBack:
		return endedError("Transaction already rolled back")
	case transactionCommitFailed:
		return endedError("Transaction is no longer active")
	}
	runtime.LockOSThread()
	err := statusToError(C.grafeo_commit(tx.handle))
	runtime.UnlockOSThread()
	if err == nil {
		tx.end(transactionCommitted)
	} else {
		// The C layer ends the transaction on any failed commit of an
		// open one.
		tx.end(transactionCommitFailed)
	}
	return err
}

// Rollback aborts the transaction. After a Commit that failed it does
// nothing and returns nil: a commit that fails ends the transaction, so a
// deferred or error-path Rollback does not hide the commit's error.
func (tx *Transaction) Rollback() error {
	defer runtime.KeepAlive(tx)
	switch tx.ended {
	case transactionCommitted:
		return endedError("Transaction already committed")
	case transactionRolledBack:
		return endedError("Transaction already rolled back")
	case transactionCommitFailed:
		return nil
	}
	runtime.LockOSThread()
	err := statusToError(C.grafeo_rollback(tx.handle))
	runtime.UnlockOSThread()
	if err == nil {
		tx.end(transactionRolledBack)
	}
	return err
}

// end records how the transaction ended and frees its C transaction, which
// the finalizer then no longer frees.
func (tx *Transaction) end(how transactionEnd) {
	runtime.SetFinalizer(tx, nil)
	tx.free()
	tx.ended = how
}

// free is the GC finalizer: it rolls back a transaction left open and frees
// its C transaction.
func (tx *Transaction) free() {
	defer runtime.KeepAlive(tx)
	if tx.handle != nil {
		C.grafeo_free_transaction(tx.handle)
		tx.handle = nil
	}
}
