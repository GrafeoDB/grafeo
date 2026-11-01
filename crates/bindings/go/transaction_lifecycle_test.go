package grafeo

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"testing"
)

// endedTransaction opens an in-memory database with Vincent's account and
// begins a transaction on it.
func endedTransaction(t *testing.T) (*Database, *Transaction) {
	t.Helper()
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if _, err := db.Execute("INSERT (:Account {name: 'Vincent', balance: 1})"); err != nil {
		t.Fatal(err)
	}
	tx, err := db.BeginTransaction()
	if err != nil {
		t.Fatal(err)
	}
	return db, tx
}

// assertGrafeoError checks that err is a Grafeo error with the message and
// GQLSTATUS the C layer gives for it.
func assertGrafeoError(t *testing.T, err error, message, status string) {
	t.Helper()
	var grafeoErr *Error
	if !errors.As(err, &grafeoErr) {
		t.Fatalf("expected a Grafeo error %q, got %v", message, err)
	}
	if grafeoErr.Message != message || grafeoErr.GQLStatus != status {
		t.Errorf("expected %q (%s), got %q (%s)", message, status, grafeoErr.Message, grafeoErr.GQLStatus)
	}
}

// A commit frees the C transaction at once, and what comes after it gives
// the errors the C layer gives, without touching the freed handle.
func TestCommitFreesTheTransaction(t *testing.T) {
	db, tx := endedTransaction(t)
	if _, err := tx.Execute("MATCH (a:Account) SET a.balance = 2"); err != nil {
		t.Fatal(err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	if tx.handle != nil {
		t.Fatal("the committed transaction still holds its C handle")
	}

	assertGrafeoError(t, tx.Commit(), "Transaction already committed", "22000")
	assertGrafeoError(t, tx.Rollback(), "Transaction already committed", "22000")
	_, err := tx.Execute("MATCH (a:Account) RETURN a.balance")
	assertGrafeoError(t, err, "Transaction is no longer active", "22000")
	_, err = tx.ExecuteWithParams("MATCH (a:Account) RETURN a.balance", "{}")
	assertGrafeoError(t, err, "Transaction is no longer active", "22000")
	_, err = tx.ExecuteLanguage("gql", "MATCH (a:Account) RETURN a.balance", "")
	assertGrafeoError(t, err, "Transaction is no longer active", "22000")

	// The finalizer, should it still run, frees nothing twice.
	tx.free()
	runtime.GC()
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
}

// A rollback frees the C transaction at once too.
func TestRollbackFreesTheTransaction(t *testing.T) {
	db, tx := endedTransaction(t)
	if _, err := tx.Execute("MATCH (a:Account) SET a.balance = 2"); err != nil {
		t.Fatal(err)
	}
	if err := tx.Rollback(); err != nil {
		t.Fatal(err)
	}
	if tx.handle != nil {
		t.Fatal("the rolled back transaction still holds its C handle")
	}
	assertGrafeoError(t, tx.Rollback(), "Transaction already rolled back", "22000")
	assertGrafeoError(t, tx.Commit(), "Transaction already rolled back", "22000")
	tx.free()

	result, err := db.Execute("MATCH (a:Account) RETURN a.balance")
	if err != nil {
		t.Fatal(err)
	}
	if got := result.Rows[0]["a.balance"]; fmt.Sprint(got) != "1" {
		t.Errorf("the rollback keeps the balance at 1, got %v", got)
	}
}

// A commit that fails (a write conflict) ends the transaction and frees it:
// a Rollback after it does nothing and returns nil, so a deferred or
// error-path Rollback does not hide the commit's error.
func TestRollbackAfterAFailedCommitDoesNothing(t *testing.T) {
	db, first := endedTransaction(t)
	second, err := db.BeginTransaction()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.Execute("MATCH (a:Account) SET a.balance = 2"); err != nil {
		t.Fatal(err)
	}
	if err := first.Commit(); err != nil {
		t.Fatal(err)
	}
	if _, err := second.Execute("MATCH (a:Account) SET a.balance = 3"); err != nil {
		t.Fatal(err)
	}

	err = second.Commit()
	if err == nil || !strings.Contains(err.Error(), "Write-write conflict") {
		t.Fatalf("expected a write-write conflict, got %v", err)
	}
	if second.handle != nil {
		t.Fatal("the transaction whose commit failed still holds its C handle")
	}
	if err := second.Rollback(); err != nil {
		t.Errorf("a rollback after a failed commit does nothing, got %v", err)
	}
	assertGrafeoError(t, second.Commit(), "Transaction is no longer active", "22000")

	result, err := db.Execute("MATCH (a:Account) RETURN a.balance")
	if err != nil {
		t.Fatal(err)
	}
	if got := result.Rows[0]["a.balance"]; fmt.Sprint(got) != "2" {
		t.Errorf("the first commit stands at 2, got %v", got)
	}
}

// A transaction left open past the database's Close is freed by its
// finalizer without harm, and a committed one has nothing left to free.
func TestTransactionsOutliveClose(t *testing.T) {
	db, open := endedTransaction(t)
	if _, err := open.Execute("INSERT (:Person {name: 'Mia'})"); err != nil {
		t.Fatal(err)
	}
	committed, err := db.BeginTransaction()
	if err != nil {
		t.Fatal(err)
	}
	if err := committed.Commit(); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	// What the finalizers do, run here so the test sees them run.
	open.free()
	committed.free()
	if open.handle != nil || committed.handle != nil {
		t.Fatal("a transaction still holds its C handle")
	}
	open.free()
	runtime.GC()
}
