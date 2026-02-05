// Package grafeo provides Go bindings for the Grafeo graph database.
//
// Pre-alpha - bindings are under development.
// See https://grafeo.dev for current status.
package grafeo

import "errors"

// ErrNotImplemented is returned when calling any function in this pre-alpha package.
var ErrNotImplemented = errors.New("grafeo-go is pre-alpha and not yet implemented; see https://grafeo.dev for status")
