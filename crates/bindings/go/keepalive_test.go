package grafeo

// A value with a finalizer that frees its native handle must stay reachable for as long as a C
// call uses that handle: the garbage collector may otherwise run the finalizer during the call,
// which then reads freed memory. Every function that passes such a handle to a C call therefore
// starts with `defer runtime.KeepAlive(x)` (one pattern for every return path and every later call
// in the body). This test reads the package's own source, so a new method is covered without a
// row here: the finalized types are the ones `runtime.SetFinalizer(_, (*T).method)` names.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// packageFiles parses the package's non-test Go files.
func packageFiles(t *testing.T) (*token.FileSet, []*ast.File) {
	t.Helper()
	names, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var files []*ast.File
	for _, name := range names {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		source, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		file, err := parser.ParseFile(fset, name, source, 0)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		files = append(files, file)
	}
	return fset, files
}

// finalizedTypes are the types whose finalizer `runtime.SetFinalizer(_, (*T).method)` sets.
func finalizedTypes(files []*ast.File) map[string]bool {
	types := map[string]bool{}
	for _, file := range files {
		ast.Inspect(file, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok || !isSelector(call.Fun, "runtime", "SetFinalizer") || len(call.Args) != 2 {
				return true
			}
			if method, ok := call.Args[1].(*ast.SelectorExpr); ok {
				if paren, ok := method.X.(*ast.ParenExpr); ok {
					if star, ok := paren.X.(*ast.StarExpr); ok {
						if ident, ok := star.X.(*ast.Ident); ok {
							types[ident.Name] = true
						}
					}
				}
			}
			return true
		})
	}
	return types
}

func isSelector(expr ast.Expr, x, name string) bool {
	selector, ok := expr.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	ident, ok := selector.X.(*ast.Ident)
	return ok && ident.Name == x && selector.Sel.Name == name
}

// finalizedNames are the receiver and parameters of fn whose type is *T for a finalized T.
func finalizedNames(fn *ast.FuncDecl, types map[string]bool) []string {
	var names []string
	add := func(fields *ast.FieldList) {
		if fields == nil {
			return
		}
		for _, field := range fields.List {
			star, ok := field.Type.(*ast.StarExpr)
			if !ok {
				continue
			}
			ident, ok := star.X.(*ast.Ident)
			if !ok || !types[ident.Name] {
				continue
			}
			for _, name := range field.Names {
				names = append(names, name.Name)
			}
		}
	}
	add(fn.Recv)
	add(fn.Type.Params)
	return names
}

// passesHandle reports whether body hands x's native handle to C: it reads a field of x (not a
// method of it), and either makes a `C.` call or passes such a field to any call. The field may
// reach C through a local it was copied to, or through a helper that takes it, so a C call whose
// own arguments never name x counts as well.
func passesHandle(body *ast.BlockStmt, x string) bool {
	readsField, passesField, callsC := false, false, false
	isField := func(expr ast.Expr) bool {
		field, ok := expr.(*ast.SelectorExpr)
		if !ok {
			return false
		}
		ident, ok := field.X.(*ast.Ident)
		return ok && ident.Name == x
	}
	methods := map[ast.Expr]bool{}
	ast.Inspect(body, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		methods[call.Fun] = true
		if selector, ok := call.Fun.(*ast.SelectorExpr); ok {
			if pkg, ok := selector.X.(*ast.Ident); ok && pkg.Name == "C" {
				callsC = true
			}
		}
		for _, arg := range call.Args {
			ast.Inspect(arg, func(inner ast.Node) bool {
				if expr, ok := inner.(ast.Expr); ok && isField(expr) {
					passesField = true
				}
				return true
			})
		}
		return true
	})
	ast.Inspect(body, func(node ast.Node) bool {
		if expr, ok := node.(ast.Expr); ok && isField(expr) && !methods[expr] {
			readsField = true
		}
		return true
	})
	return readsField && (callsC || passesField)
}

// startsWithKeepAlive reports whether body's first statement is `defer runtime.KeepAlive(x)`.
func startsWithKeepAlive(body *ast.BlockStmt, x string) bool {
	if len(body.List) == 0 {
		return false
	}
	deferred, ok := body.List[0].(*ast.DeferStmt)
	if !ok || !isSelector(deferred.Call.Fun, "runtime", "KeepAlive") || len(deferred.Call.Args) != 1 {
		return false
	}
	ident, ok := deferred.Call.Args[0].(*ast.Ident)
	return ok && ident.Name == x
}

func TestEveryNativeCallKeepsItsHandleAlive(t *testing.T) {
	fset, files := packageFiles(t)
	types := finalizedTypes(files)
	for _, want := range []string{"Database", "Graph", "ResultStream", "AsOfView", "Transaction"} {
		if !types[want] {
			t.Errorf("%s has a finalizer, but the scan did not find it: %v", want, types)
		}
	}
	var wrong []string
	checked := 0
	for _, file := range files {
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			for _, x := range finalizedNames(fn, types) {
				if !passesHandle(fn.Body, x) {
					continue
				}
				checked++
				if !startsWithKeepAlive(fn.Body, x) {
					wrong = append(wrong, fset.Position(fn.Pos()).String()+": "+fn.Name.Name+
						" passes "+x+"'s handle to C without `defer runtime.KeepAlive("+x+")` first")
				}
			}
		}
	}
	if checked == 0 {
		t.Fatal("no function passes a finalized handle to C: the scan reads nothing")
	}
	sort.Strings(wrong)
	for _, line := range wrong {
		t.Error(line)
	}
}

// The scan counts a handle that reaches C through a local or a helper, and leaves out a function
// that only calls methods of x or makes no call with its fields.
func TestTheHandleScanFollowsCopiesAndHelpers(t *testing.T) {
	source := `package grafeo

func direct(s *ResultStream) { C.grafeo_stream_next(s.handle) }

func copied(s *ResultStream) {
	h := s.handle
	C.grafeo_stream_next(h)
}

func helper(s *ResultStream) { next(s.handle) }

func methodOnly(s *ResultStream) { s.Close() }

func fieldOnly(s *ResultStream) bool { return s.closed }
`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "sample.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{"direct": true, "copied": true, "helper": true, "methodOnly": false, "fieldOnly": false}
	for _, decl := range file.Decls {
		fn := decl.(*ast.FuncDecl)
		if got := passesHandle(fn.Body, "s"); got != want[fn.Name.Name] {
			t.Errorf("%s: passesHandle %v, want %v", fn.Name.Name, got, want[fn.Name.Name])
		}
	}
}
