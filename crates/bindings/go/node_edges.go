package grafeo

/*
#include "grafeo.h"
#include <stdlib.h>
*/
import "C"
import "runtime"

// Direction selects the edges GetNodeEdges lists.
type Direction int32

const (
	// DirectionOut lists the edges that leave the node.
	DirectionOut Direction = C.GRAFEO_DIRECTION_OUT
	// DirectionIn lists the edges that reach the node.
	DirectionIn Direction = C.GRAFEO_DIRECTION_IN
	// DirectionBoth lists both, a self-loop once.
	DirectionBoth Direction = C.GRAFEO_DIRECTION_BOTH
)

// NodeEdge is one edge of a node and the neighbor it reaches.
type NodeEdge struct {
	NeighborID uint64
	EdgeID     uint64
}

// GetNodeEdges lists the committed edges of a node in direction without a
// query, only those of edgeTypes when any are given. A node that does not
// exist has none.
func (db *Database) GetNodeEdges(nodeID uint64, direction Direction, edgeTypes ...string) ([]NodeEdge, error) {
	return nodeEdgesCall(edgeTypes, func(types *C.char, neighbors, edges **C.uint64_t, count *C.size_t) C.GrafeoStatus {
		return C.grafeo_get_node_edges(db.handle, C.uint64_t(nodeID), C.GrafeoDirection(direction), types, neighbors, edges, count)
	})
}

// GetNodeEdges is Database.GetNodeEdges in the graph.
func (g *Graph) GetNodeEdges(nodeID uint64, direction Direction, edgeTypes ...string) ([]NodeEdge, error) {
	return nodeEdgesCall(edgeTypes, func(types *C.char, neighbors, edges **C.uint64_t, count *C.size_t) C.GrafeoStatus {
		return C.grafeo_graph_get_node_edges(g.handle, C.uint64_t(nodeID), C.GrafeoDirection(direction), types, neighbors, edges, count)
	})
}

// nodeEdgesCall runs a call that hands out neighbor and edge arrays, on one OS
// thread, and copies and frees them.
func nodeEdgesCall(edgeTypes []string, call func(types *C.char, neighbors, edges **C.uint64_t, count *C.size_t) C.GrafeoStatus) ([]NodeEdge, error) {
	var cTypes *C.char
	if len(edgeTypes) > 0 {
		var err error
		if cTypes, err = cJSON(edgeTypes); err != nil {
			return nil, err
		}
		defer freeC(cTypes)
	}
	var neighbors, edges *C.uint64_t
	var count C.size_t
	runtime.LockOSThread()
	status := call(cTypes, &neighbors, &edges, &count)
	if status != C.GRAFEO_OK {
		err := statusToError(status)
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	neighborIDs := takeIDs(neighbors, count)
	edgeIDs := takeIDs(edges, count)
	result := make([]NodeEdge, len(edgeIDs))
	for i := range edgeIDs {
		result[i] = NodeEdge{NeighborID: neighborIDs[i], EdgeID: edgeIDs[i]}
	}
	return result, nil
}
