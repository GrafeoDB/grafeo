"""Every call on a closed database fails.

`close()` releases what the database holds. Every later call on it raises
`DatabaseClosedError` (a `GrafeoError` with code `GRAFEO-T007`), reads,
getters and `memory_usage()` included, and so does every call on what it
handed out before the close: sessions, transactions, result streams, graph
handles, time-travel views and the algorithm adapters. Only `close()` answers,
as a no-op. Before 0.6.4 reads went on after `close()`, and an in-memory
database kept taking writes.
"""

import asyncio
import inspect

import grafeo
import pytest

# Methods that do not act on the database instance: the opening constructors,
# and `close()`, which answers on a closed database as a no-op.
NOT_ON_A_CLOSED_INSTANCE = {
    "close",
    "open",
    "open_read_only",
    "open_in_memory",
    "restore_to_epoch",
}


def settle(result):
    """Awaits what an async call returns, so its failure surfaces here."""
    if inspect.isawaitable(result):

        async def wait():
            return await result

        return asyncio.run(wait())
    return result


def assert_closed_error(call, name):
    with pytest.raises(grafeo.DatabaseClosedError) as raised:
        settle(call())
    assert raised.value.error_code == "GRAFEO-T007", name


def public_names(cls):
    return {name for name in dir(cls) if not name.startswith("_")}


def database_calls(db, tmp_path):
    """One call per public method of `GrafeoDB`, with arguments its signature takes."""
    path = str(tmp_path / "copy.grafeo")
    csv = str(tmp_path / "rows.csv")
    vector = [0.1, 0.2, 0.3, 0.4]
    return {
        "storage_tiers": lambda: db.storage_tiers(),
        "reload_eligible": lambda: db.reload_eligible(),
        "execute": lambda: db.execute("MATCH (p:Person) RETURN p.name"),
        "execute_lazy": lambda: next(iter(db.execute_lazy("MATCH (p:Person) RETURN p.name"))),
        "execute_at_epoch": lambda: db.execute_at_epoch("MATCH (p:Person) RETURN p.name", 0),
        "as_of": lambda: db.as_of(0),
        "execute_as_of": lambda: db.execute_as_of("MATCH (p:Person) RETURN p.name", 0),
        "epoch_as_of": lambda: db.epoch_as_of(0),
        "query": lambda: db.query("MATCH (p:Person) RETURN p.name"),
        "execute_cypher": lambda: db.execute_cypher("MATCH (p:Person) RETURN p.name"),
        "execute_sql": lambda: db.execute_sql(
            "SELECT * FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))"
        ),
        "execute_async": lambda: db.execute_async("MATCH (p:Person) RETURN p.name"),
        "execute_gremlin": lambda: db.execute_gremlin("g.V()"),
        "execute_graphql": lambda: db.execute_graphql("{ Person { name } }"),
        "execute_sparql": lambda: db.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }"),
        "explain_sparql": lambda: db.explain_sparql("SELECT ?s WHERE { ?s ?p ?o }"),
        "explain": lambda: db.explain("MATCH (p:Person) RETURN p.name"),
        "explain_cypher": lambda: db.explain_cypher("MATCH (p:Person) RETURN p.name"),
        "explain_sql": lambda: db.explain_sql(
            "SELECT * FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))"
        ),
        "explain_gremlin": lambda: db.explain_gremlin("g.V()"),
        "validate_shacl": lambda: db.validate_shacl("shapes"),
        "execute_language": lambda: db.execute_language("gql", "MATCH (p:Person) RETURN p.name"),
        "create_node": lambda: db.create_node(["Person"], {"name": "Vincent"}),
        "create_edge": lambda: db.create_edge(0, 1, "KNOWS"),
        "get_node": lambda: db.get_node(0),
        "get_edge": lambda: db.get_edge(0),
        "get_node_edges": lambda: db.get_node_edges(0),
        "get_node_at_epoch": lambda: db.get_node_at_epoch(0, 0),
        "get_edge_at_epoch": lambda: db.get_edge_at_epoch(0, 0),
        "get_node_history": lambda: db.get_node_history(0),
        "get_edge_history": lambda: db.get_edge_history(0),
        "get_node_property_at_epoch": lambda: db.get_node_property_at_epoch(0, "name", 0),
        "get_node_property_history": lambda: db.get_node_property_history(0, "name"),
        "get_all_node_property_history": lambda: db.get_all_node_property_history(0),
        "current_epoch": lambda: db.current_epoch(),
        "get_nodes_by_label": lambda: db.get_nodes_by_label("Person"),
        "get_property_batch": lambda: db.get_property_batch([0], "name"),
        "get_properties_batch": lambda: db.get_properties_batch([0], ["name"]),
        "delete_node": lambda: db.delete_node(0),
        "detach_delete_node": lambda: db.detach_delete_node(0),
        "delete_edge": lambda: db.delete_edge(0),
        "set_node_property": lambda: db.set_node_property(0, "city", "Amsterdam"),
        "add_node_label": lambda: db.add_node_label(0, "Director"),
        "remove_node_label": lambda: db.remove_node_label(0, "Person"),
        "get_node_labels": lambda: db.get_node_labels(0),
        "set_edge_property": lambda: db.set_edge_property(0, "since", 2020),
        "remove_node_property": lambda: db.remove_node_property(0, "name"),
        "remove_edge_property": lambda: db.remove_edge_property(0, "since"),
        "create_property_index": lambda: db.create_property_index("name"),
        "create_vector_index": lambda: db.create_vector_index("Item", "embedding", dimensions=4),
        "drop_vector_index": lambda: db.drop_vector_index("Item", "embedding"),
        "rebuild_vector_index": lambda: db.rebuild_vector_index("Item", "embedding"),
        "vector_search": lambda: db.vector_search("Item", "embedding", vector, 1),
        "batch_create_nodes": lambda: db.batch_create_nodes("Item", "embedding", [vector]),
        "upsert_nodes": lambda: db.upsert_nodes(["Person"], [{"id": 1, "name": "Jules"}]),
        "update_nodes": lambda: db.update_nodes(["Person"], [{"id": 1, "name": "Jules"}]),
        "upsert_edges": lambda: db.upsert_edges("KNOWS", [{"id": 1, "src": 1, "dst": 2}]),
        "batch_create_nodes_with_props": lambda: db.batch_create_nodes_with_props(
            "Person", [{"name": "Mia"}]
        ),
        "batch_create_edges": lambda: db.batch_create_edges([(0, 1, "KNOWS")]),
        "load_nodes_csv": lambda: db.load_nodes_csv(csv, "Person"),
        "load_edges_csv": lambda: db.load_edges_csv(
            csv, "KNOWS", ("src", "Person", "id"), ("dst", "Person", "id")
        ),
        "load_rdf": lambda: db.load_rdf(str(tmp_path / "triples.ttl")),
        "load_rdf_text": lambda: db.load_rdf_text(
            '<http://ex.org/alix> <http://ex.org/city> "Amsterdam" .', "ntriples"
        ),
        "batch_vector_search": lambda: db.batch_vector_search("Item", "embedding", [vector], 1),
        "mmr_search": lambda: db.mmr_search("Item", "embedding", vector, 1),
        "create_text_index": lambda: db.create_text_index("Person", "name"),
        "drop_text_index": lambda: db.drop_text_index("Person", "name"),
        "rebuild_text_index": lambda: db.rebuild_text_index("Person", "name"),
        "text_search": lambda: db.text_search("Person", "name", "Alix", 1),
        "hybrid_search": lambda: db.hybrid_search("Item", "name", "embedding", "Alix", 1),
        "register_embedding_model": lambda: db.register_embedding_model(
            "model", str(tmp_path / "model.onnx"), str(tmp_path / "tokenizer.json")
        ),
        "embed_text": lambda: db.embed_text("model", ["Amsterdam"]),
        "vector_search_text": lambda: db.vector_search_text(
            "Item", "embedding", "model", "Amsterdam", 1
        ),
        "drop_property_index": lambda: db.drop_property_index("name"),
        "has_property_index": lambda: db.has_property_index("name"),
        "find_nodes_by_property": lambda: db.find_nodes_by_property("name", "Alix"),
        "begin_transaction": lambda: db.begin_transaction(),
        "session": lambda: db.session(),
        "begin_transaction_with_cdc": lambda: db.begin_transaction_with_cdc(True),
        "gc": lambda: db.gc(),
        "stats": lambda: db.stats(),
        "info": lambda: db.info(),
        "detailed_stats": lambda: db.detailed_stats(),
        "memory_usage": lambda: db.memory_usage(),
        "metrics": lambda: db.metrics(),
        "metrics_prometheus": lambda: db.metrics_prometheus(),
        "reset_metrics": lambda: db.reset_metrics(),
        "schema": lambda: db.schema(),
        "validate": lambda: db.validate(),
        "wal_status": lambda: db.wal_status(),
        "set_checkpoint_triggers": lambda: db.set_checkpoint_triggers(interval_secs=60.0),
        "checkpoint_status": lambda: db.checkpoint_status(),
        "wal_checkpoint": lambda: db.wal_checkpoint(),
        "save": lambda: db.save(path),
        "backup_full": lambda: db.backup_full(str(tmp_path / "backups")),
        "backup_incremental": lambda: db.backup_incremental(str(tmp_path / "backups")),
        "release_backup_cursor": lambda: db.release_backup_cursor(),
        "to_memory": lambda: db.to_memory(),
        "is_persistent": lambda: db.is_persistent,
        "path": lambda: db.path,
        "clear_plan_cache": lambda: db.clear_plan_cache(),
        "compact": lambda: db.compact(),
        "algorithms": lambda: db.algorithms,
        "as_networkx": lambda: db.as_networkx(),
        "as_solvor": lambda: db.as_solvor(),
        "node_count": lambda: db.node_count,
        "edge_count": lambda: db.edge_count,
        "nodes_to_arrow": lambda: db.nodes_to_arrow(),
        "nodes_to_polars": lambda: db.nodes_to_polars(),
        "nodes_to_pandas": lambda: db.nodes_to_pandas(),
        "nodes_df": lambda: db.nodes_df(),
        "edges_to_arrow": lambda: db.edges_to_arrow(),
        "edges_to_polars": lambda: db.edges_to_polars(),
        "edges_to_pandas": lambda: db.edges_to_pandas(),
        "edges_df": lambda: db.edges_df(),
        "import_df": lambda: db.import_df(None, "nodes", label="Person"),
        "import_csv": lambda: db.import_csv(csv),
        "import_jsonl": lambda: db.import_jsonl(str(tmp_path / "rows.jsonl")),
        "enable_cdc": lambda: db.enable_cdc(),
        "disable_cdc": lambda: db.disable_cdc(),
        "cdc_enabled": lambda: db.cdc_enabled,
        "node_history": lambda: db.node_history(0),
        "edge_history": lambda: db.edge_history(0),
        "node_history_since": lambda: db.node_history_since(0, 0),
        "changes_between": lambda: db.changes_between(0, 1),
        "set_schema": lambda: db.set_schema("berlin"),
        "reset_schema": lambda: db.reset_schema(),
        "current_schema": lambda: db.current_schema(),
        "create_graph": lambda: db.create_graph("prague"),
        "drop_graph": lambda: db.drop_graph("berlin"),
        "list_graphs": lambda: db.list_graphs(),
        "create_projection": lambda: db.create_projection("people", node_labels=["Person"]),
        "drop_projection": lambda: db.drop_projection("people"),
        "list_projections": lambda: db.list_projections(),
        "graph": lambda: db.graph("berlin"),
        "set_graph": lambda: db.set_graph("berlin"),
        "reset_graph": lambda: db.reset_graph(),
        "current_graph": lambda: db.current_graph(),
    }


def graph_handle_calls(graph):
    """One call per public method of a `GraphHandle`."""
    return {
        "algorithms": lambda: graph.algorithms,
        "name": lambda: graph.name,
        "execute": lambda: graph.execute("MATCH (p:Person) RETURN p.name"),
        "execute_cypher": lambda: graph.execute_cypher("MATCH (p:Person) RETURN p.name"),
        "execute_language": lambda: graph.execute_language("gql", "MATCH (p:Person) RETURN p.name"),
        "begin_transaction": lambda: graph.begin_transaction(),
        "create_node": lambda: graph.create_node(["Person"], {"name": "Butch"}),
        "create_edge": lambda: graph.create_edge(0, 1, "KNOWS"),
        "batch_create_nodes_with_props": lambda: graph.batch_create_nodes_with_props(
            "Person", [{"name": "Django"}]
        ),
        "batch_create_edges": lambda: graph.batch_create_edges([(0, 1, "KNOWS")]),
        "upsert_nodes": lambda: graph.upsert_nodes(["Person"], [{"id": 1, "name": "Hans"}]),
        "update_nodes": lambda: graph.update_nodes(["Person"], [{"id": 1, "name": "Hans"}]),
        "upsert_edges": lambda: graph.upsert_edges("KNOWS", [{"id": 1, "src": 1, "dst": 2}]),
        "get_node_edges": lambda: graph.get_node_edges(0),
        "get_node": lambda: graph.get_node(0),
        "get_edge": lambda: graph.get_edge(0),
        "set_node_property": lambda: graph.set_node_property(0, "city", "Berlin"),
        "set_edge_property": lambda: graph.set_edge_property(0, "since", 2020),
        "remove_node_property": lambda: graph.remove_node_property(0, "name"),
        "remove_edge_property": lambda: graph.remove_edge_property(0, "since"),
        "add_node_label": lambda: graph.add_node_label(0, "Director"),
        "remove_node_label": lambda: graph.remove_node_label(0, "Person"),
        "delete_node": lambda: graph.delete_node(0),
        "detach_delete_node": lambda: graph.detach_delete_node(0),
        "delete_edge": lambda: graph.delete_edge(0),
        "find_nodes_by_property": lambda: graph.find_nodes_by_property("name", "Alix"),
        "create_property_index": lambda: graph.create_property_index("name"),
        "drop_property_index": lambda: graph.drop_property_index("name"),
        "has_property_index": lambda: graph.has_property_index("name"),
    }


def open_database(kind, tmp_path):
    db = (
        grafeo.GrafeoDB(str(tmp_path / "amsterdam.grafeo"))
        if kind == "persistent"
        else grafeo.GrafeoDB()
    )
    db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
    db.create_graph("berlin")
    return db


def failures(calls, available):
    """The rows whose call did not raise `DatabaseClosedError` with `GRAFEO-T007`."""
    missed = []
    for name, call in sorted(calls.items()):
        if name not in available:
            continue
        try:
            settle(call())
        except grafeo.DatabaseClosedError as error:
            if error.error_code != "GRAFEO-T007":
                missed.append(f"{name}: {error.error_code}")
            continue
        except Exception as error:  # noqa: BLE001 - every other outcome is a miss
            missed.append(f"{name}: {type(error).__name__}: {error}")
            continue
        missed.append(f"{name}: answered")
    return missed


@pytest.mark.parametrize("kind", ["persistent", "in_memory"])
def test_every_method_of_a_closed_database_raises_database_closed_error(kind, tmp_path):
    db = open_database(kind, tmp_path)
    db.close()

    calls = database_calls(db, tmp_path)
    available = public_names(grafeo.GrafeoDB)
    unlisted = available - set(calls) - NOT_ON_A_CLOSED_INSTANCE
    assert not unlisted, f"public methods without a row in this table: {sorted(unlisted)}"
    assert failures(calls, available) == []


def test_the_closed_error_names_its_code_and_what_to_do(tmp_path):
    db = open_database("persistent", tmp_path)
    db.close()

    with pytest.raises(grafeo.DatabaseClosedError) as raised:
        db.execute("MATCH (p:Person) RETURN p.name")
    error = raised.value
    assert isinstance(error, grafeo.GrafeoError)
    assert isinstance(error, RuntimeError)
    assert error.error_code == "GRAFEO-T007"
    assert error.is_retryable is False
    assert "the database is closed: open it again" in str(error)


def test_close_twice_is_a_no_op(tmp_path):
    db = open_database("persistent", tmp_path)
    db.close()
    db.close()
    with db:
        pass


def test_a_closed_database_wrote_nothing_after_its_close(tmp_path):
    path = str(tmp_path / "prague.grafeo")
    db = grafeo.GrafeoDB(path)
    db.execute("INSERT (:Person {name: 'Alix'})")
    db.create_graph("berlin")
    sparql = hasattr(db, "execute_sparql")
    if sparql:
        db.execute_sparql('INSERT DATA { <http://ex.org/alix> <http://ex.org/city> "Amsterdam" . }')
    db.close()

    calls = [
        lambda: db.execute("INSERT (:Person {name: 'Gus'})"),
        lambda: db.create_node(["Person"], {"name": "Vincent"}),
        lambda: db.execute("CREATE CONSTRAINT person_name FOR (p:Person) ON (p.name) UNIQUE"),
        lambda: db.create_graph("prague"),
        lambda: db.drop_graph("berlin"),
        lambda: db.create_property_index("name"),
        lambda: db.save(str(tmp_path / "copy.grafeo")),
    ]
    if sparql:
        calls.append(
            lambda: db.execute_sparql(
                'INSERT DATA { <http://ex.org/gus> <http://ex.org/city> "Berlin" . }'
            )
        )
    for call in calls:
        with pytest.raises(grafeo.DatabaseClosedError):
            call()
    assert not (tmp_path / "copy.grafeo").exists(), "nothing was saved"

    reopened = grafeo.GrafeoDB(path)
    names = [row["name"] for row in reopened.execute("MATCH (p:Person) RETURN p.name AS name")]
    assert names == ["Alix"]
    assert reopened.list_graphs() == ["berlin"]
    assert not reopened.has_property_index("name")
    constraints = reopened.execute("SHOW CONSTRAINTS")
    assert len(constraints) == 0, "no constraint in the file"
    if sparql:
        triples = reopened.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
        assert len(triples) == 1, "the file holds Alix's triple only"
    reopened.close()


def test_every_method_of_a_graph_handle_held_across_close_raises(tmp_path):
    db = open_database("persistent", tmp_path)
    graph = db.graph("berlin")
    db.close()

    calls = graph_handle_calls(graph)
    available = public_names(type(graph))
    unlisted = available - set(calls)
    assert not unlisted, f"public methods without a row in this table: {sorted(unlisted)}"
    assert failures(calls, available) == []


def handed_out_calls(session, transaction, view, stream):
    """One call per public method of what the database hands out, by class."""
    query = "MATCH (p:Person) RETURN p.name"
    calls = {
        "Session": {
            "execute": lambda: session.execute(query),
            "execute_cypher": lambda: session.execute_cypher(query),
            "execute_sql": lambda: session.execute_sql(
                "SELECT * FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))"
            ),
            "execute_gremlin": lambda: session.execute_gremlin("g.V()"),
            "execute_graphql": lambda: session.execute_graphql("{ Person { name } }"),
            "execute_sparql": lambda: session.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }"),
            "execute_language": lambda: session.execute_language("gql", query),
            "execute_lazy": lambda: session.execute_lazy(query),
            "begin_transaction": lambda: session.begin_transaction(),
        },
        "Transaction": {
            "isolation_level": lambda: transaction.isolation_level,
            "is_active": lambda: transaction.is_active,
            "execute": lambda: transaction.execute(query),
            "execute_cypher": lambda: transaction.execute_cypher(query),
            "execute_sql": lambda: transaction.execute_sql(
                "SELECT * FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))"
            ),
            "execute_gremlin": lambda: transaction.execute_gremlin("g.V()"),
            "execute_graphql": lambda: transaction.execute_graphql("{ Person { name } }"),
            "execute_sparql": lambda: transaction.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }"),
            "execute_language": lambda: transaction.execute_language("gql", query),
            "savepoint": lambda: transaction.savepoint("before"),
            "rollback_to_savepoint": lambda: transaction.rollback_to_savepoint("before"),
            "release_savepoint": lambda: transaction.release_savepoint("before"),
            "commit": lambda: transaction.commit(),
            "rollback": lambda: transaction.rollback(),
        },
        "AsOfView": {
            "epoch": lambda: view.epoch,
            "execute": lambda: view.execute(query),
            "get_node": lambda: view.get_node(0),
            "get_edge": lambda: view.get_edge(0),
            "get_node_property": lambda: view.get_node_property(0, "name"),
        },
    }
    if stream is not None:
        calls["ResultStream"] = {
            "columns": lambda: stream.columns,
            "warnings": lambda: stream.warnings,
        }
    return calls


# What answers on a handed-out object after the database's close(): its own close(), a no-op.
HANDED_OUT_ANSWERS = {"close"}


def test_what_the_database_handed_out_before_close_raises_after_it(tmp_path):
    """Every public method of what the database hands out, each object obtained before
    `close()`: sessions, transactions, views and result streams, plus the adapters."""
    db = open_database("persistent", tmp_path)
    session = db.session()
    transaction = db.begin_transaction()
    transaction.execute("INSERT (:Person {name: 'Mia'})")
    view = db.as_of(db.current_epoch())
    stream = (
        db.execute_lazy("MATCH (p:Person) RETURN p.name") if hasattr(db, "execute_lazy") else None
    )
    algorithms = db.algorithms if hasattr(db, "algorithms") else None
    networkx = db.as_networkx() if hasattr(db, "as_networkx") else None
    db.close()

    calls = handed_out_calls(session, transaction, view, stream)
    objects = {
        "Session": session,
        "Transaction": transaction,
        "AsOfView": view,
        "ResultStream": stream,
    }
    for kind, rows in calls.items():
        available = public_names(type(objects[kind]))
        unlisted = available - set(rows) - HANDED_OUT_ANSWERS
        assert not unlisted, (
            f"{kind}: public methods without a row in this table: {sorted(unlisted)}"
        )
        assert failures(rows, available) == [], kind
    if stream is not None:
        assert failures({"__next__": lambda: next(stream)}, {"__next__"}) == []

    # The adapters reach the database for every call: one row each.
    adapters = {}
    if algorithms is not None:
        adapters["algorithms.pagerank"] = lambda: algorithms.pagerank()
        adapters["algorithms.connected_components"] = lambda: algorithms.connected_components()
    if networkx is not None:
        adapters["networkx.number_of_nodes"] = lambda: networkx.number_of_nodes
    assert failures(adapters, set(adapters)) == []

    reopened = grafeo.GrafeoDB(str(tmp_path / "amsterdam.grafeo"))
    names = sorted(
        row["name"] for row in reopened.execute("MATCH (p:Person) RETURN p.name AS name")
    )
    assert names == ["Alix", "Gus"], "the transaction open across close() left nothing"
    reopened.close()


@pytest.mark.requires("vector-index")
def test_a_spilled_vector_is_not_read_after_close(tmp_path):
    path = str(tmp_path / "rotterdam.grafeo")
    db = grafeo.GrafeoDB(path)
    db.execute("INSERT (:Item {embedding: [0.1, 0.2, 0.3, 0.4]})")
    db.create_vector_index("Item", "embedding", dimensions=4)
    db.close()

    db = grafeo.GrafeoDB(path, section_tiers={"VectorStore": "force_disk"})
    assert db.storage_tiers().get("VectorStore") == "on_disk", "the vectors are spilled"
    assert len(db.vector_search("Item", "embedding", [0.1, 0.2, 0.3, 0.4], 1)) == 1
    db.close()

    with pytest.raises(grafeo.DatabaseClosedError):
        db.vector_search("Item", "embedding", [0.1, 0.2, 0.3, 0.4], 1)
    with pytest.raises(grafeo.DatabaseClosedError):
        db.execute("MATCH (i:Item) RETURN i.embedding")
