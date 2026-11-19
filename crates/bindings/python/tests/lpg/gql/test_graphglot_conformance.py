"""GraphGlot cross-validation: GQL conformance tests.

Compares Grafeo's GQL parser against GraphGlot (ISO/IEC 39075:2024
reference parser) using a comprehensive corpus of GQL queries.

Each query is classified into one of four buckets:
- both_pass: both parsers accept the query (ideal for GQL-standard queries)
- both_fail: neither parser accepts (malformed queries)
- grafeo_only: Grafeo parses but GraphGlot rejects (Cypher extensions / superset)
- graphglot_only: GraphGlot parses but Grafeo rejects (conformance gaps to fix)

Requires: uv pip install graphglot
"""

from __future__ import annotations

import pytest

graphglot_mod = pytest.importorskip("graphglot", reason="graphglot not installed")
from graphglot.dialect.fullgql import FullGQL  # noqa: E402

# ---------------------------------------------------------------------------
# Corpus: GQL queries organized by ISO feature category
# ---------------------------------------------------------------------------

CORPUS: dict[str, list[dict]] = {
    # -------------------------------------------------------------------
    # 1. Basic MATCH / RETURN
    # -------------------------------------------------------------------
    "match_basic": [
        {"id": "match_001", "query": "MATCH (n) RETURN n"},
        {"id": "match_002", "query": "MATCH (n:Person) RETURN n"},
        {"id": "match_003", "query": "MATCH (n:Person:Employee) RETURN n"},
        {"id": "match_004", "query": "MATCH (n {name: 'Alix'}) RETURN n"},
        {"id": "match_005", "query": "MATCH (n:Person {name: 'Alix', age: 30}) RETURN n"},
        {"id": "match_006", "query": "MATCH (n) RETURN n.name"},
        {"id": "match_007", "query": "MATCH (n) RETURN n.name, n.age"},
        {"id": "match_008", "query": "MATCH (n) RETURN DISTINCT n.name"},
        {"id": "match_009", "query": "MATCH (n) RETURN *"},
        {"id": "match_010", "query": "OPTIONAL MATCH (n:Person) RETURN n"},
    ],
    # -------------------------------------------------------------------
    # 2. IS label expressions (GQL-specific, ISO 39075)
    # -------------------------------------------------------------------
    "label_expression": [
        {"id": "label_001", "query": "MATCH (n IS Person) RETURN n"},
        {"id": "label_002", "query": "MATCH (n IS Person | Employee) RETURN n"},
        {"id": "label_003", "query": "MATCH (n IS Person & Employee) RETURN n"},
        {"id": "label_004", "query": "MATCH (n IS !Person) RETURN n"},
        {"id": "label_005", "query": "MATCH (n IS %) RETURN n"},
        {"id": "label_006", "query": "MATCH (n IS Person & !Employee) RETURN n"},
        {"id": "label_007", "query": "MATCH (n IS (Person | Employee) & !Contractor) RETURN n"},
    ],
    # -------------------------------------------------------------------
    # 3. Edge patterns and path patterns
    # -------------------------------------------------------------------
    "edge_patterns": [
        {"id": "edge_001", "query": "MATCH (a)-[r:KNOWS]->(b) RETURN a, b"},
        {"id": "edge_002", "query": "MATCH (a)<-[r:KNOWS]-(b) RETURN a, b"},
        {"id": "edge_003", "query": "MATCH (a)-[r:KNOWS]-(b) RETURN a, b"},
        {"id": "edge_004", "query": "MATCH (a)-[r:KNOWS|LIKES]->(b) RETURN a, b"},
        {"id": "edge_005", "query": "MATCH (a)-[r]->(b) RETURN a, b"},
        {"id": "edge_006", "query": "MATCH (a)-->(b) RETURN a, b"},
        {"id": "edge_007", "query": "MATCH (a)-[r {since: 2020}]->(b) RETURN a, b"},
        {"id": "edge_008", "query": "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, b.name"},
        {"id": "edge_009", "query": "MATCH (a)-[r:KNOWS]->(b)-[s:LIKES]->(c) RETURN a, b, c"},
        {"id": "edge_010", "query": "MATCH (a)~[r:KNOWS]~(b) RETURN a, b"},
    ],
    # -------------------------------------------------------------------
    # 4. Variable-length paths
    # -------------------------------------------------------------------
    "variable_length_paths": [
        {"id": "vpath_001", "query": "MATCH (a)-[r:KNOWS*1..3]->(b) RETURN a, b"},
        {"id": "vpath_002", "query": "MATCH (a)-[r:KNOWS*2]->(b) RETURN a, b"},
        {"id": "vpath_003", "query": "MATCH (a)-[r:KNOWS*]->(b) RETURN a, b"},
        {"id": "vpath_004", "query": "MATCH (a)-[r:KNOWS*0..5]->(b) RETURN a, b"},
        {"id": "vpath_005", "query": "MATCH (a)-[:KNOWS*1..]->(b) RETURN a, b"},
    ],
    # -------------------------------------------------------------------
    # 5. Quantified path patterns (ISO GQL)
    # -------------------------------------------------------------------
    "quantified_paths": [
        {"id": "qpath_001", "query": "MATCH (a)-[r:KNOWS]->{1,3}(b) RETURN a, b"},
        {"id": "qpath_002", "query": "MATCH (a)-[r:KNOWS]->{2}(b) RETURN a, b"},
        {"id": "qpath_003", "query": "MATCH (a)-[r]->{1,}(b) RETURN a, b"},
        {"id": "qpath_004", "query": "MATCH (a)(-[r:KNOWS]->(x)){2,5}(b) RETURN a, b"},
        {"id": "qpath_005", "query": "MATCH (a)-[]->{0,10}(b) RETURN a, b"},
        {"id": "qpath_006", "query": "MATCH (a)-[]->+(b) RETURN a, b"},
        {"id": "qpath_007", "query": "MATCH (a)-[]->*(b) RETURN a, b"},
    ],
    # -------------------------------------------------------------------
    # 6. Path modes (WALK, TRAIL, SIMPLE, ACYCLIC)
    # -------------------------------------------------------------------
    "path_modes": [
        {"id": "pmode_001", "query": "MATCH WALK (a)-[]->{1,5}(b) RETURN a, b"},
        {"id": "pmode_002", "query": "MATCH TRAIL (a)-[]->{1,5}(b) RETURN a, b"},
        {"id": "pmode_003", "query": "MATCH SIMPLE (a)-[]->{1,5}(b) RETURN a, b"},
        {"id": "pmode_004", "query": "MATCH ACYCLIC (a)-[]->{1,5}(b) RETURN a, b"},
    ],
    # -------------------------------------------------------------------
    # 7. Path search prefixes
    # -------------------------------------------------------------------
    "path_search": [
        {"id": "psearch_001", "query": "MATCH ANY SHORTEST (a)-[]->{1,5}(b) RETURN a, b"},
        {"id": "psearch_002", "query": "MATCH ALL SHORTEST (a)-[]->{1,5}(b) RETURN a, b"},
        {"id": "psearch_003", "query": "MATCH SHORTEST 3 (a)-[]->{1,5}(b) RETURN a, b"},
        {"id": "psearch_004", "query": "MATCH ANY (a)-[]->{1,5}(b) RETURN a, b"},
        {"id": "psearch_005", "query": "MATCH ALL (a)-[]->{1,5}(b) RETURN a, b"},
        {"id": "psearch_006", "query": "MATCH p = ANY SHORTEST (a)-[]->{1,5}(b) RETURN p"},
    ],
    # -------------------------------------------------------------------
    # 8. Match modes
    # -------------------------------------------------------------------
    "match_modes": [
        {"id": "mmode_001", "query": "MATCH DIFFERENT EDGES (a)-[e]->(b) RETURN a, b"},
        {"id": "mmode_002", "query": "MATCH REPEATABLE ELEMENTS (a)-[e]->(b) RETURN a, b"},
    ],
    # -------------------------------------------------------------------
    # 9. Named paths
    # -------------------------------------------------------------------
    "named_paths": [
        {"id": "npath_001", "query": "MATCH p = (a)-[:KNOWS]->(b) RETURN p"},
        {"id": "npath_002", "query": "MATCH p = (a)-[:KNOWS*1..3]->(b) RETURN p"},
    ],
    # -------------------------------------------------------------------
    # 10. WHERE clause
    # -------------------------------------------------------------------
    "where_clause": [
        {"id": "where_001", "query": "MATCH (n) WHERE n.age > 30 RETURN n"},
        {"id": "where_002", "query": "MATCH (n) WHERE n.age >= 18 AND n.age <= 65 RETURN n"},
        {"id": "where_003", "query": "MATCH (n) WHERE n.name = 'Alix' OR n.name = 'Gus' RETURN n"},
        {"id": "where_004", "query": "MATCH (n) WHERE NOT n.active RETURN n"},
        {"id": "where_005", "query": "MATCH (n) WHERE n.name IS NOT NULL RETURN n"},
        {"id": "where_006", "query": "MATCH (n) WHERE n.name IS NULL RETURN n"},
        {"id": "where_007", "query": "MATCH (n) WHERE n.name STARTS WITH 'A' RETURN n"},
        {"id": "where_008", "query": "MATCH (n) WHERE n.name ENDS WITH 'x' RETURN n"},
        {"id": "where_009", "query": "MATCH (n) WHERE n.name CONTAINS 'li' RETURN n"},
        {"id": "where_010", "query": "MATCH (n) WHERE n.age IN [25, 30, 35] RETURN n"},
        {"id": "where_011", "query": "MATCH (n) WHERE n.name LIKE 'A%' RETURN n"},
    ],
    # -------------------------------------------------------------------
    # 11. Element-pattern WHERE (inline)
    # -------------------------------------------------------------------
    "inline_where": [
        {"id": "iwhere_001", "query": "MATCH (n:Person WHERE n.age > 30) RETURN n"},
        {"id": "iwhere_002", "query": "MATCH (a)-[r:KNOWS WHERE r.since > 2020]->(b) RETURN a, b"},
    ],
    # -------------------------------------------------------------------
    # 12. RETURN clause features
    # -------------------------------------------------------------------
    "return_clause": [
        {"id": "ret_001", "query": "MATCH (n) RETURN n ORDER BY n.name"},
        {"id": "ret_002", "query": "MATCH (n) RETURN n ORDER BY n.name ASC"},
        {"id": "ret_003", "query": "MATCH (n) RETURN n ORDER BY n.name DESC"},
        {"id": "ret_004", "query": "MATCH (n) RETURN n ORDER BY n.name NULLS FIRST"},
        {"id": "ret_005", "query": "MATCH (n) RETURN n ORDER BY n.name NULLS LAST"},
        {"id": "ret_006", "query": "MATCH (n) RETURN n LIMIT 10"},
        {"id": "ret_007", "query": "MATCH (n) RETURN n SKIP 5"},
        {"id": "ret_008", "query": "MATCH (n) RETURN n SKIP 5 LIMIT 10"},
        {"id": "ret_009", "query": "MATCH (n) RETURN n OFFSET 5 LIMIT 10"},
        {"id": "ret_010", "query": "MATCH (n) RETURN n.name AS name, n.age AS age"},
        {"id": "ret_011", "query": "MATCH (n) RETURN n ORDER BY n.age DESC, n.name ASC LIMIT 5"},
        {"id": "ret_012", "query": "MATCH (n) RETURN n FETCH FIRST 10 ROWS ONLY"},
        {"id": "ret_013", "query": "MATCH (n) RETURN n FETCH NEXT 5 ROWS ONLY"},
    ],
    # -------------------------------------------------------------------
    # 13. Aggregation functions
    # -------------------------------------------------------------------
    "aggregation": [
        {"id": "agg_001", "query": "MATCH (n) RETURN COUNT(*)"},
        {"id": "agg_002", "query": "MATCH (n) RETURN COUNT(n)"},
        {"id": "agg_003", "query": "MATCH (n) RETURN COUNT(DISTINCT n.name)"},
        {"id": "agg_004", "query": "MATCH (n) RETURN SUM(n.age)"},
        {"id": "agg_005", "query": "MATCH (n) RETURN AVG(n.age)"},
        {"id": "agg_006", "query": "MATCH (n) RETURN MIN(n.age)"},
        {"id": "agg_007", "query": "MATCH (n) RETURN MAX(n.age)"},
        {"id": "agg_008", "query": "MATCH (n) RETURN COLLECT_LIST(n.name)"},
        {"id": "agg_009", "query": "MATCH (n) RETURN STDDEV_SAMP(n.age)"},
        {"id": "agg_010", "query": "MATCH (n) RETURN STDDEV_POP(n.age)"},
        {"id": "agg_011", "query": "MATCH (n) RETURN PERCENTILE_CONT(0.5, n.age)"},
        {"id": "agg_012", "query": "MATCH (n) RETURN PERCENTILE_DISC(0.5, n.age)"},
    ],
    # -------------------------------------------------------------------
    # 14. GROUP BY / HAVING
    # -------------------------------------------------------------------
    "group_by": [
        {
            "id": "grp_001",
            "query": "MATCH (n:Person) RETURN n.city, COUNT(*) GROUP BY n.city",
        },
        {
            "id": "grp_002",
            "query": (
                "MATCH (n:Person) RETURN n.city, AVG(n.age) GROUP BY n.city HAVING AVG(n.age) > 30"
            ),
        },
        {
            "id": "grp_003",
            "query": (
                "MATCH (n:Person) RETURN n.city, n.country, COUNT(*) GROUP BY n.city, n.country"
            ),
        },
    ],
    # -------------------------------------------------------------------
    # 15. WITH clause
    # -------------------------------------------------------------------
    "with_clause": [
        {"id": "with_001", "query": "MATCH (n) WITH n RETURN n"},
        {"id": "with_002", "query": "MATCH (n) WITH n WHERE n.age > 30 RETURN n"},
        {"id": "with_003", "query": "MATCH (n) WITH n.name AS name RETURN name"},
        {"id": "with_004", "query": "MATCH (n) WITH * RETURN n"},
        {"id": "with_005", "query": "MATCH (n) WITH DISTINCT n.city AS city RETURN city"},
    ],
    # -------------------------------------------------------------------
    # 16. UNWIND / FOR
    # -------------------------------------------------------------------
    "unwind_for": [
        {"id": "unwind_001", "query": "UNWIND [1, 2, 3] AS x RETURN x"},
        {"id": "unwind_002", "query": "MATCH (n) UNWIND n.tags AS tag RETURN tag"},
        {"id": "for_001", "query": "FOR x IN [1, 2, 3] RETURN x"},
        {"id": "for_002", "query": "FOR x IN [1, 2, 3] WITH ORDINALITY i RETURN x, i"},
        {"id": "for_003", "query": "FOR x IN [1, 2, 3] WITH OFFSET i RETURN x, i"},
    ],
    # -------------------------------------------------------------------
    # 17. Data modification
    # -------------------------------------------------------------------
    "data_modification": [
        {"id": "dml_001", "query": "INSERT (:Person {name: 'Alix', age: 30})"},
        {
            "id": "dml_002",
            "query": "INSERT (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person {name: 'Gus'})",
        },
        {"id": "dml_003", "query": "MATCH (n:Person {name: 'Alix'}) SET n.age = 31"},
        {
            "id": "dml_004",
            "query": "MATCH (n:Person {name: 'Alix'}) SET n = {name: 'Alix', age: 31}",
        },
        {"id": "dml_005", "query": "MATCH (n:Person {name: 'Alix'}) SET n += {age: 31}"},
        {"id": "dml_006", "query": "MATCH (n:Person {name: 'Alix'}) SET n:Employee"},
        {"id": "dml_007", "query": "MATCH (n:Person {name: 'Alix'}) REMOVE n.age"},
        {"id": "dml_008", "query": "MATCH (n:Person {name: 'Alix'}) REMOVE n:Employee"},
        {"id": "dml_009", "query": "MATCH (n:Person {name: 'Alix'}) DELETE n"},
        {"id": "dml_010", "query": "MATCH (n:Person {name: 'Alix'}) DETACH DELETE n"},
        {"id": "dml_011", "query": "MATCH (n:Person {name: 'Alix'}) NODETACH DELETE n"},
        {
            "id": "dml_012",
            "query": (
                "MERGE (n:Person {name: 'Alix'}) "
                "ON CREATE SET n.created = true "
                "ON MATCH SET n.seen = true"
            ),
        },
    ],
    # -------------------------------------------------------------------
    # 18. Composite queries
    # -------------------------------------------------------------------
    "composite_queries": [
        {
            "id": "comp_001",
            "query": "MATCH (n:Person) RETURN n.name UNION MATCH (n:Company) RETURN n.name",
        },
        {
            "id": "comp_002",
            "query": "MATCH (n:Person) RETURN n.name UNION ALL MATCH (n:Company) RETURN n.name",
        },
        {
            "id": "comp_003",
            "query": "MATCH (n:Person) RETURN n.name EXCEPT MATCH (n:Employee) RETURN n.name",
        },
        {
            "id": "comp_004",
            "query": ("MATCH (n:Person) RETURN n.name EXCEPT ALL MATCH (n:Employee) RETURN n.name"),
        },
        {
            "id": "comp_005",
            "query": ("MATCH (n:Person) RETURN n.name INTERSECT MATCH (n:Employee) RETURN n.name"),
        },
        {
            "id": "comp_006",
            "query": (
                "MATCH (n:Person) RETURN n.name INTERSECT ALL MATCH (n:Employee) RETURN n.name"
            ),
        },
        {
            "id": "comp_007",
            "query": ("MATCH (n:Person) RETURN n.name OTHERWISE MATCH (n:Company) RETURN n.name"),
        },
    ],
    # -------------------------------------------------------------------
    # 19. Subqueries (EXISTS, COUNT, inline CALL)
    # -------------------------------------------------------------------
    "subqueries": [
        {
            "id": "subq_001",
            "query": "MATCH (n) WHERE EXISTS { MATCH (n)-[:KNOWS]->(m) } RETURN n",
        },
        {
            "id": "subq_002",
            "query": "MATCH (n) WHERE COUNT { MATCH (n)-[:KNOWS]->(m) } > 3 RETURN n",
        },
        {
            "id": "subq_003",
            "query": (
                "MATCH (n) CALL { MATCH (n)-[:KNOWS]->(m) RETURN COUNT(m) AS cnt } RETURN n, cnt"
            ),
        },
    ],
    # -------------------------------------------------------------------
    # 20. Expressions and operators
    # -------------------------------------------------------------------
    "expressions": [
        {"id": "expr_001", "query": "RETURN 1 + 2"},
        {"id": "expr_002", "query": "RETURN 10 - 3"},
        {"id": "expr_003", "query": "RETURN 2 * 5"},
        {"id": "expr_004", "query": "RETURN 10 / 3"},
        {"id": "expr_005", "query": "RETURN 10 MOD 3"},
        {"id": "expr_006", "query": "RETURN -5"},
        {"id": "expr_007", "query": "RETURN NOT true"},
        {"id": "expr_008", "query": "RETURN true AND false"},
        {"id": "expr_009", "query": "RETURN true OR false"},
        {"id": "expr_010", "query": "RETURN true XOR false"},
        {"id": "expr_011", "query": "RETURN 'hello' || ' world'"},
        {"id": "expr_012", "query": "RETURN [1, 2] || [3, 4]"},
        {"id": "expr_013", "query": "RETURN CASE WHEN 1 > 0 THEN 'yes' ELSE 'no' END"},
        {
            "id": "expr_014",
            "query": "RETURN CASE 1 WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END",
        },
        {"id": "expr_015", "query": "RETURN NULLIF(1, 1)"},
        {"id": "expr_016", "query": "RETURN COALESCE(null, 1, 2)"},
    ],
    # -------------------------------------------------------------------
    # 21. List expressions
    # -------------------------------------------------------------------
    "list_expressions": [
        {"id": "list_001", "query": "RETURN [1, 2, 3]"},
        {"id": "list_002", "query": "RETURN [1, 2, 3][0]"},
        {"id": "list_003", "query": "MATCH (n) RETURN [x IN n.scores WHERE x > 50]"},
        {"id": "list_004", "query": "MATCH (n) RETURN [x IN n.scores WHERE x > 50 | x * 2]"},
        {"id": "list_005", "query": "MATCH (n) WHERE all(x IN n.scores WHERE x > 50) RETURN n"},
        {"id": "list_006", "query": "MATCH (n) WHERE any(x IN n.scores WHERE x > 50) RETURN n"},
        {"id": "list_007", "query": "MATCH (n) WHERE none(x IN n.scores WHERE x > 50) RETURN n"},
        {
            "id": "list_008",
            "query": "MATCH (n) WHERE single(x IN n.scores WHERE x > 50) RETURN n",
        },
    ],
    # -------------------------------------------------------------------
    # 22. Type functions and CAST
    # -------------------------------------------------------------------
    "type_cast": [
        {"id": "cast_001", "query": "RETURN CAST(42 AS STRING)"},
        {"id": "cast_002", "query": "RETURN CAST('42' AS INTEGER)"},
        {"id": "cast_003", "query": "RETURN CAST(3.14 AS INTEGER)"},
        {"id": "cast_004", "query": "RETURN CAST('true' AS BOOLEAN)"},
        {"id": "cast_005", "query": "RETURN CAST(42 AS DOUBLE)"},
    ],
    # -------------------------------------------------------------------
    # 23. Temporal literals and functions
    # -------------------------------------------------------------------
    "temporal": [
        {"id": "temp_001", "query": "RETURN DATE '2024-01-15'"},
        {"id": "temp_002", "query": "RETURN TIME '14:30:00'"},
        {"id": "temp_003", "query": "RETURN DATETIME '2024-01-15T14:30:00'"},
        {"id": "temp_004", "query": "RETURN DURATION 'P1Y2M3D'"},
        {"id": "temp_005", "query": "RETURN DURATION 'PT1H30M'"},
        {"id": "temp_006", "query": "RETURN CURRENT_DATE"},
        {"id": "temp_007", "query": "RETURN CURRENT_TIME"},
        {"id": "temp_008", "query": "RETURN CURRENT_TIMESTAMP"},
    ],
    # -------------------------------------------------------------------
    # 24. String functions
    # -------------------------------------------------------------------
    "string_functions": [
        {"id": "sfn_001", "query": "RETURN UPPER('hello')"},
        {"id": "sfn_002", "query": "RETURN LOWER('HELLO')"},
        {"id": "sfn_003", "query": "RETURN LEFT('abcdef', 3)"},
        {"id": "sfn_004", "query": "RETURN RIGHT('abcdef', 3)"},
        {"id": "sfn_005", "query": "RETURN CHAR_LENGTH('hello')"},
        {"id": "sfn_006", "query": "RETURN TRIM(LEADING ' ' FROM '  hello  ')"},
    ],
    # -------------------------------------------------------------------
    # 25. Numeric functions
    # -------------------------------------------------------------------
    "numeric_functions": [
        {"id": "nfn_001", "query": "RETURN ABS(-5)"},
        {"id": "nfn_002", "query": "RETURN FLOOR(3.7)"},
        {"id": "nfn_003", "query": "RETURN CEILING(3.2)"},
        {"id": "nfn_004", "query": "RETURN SQRT(16)"},
        {"id": "nfn_005", "query": "RETURN POWER(2, 10)"},
        {"id": "nfn_006", "query": "RETURN MOD(10, 3)"},
        {"id": "nfn_007", "query": "RETURN LOG(100, 10)"},
        {"id": "nfn_008", "query": "RETURN LOG10(100)"},
        {"id": "nfn_009", "query": "RETURN LN(2.718)"},
        {"id": "nfn_010", "query": "RETURN EXP(1)"},
        {"id": "nfn_011", "query": "RETURN SIN(3.14159)"},
    ],
    # -------------------------------------------------------------------
    # 26. Parameters
    # -------------------------------------------------------------------
    "parameters": [
        {"id": "param_001", "query": "MATCH (n {name: $name}) RETURN n"},
        {"id": "param_002", "query": "MATCH (n) WHERE n.age > $minAge RETURN n"},
    ],
    # -------------------------------------------------------------------
    # 27. CALL procedure
    # -------------------------------------------------------------------
    "procedures": [
        {"id": "proc_001", "query": "CALL my_procedure() YIELD result RETURN result"},
        {"id": "proc_002", "query": "CALL db.schema.nodeTypes() YIELD nodeType RETURN nodeType"},
    ],
    # -------------------------------------------------------------------
    # 28. Session commands
    # -------------------------------------------------------------------
    "session_commands": [
        {"id": "sess_001", "query": "USE GRAPH myGraph"},
        {"id": "sess_002", "query": "SESSION SET GRAPH myGraph"},
        {"id": "sess_003", "query": "SESSION SET SCHEMA mySchema"},
        {"id": "sess_004", "query": "SESSION SET TIME ZONE 'UTC'"},
        {"id": "sess_005", "query": "SESSION RESET"},
        {"id": "sess_006", "query": "SESSION CLOSE"},
    ],
    # -------------------------------------------------------------------
    # 29. Transaction commands
    # -------------------------------------------------------------------
    "transaction_commands": [
        {"id": "tx_001", "query": "START TRANSACTION"},
        {"id": "tx_002", "query": "START TRANSACTION READ ONLY"},
        {"id": "tx_003", "query": "START TRANSACTION READ WRITE"},
        {"id": "tx_004", "query": "COMMIT"},
        {"id": "tx_005", "query": "ROLLBACK"},
    ],
    # -------------------------------------------------------------------
    # 30. Schema DDL
    # -------------------------------------------------------------------
    "schema_ddl": [
        {"id": "ddl_001", "query": "CREATE GRAPH myGraph"},
        {"id": "ddl_002", "query": "CREATE GRAPH IF NOT EXISTS myGraph"},
        {"id": "ddl_003", "query": "DROP GRAPH myGraph"},
        {"id": "ddl_004", "query": "DROP GRAPH IF EXISTS myGraph"},
        {"id": "ddl_005", "query": "CREATE PROPERTY GRAPH myGraph LIKE CURRENT_GRAPH"},
    ],
    # -------------------------------------------------------------------
    # 31. Graph type DDL
    # -------------------------------------------------------------------
    "graph_type_ddl": [
        {
            "id": "gtype_001",
            "query": "CREATE GRAPH TYPE myType { (n: Person {name STRING, age INTEGER}) }",
        },
        {
            "id": "gtype_002",
            "query": (
                "CREATE GRAPH TYPE myType { (n: Person {name STRING})-[r: KNOWS]->(m: Person) }"
            ),
        },
        {"id": "gtype_003", "query": "DROP GRAPH TYPE myType"},
        {"id": "gtype_004", "query": "DROP GRAPH TYPE IF EXISTS myType"},
    ],
    # -------------------------------------------------------------------
    # 32. FILTER statement (GQL-specific)
    # -------------------------------------------------------------------
    "filter_statement": [
        {"id": "filter_001", "query": "MATCH (n) FILTER WHERE n.age > 30 RETURN n"},
        {
            "id": "filter_002",
            "query": "MATCH (n) FILTER WHERE n.age > 30 AND n.name <> 'Gus' RETURN n",
        },
    ],
    # -------------------------------------------------------------------
    # 33. LET bindings
    # -------------------------------------------------------------------
    "let_bindings": [
        {"id": "let_001", "query": "MATCH (n) LET x = n.age * 2 RETURN x"},
        {"id": "let_002", "query": "RETURN LET x = 1 IN [x, x + 1, x * 10] END"},
    ],
    # -------------------------------------------------------------------
    # 34. SELECT statement (SQL-style GQL)
    # -------------------------------------------------------------------
    "select_statement": [
        {"id": "sel_001", "query": "SELECT * FROM CURRENT_GRAPH MATCH (n)"},
        {"id": "sel_002", "query": "SELECT n.name FROM CURRENT_GRAPH MATCH (n)"},
        {
            "id": "sel_003",
            "query": ("SELECT n.name, COUNT(*) FROM CURRENT_GRAPH MATCH (n) GROUP BY n.name"),
        },
    ],
    # -------------------------------------------------------------------
    # 35. Literal types
    # -------------------------------------------------------------------
    "literals": [
        {"id": "lit_001", "query": "RETURN true"},
        {"id": "lit_002", "query": "RETURN false"},
        {"id": "lit_003", "query": "RETURN null"},
        {"id": "lit_004", "query": "RETURN 42"},
        {"id": "lit_005", "query": "RETURN 3.14"},
        {"id": "lit_006", "query": "RETURN 'hello'"},
        {"id": "lit_007", "query": "RETURN 0xFF"},
        {"id": "lit_008", "query": "RETURN 1.5e10"},
        {"id": "lit_009", "query": "RETURN [1, 'two', true, null]"},
        {"id": "lit_010", "query": "RETURN {name: 'Alix', age: 30}"},
    ],
    # -------------------------------------------------------------------
    # 36. Predicates (GQL-specific)
    # -------------------------------------------------------------------
    "predicates": [
        {"id": "pred_001", "query": "MATCH (n) WHERE PROPERTY_EXISTS(n, 'name') RETURN n"},
        {
            "id": "pred_002",
            "query": "MATCH (n) WHERE EXISTS { MATCH (n)-[:KNOWS]->() } RETURN n",
        },
    ],
    # -------------------------------------------------------------------
    # 37. EXPLAIN / PROFILE
    # -------------------------------------------------------------------
    "explain_profile": [
        {"id": "expl_001", "query": "EXPLAIN MATCH (n) RETURN n"},
        {"id": "expl_002", "query": "PROFILE MATCH (n) RETURN n"},
    ],
    # -------------------------------------------------------------------
    # 38. Path pattern union/multiset alternation
    # -------------------------------------------------------------------
    "path_alternation": [
        {"id": "palt_001", "query": "MATCH (a)-[:T1]->(b) | (a)-[:T2]->(c) RETURN a"},
        {"id": "palt_002", "query": "MATCH (a)-[:T1]->(b) |+| (a)-[:T2]->(c) RETURN a"},
    ],
    # -------------------------------------------------------------------
    # 39. Complex multi-clause queries
    # -------------------------------------------------------------------
    "complex_queries": [
        {
            "id": "cmplx_001",
            "query": (
                "MATCH (p:Person)-[:LIVES_IN]->(c:City) WHERE p.age > 25 "
                "WITH c.name AS city, COUNT(*) AS pop ORDER BY pop DESC LIMIT 5 "
                "RETURN city, pop"
            ),
        },
        {
            "id": "cmplx_002",
            "query": (
                "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
                "WHERE a <> c AND NOT EXISTS { MATCH (a)-[:KNOWS]->(c) } "
                "RETURN a.name, c.name"
            ),
        },
        {
            "id": "cmplx_003",
            "query": (
                "MATCH (n:Person) WITH n ORDER BY n.age DESC LIMIT 10 "
                "MATCH (n)-[:WORKS_AT]->(c:Company) RETURN n.name, c.name"
            ),
        },
    ],
    # -------------------------------------------------------------------
    # 40. OPTIONAL blocks (GQL-specific)
    # -------------------------------------------------------------------
    "optional_blocks": [
        {
            "id": "opt_001",
            "query": "MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(m) RETURN n, m",
        },
    ],
    # -------------------------------------------------------------------
    # 41. Savepoints
    # -------------------------------------------------------------------
    "savepoints": [
        {"id": "sp_001", "query": "SAVEPOINT sp1"},
        {"id": "sp_002", "query": "ROLLBACK TO SAVEPOINT sp1"},
        {"id": "sp_003", "query": "RELEASE SAVEPOINT sp1"},
    ],
}

# All categories in CORPUS
ALL_CATEGORIES = list(CORPUS.keys())

# Markers in Grafeo error messages that indicate a parse-level failure.
# Execution errors (semantic, runtime) mean the parser accepted the query.
_PARSE_ERROR_MARKERS = ("syntax error", "lexer error")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_parse_error(error_msg: str) -> bool:
    """True if the error message indicates a parser rejection."""
    lower = error_msg.lower()
    return any(marker in lower for marker in _PARSE_ERROR_MARKERS)


def _graphglot_accepts(dialect: FullGQL, query: str) -> tuple[bool, str | None, str | None]:
    """Parse *query* with GraphGlot. Returns (success, normalized_form, error)."""
    try:
        parsed = dialect.parse(query)
        vr = dialect.validate(query)
        if parsed and vr.success:
            try:
                normalized = dialect.generate(parsed[0])
            except Exception:
                normalized = None
            return True, normalized, None
        error_msg = str(vr.error) if vr.error else "parse returned empty"
        return False, None, error_msg
    except Exception as e:
        return False, None, str(e)


def _grafeo_accepts(db, query: str) -> tuple[bool, str | None]:
    """Execute *query* via Grafeo. Returns (parser_accepted, error_message).

    A query is considered parser-accepted if execution succeeds OR if it fails
    with a non-parse error (semantic, execution, transaction, etc.).
    """
    try:
        list(db.execute(query))
        return True, None
    except RuntimeError as e:
        msg = str(e)
        if _is_parse_error(msg):
            return False, msg
        # Non-parse error: the parser accepted it, execution failed
        return True, None
    except Exception:
        # Non-query errors (ValueError, etc.): assume parser accepted
        return True, None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dialect():
    """Shared GraphGlot FullGQL dialect instance."""
    return FullGQL()


@pytest.fixture(scope="module")
def grafeo_db():
    """Shared Grafeo DB for parse-level testing (empty, no data needed)."""
    import grafeo

    return grafeo.GrafeoDB()


# ---------------------------------------------------------------------------
# Master conformance report (run with pytest -s to see output)
# ---------------------------------------------------------------------------


class TestGraphGlotConformance:
    """Cross-validation report comparing Grafeo and GraphGlot parsers."""

    def test_conformance_report(self, dialect, grafeo_db):
        both_pass = []
        both_fail = []
        grafeo_only = []
        graphglot_only = []

        for category, queries in CORPUS.items():
            for entry in queries:
                qid = entry["id"]
                query = entry["query"]

                gg_ok, gg_normalized, gg_err = _graphglot_accepts(dialect, query)
                gr_ok, gr_err = _grafeo_accepts(grafeo_db, query)

                record = {
                    "id": qid,
                    "category": category,
                    "query": query,
                    "graphglot_ok": gg_ok,
                    "graphglot_normalized": gg_normalized,
                    "graphglot_error": gg_err,
                    "grafeo_ok": gr_ok,
                    "grafeo_error": gr_err,
                }

                match (gr_ok, gg_ok):
                    case (True, True):
                        both_pass.append(record)
                    case (False, False):
                        both_fail.append(record)
                    case (True, False):
                        grafeo_only.append(record)
                    case (False, True):
                        graphglot_only.append(record)

        total = sum(len(q) for q in CORPUS.values())

        print(f"\n{'=' * 72}")
        print("  GraphGlot Cross-Validation Report")
        print(f"  Corpus: {total} queries")
        print("=" * 72)

        pct = lambda n: f"{n / total * 100:.0f}%" if total else "0%"  # noqa: E731
        print(
            f"\n  Both pass (standard GQL, both agree):  {len(both_pass)} ({pct(len(both_pass))})"
        )
        print(f"  Both fail (expected parse errors):      {len(both_fail)} ({pct(len(both_fail))})")
        go = len(grafeo_only)
        glo = len(graphglot_only)
        print(f"  Grafeo only (Cypher superset):          {go} ({pct(go)})")
        print(f"  GraphGlot only (conformance gaps):      {glo} ({pct(glo)})")

        if both_fail:
            print(f"\n{'-' * 72}")
            print("  BOTH FAIL (neither parser accepts):")
            print("-" * 72)
            for r in both_fail:
                print(f"  [{r['id']}] {r['category']}")
                print(f"    Query: {r['query']}")
                if r["grafeo_error"]:
                    print(f"    Grafeo error: {r['grafeo_error']}")
                if r["graphglot_error"]:
                    print(f"    GraphGlot error: {r['graphglot_error']}")
                print()

        if graphglot_only:
            print(f"\n{'-' * 72}")
            print("  CONFORMANCE GAPS (GraphGlot parses, Grafeo fails):")
            print("-" * 72)
            for r in graphglot_only:
                print(f"  [{r['id']}] {r['category']}")
                print(f"    Query: {r['query']}")
                if r["grafeo_error"]:
                    print(f"    Grafeo error: {r['grafeo_error']}")
                if r["graphglot_normalized"]:
                    print(f"    GraphGlot normalized: {r['graphglot_normalized']}")
                print()

        if grafeo_only:
            print(f"{'-' * 72}")
            print("  CYPHER SUPERSET (Grafeo parses, GraphGlot rejects):")
            print("-" * 72)
            for r in grafeo_only:
                print(f"  [{r['id']}] {r['category']} :: {r['query']}")
            print()

        # Round-trip check
        roundtrip_ok = 0
        roundtrip_fail = []
        for r in both_pass:
            normalized = r["graphglot_normalized"]
            if normalized and normalized.strip() != r["query"].strip():
                gr_ok, gr_err = _grafeo_accepts(grafeo_db, normalized.strip())
                if gr_ok:
                    roundtrip_ok += 1
                else:
                    roundtrip_fail.append(r | {"roundtrip_error": gr_err})

        if roundtrip_fail:
            print(f"{'-' * 72}")
            print("  ROUND-TRIP FAILURES (Grafeo can't parse GraphGlot's normalized form):")
            print("-" * 72)
            for r in roundtrip_fail:
                print(f"  [{r['id']}] {r['category']}")
                print(f"    Original:   {r['query']}")
                print(f"    Normalized: {r['graphglot_normalized']}")
                print(f"    Error: {r['roundtrip_error']}")
                print()

        print("=" * 72)
        print(f"  Round-trip (normalized form): {roundtrip_ok} OK, {len(roundtrip_fail)} FAIL")
        print("=" * 72)


# ---------------------------------------------------------------------------
# Per-category tests
# ---------------------------------------------------------------------------


# Queries GraphGlot accepts and Grafeo refuses by a syntax rule GraphGlot does not check:
# ISO/IEC 39075 16.4, syntax rule 14 (an unbounded quantifier needs a path mode or a
# selector). Grafeo's parser reads them; the refusal (42001) comes before they run.
REFUSED_BY_A_SYNTAX_RULE = {
    "qpath_003": "no upper bound",
    "qpath_006": "no upper bound",
    "qpath_007": "no upper bound",
}


def _assert_category(dialect: FullGQL, db, category: str) -> None:
    """Assert all GraphGlot-accepted queries in *category* also parse in Grafeo, except those
    a syntax rule refuses (``REFUSED_BY_A_SYNTAX_RULE``), which must be refused for it."""
    queries = CORPUS.get(category, [])

    # Only test queries that GraphGlot accepts (valid GQL)
    gg_valid = []
    for entry in queries:
        ok, _, _ = _graphglot_accepts(dialect, entry["query"])
        if ok:
            gg_valid.append(entry)

    if not gg_valid:
        return

    failures = []
    for entry in gg_valid:
        gr_ok, gr_err = _grafeo_accepts(db, entry["query"])
        refused_for = REFUSED_BY_A_SYNTAX_RULE.get(entry["id"])
        if refused_for is not None:
            if gr_ok or refused_for not in (gr_err or ""):
                failures.append(
                    f"[{entry['id']}] {entry['query']}\n  Expected the refusal: {gr_err}"
                )
            continue
        if not gr_ok:
            failures.append(f"[{entry['id']}] {entry['query']}\n  Error: {gr_err}")

    assert not failures, (
        f"Grafeo failed to parse {len(failures)}/{len(gg_valid)} "
        f"GraphGlot-valid queries in '{category}':\n" + "\n".join(failures)
    )


class TestGqlCategories:
    """Per-category conformance: each GraphGlot-valid query must parse in Grafeo."""

    def test_match_basic(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "match_basic")

    def test_label_expressions(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "label_expression")

    def test_edge_patterns(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "edge_patterns")

    def test_variable_length_paths(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "variable_length_paths")

    def test_quantified_paths(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "quantified_paths")

    def test_path_modes(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "path_modes")

    def test_path_search(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "path_search")

    def test_match_modes(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "match_modes")

    def test_named_paths(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "named_paths")

    def test_where_clause(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "where_clause")

    def test_inline_where(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "inline_where")

    def test_return_clause(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "return_clause")

    def test_aggregation(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "aggregation")

    def test_group_by(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "group_by")

    def test_with_clause(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "with_clause")

    def test_unwind_for(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "unwind_for")

    def test_data_modification(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "data_modification")

    def test_composite_queries(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "composite_queries")

    def test_subqueries(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "subqueries")

    def test_expressions(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "expressions")

    def test_list_expressions(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "list_expressions")

    def test_type_cast(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "type_cast")

    def test_temporal(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "temporal")

    def test_string_functions(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "string_functions")

    def test_numeric_functions(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "numeric_functions")

    def test_parameters(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "parameters")

    def test_procedures(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "procedures")

    def test_session_commands(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "session_commands")

    def test_transaction_commands(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "transaction_commands")

    def test_schema_ddl(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "schema_ddl")

    def test_graph_type_ddl(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "graph_type_ddl")

    def test_filter_statement(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "filter_statement")

    def test_let_bindings(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "let_bindings")

    def test_select_statement(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "select_statement")

    def test_literals(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "literals")

    def test_predicates(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "predicates")

    def test_explain_profile(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "explain_profile")

    def test_path_alternation(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "path_alternation")

    def test_complex_queries(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "complex_queries")

    def test_optional_blocks(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "optional_blocks")

    def test_savepoints(self, dialect, grafeo_db):
        _assert_category(dialect, grafeo_db, "savepoints")
