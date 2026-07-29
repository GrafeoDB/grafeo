//! Integration tests for WAL-based persistence and recovery.
//!
//! Covers: save/restore cycle, WAL recovery of nodes/edges/properties/labels,
//! and checkpoint operations.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test wal_recovery
//! ```

#[cfg(feature = "wal")]
mod wal {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn test_persistent_db_roundtrip() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test.grafeo");

        // Create and populate
        {
            let db = GrafeoDB::open(&path).expect("open for write");
            let a = db.create_node(&["Person"]).unwrap();
            db.set_node_property(a, "name", Value::String("Alix".into()))
                .unwrap();
            db.set_node_property(a, "age", Value::Int64(30)).unwrap();
            let b = db.create_node(&["Person"]).unwrap();
            db.set_node_property(b, "name", Value::String("Gus".into()))
                .unwrap();
            db.create_edge(a, b, "KNOWS");
            db.close().expect("close");
        }

        // Reopen and verify
        {
            let db = GrafeoDB::open(&path).expect("reopen");
            assert_eq!(db.node_count(), 2);
            assert_eq!(db.edge_count(), 1);

            let session = db.session();
            let result = session
                .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age")
                .unwrap();
            assert_eq!(result.rows().len(), 1);
            assert_eq!(result.rows()[0][0], Value::Int64(30));

            db.close().expect("close");
        }
    }

    #[test]
    fn test_wal_recovery_labels() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("labels.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let n = db.create_node(&["Person"]).unwrap();
            db.add_node_label(n, "Employee");
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();
            let result = session.execute("MATCH (n:Employee) RETURN n").unwrap();
            assert_eq!(result.rows().len(), 1, "label should survive WAL recovery");
            db.close().expect("close");
        }
    }

    #[test]
    fn test_wal_recovery_deletes() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("deletes.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let a = db.create_node(&["Person"]).unwrap();
            db.set_node_property(a, "name", Value::String("Alix".into()))
                .unwrap();
            let b = db.create_node(&["Person"]).unwrap();
            db.set_node_property(b, "name", Value::String("Gus".into()))
                .unwrap();
            let c = db.create_node(&["Person"]).unwrap();
            db.set_node_property(c, "name", Value::String("Harm".into()))
                .unwrap();
            db.create_edge(a, b, "KNOWS");

            // Delete Harm (no edges)
            db.delete_node(c).unwrap();
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            assert_eq!(
                db.node_count(),
                2,
                "deleted node should not survive recovery"
            );
            assert_eq!(db.edge_count(), 1);
            db.close().expect("close");
        }
    }

    #[test]
    fn test_save_to_new_location() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let save_path = dir.path().join("saved.grafeo");

        let db = GrafeoDB::new_in_memory();
        let a = db.create_node(&["Person"]).unwrap();
        db.set_node_property(a, "name", Value::String("Alix".into()))
            .unwrap();
        let b = db.create_node(&["Person"]).unwrap();
        db.set_node_property(b, "name", Value::String("Gus".into()))
            .unwrap();
        db.create_edge(a, b, "KNOWS");

        // Save in-memory database to disk
        db.save(&save_path).expect("save should succeed");

        // Open the saved database
        let restored = GrafeoDB::open(&save_path).expect("open saved");
        assert_eq!(restored.node_count(), 2);
        assert_eq!(restored.edge_count(), 1);
        restored.close().expect("close");
    }

    #[test]
    fn test_open_in_memory_from_file() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("source.grafeo");

        // Create persistent database
        {
            let db = GrafeoDB::open(&path).expect("open");
            let n = db.create_node(&["Person"]).unwrap();
            db.set_node_property(n, "name", Value::String("Alix".into()))
                .unwrap();
            db.close().expect("close");
        }

        // Load into memory (detached from file)
        let mem_db = GrafeoDB::open_in_memory(&path).expect("open_in_memory");
        assert_eq!(mem_db.node_count(), 1);
        assert!(
            !mem_db.is_persistent(),
            "should be in-memory after open_in_memory"
        );

        // Modifications to the in-memory copy don't affect the file
        mem_db.create_node(&["Extra"]).expect("create extra node");
        assert_eq!(mem_db.node_count(), 2);

        // Reopening the file should still show 1 node
        let file_db = GrafeoDB::open(&path).expect("reopen");
        assert_eq!(file_db.node_count(), 1);
        file_db.close().expect("close");
    }

    #[test]
    fn test_wal_status_persistent() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("status.grafeo");

        let db = GrafeoDB::open(&path).expect("open");
        db.create_node(&["N"]).unwrap();

        let status = db.wal_status();
        assert!(status.enabled, "persistent db should have WAL enabled");

        db.close().expect("close");
    }

    #[test]
    fn test_query_mutations_persist() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("query_persist.grafeo");

        // Create data via GQL queries (not CRUD API)
        {
            let db = GrafeoDB::open(&path).expect("open");
            let session = db.session();
            session
                .execute("INSERT (:Person {name: 'Alix', age: 30})")
                .expect("insert Alix");
            session
                .execute("INSERT (:Person {name: 'Gus', age: 25})")
                .expect("insert Gus");
            session
                .execute(
                    "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                     INSERT (a)-[:KNOWS {since: 2020}]->(b)",
                )
                .expect("insert edge");

            // Verify data exists before close
            let result = session
                .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
                .unwrap();
            assert_eq!(result.rows().len(), 2);

            db.close().expect("close");
        }

        // Reopen and verify all query-created data survived
        {
            let db = GrafeoDB::open(&path).expect("reopen");
            assert_eq!(db.node_count(), 2, "query-created nodes should persist");
            assert_eq!(db.edge_count(), 1, "query-created edges should persist");

            let session = db.session();
            let result = session
                .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age")
                .unwrap();
            assert_eq!(result.rows().len(), 1, "Alix should be queryable");
            assert_eq!(result.rows()[0][0], Value::Int64(30));

            let result = session
                .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
                .unwrap();
            assert_eq!(result.rows().len(), 1, "KNOWS edge should persist");

            db.close().expect("close");
        }
    }

    #[test]
    fn test_query_delete_persists() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("query_delete.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let session = db.session();
            session
                .execute("INSERT (:Person {name: 'Alix'})")
                .expect("insert");
            session
                .execute("INSERT (:Person {name: 'Gus'})")
                .expect("insert");
            session
                .execute("MATCH (n:Person {name: 'Gus'}) DELETE n")
                .expect("delete");
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            assert_eq!(db.node_count(), 1, "delete should persist");
            let session = db.session();
            let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
            assert_eq!(result.rows().len(), 1);
            assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
            db.close().expect("close");
        }
    }

    #[test]
    fn test_query_set_property_persists() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("query_set.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let session = db.session();
            session
                .execute("INSERT (:Person {name: 'Alix', age: 30})")
                .expect("insert");
            session
                .execute("MATCH (n:Person {name: 'Alix'}) SET n.age = 31")
                .expect("update");
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();
            let result = session
                .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age")
                .unwrap();
            assert_eq!(result.rows().len(), 1);
            assert_eq!(result.rows()[0][0], Value::Int64(31), "SET should persist");
            db.close().expect("close");
        }
    }

    #[test]
    fn test_wal_checkpoint_succeeds() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("checkpoint.grafeo");

        let db = GrafeoDB::open(&path).expect("open");
        let n = db.create_node(&["Person"]).unwrap();
        db.set_node_property(n, "name", Value::String("Alix".into()))
            .unwrap();

        // Explicit checkpoint should not error
        db.wal_checkpoint().expect("checkpoint should succeed");

        // Verify data is still accessible after checkpoint
        assert_eq!(db.node_count(), 1);

        db.close().expect("close");
    }

    // =========================================================================
    // Named Graph WAL Recovery
    // =========================================================================

    #[test]
    fn test_named_graph_persists_across_restart() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("named_graph.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let session = db.session();
            session.execute("CREATE GRAPH analytics").unwrap();
            session.execute("USE GRAPH analytics").unwrap();
            session
                .execute("INSERT (:KPI {name: 'pageviews', count: 42})")
                .unwrap();
            assert_eq!(db.node_count(), 0, "default graph should be empty");
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();
            // Default graph should still be empty
            assert_eq!(db.node_count(), 0);

            // Named graph should have the data
            session.execute("USE GRAPH analytics").unwrap();
            let result = session
                .execute("MATCH (m:KPI) RETURN m.name, m.count")
                .unwrap();
            assert_eq!(result.rows().len(), 1);
            assert_eq!(result.rows()[0][0], Value::String("pageviews".into()));
            assert_eq!(result.rows()[0][1], Value::Int64(42));
            db.close().expect("close");
        }
    }

    #[test]
    fn test_drop_named_graph_persists() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("drop_graph.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let session = db.session();
            session.execute("CREATE GRAPH temp_graph").unwrap();
            session.execute("USE GRAPH temp_graph").unwrap();
            session.execute("INSERT (:Temp {val: 1})").unwrap();
            // Drop the graph we just created
            session.execute("USE GRAPH default").unwrap();
            session.execute("DROP GRAPH temp_graph").unwrap();
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();
            // Trying to use the dropped graph should fail
            let result = session.execute("USE GRAPH temp_graph");
            assert!(
                result.is_err(),
                "dropped graph should not exist after recovery"
            );
            db.close().expect("close");
        }
    }

    #[test]
    fn test_multiple_named_graphs_persist() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("multi_graph.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let session = db.session();

            // Create data in default graph
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

            // Create data in graph "alpha"
            session.execute("CREATE GRAPH alpha").unwrap();
            session.execute("USE GRAPH alpha").unwrap();
            session.execute("INSERT (:Item {name: 'Widget'})").unwrap();

            // Create data in graph "beta"
            session.execute("USE GRAPH default").unwrap();
            session.execute("CREATE GRAPH beta").unwrap();
            session.execute("USE GRAPH beta").unwrap();
            session
                .execute("INSERT (:City {name: 'Amsterdam'})")
                .unwrap();

            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();

            // Check default graph
            let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
            assert_eq!(result.rows().len(), 1);
            assert_eq!(result.rows()[0][0], Value::String("Alix".into()));

            // Check alpha graph
            session.execute("USE GRAPH alpha").unwrap();
            let result = session.execute("MATCH (i:Item) RETURN i.name").unwrap();
            assert_eq!(result.rows().len(), 1);
            assert_eq!(result.rows()[0][0], Value::String("Widget".into()));

            // Check beta graph
            session.execute("USE GRAPH beta").unwrap();
            let result = session.execute("MATCH (c:City) RETURN c.name").unwrap();
            assert_eq!(result.rows().len(), 1);
            assert_eq!(result.rows()[0][0], Value::String("Amsterdam".into()));

            db.close().expect("close");
        }
    }

    #[test]
    fn test_named_graph_wal_interleaved_with_default() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("interleaved.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let session = db.session();

            // Interleave mutations between default and named graph
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            session.execute("CREATE GRAPH other").unwrap();
            session.execute("USE GRAPH other").unwrap();
            session.execute("INSERT (:Robot {name: 'R2D2'})").unwrap();
            session.execute("USE GRAPH default").unwrap();
            session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
            session.execute("USE GRAPH other").unwrap();
            session.execute("INSERT (:Robot {name: 'C3PO'})").unwrap();

            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();

            // Default graph: 2 persons
            let result = session
                .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
                .unwrap();
            assert_eq!(result.rows().len(), 2);
            assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
            assert_eq!(result.rows()[1][0], Value::String("Gus".into()));

            // Other graph: 2 robots
            session.execute("USE GRAPH other").unwrap();
            let result = session
                .execute("MATCH (r:Robot) RETURN r.name ORDER BY r.name")
                .unwrap();
            assert_eq!(result.rows().len(), 2);
            assert_eq!(result.rows()[0][0], Value::String("C3PO".into()));
            assert_eq!(result.rows()[1][0], Value::String("R2D2".into()));

            db.close().expect("close");
        }
    }

    // =========================================================================
    // T1-05: Temporal & Complex Value WAL Recovery
    // =========================================================================

    #[test]
    fn test_wal_recovery_map_property() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("map_prop.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let n = db.create_node(&["Config"]).unwrap();
            db.set_node_property(n, "name", Value::String("settings".into()))
                .unwrap();
            let mut map = std::collections::BTreeMap::new();
            map.insert(
                grafeo_common::types::PropertyKey::from("theme"),
                Value::String("dark".into()),
            );
            map.insert(
                grafeo_common::types::PropertyKey::from("font_size"),
                Value::Int64(14),
            );
            db.set_node_property(n, "prefs", Value::Map(std::sync::Arc::new(map)))
                .unwrap();
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();
            let result = session
                .execute("MATCH (c:Config) RETURN c.prefs AS prefs")
                .unwrap();
            assert_eq!(result.rows().len(), 1);
            match &result.rows()[0][0] {
                Value::Map(m) => {
                    assert_eq!(m.len(), 2);
                    assert_eq!(
                        m[&grafeo_common::types::PropertyKey::from("theme")],
                        Value::String("dark".into())
                    );
                }
                other => panic!("Expected Map, got {other:?}"),
            }
            db.close().expect("close");
        }
    }

    #[test]
    fn test_wal_recovery_vector_property() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("vector_prop.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let n = db.create_node(&["Document"]).unwrap();
            db.set_node_property(n, "name", Value::String("doc1".into()))
                .unwrap();
            let embedding: std::sync::Arc<[f32]> =
                std::sync::Arc::from(vec![0.1_f32, 0.2, 0.3, 0.4]);
            db.set_node_property(n, "embedding", Value::Vector(embedding))
                .unwrap();
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();
            let result = session
                .execute("MATCH (d:Document) RETURN d.embedding AS emb")
                .unwrap();
            assert_eq!(result.rows().len(), 1);
            match &result.rows()[0][0] {
                Value::Vector(v) => {
                    assert_eq!(v.len(), 4);
                    assert!((v[0] - 0.1).abs() < f32::EPSILON);
                }
                other => panic!("Expected Vector, got {other:?}"),
            }
            db.close().expect("close");
        }
    }

    #[test]
    fn test_wal_recovery_timestamp_property() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("timestamp_prop.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let n = db.create_node(&["Event"]).unwrap();
            db.set_node_property(n, "name", Value::String("launch".into()))
                .unwrap();
            let ts = grafeo_common::types::Timestamp::from_secs(1_700_000_000);
            db.set_node_property(n, "occurred_at", Value::Timestamp(ts))
                .unwrap();
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();
            let result = session
                .execute("MATCH (e:Event) RETURN e.occurred_at AS ts")
                .unwrap();
            assert_eq!(result.rows().len(), 1);
            match &result.rows()[0][0] {
                Value::Timestamp(t) => {
                    assert_eq!(t.as_secs(), 1_700_000_000);
                }
                other => panic!("Expected Timestamp, got {other:?}"),
            }
            db.close().expect("close");
        }
    }

    #[test]
    fn test_wal_recovery_zoned_datetime_property() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("zdt_prop.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let n = db.create_node(&["Meeting"]).unwrap();
            db.set_node_property(n, "name", Value::String("standup".into()))
                .unwrap();
            let ts = grafeo_common::types::Timestamp::from_secs(1_700_000_000);
            let zdt = grafeo_common::types::ZonedDatetime::from_timestamp_offset(ts, 3600);
            db.set_node_property(n, "scheduled_at", Value::ZonedDatetime(zdt))
                .unwrap();
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();
            let result = session
                .execute("MATCH (m:Meeting) RETURN m.scheduled_at AS zdt")
                .unwrap();
            assert_eq!(result.rows().len(), 1);
            match &result.rows()[0][0] {
                Value::ZonedDatetime(z) => {
                    assert_eq!(z.as_timestamp().as_secs(), 1_700_000_000);
                    assert_eq!(z.offset_seconds(), 3600);
                }
                other => panic!("Expected ZonedDatetime, got {other:?}"),
            }
            db.close().expect("close");
        }
    }

    #[test]
    fn test_wal_recovery_duration_property() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("duration_prop.grafeo");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let n = db.create_node(&["Task"]).unwrap();
            db.set_node_property(n, "name", Value::String("sprint".into()))
                .unwrap();
            let dur = grafeo_common::types::Duration::new(0, 14, 0); // 14 days
            db.set_node_property(n, "duration", Value::Duration(dur))
                .unwrap();
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();
            let result = session
                .execute("MATCH (t:Task) RETURN t.duration AS dur")
                .unwrap();
            assert_eq!(result.rows().len(), 1);
            match &result.rows()[0][0] {
                Value::Duration(d) => {
                    assert_eq!(d.days(), 14);
                    assert_eq!(d.months(), 0);
                }
                other => panic!("Expected Duration, got {other:?}"),
            }
            db.close().expect("close");
        }
    }

    // =========================================================================
    // T1-06: Schema DDL Persistence (WAL Replay)
    // =========================================================================

    #[test]
    fn test_create_node_type_persists() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("ddl_node_type");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let session = db.session();
            session
                .execute("CREATE NODE TYPE Vehicle (make STRING NOT NULL, year INTEGER)")
                .unwrap();
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();
            let result = session.execute("SHOW NODE TYPES").unwrap();
            let type_names: Vec<&str> = result
                .rows()
                .iter()
                .filter_map(|row| match &row[0] {
                    Value::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                type_names.contains(&"Vehicle"),
                "Node type Vehicle should survive WAL replay, got: {type_names:?}"
            );
            db.close().expect("close");
        }
    }

    #[test]
    fn test_create_edge_type_persists() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("ddl_edge_type");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let session = db.session();
            session
                .execute("CREATE EDGE TYPE SUPPLIES (quantity INTEGER)")
                .unwrap();
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();
            let result = session.execute("SHOW EDGE TYPES").unwrap();
            let type_names: Vec<&str> = result
                .rows()
                .iter()
                .filter_map(|row| match &row[0] {
                    Value::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                type_names.contains(&"SUPPLIES"),
                "Edge type SUPPLIES should survive WAL replay, got: {type_names:?}"
            );
            db.close().expect("close");
        }
    }

    #[test]
    fn test_drop_node_type_persists() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("ddl_drop_type");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let session = db.session();
            session
                .execute("CREATE NODE TYPE Temp (name STRING)")
                .unwrap();
            session.execute("DROP NODE TYPE Temp").unwrap();
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();
            let result = session.execute("SHOW NODE TYPES").unwrap();
            let type_names: Vec<&str> = result
                .rows()
                .iter()
                .filter_map(|row| match &row[0] {
                    Value::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                !type_names.contains(&"Temp"),
                "Dropped node type should not survive WAL replay, got: {type_names:?}"
            );
            db.close().expect("close");
        }
    }

    #[test]
    fn test_schema_with_data_persists() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("ddl_with_data");

        {
            let db = GrafeoDB::open(&path).expect("open");
            let session = db.session();
            session
                .execute("CREATE NODE TYPE Person (name STRING NOT NULL, age INTEGER)")
                .unwrap();
            session
                .execute("INSERT (:Person {name: 'Alix', age: 30})")
                .unwrap();
            session
                .execute("INSERT (:Person {name: 'Gus', age: 25})")
                .unwrap();
            db.close().expect("close");
        }

        {
            let db = GrafeoDB::open(&path).expect("reopen");
            let session = db.session();

            // Schema should be present
            let types = session.execute("SHOW NODE TYPES").unwrap();
            let type_names: Vec<&str> = types
                .rows()
                .iter()
                .filter_map(|row| match &row[0] {
                    Value::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect();
            assert!(type_names.contains(&"Person"));

            // Data should be present
            let data = session
                .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
                .unwrap();
            assert_eq!(data.rows().len(), 2);
            assert_eq!(data.rows()[0][0], Value::String("Alix".into()));
            assert_eq!(data.rows()[1][0], Value::String("Gus".into()));

            db.close().expect("close");
        }
    }

    // =========================================================================
    // RDF WAL Recovery
    // =========================================================================

    #[cfg(all(feature = "sparql", feature = "triple-store"))]
    #[test]
    fn test_rdf_triples_persist_across_restart() {
        use grafeo_engine::{Config, GraphModel};

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("rdf.grafeo");

        {
            let config = Config::persistent(&path).with_graph_model(GraphModel::Rdf);
            let db = GrafeoDB::with_config(config).expect("open");
            let session = db.session();
            session
                .execute_sparql(
                    r#"INSERT DATA {
                        <http://ex.org/alix> <http://ex.org/name> "Alix" .
                        <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    }"#,
                )
                .unwrap();
            db.close().expect("close");
        }

        {
            let config = Config::persistent(&path).with_graph_model(GraphModel::Rdf);
            let db = GrafeoDB::with_config(config).expect("reopen");
            let session = db.session();
            let result = session
                .execute_sparql(
                    "SELECT ?name WHERE { ?s <http://ex.org/name> ?name } ORDER BY ?name",
                )
                .unwrap();
            assert_eq!(result.rows().len(), 2, "RDF triples should survive restart");
            assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
            assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
            db.close().expect("close");
        }
    }

    #[cfg(all(feature = "sparql", feature = "triple-store"))]
    #[test]
    fn test_rdf_named_graph_persists_across_restart() {
        use grafeo_engine::{Config, GraphModel};

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("rdf_named.grafeo");

        {
            let config = Config::persistent(&path).with_graph_model(GraphModel::Rdf);
            let db = GrafeoDB::with_config(config).expect("open");
            let session = db.session();
            session
                .execute_sparql(
                    r#"INSERT DATA {
                        GRAPH <http://ex.org/g1> {
                            <http://ex.org/alix> <http://ex.org/name> "Alix" .
                        }
                    }"#,
                )
                .unwrap();
            db.close().expect("close");
        }

        {
            let config = Config::persistent(&path).with_graph_model(GraphModel::Rdf);
            let db = GrafeoDB::with_config(config).expect("reopen");
            let session = db.session();
            let result = session
                .execute_sparql(
                    r#"SELECT ?name WHERE {
                        GRAPH <http://ex.org/g1> { ?s <http://ex.org/name> ?name }
                    }"#,
                )
                .unwrap();
            assert_eq!(
                result.rows().len(),
                1,
                "RDF named graph should survive restart"
            );
            assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
            db.close().expect("close");
        }
    }

    #[cfg(all(feature = "sparql", feature = "triple-store"))]
    #[test]
    fn test_rdf_delete_persists_across_restart() {
        use grafeo_engine::{Config, GraphModel};

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("rdf_delete.grafeo");

        {
            let config = Config::persistent(&path).with_graph_model(GraphModel::Rdf);
            let db = GrafeoDB::with_config(config).expect("open");
            let session = db.session();
            session
                .execute_sparql(
                    r#"INSERT DATA {
                        <http://ex.org/alix> <http://ex.org/name> "Alix" .
                        <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    }"#,
                )
                .unwrap();
            session
                .execute_sparql(
                    r#"DELETE DATA {
                        <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    }"#,
                )
                .unwrap();
            db.close().expect("close");
        }

        {
            let config = Config::persistent(&path).with_graph_model(GraphModel::Rdf);
            let db = GrafeoDB::with_config(config).expect("reopen");
            let session = db.session();
            let result = session
                .execute_sparql("SELECT ?name WHERE { ?s <http://ex.org/name> ?name }")
                .unwrap();
            assert_eq!(result.rows().len(), 1, "delete should persist");
            assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
            db.close().expect("close");
        }
    }

    #[cfg(all(feature = "sparql", feature = "triple-store"))]
    #[test]
    fn test_rdf_clear_graph_persists_across_restart() {
        use grafeo_engine::{Config, GraphModel};

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("rdf_clear.grafeo");

        {
            let config = Config::persistent(&path).with_graph_model(GraphModel::Rdf);
            let db = GrafeoDB::with_config(config).expect("open");
            let session = db.session();
            session
                .execute_sparql(
                    r#"INSERT DATA {
                        <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    }"#,
                )
                .unwrap();
            session.execute_sparql("CLEAR DEFAULT").unwrap();
            db.close().expect("close");
        }

        {
            let config = Config::persistent(&path).with_graph_model(GraphModel::Rdf);
            let db = GrafeoDB::with_config(config).expect("reopen");
            let session = db.session();
            let result = session
                .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
                .unwrap();
            assert_eq!(result.rows().len(), 0, "CLEAR should persist");
            db.close().expect("close");
        }
    }

    #[cfg(all(feature = "sparql", feature = "triple-store"))]
    #[test]
    fn test_rdf_create_drop_graph_persists() {
        use grafeo_engine::{Config, GraphModel};

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("rdf_create_drop.grafeo");

        {
            let config = Config::persistent(&path).with_graph_model(GraphModel::Rdf);
            let db = GrafeoDB::with_config(config).expect("open");
            let session = db.session();
            session
                .execute_sparql("CREATE GRAPH <http://ex.org/g1>")
                .unwrap();
            session
                .execute_sparql(
                    r#"INSERT DATA {
                        GRAPH <http://ex.org/g1> {
                            <http://ex.org/a> <http://ex.org/p> "val" .
                        }
                    }"#,
                )
                .unwrap();
            session
                .execute_sparql("DROP GRAPH <http://ex.org/g1>")
                .unwrap();
            db.close().expect("close");
        }

        {
            let config = Config::persistent(&path).with_graph_model(GraphModel::Rdf);
            let db = GrafeoDB::with_config(config).expect("reopen");
            let session = db.session();
            // Graph was dropped, so querying it should yield nothing or error
            let result = session
                .execute_sparql(
                    r#"SELECT ?s WHERE {
                        GRAPH <http://ex.org/g1> { ?s ?p ?o }
                    }"#,
                )
                .unwrap();
            assert_eq!(
                result.rows().len(),
                0,
                "dropped RDF graph should not survive"
            );
            db.close().expect("close");
        }
    }

    #[cfg(all(feature = "sparql", feature = "triple-store"))]
    #[test]
    fn test_rdf_save_preserves_triples() {
        use grafeo_engine::{Config, GraphModel};

        let dir = tempfile::tempdir().expect("create temp dir");
        let save_path = dir.path().join("rdf_saved.grafeo");

        let config = Config::in_memory().with_graph_model(GraphModel::Rdf);
        let db = GrafeoDB::with_config(config).expect("create");
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    GRAPH <http://ex.org/g1> {
                        <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    }
                }"#,
            )
            .unwrap();

        db.save(&save_path).expect("save");

        let restored = GrafeoDB::open(&save_path).expect("open saved");
        let session2 = restored.session();
        let result = session2
            .execute_sparql("SELECT ?name WHERE { ?s <http://ex.org/name> ?name }")
            .unwrap();
        assert_eq!(result.rows().len(), 1, "default RDF graph saved");
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));

        let result = session2
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    GRAPH <http://ex.org/g1> { ?s <http://ex.org/name> ?name }
                }"#,
            )
            .unwrap();
        assert_eq!(result.rows().len(), 1, "named RDF graph saved");
        assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
        restored.close().expect("close");
    }

    #[test]
    fn test_save_preserves_named_graphs() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let save_path = dir.path().join("saved_graphs.grafeo");

        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("CREATE GRAPH analytics").unwrap();
        session.execute("USE GRAPH analytics").unwrap();
        session
            .execute("INSERT (:KPI {name: 'views', count: 100})")
            .unwrap();

        db.save(&save_path).expect("save should succeed");

        let restored = GrafeoDB::open(&save_path).expect("open saved");
        assert_eq!(restored.node_count(), 1, "default graph: 1 person");

        let session2 = restored.session();
        session2.execute("USE GRAPH analytics").unwrap();
        let result = session2.execute("MATCH (m:KPI) RETURN m.name").unwrap();
        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0][0], Value::String("views".into()));
        restored.close().expect("close");
    }
}
