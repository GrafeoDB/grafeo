# sv_people, and Richard's name also in FOAF: `name` is then a local name of two IRIs
# One SPARQL Update per line. Used by: rdf/graphql/local_names.gtest

PREFIX schema: <https://schema.org/> PREFIX sv: <https://docs.grafeo.dev/datasets/silicon-valley/vocab#> PREFIX svr: <https://docs.grafeo.dev/datasets/silicon-valley/id#> INSERT DATA { svr:richard a schema:Person ; schema:name "Richard Hendricks" ; sv:worksAt svr:pied-piper . }
PREFIX schema: <https://schema.org/> PREFIX sv: <https://docs.grafeo.dev/datasets/silicon-valley/vocab#> PREFIX svr: <https://docs.grafeo.dev/datasets/silicon-valley/id#> INSERT DATA { svr:gilfoyle a schema:Person ; schema:name "Bertram Gilfoyle" ; sv:worksAt svr:pied-piper . }
PREFIX schema: <https://schema.org/> PREFIX sv: <https://docs.grafeo.dev/datasets/silicon-valley/vocab#> PREFIX svr: <https://docs.grafeo.dev/datasets/silicon-valley/id#> INSERT DATA { svr:pied-piper a schema:Organization ; schema:name "Pied Piper" . }
PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX svr: <https://docs.grafeo.dev/datasets/silicon-valley/id#> INSERT DATA { svr:richard foaf:name "Richard" . }
