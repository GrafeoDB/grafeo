# Silicon Valley people (RDF), in schema.org, with a link of the dataset's own vocabulary
# 2 schema:Person, 1 schema:Organization, 2 sv:worksAt links
# One SPARQL Update per line. Used by: rdf/graphql/local_names.gtest

PREFIX schema: <https://schema.org/> PREFIX sv: <https://docs.grafeo.dev/datasets/silicon-valley/vocab#> PREFIX svr: <https://docs.grafeo.dev/datasets/silicon-valley/id#> INSERT DATA { svr:richard a schema:Person ; schema:name "Richard Hendricks" ; sv:worksAt svr:pied-piper . }
PREFIX schema: <https://schema.org/> PREFIX sv: <https://docs.grafeo.dev/datasets/silicon-valley/vocab#> PREFIX svr: <https://docs.grafeo.dev/datasets/silicon-valley/id#> INSERT DATA { svr:gilfoyle a schema:Person ; schema:name "Bertram Gilfoyle" ; sv:worksAt svr:pied-piper . }
PREFIX schema: <https://schema.org/> PREFIX sv: <https://docs.grafeo.dev/datasets/silicon-valley/vocab#> PREFIX svr: <https://docs.grafeo.dev/datasets/silicon-valley/id#> INSERT DATA { svr:pied-piper a schema:Organization ; schema:name "Pied Piper" . }
