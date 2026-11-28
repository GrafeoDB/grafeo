# Example People Dataset (RDF), in the example.org vocabulary GraphQL over RDF reads
# 3 ex:Person, 1 ex:Organization, 3 ex:knows links, 1 ex:worksAt link
# One SPARQL Update per line. Used by: rdf/graphql/*

PREFIX ex: <http://example.org/> INSERT DATA { ex:alix a ex:Person ; ex:name "Alix" ; ex:age 30 ; ex:city "Amsterdam" ; ex:knows ex:gus, ex:mia ; ex:worksAt ex:grafeo . }
PREFIX ex: <http://example.org/> INSERT DATA { ex:gus a ex:Person ; ex:name "Gus" ; ex:age 25 ; ex:city "Berlin" ; ex:knows ex:mia . }
PREFIX ex: <http://example.org/> INSERT DATA { ex:mia a ex:Person ; ex:name "Mia" ; ex:age 32 ; ex:city "Prague" . }
PREFIX ex: <http://example.org/> INSERT DATA { ex:grafeo a ex:Organization ; ex:name "Grafeo OSS" ; ex:city "Amsterdam" . }
