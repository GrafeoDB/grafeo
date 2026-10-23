"""RDF terms as SPARQL results give them.

Each term is a ``str`` of its text, so it formats, indexes and takes the
``str`` methods like the string it holds, and carries what else it is:

- ``IRI``: the IRI.
- ``BlankNode``: the blank node's label, without ``_:`` (as rdflib's
  ``BNode``).
- ``Literal``: the lexical form, with ``language`` (``None`` when it has no
  language tag, lowercase otherwise) and ``datatype`` (an ``IRI``:
  ``xsd:string`` for a simple literal, ``rdf:langString`` for a tagged one).

Terms compare and hash as RDF terms, as rdflib's do: two terms are equal when
they are of the same kind with the same text, language tag and datatype. A
term never equals a plain ``str`` (``IRI("http://x") != "http://x"``; compare
``str(term)`` for the text), and an ``IRI`` never equals a ``Literal`` of the
same text. Language tags compare case-insensitively: a ``Literal`` keeps its
tag in lowercase, as RDF 1.2 does.

A term passed back as a query parameter is that exact term.
"""

from __future__ import annotations

XSD = "http://www.w3.org/2001/XMLSchema#"
XSD_STRING = XSD + "string"
RDF_LANG_STRING = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"

__all__ = ["IRI", "BlankNode", "Literal"]

_INTEGER_TYPES = frozenset(
    {
        "integer",
        "int",
        "long",
        "short",
        "byte",
        "nonNegativeInteger",
        "positiveInteger",
        "negativeInteger",
        "nonPositiveInteger",
        "unsignedLong",
        "unsignedInt",
        "unsignedShort",
        "unsignedByte",
    }
)


class _Term(str):
    """A ``str`` that compares and hashes as an RDF term: by its kind, text,
    language tag and datatype, never equal to a plain ``str``."""

    __slots__ = ()
    _kind = ""

    def _term_key(self) -> tuple[object, ...]:
        return (self._kind, str(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _Term):
            return self._term_key() == other._term_key()
        if isinstance(other, str):
            # Not NotImplemented: Python would then ask str, which compares
            # the text alone.
            return False
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal

    def __hash__(self) -> int:
        return hash(self._term_key())


class IRI(_Term):
    """An IRI: a ``str`` of the IRI."""

    __slots__ = ()
    _kind = "IRI"

    def __repr__(self) -> str:
        return f"IRI({str.__repr__(self)})"

    def n3(self) -> str:
        """The IRI as N-Triples writes it: ``<...>``."""
        return f"<{self}>"


class BlankNode(_Term):
    """A blank node: a ``str`` of its label, without ``_:``."""

    __slots__ = ()
    _kind = "BlankNode"

    def __repr__(self) -> str:
        return f"BlankNode({str.__repr__(self)})"

    def n3(self) -> str:
        """The blank node as N-Triples writes it: ``_:label``."""
        return f"_:{self}"


class Literal(_Term):
    """A literal: a ``str`` of its lexical form, with ``language`` and
    ``datatype``.

    ``Literal("Pomme", language="fr")`` is a tagged literal (its datatype
    ``rdf:langString``), ``Literal("19.50", datatype=XSD + "decimal")`` a typed
    one, and ``Literal("Apple")`` a simple literal (``xsd:string``, the same
    literal as ``Literal("Apple", datatype=XSD + "string")``). The language
    tag is kept in lowercase (``Literal("Pomme", language="FR").language`` is
    ``"fr"``). ``language`` and ``datatype`` are read-only.
    """

    def __new__(
        cls,
        value: str,
        language: str | None = None,
        datatype: str | None = None,
    ) -> Literal:
        if language is not None and datatype is not None and str(datatype) != RDF_LANG_STRING:
            raise ValueError("a literal with a language tag has the datatype rdf:langString")
        literal = super().__new__(cls, value)
        tag = language.lower() if language else None
        literal.__dict__["_language"] = tag
        literal.__dict__["_datatype"] = IRI(
            RDF_LANG_STRING if tag else (str(datatype) if datatype else XSD_STRING)
        )
        return literal

    @property
    def language(self) -> str | None:
        """The language tag, in lowercase, or ``None``."""
        return self.__dict__["_language"]

    @property
    def datatype(self) -> IRI:
        """The datatype IRI: ``rdf:langString`` for a tagged literal,
        ``xsd:string`` for a simple one."""
        return self.__dict__["_datatype"]

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"a Literal is immutable: cannot set {name!r}")

    def _term_key(self) -> tuple[object, ...]:
        return ("Literal", str(self), self.language, str(self.datatype))

    def __repr__(self) -> str:
        if self.language:
            return f"Literal({str.__repr__(self)}, language={self.language!r})"
        if str(self.datatype) != XSD_STRING:
            return f"Literal({str.__repr__(self)}, datatype={str(self.datatype)!r})"
        return f"Literal({str.__repr__(self)})"

    def __reduce__(self):
        return (Literal, (str(self), self.language, str(self.datatype)))

    def to_python(self) -> object:
        """The value of the literal as Python has it: an ``int`` for
        ``xsd:integer`` and the integer types derived from it, a
        ``decimal.Decimal`` for ``xsd:decimal``, a ``float`` for
        ``xsd:double`` and ``xsd:float``, a ``bool`` for ``xsd:boolean``, a
        ``datetime.date`` for ``xsd:date`` and a ``datetime.datetime`` for
        ``xsd:dateTime``; the text for any other literal. Raises
        ``ValueError`` for a text that is not of its type
        (``"abc"^^xsd:integer``).
        """
        import datetime
        import decimal

        name = str(self.datatype).removeprefix(XSD)
        text = str(self).strip()
        if str(self.datatype).startswith(XSD):
            if name in _INTEGER_TYPES:
                return int(text)
            if name == "decimal":
                try:
                    return decimal.Decimal(text)
                except decimal.InvalidOperation as error:
                    raise ValueError(f"not an xsd:decimal: {text!r}") from error
            if name in ("double", "float"):
                return float({"INF": "inf", "+INF": "inf", "-INF": "-inf"}.get(text, text))
            if name == "boolean":
                if text in ("true", "1"):
                    return True
                if text in ("false", "0"):
                    return False
                raise ValueError(f"not an xsd:boolean: {text!r}")
            if name == "date":
                return datetime.date.fromisoformat(text)
            if name == "dateTime":
                return datetime.datetime.fromisoformat(text)
        return str(self)

    def n3(self) -> str:
        """The literal as N-Triples writes it."""
        escaped = (
            str(self)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        if self.language:
            return f'"{escaped}"@{self.language}'
        if str(self.datatype) != XSD_STRING:
            return f'"{escaped}"^^<{self.datatype}>'
        return f'"{escaped}"'
