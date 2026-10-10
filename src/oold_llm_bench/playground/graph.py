"""Turning one answer into a graph, including the parts that failed.

The predecessor drew entities and drew a property whose value looked like an
IRI as an edge. That was enough for a demo, because nothing scored the result.
Here an edge is a measured thing: it is the difference between a correct graph
and a correct bag of unlinked entities, which the grader now separates and the
picture has to separate too.

So a link reaching an entity that was emitted and a link reaching one that was
not are drawn as two different things. Collapsing them would put the failure
this view exists to show back under the same colour as the success.

Nothing here imports the agent package. The links are read off whatever object
carries ``source``, ``prop`` and ``target``, so the conversion is testable with
no provider, no client and no optional dependency installed.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Container, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from oold_llm_bench.grading.triples import Quantity, Reference, TripleSet, normalise_property
from oold_llm_bench.playground.identity import (
    CLOSE_MATCH,
    MERGE,
    NO_ACTION,
    Comparable,
    Decision,
    Judge,
    Ledger,
    action_for,
    decide,
)

__all__ = [
    "VIS_OPTIONS",
    "Diff",
    "Edge",
    "Graph",
    "GraphState",
    "Link",
    "Node",
    "build_graph",
    "links_of",
]


class Link(Protocol):
    """One edge as the agent records it, by the ids the plan handed out."""

    source: str
    prop: str
    target: str


NODE_TYPES: dict[str, str] = {
    "entity": "instance",
    "class": "class",
    "missing": "input",
    "literal": "input",
}
"""How a node kind maps onto the three the renderer colours.

The renderer offers instance, class and input and nothing else, so the fourth
kind has to be carried some other way. ``missing`` takes the input colour and
:data:`MISSING_COLOUR` on top, because it is the one kind that has to be read
at a glance: it is an entity a link names at either end and no answer
reported, so it is the shape of a failure rather than of a thing.
"""

MISSING_COLOUR: dict[str, str] = {"background": "#c0392b", "border": "#7b241c"}

CLOSE_MATCH_COLOUR: dict[str, str] = {"color": "#8e44ad", "highlight": "#8e44ad"}
"""A deferred identity, drawn apart from a link and from a broken one.

It is neither an assertion the answer made nor a failure: it is the one edge on
screen that is a question, and the thing a person is here to resolve.
"""

PARALLEL_ROUNDNESS = 0.3
"""How far each further edge between one pair of nodes is bent.

The renderer draws every edge between two nodes along the same path, so a
second one lands under the first and the two labels land on each other.
``multi_step`` reaches that on one sentence: "Andrea works at ExampleCorp" comes
back as ``worksFor`` one way and ``employee`` the other, which is two
assertions and, drawn straight, one line nobody can read.

Whether an answer should state the inverse at all is a question for the
benchmark. The picture's job is that both are legible either way.
"""

SELF_ANGLE = math.pi / 2
"""How far apart two loops on one node are turned.

A loop is not drawn as a curve between two points, so the roundness that
separates parallel edges does nothing to it. The renderer turns it around its
node instead, and that is the knob this uses. The first loop keeps the
renderer's own quarter turn, so a node carrying one is drawn as before.
"""

CONNECTING_KINDS: frozenset[str] = frozenset({"link", "dangling"})
"""Edge kinds that join an entity to something other than itself.

A self-loop is left out because it connects nothing: an entity whose only
edge points at itself is as unreachable as one with no edge at all. A type
edge joins an entity to its class rather than to another entity, and an
identity edge is the session's verdict rather than an assertion the answer
made.
"""

ASSERTED_KINDS: frozenset[str] = frozenset({"link", "dangling", "self"})
"""Edge kinds the answer itself asserted, as against the ones drawn about it.

A dangling edge and a self-loop are both wrong and both stated: the answer
pointed at an entity it never reported, or at itself. A ``type`` edge restates
the class the node already carries, and an identity edge is the session's
verdict on two answers rather than anything either one said.
"""

NAME_KEYS: tuple[str, ...] = ("name", "legal_name", "alternate_name", "title", "headline", "identifier")
"""Properties a label is read from, in order of preference.

Normalised spellings, because that is what the extractor keys triples on.
"""


@dataclass(frozen=True)
class Node:
    """One node, in the flat form the renderer's action list takes."""

    id: str
    label: str
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    """What the entity holds, rendered as the hover tooltip.

    The values live here and not in the picture. A graph that drew every
    literal would be unreadable at four properties an entity, and the
    properties are what the score is counted over, so they have to be
    reachable and do not have to be on screen.
    """
    class_path: str | None = None
    """The class the answer claimed, kept apart from the label.

    Drawn as one, an entity of class ``Person`` reads "Person": two people
    are two nodes reading the same word, an edge between them reads
    ``Person -> Person``, and a class node beside its instance reads the same.
    None of those is a self-loop and all three look like one.
    """
    named: bool = False
    """Whether :attr:`label` names the thing or only stands in for it.

    Set where the node is built, from whether anything named the entity at
    all. A stand-in label is not distinguishable from a real one by looking
    at it, and the resolver must never be offered one as a designator: two
    entities sharing a stand-in would agree on a name neither of them has.
    """

    def action(self, verb: str = "addNode") -> dict[str, Any]:
        node: dict[str, Any] = {
            "action": verb,
            "id": self.id,
            "label": self.label,
            "type": NODE_TYPES.get(self.kind, "instance"),
        }
        if self.kind == "missing":
            node["state"] = "new"
        if self.data:
            node["json_data"] = self.data
        return node

    def overlay(self) -> dict[str, Any] | None:
        """A direct node update, for what the flat format cannot say."""
        if self.kind != "missing":
            return None
        return {"id": self.id, "color": MISSING_COLOUR, "shape": "diamond"}


@dataclass(frozen=True)
class Edge:
    """One edge, in the flat form the renderer's action list takes."""

    source: str
    target: str
    label: str
    kind: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.label)

    @property
    def id(self) -> str:
        return "|".join(self.key)

    def action(self, verb: str = "addEdge") -> dict[str, Any]:
        return {
            "action": verb,
            "id": self.id,
            "from": self.source,
            "to": self.target,
            "label": self.label,
            # Dashed means the edge is not a plain assertion between two
            # reported entities: it reaches nothing, it points at its own
            # source, or it is a deferred identity nobody has confirmed.
            "dashed": self.kind in ("dangling", "self", "closeMatch"),
        }

    def overlay(self) -> dict[str, Any] | None:
        """A direct edge update, for what the flat format cannot say.

        The step format carries a label and a dash and no colour, and three
        kinds of edge are now dashed. A deferred identity is the one a person
        is here to act on, so it is the one that gets a colour of its own.
        """
        if self.kind != "closeMatch":
            return None
        return {"id": self.id, "from": self.source, "to": self.target, "color": CLOSE_MATCH_COLOUR}


@dataclass
class Graph:
    """Everything one answer drew, keyed so a second answer can be diffed."""

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], Edge] = field(default_factory=dict)

    def of_kind(self, kind: str) -> list[Node]:
        return [node for node in self.nodes.values() if node.kind == kind]

    def edges_of_kind(self, kind: str) -> list[Edge]:
        return [edge for edge in self.edges.values() if edge.kind == kind]

    @property
    def dangling(self) -> list[Edge]:
        return self.edges_of_kind("dangling")

    @property
    def isolated(self) -> list[str]:
        """Entities nothing reaches and nothing is reached from.

        A plan the extraction did not carry out. The detect step said the
        document names this thing; no property anybody filled turned out to
        be about it, so it sits in the picture connected to nothing.

        Counted apart from ``dangling``, which is the opposite failure: an
        edge that reaches nothing still says the source asserted something.
        """
        return _isolated(self.nodes, self.edges.values())

    def describe(self) -> dict[str, Any]:
        """The counts a test asserts and a reader checks against the picture."""
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "entities": len(self.of_kind("entity")),
            "classes": len(self.of_kind("class")),
            "missing": len(self.of_kind("missing")),
            "links": len(self.edges_of_kind("link")),
            "dangling": len(self.dangling),
            "self_loops": len(self.edges_of_kind("self")),
            "isolated": len(self.isolated),
        }


def _isolated(nodes: Mapping[str, Node], edges: Iterable[Edge]) -> list[str]:
    """Entity ids no connecting edge touches, at either end."""
    touched: set[str] = set()
    for edge in edges:
        if edge.kind in CONNECTING_KINDS:
            touched.add(edge.source)
            touched.add(edge.target)
    return sorted(key for key, node in nodes.items() if node.kind == "entity" and key not in touched)


def links_of(result: Any) -> tuple[list[Link], list[Link]]:
    """The links and the dangling ones an extraction result carries.

    Read by attribute rather than by type, so this module never has to import
    the agent package to draw what the agent produced.
    """
    return list(getattr(result, "links", ()) or ()), list(getattr(result, "dangling", ()) or ())


def _label_of(value: Any) -> str:
    if isinstance(value, Quantity):
        return f"{value.magnitude} {value.unit}"
    return str(value)


def _name_of(values: list[tuple[str, Any]]) -> str | None:
    """The words the answer called this entity, if it called it anything.

    Preferred over the class for the label. A graph of "Person" and
    "Organization" says what the model classified and nothing about what the
    document said, which is the half a reader is checking.

    A ``Person`` recorded by a split ``givenName``/``familyName`` rather than
    one ``name`` falls back to the class otherwise: found live, 10 of 208
    linked-articles sources are a person named that way and none of
    :data:`NAME_KEYS` reads it, so every one of them drew as a node labelled
    "Person" rather than by anyone's actual name. Composed only when neither
    half is independently in :data:`NAME_KEYS`'s stronger keys, so a direct
    ``name`` or ``title`` still wins when the answer gave one.
    """
    held = {normalise_property(prop): value for prop, value in values if isinstance(value, str) and value.strip()}
    for key in NAME_KEYS:
        found = held.get(key)
        if found:
            return found.strip()
    composed = " ".join(part.strip() for part in (held.get("given_name"), held.get("family_name")) if part)
    return composed or None


def _placeholder_label(class_path: str | None, node_id: str) -> str:
    """What to call an entity nothing named, once it has a graph id.

    The class and the id the graph handed out. The class says what the node
    is and the id distinguishes it from the next unnamed one of its kind,
    which an answer-scoped key cannot: that key is scoped to one call and
    every turn hands out the same few.

    Only the id where no class was claimed, rather than a word like "Thing"
    standing in for one, because the absence of a class is itself worth
    seeing.
    """
    return f"{class_path} {node_id}" if class_path else node_id


def _short_uuid() -> str:
    """A fresh id for a node entering the graph for the first time.

    Not the orchestration's own ``e1``/``e2``: those are scoped to one call
    and collide the moment a second turn's entity also answers as ``e1``,
    which is what the ``#2``-suffixed id this replaces was papering over. A
    graph node's identity is the session's to hand out, not the answer's, so
    it is never reused and never needs disambiguating against a turn number.
    Eight hex characters is short enough to read in a tooltip and collision
    only matters within one session's node count, not at any wider scale.
    """
    return uuid.uuid4().hex[:8]


def _data_of(
    key: str, class_path: str | None, values: list[tuple[str, Any]], *, mention: str | None = None
) -> dict[str, Any]:
    """One entity as the tooltip and the YAML view read it.

    The mention is kept apart from the properties and never written into a
    ``name`` slot. It is what the document called the entity, not a value the
    document stated, and folding it in would make the YAML claim an extraction
    that did not happen and the score count a property nobody filled.
    """
    data: dict[str, Any] = {"id": key}
    if class_path:
        data["type"] = class_path
    if mention:
        data["mention"] = mention
    for prop, value in sorted(values, key=lambda pair: pair[0]):
        _put(data, prop, _label_of(value))
    return data


def _put(data: dict[str, Any], prop: str, value: Any) -> None:
    """Add one value, keeping both readings where a property is stated twice.

    A second value becomes a list rather than replacing the first. Which of
    two readings is right is not a question a view gets to answer, and showing
    one of them would hide that the answer held both.
    """
    existing = data.get(prop)
    if existing is None:
        data[prop] = value
    elif isinstance(existing, list):
        existing.append(value)
    else:
        data[prop] = [existing, value]


def build_graph(
    produced: TripleSet,
    *,
    links: Iterable[Link] = (),
    dangling: Iterable[Link] = (),
    mentions: Mapping[str, str] | None = None,
    show_classes: bool = False,
    show_literals: bool = False,
) -> Graph:
    """Convert one answer into nodes and edges.

    Four sources feed this and each says something the others cannot.

    The triples give the entities and, where a reference resolved, the edges
    between them. The agent's ``links`` give the edges an orchestration that
    hands out ids recorded for itself, which includes edges the extractor
    cannot see: a link whose target was never emitted stays an ordinary string
    in the triples, because the extractor only resolves an id some entity
    claimed. The agent's ``dangling`` names exactly those.

    The agent's ``mentions`` give what the document called each entity, which
    is the one thing an extracted property cannot be relied on for. Nobody
    writes "Jane is named Jane", so a property step asked whether the document
    states a ``name`` for "Jane works at ExampleCorp" is right to answer no:
    claude-haiku-4-5 chooses ``worksFor`` and ``employee`` on 8 of 8 runs and
    ``name`` on none. The plan step read "Jane" correctly every one of those
    times, and this is where that reading becomes the label.

    ``show_classes`` draws the claimed class as its own node, which makes two
    entities landing on one class visible as a shared node. Off by default,
    because the class is already the entity's label and a second node per
    entity competes with the distinction this view exists to draw.

    An entity that stated a class and no values is drawn as well. A triple
    carries a value, so an entity that gave none leaves none to be found
    under, and ``multi_step`` arrives there on purpose: an entity its property
    step chose nothing for is extracted with its class and its id alone.
    Reading the entities off the triples alone dropped exactly those, which
    left their class edge hanging off a node nothing had drawn.

    Every edge this returns has a node at both ends, whatever the answer left
    out. :class:`GraphState` relies on that to place an edge by the ids it
    already resolved, and the renderer needs it to draw one at all.
    """
    graph = Graph()
    # The agent names a link by the property the answer wrote, the extractor
    # by the normalised form it keys triples on. Left alone, the same edge
    # arrives twice under two spellings and the picture doubles every link.
    dangling_keys = {(link.source, normalise_property(link.prop), link.target) for link in dangling}

    by_entity: dict[str, list[tuple[str, Any]]] = {key: [] for key in (*produced.entities(), *produced.classes)}
    for triple in produced.triples:
        by_entity.setdefault(triple.entity, []).append((triple.prop, triple.value))

    read = dict(mentions or {})
    for key, values in by_entity.items():
        class_path = produced.classes.get(key)
        literals = [(p, v) for p, v in values if not isinstance(v, Reference)]
        # An extracted name first, then what the plan read the entity from.
        # The two are not the same claim: a name is a property the document
        # states, a mention is the words it refers to the thing by, and only
        # the first is scored. Both name the thing well enough to draw and to
        # compare, which is what a label is for.
        named = _name_of(literals) or read.get(key)
        graph.nodes[key] = Node(
            id=key,
            # The key where neither named this entity, which holds only until
            # `GraphState` hands the node a graph id and composes a stand-in
            # from it. A class on its own would not distinguish two unnamed
            # entities of one class.
            label=named or key,
            kind="entity",
            data=_data_of(key, class_path, literals, mention=read.get(key)),
            class_path=class_path,
            named=named is not None,
        )

    if show_classes:
        for key, class_path in produced.classes.items():
            node_id = f"class:{class_path}"
            graph.nodes.setdefault(node_id, Node(id=node_id, label=class_path, kind="class"))
            edge = Edge(source=key, target=node_id, label="type", kind="type")
            graph.edges[edge.key] = edge

    for key, values in by_entity.items():
        for prop, value in values:
            if isinstance(value, Reference):
                _add_link(graph, key, prop, value.key, dangling_keys)
            elif show_literals:
                _add_literal(graph, key, prop, value)

    for link in links:
        _add_link(graph, link.source, normalise_property(link.prop), link.target, dangling_keys)

    return graph


def _add_link(
    graph: Graph,
    source: str,
    prop: str,
    target: str,
    dangling_keys: set[tuple[str, str, str]],
) -> None:
    """Draw one edge, creating either end only when nothing reported it.

    An end that was never emitted still gets a node, because an edge with one
    end missing cannot be drawn and dropping the edge would make a link into
    nothing look like no link at all.

    Both ends, and not only the target the agent calls dangling. The agent
    reads its links off the answer, the picture is built from what the
    extractor made of that same answer, and the two need not agree on which
    ids exist: an id filed under another spelling, or an entity the extractor
    could not read at all, leaves a link whose source nothing drew. That used
    to be an ordinary link out of a node that was not there, which the
    renderer cannot draw and the fold had nowhere to place.

    An entity pointing at itself is drawn as its own kind. The extractor
    already refuses to read one out of a value, but an orchestration that
    offers the plan's ids as an enum lets a model answer ``worksFor`` with the
    organisation's own id, and nano does: five of the seven links one run
    asserted about "ExampleCorp" pointed at "ExampleCorp". Counting those as links
    would report an answer about nothing as an answer with edges.
    """
    for end in (source, target):
        if end not in graph.nodes:
            graph.nodes[end] = Node(
                id=end,
                label=end,
                kind="missing",
                data={"id": end, "reported": "no", "note": "named by a link, never emitted"},
            )
    if source == target:
        edge = Edge(source=source, target=target, label=prop, kind="self")
        graph.edges[edge.key] = edge
        return
    broken = (
        (source, prop, target) in dangling_keys
        or graph.nodes[source].kind == "missing"
        or graph.nodes[target].kind == "missing"
    )
    edge = Edge(source=source, target=target, label=prop, kind="dangling" if broken else "link")
    graph.edges[edge.key] = edge


def _add_literal(graph: Graph, key: str, prop: str, value: Any) -> None:
    label = _label_of(value)
    node_id = f"lit:{key}:{prop}:{label}"
    graph.nodes[node_id] = Node(
        id=node_id,
        label=label if len(label) <= 40 else label[:37] + "...",
        kind="literal",
        data={prop: label},
    )
    edge = Edge(source=key, target=node_id, label=prop, kind="value")
    graph.edges[edge.key] = edge


@dataclass
class Diff:
    """What changed between two answers, as the renderer's action list."""

    actions: list[dict[str, Any]] = field(default_factory=list)
    created_nodes: list[str] = field(default_factory=list)
    updated_nodes: list[str] = field(default_factory=list)
    created_edges: list[str] = field(default_factory=list)
    stored_nodes: list[str] = field(default_factory=list)
    merged_nodes: list[str] = field(default_factory=list)
    """Nodes this answer was folded into rather than drawn beside."""
    decisions: list[Decision] = field(default_factory=list)
    """Every identity comparison this answer caused."""
    resolved_placeholders: list[str] = field(default_factory=list)
    """Entities that arrived and filled a node a link had only named.

    Worth reporting rather than counting as an ordinary update: it is the
    graph healing, and before it existed the placeholder stayed red for good
    while the real record was drawn beside it under a suffixed id."""

    def summary(self) -> str:
        return (
            f"{len(self.created_nodes)} new nodes, "
            f"{len(self.updated_nodes)} changed, "
            f"{len(self.stored_nodes)} unchanged, "
            f"{len(self.created_edges)} new edges, "
            f"{len(self.merged_nodes)} merged"
        )


class GraphState:
    """The graph as it stands, and the diff the next answer implies.

    Kept incremental because the point of the view is that a document grows a
    graph: redrawing from scratch on every turn loses the layout and with it
    any sense of what the last answer added.

    Two answers hand out the same ids. An orchestration names its entities
    ``e1`` and ``e2`` every time, so folding turn two in by key alone wrote
    turn two's first entity over turn one's and the graph stopped growing at
    two nodes. An incoming key therefore resolves against the entities already
    drawn, by :func:`~oold_llm_bench.playground.identity.decide`, and what
    :func:`~oold_llm_bench.playground.identity.action_for` says to do about
    the decision: an ``exactMatch`` reuses a node where
    :attr:`auto_merge_exact_match` leaves it to, and anything else, including
    an ``exactMatch`` the toggle holds back, is a new node under a
    turn-suffixed id with a ``closeMatch`` edge of its own between the two.

    One answer resolves against itself on the same rule. Asked for "Alice works
    at Example Corp", gpt-5-nano reported the organisation twice in a single
    answer, and two nodes reading "Example Corp" is no more useful to a reader
    than two turns' worth would have been. The extraction result is untouched,
    so a score still counts what was answered; only the picture is one graph.
    """

    def __init__(self, judge: Judge | None = None, *, auto_merge_exact_match: bool = True) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[tuple[str, str, str], Edge] = {}
        self.turn = 0
        self.judge = judge
        self.auto_merge_exact_match = auto_merge_exact_match
        """Whether an ``exactMatch`` folds on sight or is only drawn.

        Read by :func:`~oold_llm_bench.playground.identity.action_for` on
        every comparison, so a reader can flip it between two turns and see
        the next ``exactMatch`` deferred instead of merged, the same toggle
        the interface exposes.
        """
        self.ledger = Ledger()
        self.conflicts: dict[str, list[str]] = {}
        """Properties a merged node holds two readings of, by node id.

        Recorded and never resolved. Deciding that two entities are one says
        nothing about which of two values for one property is right, and
        ``{**existing, **new}`` is what taking the newer one looks like."""

    def update(self, graph: Graph) -> Diff:
        self.turn += 1
        diff = Diff()
        mapping, deferred = self._resolve(graph, diff)

        for key, node in graph.nodes.items():
            target = mapping[key]
            current = self.nodes.get(target)
            folded = node if current is None else _fold(current, node, self.conflicts.setdefault(target, []))
            # The id the graph handed out, in the record as well as on the
            # node. `data` is what the tooltip and the YAML view read, so the
            # id printed there has to be the one the rest of the graph uses
            # and not the answer-scoped key the node arrived under.
            #
            # An unnamed entity's label is composed here for the same reason:
            # it stands in for a name, and this is the first point at which
            # there is an id worth standing in with. A class or a `missing`
            # node keeps the label it was drawn with; both already say what
            # they are.
            label = folded.label
            if folded.kind == "entity" and not folded.named:
                label = _placeholder_label(folded.class_path, target)
            placed = replace(folded, id=target, label=label, data={**folded.data, "id": target})
            if current is None:
                diff.actions.append(placed.action())
                diff.created_nodes.append(target)
            elif current != placed:
                diff.actions.append(placed.action("updateNode"))
                diff.updated_nodes.append(target)
            self.nodes[target] = placed

        drawn = [
            replace(edge, source=mapping[edge.source], target=mapping[edge.target]) for edge in graph.edges.values()
        ]
        for edge in [*drawn, *deferred]:
            placed = replace(edge, kind="self") if edge.source == edge.target and edge.kind == "link" else edge
            if placed.key not in self.edges:
                diff.actions.append(placed.action())
                diff.created_edges.append(placed.id)
            self.edges[placed.key] = placed

        touched = set(diff.created_nodes) | set(diff.updated_nodes)
        diff.stored_nodes = [key for key in self.nodes if key not in touched]
        if diff.stored_nodes:
            diff.actions.append({"action": "updateNodeState", "nodeIds": diff.stored_nodes, "state": "stored"})
        if touched:
            diff.actions.append({"action": "updateNodeState", "nodeIds": sorted(touched), "state": "modified"})
        return diff

    def _resolve(self, graph: Graph, diff: Diff) -> tuple[dict[str, str], list[Edge]]:
        """Where each incoming node id lands, once identity has been decided.

        Returns the deferred relations with it, because a ``closeMatch`` edge
        needs the id the new node ends up under and that is only settled here.
        """
        mapping: dict[str, str] = {}
        taken: set[str] = set()
        deferred: list[Edge] = []
        # Entities placed by this answer are candidates for the ones after
        # them, which is what makes an answer that reported one thing twice
        # resolve to one node.
        candidates = {node_id: node for node_id, node in self.nodes.items() if node.kind == "entity"}
        for key, node in graph.nodes.items():
            if node.kind == "class":
                # A class node is the class itself and not an instance of
                # anything, so two turns naming one class mean one node and
                # there is nothing to decide about it.
                mapping[key] = key
                continue
            # An entity reported under the id a placeholder is holding is that
            # placeholder. The id is the only thing a `missing` node carries,
            # because it stands for an entity a link named and nothing
            # reported, so the claim is by construction and never a judgement:
            # comparing an empty node by its values would spend a call to
            # decide nothing. Without this the placeholder kept the id for
            # good and the real record was pushed to `e2#2`.
            standing = self.nodes.get(key)
            if node.kind == "entity" and standing is not None and standing.kind == "missing":
                mapping[key] = key
                taken.add(key)
                candidates[key] = node
                diff.resolved_placeholders.append(key)
                continue

            merged, unclear = (
                self._compare(key, node, candidates, diff, graph.edges.values())
                if node.kind == "entity"
                else (None, [])
            )
            if merged is not None:
                mapping[key] = merged
                diff.merged_nodes.append(merged)
                # What merged in is part of the candidate from now on. The map
                # is built once and `self.nodes` is not written until after
                # this loop, so without it a later entity in the same turn is
                # compared against the node as it stood before the merge: a
                # pair that conflicts against the folded record, and would
                # therefore be a question, agreed with the stale copy and was
                # merged with no call.
                candidates[merged] = _fold(candidates[merged], node, []) if merged in candidates else node
                # A deferred pair is a question for a person and it survives
                # the merge that answered a different question. Dropping it
                # here is why the chat could report "1 pair left to resolve"
                # with nothing on screen to resolve.
                deferred.extend(
                    Edge(source=merged, target=other, label=CLOSE_MATCH, kind="closeMatch")
                    for other in unclear
                    if other != merged
                )
                continue
            # A `missing` node keeps the local key it arrived under rather
            # than drawing a fresh one: that key is the only identity it has,
            # and it is what the healing branch above looks `self.nodes` up
            # by once a later turn reports the same key as a real entity.
            # Handing it a random id the moment it is created would make that
            # lookup fail forever, and the placeholder would never heal.
            free = key if node.kind == "missing" else _short_uuid()
            mapping[key] = free
            taken.add(free)
            if node.kind == "entity":
                candidates[free] = node
            deferred.extend(Edge(source=free, target=other, label=CLOSE_MATCH, kind="closeMatch") for other in unclear)
        return mapping, deferred

    def _compare(
        self,
        key: str,
        node: Node,
        candidates: dict[str, Node],
        diff: Diff,
        incoming_edges: Iterable[Edge] = (),
    ) -> tuple[str | None, list[str]]:
        """The node this entity already is, and the ones nobody could decide.

        The first entity :func:`~oold_llm_bench.playground.identity.action_for`
        says to merge wins and the rest are not asked. What is left is one
        comparison per entity already drawn, and a comparison the class rule
        and exact agreement cannot settle is a call, so a graph that grows
        gets dearer to grow. The Identity read-out reports the two routes apart
        for that reason: the cost is on screen rather than in a bill.
        """
        incoming = _comparable(key, node, _link_props(incoming_edges, key))
        drawn = list(self.edges.values())
        unclear: list[str] = []
        for other_id, other in list(candidates.items()):
            against = _comparable(other_id, other, _link_props(drawn, other_id))
            decision = self.ledger.record(decide(incoming, against, self.judge))
            diff.decisions.append(decision)
            action = action_for(decision, auto_merge_exact_match=self.auto_merge_exact_match)
            if action == MERGE:
                return other_id, unclear
            if action != NO_ACTION:
                unclear.append(other_id)
        return None, unclear

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.conflicts.clear()
        self.ledger = Ledger()
        self.turn = 0

    def overlays(self) -> list[dict[str, Any]]:
        """Node updates the flat action format cannot carry.

        Applied after a diff, because the state update at the end of one would
        otherwise repaint a missing node as an ordinary stored one.
        """
        return [overlay for node in self.nodes.values() if (overlay := node.overlay()) is not None]

    def edge_overlays(self) -> list[dict[str, Any]]:
        """Edge updates the flat action format cannot carry.

        The curve belongs here and not on :class:`Edge`, because how far an
        edge has to bend depends on the other edges between the same two
        nodes, and an edge on its own cannot know of them.
        """
        curves = _spread(self.edges.values())
        overlays: list[dict[str, Any]] = []
        for edge in self.edges.values():
            parts = [part for part in (edge.overlay(), curves.get(edge.key)) if part is not None]
            if not parts:
                continue
            overlay: dict[str, Any] = {"id": edge.id, "from": edge.source, "to": edge.target}
            for part in parts:
                overlay.update(part)
            overlays.append(overlay)
        return overlays

    def documents(self) -> list[dict[str, Any]]:
        """Every entity the graph holds, written out as its own document.

        The node's own values plus the edges leaving it. A link is something
        the answer asserted about the entity, so a document without them
        states half of what was extracted, and the half it drops is the half
        a graph exists for.

        A link carries the id of the node it reaches, which is the id that
        node's own document is headed by. That is what makes the dump
        traversable rather than a list of unconnected records.

        Identity edges are left out. ``skos:closeMatch`` is what the session
        decided about two documents, not what either document says.
        """
        documents: list[dict[str, Any]] = []
        for node in self.nodes.values():
            if node.kind != "entity":
                continue
            data = dict(node.data)
            leaving = (e for e in self.edges.values() if e.source == node.id and e.kind in ASSERTED_KINDS)
            for edge in sorted(leaving, key=lambda e: (e.label, e.target)):
                _put(data, edge.label, edge.target)
            documents.append(data)
        return documents

    def describe(self) -> dict[str, Any]:
        """The state a test reads instead of reading pixels."""
        kinds: Mapping[str, int] = {
            kind: sum(1 for node in self.nodes.values() if node.kind == kind)
            for kind in ("entity", "class", "missing", "literal")
        }
        entities = [node for node in self.nodes.values() if node.kind == "entity"]
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "turns": self.turn,
            "node_ids": sorted(self.nodes),
            "entities": sorted(node.id for node in entities),
            "labels": {node.id: node.label for node in sorted(entities, key=lambda n: n.id)},
            "types": {node.id: node.class_path for node in sorted(entities, key=lambda n: n.id)},
            "classes": sorted({n.label for n in self.nodes.values() if n.kind == "class"}),
            "links": sorted("|".join(e.key) for e in self.edges.values() if e.kind == "link"),
            "dangling": sorted("|".join(e.key) for e in self.edges.values() if e.kind == "dangling"),
            "self_loops": sorted("|".join(e.key) for e in self.edges.values() if e.kind == "self"),
            "isolated": _isolated(self.nodes, self.edges.values()),
            "close_matches": sorted("|".join(e.key) for e in self.edges.values() if e.kind == "closeMatch"),
            "conflicts": {key: sorted(props) for key, props in sorted(self.conflicts.items()) if props},
            "identity": self.ledger.describe(),
            **{f"n_{kind}": count for kind, count in kinds.items()},
        }


def _spread(edges: Iterable[Edge]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """A curve for each edge that shares both its ends with another.

    Nothing is returned for an edge alone between its nodes, so the ordinary
    case keeps the straight path and only the edges that would cover each
    other are bent.

    Direction is folded out before the edges are counted. An edge and its
    inverse connect the same two nodes and would be drawn on the same path, so
    they have to be spread against each other and not each against itself; the
    handedness is then flipped back for the inverse, which is what puts the
    two on either side of the line rather than both on one.
    """
    bundles: dict[tuple[str, ...], list[Edge]] = {}
    for edge in edges:
        bundles.setdefault(tuple(sorted((edge.source, edge.target))), []).append(edge)

    curves: dict[tuple[str, str, str], dict[str, Any]] = {}
    for pair, bundle in bundles.items():
        if len(bundle) < 2:
            continue
        for index, edge in enumerate(sorted(bundle, key=lambda item: item.id)):
            if edge.source == edge.target:
                curves[edge.key] = {"selfReference": {"angle": math.pi / 4 + index * SELF_ANGLE}}
                continue
            forward = (edge.source, edge.target) == pair
            curves[edge.key] = {
                "smooth": {
                    "enabled": True,
                    "type": "curvedCW" if (index % 2 == 0) == forward else "curvedCCW",
                    "roundness": min(PARALLEL_ROUNDNESS * (index // 2 + 1), 1.0),
                }
            }
    return curves


def _comparable(key: str, node: Node, links: Container[str] = ()) -> Comparable:
    """One drawn node as the identity step compares it: class and literals.

    ``links`` names this node's reference properties, and they are left out.
    A link's value is the id of whatever the plan called the target, so two
    records of one thing written in two turns carry different ids for the same
    relation and two unrelated entities can carry the same id for different
    ones. Comparing them asks whether two answers agreed about a name the
    pipeline invented, which is not a question about identity.

    Excluded at this stage rather than for good. What an entity is linked to
    is evidence about what it is, and a later judge may be given the edges;
    what it cannot be is evidence of agreement between two raw id strings.
    """
    values: dict[str, str] = {}
    for prop, value in node.data.items():
        if prop in ("id", "type") or prop in links:
            continue
        # A property stated twice arrives as a list, which is how `_data_of`
        # keeps both readings. Skipping those made an entity whose name was
        # emitted twice compare on nothing at all, so the one property most
        # likely to settle an identity was the one most likely to be dropped.
        readings = value if isinstance(value, list) else [value]
        texts = sorted({str(item).strip() for item in readings if isinstance(item, str) and str(item).strip()})
        if texts:
            values[prop] = texts[0] if len(texts) == 1 else " | ".join(texts)
    # `node.label` already resolved a split `givenName`/`familyName` into one
    # name, or fell back to the node's own id where nothing named it; passed
    # through so a resolver comparison designates the entity the same way the
    # picture already does, rather than re-deriving it from `values` alone.
    return Comparable(key=key, class_path=node.class_path, values=values, label=node.label if _is_named(node) else None)


def _is_named(node: Node) -> bool:
    """Whether this node's label names the thing or stands in for one.

    A node nothing named still carries a label, so the flag decides this and
    not the text. A turn that supplies a real name upgrades the node; a
    second stand-in leaves it as it was.
    """
    return node.named and bool(node.label)


def _link_props(edges: Iterable[Edge], key: str) -> frozenset[str]:
    """The property names this node points at something with."""
    return frozenset(edge.label for edge in edges if edge.source == key and edge.kind != "closeMatch")


def _fold(current: Node, incoming: Node, conflicts: list[str]) -> Node:
    """One entity's values folded into the node it was decided to be.

    The values already drawn win and the disagreement is recorded. Letting the
    newer answer overwrite would make the graph report whichever turn ran last
    rather than what was extracted, which is the ``{**existing, **new}`` the
    predecessor called a merge.

    A class and a label are the exception, and only where the node has none. An
    entity that stated neither is the nested-object case, and keeping its
    silence over the other's answer would draw a typed thing as an untyped one.
    """
    data = dict(current.data)
    for prop, value in incoming.data.items():
        if prop not in data:
            data[prop] = value
        elif data[prop] != value and prop != "id":
            if prop not in conflicts:
                conflicts.append(prop)
            data[f"{prop} (conflict)"] = value
    class_path = current.class_path or incoming.class_path
    if class_path:
        data["type"] = class_path
        data.pop("type (conflict)", None)
    # A stand-in is not a label. The node keeps whichever reading actually
    # names the thing, so a turn that finally supplies a name upgrades a node
    # that had only been standing in for one.
    named = current.named or incoming.named
    label = current.label if _is_named(current) else incoming.label
    # A healed placeholder is the one direction kind has to change: a
    # `missing` node is a stand-in for an entity a link named and nothing
    # reported, and the entity reporting it later is exactly what it was
    # waiting for. Without this a healed placeholder stayed drawn as
    # "missing" forever, even once a real entity had filled it in.
    kind = incoming.kind if current.kind == "missing" else current.kind
    return replace(current, data=data, class_path=class_path, label=label, kind=kind, named=named)


VIS_OPTIONS: dict[str, Any] = {
    "nodes": {
        "shape": "dot",
        "size": 18,
        "font": {"size": 14, "strokeWidth": 3, "strokeColor": "#ffffff"},
        "borderWidth": 2,
    },
    "edges": {
        "arrows": {"to": {"enabled": True}},
        "font": {"size": 10, "align": "middle", "strokeWidth": 3, "strokeColor": "#ffffff"},
        "smooth": {"type": "cubicBezier"},
    },
    "physics": {
        "enabled": True,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
            "gravitationalConstant": -200,
            "centralGravity": 0.005,
            "springLength": 150,
            "springConstant": 0.05,
            "damping": 0.4,
            "avoidOverlap": 1,
        },
        "stabilization": {"enabled": True, "iterations": 200, "updateInterval": 25},
    },
    "interaction": {"hover": True, "tooltipDelay": 200},
}
"""Layout for the VisNetwork renderer, carried over from the predecessor.

Stabilisation is left on. The test that drives this waits for the node count
to settle rather than for the physics to stop, so a running simulation costs
nothing there and a still picture is worth more to a reader.
"""
