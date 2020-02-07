import argparse
import json
import logging
import pickle
import re
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tqdm

import numpy as np
from nltk.corpus import stopwords

LOGGER = logging.getLogger(__name__)
STOPWORDS = set(stopwords.words("english") + [".", ",", '"'])
PAREN = re.compile(r"([^\(\s]+)\s\([^\)]+\)")  # Parenthesis removal


class Node:
    def __init__(
        self,
        id: str,
        relations: List[str],
        depth: int,
        head_id: str,
        surfaces: List[str] = [],
    ):
        self.id = id
        self.relations = relations
        self.depth = depth
        self.head_id = head_id
        self.surfaces = surfaces

    def add_surfaces(self, surfaces):
        self.surfaces = surfaces


class Wikidata:

    """A class object containing the full wikidata KB."""

    def __init__(self, id2name, id2nb, id2date):
        self.id2name = id2name
        self.id2date = id2date
        self.id2nb = id2nb

    def retrieve_neighbors(self, query: Node, distance: int = 1) -> List[Node]:
        """get a neighbor of an entity with BFS

        :param query: query node.
        :param distance: k-hop neighbors.
        """
        assert distance > 0
        depth_limit = query.depth + distance
        retrieved = [
            self.get_surface_forms(Node(n, [rel], query.depth + 1, query.id))
            for rel, n in self.id2nb.get(query.id, [])
        ]

        if distance == 1:
            return retrieved

        q = deque(retrieved)
        while len(q) > 0:
            node = q.popleft()
            if node.depth > depth_limit:
                break

            retrieved.append(node)
            for r, n in self.id2nb.get(node.id, []):
                # Construct a new node and register surface forms for the node
                neighbor = self.get_surface_forms(
                    Node(n, node.relations + [r], node.depth + 1, node.id)
                )
                q.append(neighbor)

        return retrieved

    def get_surface_forms(self, node: Node) -> Node:
        """get surface forms of a given node and return the same node."""
        surface_forms = []
        expressions = self.id2name.get(node.id)
        if expressions is not None:
            canonical = re.sub(PAREN, r"\1", expressions[0])
            aliases = list(
                set(re.sub(PAREN, r"\1", e.strip()) for e in expressions[1:])
            )  # remove duplicates
            node.add_surfaces([canonical] + aliases)

        return node


class SubGraph:

    """A subgraph for an article."""

    def __init__(self, root: Node):
        self.root = root
        self.heads: Dict[str, Node] = {root.id: root}
        self.entities = []

    def expand(self, ent_id: str, kg: Wikidata):
        """retrieve one-hop neighbors from a given query wikidata ID.

            ent_id: Wikidata ID.
            kg: Wikidata.
        """
        # Search the graph from the title entity if ent_id is not given.
        if ent_id is None:
            ent_id = self.root.id
            query_node = self.root
        elif ent_id in self.heads:
            query_node = self.heads[ent_id]

        retrieved = kg.retrieve_neighbors(query_node, distance=1)
        head_candidates = set([n.id for n in retrieved])
        for hid in head_candidates:
            if hid not in self.heads:
                self.heads[hid] = next(n for n in retrieved if n.id == hid)

        self.entities += retrieved


def load_corpus(wikitext_path: Path):
    """reads the dataset preprocessed by the split_wikitext.py."""
    dataset, skipped = {}, {}
    for split in ["train", "valid", "test"]:
        with (wikitext_path / f"{split}.json").open("r") as f:
            data = json.load(f)

        loaded_articles, skipped_articles = [], []
        for i, article in enumerate(data):
            if article["id"] == "":
                skipped_articles[split].append(article)

            # Join section text into single string
            full_body = f"= {article['title'].strip()} = "
            for head, sec in article["sections"]:
                head = head.strip()
                if head != "":
                    head = f"= = {head.strip()} = = "
                sec = sec.strip() + " "
                if sec.strip() == "":
                    sec = ""
                full_body += head + re.sub(r"\s+", " ", sec)

            data[i]["tokens"] = full_body
            loaded_articles.append(article)

        dataset[split] = data
        skipped[split] = skipped_articles

    return dataset, skipped


def find_surface_matches(
    article: str, token_array: np.ndarray, entities: List[Node], match_stopwords, lower
):
    """looks for surface matches for given entities over an article.

    :param article: Article text in a single string.
    :param token_array: Token-split numpy array of the article.
    :param entities: List of entity nodes to match the surfaces against.
    :param match_stopwords: A flag for whether or not to include stopwords.
    :param lower: A flag for whether or not to match lowercased.
    """
    found_spans = []
    for entity in entities:

        for surface in entity.surfaces:
            if len(surface.strip()) == 0:
                continue

            # single word entity
            if len(surface.split(" ")) == 1:
                if not match_stopwords and surface in STOPWORDS:
                    continue

                match_indices = np.asarray(
                    token_array == (surface.lower() if lower else surface)
                ).nonzero()[0]
                if match_indices.size > 0:  # Couldn't be found.
                    found_spans += [
                        (
                            entity.id,
                            entity.head_id,
                            (i, i + 1),
                            entity.relations,
                            surface,
                            entity.surfaces,
                        )
                        for i in match_indices
                    ]

            # multi-word entity: concatenate it and match, and then split the string back
            else:
                # multi-word entity matching
                concat_surface = surface.replace(" ", "####")
                if lower:
                    concat_surface = concat_surface.lower()

                try:
                    # Insert concat chars in the original article, and split again if there's any merged.
                    subbed_article, sub_count = re.subn(
                        re.escape(surface.lower() if lower else surface),
                        concat_surface,
                        article,
                    )
                    if sub_count > 0:
                        phrase_len = len(surface.split(" "))
                        subbed_token_array = np.array(subbed_article.split(" "))
                        match_indices = np.asarray(
                            subbed_token_array == concat_surface
                        ).nonzero()[0]
                        # token-level matched indices if it were not concat'd
                        match_token_indices = [
                            range(
                                pos + (phrase_len - 1) * i,
                                pos + (phrase_len - 1) * i + phrase_len,
                            )
                            for i, pos in enumerate(match_indices)
                        ]
                        found_spans += [
                            (
                                entity.id,
                                entity.head_id,
                                (i.start, i.stop),
                                entity.relations,
                                surface,
                                entity.surfaces,
                            )
                            for i in match_token_indices
                        ]

                # TODO: more specific exception handling.
                except Exception:
                    LOGGER.info(f"Bad name : {surface}")

    # sort by the starting position
    found_spans = sorted(found_spans, key=lambda x: x[2][0])

    return found_spans


def annotate_articles(
    dataset,
    canonical_forms,
    kg: Wikidata,
    match_title: bool = True,
    match_date: bool = True,
    match_stopwords: bool = False,
    lower: bool = False,
):
    """annotates entity matches in each article.

    :param dataset: Dictionary of list of instances.
    :param kg: KG object containing the loaded wikidata dictionaries.
    :param match_title: A flag for whether or not to match the title itself.
    :param match_date: A flag for whether or not to match the date info if present.
    :param match_stopwords: A flag for whether or not to match the stopwords.
    :param lower: A flag for whether or not to match lowercased.

    """

    def remove_duplicate_occurrence(matches) -> Dict[str, int]:
        """finds the earliest occurrences of matched entities."""
        earliest_span = {}
        for ent_id, _, (bg, ed), _, _, _ in matches:
            if ent_id != "" and (
                ent_id not in earliest_span or bg < earliest_span[ent_id][0]
            ):
                earliest_span[ent_id] = (bg, ed)
        return earliest_span

    spans = []

    for idx, article in tqdm.tqdm(
        enumerate(dataset), ncols=80, desc="Searching over dataset", ascii=True
    ):
        title, title_id, article_body = (
            article["title"],
            article["id"],
            article["tokens"],
        )

        # prepare a graph with the title entity
        graph = SubGraph(Node(title_id, [], 0, ""))

        if lower:
            article_body = article_body.lower()

        tokens = article_body.strip().split(" ")

        token_array = np.array(tokens)
        n_tokens = len(token_array)

        ent_id = graph.heads[article["id"]].id  # first entity to check
        graph.expand(None, kg)  # expand and get the new surfaces
        entities_tobe_searched = graph.entities
        if article["id"] == "Q2353693":
            import pdb

            pdb.set_trace()

        if match_title:
            # ALWAYS put canonical forms at the first index of the surface forms.
            canonical_form = canonical_forms[idx]
            if canonical_form == title:
                surfaces = [title]
            else:
                surfaces = [canonical_form, title]

            entities_tobe_searched += [
                Node(
                    id="",
                    relations=["@TITLE@"],
                    depth=1,
                    head_id=ent_id,
                    surfaces=surfaces,
                )
            ]

        if match_date:
            if ent_id in kg.id2date and kg.id2date[ent_id][1] == 0:
                dates = kg.id2date.get(ent_id)[0]
                kg.id2date[ent_id][1] = 1
            else:
                dates = []

            entities_tobe_searched += [
                Node(id="", relations=d[1], depth=d[2], head_id=ent_id, surfaces=d[0])
                for d in dates
            ]

        found = find_surface_matches(
            article_body,
            token_array,
            entities=entities_tobe_searched,
            match_stopwords=match_stopwords,
            lower=lower,
        )
        spans.append(found)

    return spans


def expand_date(date):
    """Get multiple surface form date expressions from a date object."""

    def convert_datetime(date_string):
        year = "%Y"
        if date_string.startswith("-"):
            year = "BC%Y"
            date_string = date_string[1:]

        # add more formats here if more variations are necessary
        fmts = [f"%-d %B {year}", f"%B %-d , {year}"]

        date_obj = datetime.strptime(date_string, "%Y-%m-%dT00:00:00Z")
        s = [datetime.strftime(date_obj, f) for f in fmts]
        return s

    name2propid = {"dob": "P569", "dod": "P570"}
    expanded_dates = []
    for k, v in date.items():
        if k in name2propid:
            try:
                expanded_dates += [[convert_datetime(v), [name2propid[k]], 1]]
            except ValueError:
                LOGGER.info(f"Date expression: {v} was not parsed.")

    return expanded_dates


def convert_ids(data, prop2name):
    """Convert relation IDs into actual surface forms."""
    converted = []
    for matches in data:
        named = [
            (
                ent_id,
                head_id,
                (int(span[0]), int(span[1])),
                (relations, [prop2name.get(r, "NO_REL") for r in relations]),
                surface,
                surfaces,
            )
            for ent_id, head_id, span, relations, surface, surfaces in matches
        ]
        converted.append(named)

    return converted


def main(args):

    args.output_dir.mkdir(exist_ok=True)
    LOGGER.info(f"Save directory is {args.output_dir}")

    dataset, _ = load_corpus(args.wikitext_dir)
    LOGGER.info("Loaded dataset.")

    # Load canonical_forms
    canonical_forms = {}
    for mode in ["train", "valid", "test"]:
        with (args.canonical_forms_dir / f"{mode}_canonical_forms.txt").open("r") as f:
            canonical_forms[mode] = [l.strip() for l in f]
            assert len(canonical_forms[mode]) == len(dataset[mode])
    LOGGER.info("Loaded canonical forms.")

    # Assuming that triples are separated in multiple files
    id2name: Dict[str, List[str]] = {}
    with (args.wikidata_dir / "items.txt").open("r") as f:
        for line in tqdm.tqdm(
            f, ncols=80, desc="Loading entities", ascii=True, total=37899556
        ):
            line = line[:-1].split("\t")
            if len(line) == 3:
                k, name, aliases = line
            else:
                k, name, aliases, _ = line

            id2name[k] = [name] + aliases.split("||")

    # only items with date info found
    id2date = {}
    with (args.wikidata_dir / "item_with_dates.bin").open("rb") as f:
        item_with_dates = pickle.load(f)
        for k, v in tqdm.tqdm(
            item_with_dates.items(),
            ncols=80,
            desc="Loading entities w/ dates",
            ascii=True,
        ):
            # item_with_dates contain dob/dod info under "n/a" key
            # Second element to keep track of whether the date info is considered or not
            id2date[k] = [expand_date(v), 0]

    LOGGER.info(f"Loaded names and dates.")

    # load triples
    id2nb: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    with (args.wikidata_dir / f"triples.txt").open("r") as f:
        for line in tqdm.tqdm(
            f, ncols=80, desc="Loading triples", ascii=True, total=270306417
        ):
            subj, rel, obj = line[:-1].split("\t")
            if obj.startswith("Q"):
                id2nb[subj].append((rel, obj))

    # load properties
    with (args.wikidata_dir / "properties.bin").open("rb") as f:
        props = pickle.load(f)
        prop2name = {k: v["name"] for k, v in props.items()}
        prop2name["@TITLE@"] = "TITLE"

    wikidata_kg = Wikidata(id2name, id2nb, id2date)

    for mode in tqdm.tqdm(dataset, ncols=80, total=3, ascii=True):
        if args.use_shards:
            shards = len(dataset[mode]) // args.shard_size
            for shard in tqdm.trange(shards + 1, ncols=80, desc="Shard", ascii=True):
                slice_ = slice(
                    args.shard_size * shard,
                    min(args.shard_size * (shard + 1), len(dataset[mode])),
                )
                matches = annotate_articles(
                    dataset[mode][slice_],
                    canonical_forms[mode][slice_],
                    wikidata_kg,
                    match_title=True,
                    match_date=True,
                    match_stopwords=False,
                    lower=args.lower,
                )

                # Insert show_example function here.
                named_matches = convert_ids(matches, prop2name)

                output = []
                for article, matched_spans in tqdm.tqdm(
                    list(zip(dataset[mode][slice_], named_matches)),
                    ncols=80,
                    ascii=True,
                ):
                    # save the triple annotations with the tokens
                    output.append((article["tokens"].strip().split(" "), matched_spans))

                # Dump the dataset splits with relation annotations
                with (args.output_dir / f"{mode}_{shard:02d}.pkl").open("wb") as f:
                    pickle.dump(output, f)

        else:
            matches = annotate_articles(
                dataset[mode],
                canonical_forms[mode],
                wikidata_kg,
                match_title=True,
                match_date=True,
                match_stopwords=False,
                lower=args.lower,
            )
            # Insert show_example function here.
            named_matches = convert_ids(matches, prop2name)

            output = []
            for article, relations in tqdm.tqdm(
                list(zip(dataset[mode], named_matches)),
                ncols=80,
                ascii=True,
                desc="Dumping",
            ):
                # save the triple annotations with the tokens
                output.append((article["tokens"].strip().split(" "), relations))

            # Dump the dataset splits with relation annotations
            with (args.output_dir / f"{mode}.pkl").open("wb") as f:
                pickle.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Perform surface matching on text using Wikidata dump."
    )
    parser.add_argument(
        "--wikitext-dir",
        type=Path,
        help="Path to Article-split WikiText-103 JSON files.",
        required=True,
    )
    parser.add_argument(
        "--wikidata-dir",
        type=Path,
        help="Path to the extrated Wikidata dump.",
        required=True,
    )
    parser.add_argument(
        "--canonical-forms-dir", type=Path, help="Path to canonincal forms."
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Path to the output directory.", required=True
    )
    parser.add_argument(
        "--use-shards",
        action="store_true",
        help="Whether to split the result files into shards.",
    )
    parser.add_argument(
        "--shard-size", type=int, help="# of examples in one shard.", default=1000
    )
    parser.add_argument(
        "--lower", action="store_true", help="If matching lowercased or not."
    )
    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logfile = args.output_dir / f"{str(datetime.now()).replace(' ', '_')}.log"

    if not args.output_dir.exists():
        args.output_dir.mkdir()
    LOGGER.info(f"Saving into {args.output_dir}.")

    logging.basicConfig(
        datefmt="%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.getLevelName(args.logging_level),
        filename=logfile,
    )
    main(args)
