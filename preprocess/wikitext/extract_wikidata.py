"""Preprocessing script for latest-truthy.nt, which is obtainable from
https://dumps.wikimedia.org/wikidatawiki/entities/

The original script is authored by Shoetsu @ University of Tokyo and modified by
Hiroaki Hayashi.

Usage:

    python extract_wikidata.py \
        --source-path /path/to/latest-truthy.nt \
        --target-dir /path/to/target_dir \
        --lang LANG \
        --required-value-types VALUE_TYPES \
        --exclude-value-types VALUE_TYPES \

* VALUE_TYPES can be specified from the following: name, aka, desc.

* Specify LANG to harvest surface forms only in the language.

"""
import argparse
import bz2
import codecs
import gzip
import pickle
import re
import sys
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm


# for important columns that have a particular value or string.
url2type = {
    "<http://schema.org/name>": "name",
    "<http://schema.org/description>": "desc",
    "<http://www.w3.org/2004/02/skos/core#altLabel>": "aka",
    "<http://www.wikidata.org/prop/direct/P569>": "dob",
    "<http://www.wikidata.org/prop/direct/P570>": "dod",
}

# the values of name, description, and aka are followed by a lang symbol. (e.g. "green plants"@en)
VALUE_TEMPLATE = r"\"(.+)\"\s*@({})"


def create_entity_template():
    return {"name": "", "desc": "", "aka": []}


def is_an_edge(triple):
    pattern = "[PQ][0-9]+"
    ids = [re.match(pattern, x.split("/")[-1][:-1]) for x in triple]
    if not None in ids:
        return [m.group(0) for m in ids]
    else:
        return None


def is_a_value(triple):
    rel = triple[1]
    return rel in url2type.keys()


def process_data(args):

    """reads `latest-truthy.nt` line by line and extract both entities and triples
    informatoin.

    Entities are stored in the form of gigantic dictionaries.
    Triples are dumped in this function as the function processes each line.
    """
    f_triple = (args.target_dir / "triples.txt").open("w")

    counts = {"item": 0, "triple": 0}
    entities = {}
    value_regx = VALUE_TEMPLATE.format(args.lang)
    date_regx = r"\"(.+)\"\^\^.+"

    if args.source_path.name.endswith("gz"):
        open_func = gzip.open
        is_byte = True
    elif args.source_path.name.endswith("bz2"):
        open_func = bz2.open
        is_byte = True
    else:
        open_func = open
        is_byte = False

    # Approximate number of lines, as of the dump from 20180921.
    for i, line in tqdm(
        enumerate(open_func(args.source_path, "r")), ncols=80, total=3127159097, ascii=True
    ):
        if is_byte:
            line = line.decode("utf8")
        line_splits = line.split(" ")
        subj_url, rel_url, obj_url = (
            line_splits[0],
            line_splits[1],
            " ".join(line_splits[2:-1]),
        )

        # Extract only (entity, prop, entity) or (entity | prop, type, value) triples.
        triple = is_an_edge((subj_url, rel_url, obj_url))
        if triple:
            print("\t".join(triple), file=f_triple)
            counts["triple"] += 1

        # Extract values (name, description, aka.)
        elif is_a_value((subj_url, rel_url, obj_url)):
            v_type = url2type[rel_url]

            subj = subj_url.split("/")[-1][:-1]  # "<.+/Q[0-9+]>" -> "Q[0-9+]"

            if v_type in ["dob", "dod"]:
                m = re.match(date_regx, obj_url)
            else:
                m = re.match(value_regx, obj_url)

            if m:  # values must be written in target lang.
                v = m.group(1)  # value
                v = " ".join([x for x in v.split() if x]).strip()
                v = codecs.decode(v, "unicode-escape")
            else:
                continue

            if subj not in entities:
                entities[subj] = create_entity_template()
                counts["item"] += 1
            if v_type == "aka":
                entities[subj][v_type].append(v)
            else:
                entities[subj][v_type] = v

        if args.debug and i > 10000000:
            break

    f_triple.close()

    return entities, counts


def filter_items(entities, required_value_types=[], exclude_value_types=[]):
    def is_complete(entity):
        if False not in [
            True if entity[v_type] else False for v_type in required_value_types
        ]:
            return True
        else:
            return False

    if required_value_types:
        # Remove incomplete (no name) entities due to lack of data
        e_complete = {k: v for k, v in entities.items() if is_complete(v)}
    else:
        e_complete = entities

    if exclude_value_types:
        # Remove fields from entities
        for vt in exclude_value_types:
            for k, e in e_complete.items():
                if vt in e:
                    del e_complete[k][vt]

    items = {k: v for k, v in e_complete.items() if k[0] == "Q"}
    item_with_dates = {k: v for k, v in items.items() if "dob" in v or "dod" in v}
    props = {k: v for k, v in e_complete.items() if k[0] == "P"}
    return items, item_with_dates, props


def dump_dict(data, path):
    with path.open("w") as f:
        if isinstance(data, dict):  # items, props
            for k, values in list(data.items()):
                name = values["name"]
                aka = values["aka"]
                if type(aka) == list:
                    aka = "||".join([x for x in aka])
                columns = [k, name, aka]
                if "desc" in values:
                    columns.append(values["desc"])
                print("\t".join(columns), file=f)
        else:
            raise Exception


def main(args):
    args.target_dir.mkdir(exist_ok=True)

    f_items = args.target_dir / "items.bin"
    f_item_with_dates = args.target_dir / "item_with_dates.bin"
    f_props = args.target_dir / "properties.bin"

    entities, counts = process_data(args)  # Triples are already dumped
    items, iwds, props = filter_items(
        entities, args.required_value_types, args.exclude_value_types
    )
    print(f"items, props, triples = ({len(items)}, {len(props)}, {counts['triple']})")

    # Additionally dump items and props line by line.
    dump_dict(items, f_items.with_suffix(".txt"))
    dump_dict(props, f_props.with_suffix(".txt"))

    if not f_items.exists():
        with f_items.open("wb") as f:
            pickle.dump(items, f)
        del items

    if not f_item_with_dates.exists():
        with f_item_with_dates.open("wb") as f:
            pickle.dump(iwds, f)
        del iwds

    if not f_props.exists():
        with f_props.open("wb") as f:
            pickle.dump(props, f)
        del props


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Wikidata dump parser.")
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default="./extracted")
    parser.add_argument("--lang", default="en")
    parser.add_argument(
        "--required_value_types",
        nargs="+",
        default=["name"],
        help="Required value types which an *entity* must have.",
    )
    parser.add_argument(
        "--exclude-value-types",
        nargs="+",
        default=[],
        help="Excluded value types which an *entity* must omit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode; only a small amount is extracted.",
    )
    args = parser.parse_args()
    main(args)
