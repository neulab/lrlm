""" split_wikitext.py: Split WikiText=103 at article-level."""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import requests
from tqdm import tqdm

WIKI_URL = "https://en.wikipedia.org/w/api.php"

TITLE_PATTERN = re.compile(r"^= ([^=]+) =$")
SECTION_PATTERN = re.compile(r"^(= ){2,}([^=]+)(= ){2,}$")


def split_into_articles(path: Path, found_ids):
    """Reads the WikiText line by line and split into articles.

        path: Path to WikiText-103.
        found_ids: title to ID mapping.
    """
    docs = []
    with path.open("r") as f:
        doc = {}
        buffer = ""
        no_record = False
        data = f.read().strip().split("\n")
        missed = []

        for idx, line in tqdm(enumerate(data), ncols=80, total=len(data), ascii=True):
            line = line.strip()

            # Only perform search when there's a symbol
            matched_title = (
                re.search(TITLE_PATTERN, line) if line.startswith("=") else None
            )

            if matched_title is None and no_record:
                continue

            # When a title is detected, the next line is a new line
            elif matched_title is not None and data[idx + 1].strip() == "":
                if "sections" in doc and len(doc["sections"]) > 0:
                    if len(buffer) > 0:
                        doc["sections"][-1][1] = buffer
                        buffer = ""

                    # Commit document
                    docs.append(doc)
                    doc = {}

                processed_title = preprocess([matched_title.group(1)])[0]

                # Ignore the whole document until next valid title is found
                if found_ids is not None and processed_title not in found_ids:
                    no_record = True
                    doc["id"] = ""
                else:
                    no_record = False
                    doc["title"] = matched_title.group(1)
                    doc["sections"] = [["", ""]]
                    doc["id"] = found_ids[processed_title]

            elif matched_title is not None:
                missed.append(matched_title.group(1))
                buffer += line + " "

            # Not a title, but with heading
            elif line.startswith("= ="):
                # Commit buffer
                if len(buffer) > 0:
                    doc["sections"][-1][1] = buffer
                    buffer = ""

                line += " "
                section_match = re.search(SECTION_PATTERN, line)
                if section_match is None:
                    section_header = ""
                else:
                    section_header = section_match.group(2)
                doc["sections"].append([section_header, ""])

            else:
                buffer += line + " "

        if len(buffer) > 0:
            doc["sections"][-1][1] = buffer
        if len(doc.keys()) > 0:
            docs.append(doc)

    return docs


def preprocess(titles: List[str]) -> List[str]:
    """Handmade text normalization."""
    preprocessed = []
    for t in titles:
        t = re.sub(r"\s@-@\s", "-", t)
        t = re.sub(r"\(\s(.+)\s\)", r"(\1)", t)
        t = re.sub(r"\s\'(\w+)", r"'\1", t)
        t = t.replace("What (ever)", "What(ever)")
        t = t.replace(" ; ", ";")
        t = t.replace(" – ", "–")  # "em dash" present in some titles
        t = t.replace("... ", "...")
        t = t.replace(" ...", "...")
        t = t.replace(" , ", ", ")
        t = t.replace(" ' ", "' ")
        t = re.sub(r"\s?'\s?([a-z]{,2})\s", r"'\1 ", t)
        t = t.replace(" !", "!")
        t = t.replace(" ?", "?")
        t = t.replace(" :", ":")
        # Only reduce spaces when single characters are ampasanded
        t = re.sub(r"(^|\s)(\w)\s\&\s(\w)(\s|$)", r"\1\2&\3\4", t)

        preprocessed.append(t)
    return preprocessed


def query_wikidata_id(titles: List[str], dest: Path) -> Dict[str, str]:
    """Queries Wikipedia (slowly) and obtain the *Wikidata* ID given the title strings.

        titles: List of title strings.
        dest: Output file name to store the found IDs, if specified.
    """

    def get_id(title: str) -> str:
        """Issues a query to obtain Wikidata ID for a given title string.
        Returns an empty string if not found.
        """
        query = {
            "action": "query",
            "prop": "pageprops",
            "titles": title,
            "format": "json",
            "maxlag": 5,
        }
        try:
            r = requests.get(WIKI_URL, params=query)
            page = json.loads(r.content.decode(r.encoding))["query"]["pages"]
            page_id = list(page.keys())[0]
            return page[page_id]["pageprops"]["wikibase_item"]
        except KeyError:  # When the title is not found for any reason
            return ""

    titles = preprocess(titles)
    # Get ids by tagging missing ones as empty string
    with tqdm(titles, ncols=80, ascii=True, desc="Querying Wikidata") as pbar:
        ids = {t: get_id(t) for t in pbar}

    notfound_count = len([k for k, v in ids.items() if len(v) == 0])

    with dest.open("w") as f, dest.with_suffix(".missing").open("w") as fnot:
        for t, id_ in ids.items():
            if len(id_) > 0:
                print(t, id_, file=f, sep="\t")
            else:
                print(t, "", file=fnot, sep="\t")
                print(t, "", file=f, sep="\t")

    print(
        f"Wikidata IDs for {notfound_count}/{len(titles)} articles were not found.\n"
        "Resolve this by visitng *.ids.missing and query by yourself and fill them"
        " into *.ids.txt."
    )
    assert len(titles) == len(ids)
    return ids


def main(args):

    args.output_dir.mkdir(exist_ok=True)

    # Considers the raw WikiText-103, which is not unk-ed
    wikitext_paths = args.wikitext_dir.glob("*.raw")

    dataset = {}
    for wpath in wikitext_paths:
        splits = wpath.with_suffix("").name.split(".")[1]

        ids_path = args.ids_dir / f"{splits}.ids.txt"
        if ids_path.exists():
            with ids_path.open("r") as f:
                found_ids: Dict[str, str] = {}
                for l in f.read().strip().split("\n"):
                    name, id_ = tuple(l.strip().split("\t"))
                    found_ids[name] = id_
        else:
            found_ids = None

        dataset[splits] = split_into_articles(wpath, found_ids)

        # Make found_ids, they have to be constructed based on the split articles
        if found_ids is None:
            titles = [d["title"] for d in dataset[splits]]
            found_ids = query_wikidata_id(titles, ids_path)

            for d in dataset[splits]:
                d["id"] = found_ids[preprocess([d["title"]])[0]]

        if args.drop_missing_ids:
            dataset[splits] = [d for d in dataset[splits] if len(d["id"]) > 0]

        print(f"{splits}: {len(dataset[splits])} articles.")

        with (Path(args.output_dir) / f"{splits}.json").open("w") as fout:
            json.dump(dataset[splits], fout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocess WikiText-103 into a list of articles.")
    parser.add_argument(
        "--wikitext-dir",
        type=Path,
        help="Directory to WikiText raw files.",
        required=True,
    )
    parser.add_argument(
        "--ids-dir",
        type=Path,
        help="Directory to title-id mapping files, if exists.",
        default="",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Where the articles are dumped.", default=None
    )
    parser.add_argument(
        "--drop-missing-ids",
        action="store_true",
        help="Whether to keep articles with missing IDs.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.wikitext_dir

    main(args)
