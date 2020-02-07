# Preprocessing Routine for WikiFacts Dataset

## Dataset Format

The dataset can be downloaded from <https://bitbucket.org/skaasj/wikifact_filmactor/overview>. The link also contains
descriptions of the data format. Simply said, the dataset contains three files for each topic (Freebase topic, i.e.
Wikipedia article title):

- `.sm`: Wikipedia summary annotated with entities.
- `.fb`: Freebase relations related to the topic. A relation is a triplet `(subject, relation_type, object)`. Each
  relation is given as a triplet of Freebase IDs.
- `.en`: The same relations, given in their English names.

For instance, the topic `Louise_Allbritton` is Freebase ID `m.0k3jp_`. An excerpt of the `.fb` file is as follows:

```
/ns/m.0k3jp_ /ns/people.person.spouse_s [/ns/m.06638z7 /ns/people.marriage.spouse /ns/m.03wl63]
/ns/m.0k3jp_ /ns/people.person.gender /ns/m.02zsn
```

And the corresponding `.en` file:

```
Louise_Allbritton Spouse_(or_domestic_partner) [NO_EN Spouse Charles_Collingwood]
Louise_Allbritton Gender Female
```

The stuff in `[]` is called a "composite value type" (CVT), which is a relation triplet in another triplet. This can be
understood as relation between an entity and a `namedtuple` of entities, and Freebase stores the relation as multiple
relations between the entity and each entity in the tuple, where the tuple entity is represented as another relation
between itself and the tuple ID.

Here is an excerpt from the `.sm` file:

```
@@louise_allbritton/f/ns/m.0k3jp_@@ (3 july 1920–16 february 1979) was an american @@actress/f/ns/m.02hrh1q@@ born in
@@oklahoma_city_oklahoma/f/ns/m.0fvzg@@.
```

Entity mentions are tagged as `@@[words]/[type]/ns/[freebase_id]@@`, where `type` can be either `f` or `a`, denoting
relations in Freebase and Wikipedia anchors. `a` type mentions are links in the Wikipedia page but does not exist
relations in Freebase.

## Processing Relations

For topic relations, we only use those whose subject is the topic _(the `.fb` file also contains relations where the
topic is the object)_. For relations with CVT as object, we replace it with a simple relation using the following rule:

```
topic rel_type [tuple_id tuple_rel_type object]
 =>
topic tuple_rel_type object
```

This is assumed from the heat map in the appendix of the paper.

Note that the object in CVT could also be the topic. Such relations are also removed.

### Indexing Relations

Relations are indexed starting from 0. Negative indices are used for special relations:

- Index -1 (Not-a-fact, NaF): As the name.
- Index -2 (Anchor): An entity in Freebase but is not has no relation with the topic. Including these actually gives the
  model an unfair advantage (it'll know these are bound to appear once in the text), but they're still included for
  completeness.
- Index -3 (Topic-itself): A self-relation. In WikiText, this is called `TITLE`.

### Indexing Entities

Entities are indexed starting from 0. Negative indices are used for special entities:

- Index -1 (UNK): An entity that is not in the pretrained embeddings.

## Processing Text

### Assigning Copying Positions

Another problem is that the surface form of words in an extracted entity may not exactly match the canonical entity
name (the first name in the list of aliases). Freebase, like Wikidata, contains an `aka` attribute which lists
alternative names, but unfortunately is not contained in the WikiFacts dataset. Since the Freebase dump is way too big
to process, we have two substitution strategies to consider instead:

- Ignore the differences and just assign copying positions sequentially.
- Replace all entity mentions with their canonical forms. This will be done for non-anchor relations only since there
  are no canonical forms for anchors.

We use the "ignore" strategy in experiments since it does not change the original text. Alternatively, you can test
the "replace" strategy by running the `process_wikifacts.py` script with the `--replace` flag.

Finally, we perform additional tokenization using Moses after replacement.

### Anchor Mentions

A post-hoc analysis of matched positions show that:

- Of all 167,287 mentions, 71,728 are anchors.
- Among non-anchor mentions, 50,073 of them are not in canonical form.

### Disambiguating Relations

When matching relations to objects, if there are multiple relation matches, simply pick the first one. This is the
paper's strategy as the authors had answered on OpenReview.
