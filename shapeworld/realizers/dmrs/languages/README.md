# Multilingual ShapeWorld

### Table of content

- [Dependency Minimal Recursion Semantics (DMRS)](#dependency-minimal-recursion-semantics-dmrs)
- [Linearized DMRS Notation (LDN)](#linearized-dmrs-notation-ldn)
    + [How to go from MRS to DMRS using LDN](#how-to-go-from-mrs-to-dmrs-using-ldn)
- [How to integrate the grammar for another language](#how-to-integrate-the-grammar-for-another-language)
    + [Caption components](#caption-components)
- [Links and references](#links-and-references)



## Dependency Minimal Recursion Semantics (DMRS)

DMRS is a graph representation of MRS structures. The transformation to and from DMRS is essentially an equivalence conversion (if the MRS satisfies a few basic assumptions). More information can be found [here](http://www.aclweb.org/anthology/E/E09/E09-1001.pdf) or, more recently, [here](http://www.lrec-conf.org/proceedings/lrec2016/pdf/634_Paper.pdf). There is a DMRS demo website available for some grammars (for instance, [ERG](http://chimpanzee.ling.washington.edu/demophin/erg/) for English, [Jacy](http://chimpanzee.ling.washington.edu/demophin/jacy/) for Japanese, [Zhong](http://chimpanzee.ling.washington.edu/demophin/zhong/) for Mandarin Chinese), where one can parse natural language sentences and inspect the visualized DMRS graph representation.



## Linearized DMRS Notation (LDN)

LDN is a formalism to specify DMRS graphs in a linearized string form, with the aim to be easily read- and writable by humans. In particular, it simplifies the definition of small, potentially underspecified or otherwise constrained DMRS subgraph snippets, which can be used, for instance, to express DMRS graph search queries, DMRS paraphrase or simplification rules, or in general graph mappings of any kind. A one-page overview over the formalism can be found [here](https://www.cl.cam.ac.uk/~aok25/files/graphlang_overview.pdf).


### How to go from MRS to DMRS using LDN

In this informal walkthrough we will look at the English example sentence *"There are three squares"*. Parsed by ACE using the ERG, we get the following MRS structure:

```
[ 
    LTOP: h0 
    INDEX: e2 
    RELS: < 
        [ _be_v_there LBL: h1 ARG0: e2 [ e SF: prop TENSE: pres MOOD: indicative PROG: - PERF: - ] ARG1: x4 ]  
        [ udef_q LBL: h5 ARG0: x4 RSTR: h6 BODY: h7 ]  
        [ card LBL: h8 CARG: "3" ARG0: e10 [ e SF: prop TENSE: untensed MOOD: indicative PERF: - PROG: - ] ARG1: x4 ]  
        [ _square_n_of LBL: h8 ARG0: x4 [ x PERS: 3 NUM: pl GEND: n IND: + ] ARG1: i11 ] 
    > 
    HCONS: < h0 qeq h1 h6 qeq h8 > 
    ICONS: <  > 
]
```

First, each MRS elementary predication (EP) is a node in the DMRS graph, with their intrinsic `ARG0` variable being the node variable. Here, quantifier EPs (and only them) are treated differently in that they do not have their own variable, since they are unambiguously associated with the instance variable of another EP. A `CARG` (constant argument) is represented as suffix to the predicate, enclosed in brackets. This gives us the following list of DMRS nodes in LDN:

```
_be_v_there e[sf=prop,tense=pres,mood=indicative,perf=-,prog=-]
udef_q
card(3) e[sf=prop,tense=untensed,mood=indicative,perf=-,prog=-]
_square_n_of x[pers=3,num=pl,gend=n,ind=+]
```



For the other arguments `ARG#` of an EP (`#` a number in 1-4), there are two possibilities. On the one hand, they might point to another node variable and hence the corresponding node, which is indicated by a link `-#->` if they do not share the same label `LBL`, or `=#=>` if they do share a label. On the other hand, they might point to a label, which is indicated by a link `-#h->` or `=#h=>` to the node corresponding to the head EP of EPs with this label. Finally, some EPs (e.g., quantifiers or connectives) have special values which, accordingly, are indicated by special link types: `RSTR` as `-->`, `#-INDEX` as `-#->` or `=#=>`, `#-HNDL` as `-#h->` or `=#h=>` (`#` either `l` or `r`).

Consequently, we can start with the following DMRS graph representation of the above sentence in LDN:

```
udef_q --> _square_n_of x[pers=3,num=pl,gend=n,ind=+] <=1= card(3) e
```

To be able to represent links for a node which is in more than two relations to other nodes, once introduced they can be referred to via repeating its predicate string (without variable definition) with a leading colon. That enables us to express the second part of the LDN representation:

```
:_square_n_of <-1- _be_v_there e[sf=prop,tense=pres,mood=indicative,perf=-,prog=-]
```

Alternatively, nodes can be annotated with and referred to by an identifier prefixing the colon, which we will do in the following full LDN representation. Both parts are combined with a newline or semicolon, and the top handle `LTOP` and index variable `INDEX` is annotated with `***`:

```
udef_q --> entity:_square_n_of x[pers=3,num=pl,gend=n,ind=+] <=1= card(3) e ;
:entity <-1- ***_be_v_there e[sf=prop,tense=pres,mood=indicative,perf=-,prog=-]
```

For the application in a compositional generation system, we need to match and unify nodes in different DMRS snippets. This is done by indicating an anchor node (possibly underspecified, compare `pred`, `x?`, `node` in the lines below) with square brackets around its identifier. We can, for instance, define a snippet for the quantifier *"three"* as follows:

```
udef_q --> [entity]:pred x?[num=pl] <=1= card(3) e
```

The corresponding snippet for the noun *"square"* is given by:

```
[entity]:_square_n_of x?[pers=3,ind=+]
```



## How to integrate the grammar for another language

To be able to generate language data in another language, you need to provide the grammar file `[LANGUAGE].dat` for the language and an identically named JSON file `[LANGUAGE].json` in this directory `shapeworld/realizers/dmrs/languages/` (see command line below). The latter mainly specifies the mapping of the ShapeWorld-internal semantic objects (attributes, relations, quantifiers, etc) to DMRS subgraphs corresponding to the MRS representation of this concept for the language in question. Moreover, it specifies the features of the sortinfo variables in the grammar and potential post-processing paraphrase rules (which, for instance, can be useful for certain language peculiarities not consistent with the composition system of the ShapeWorld framework).

The directory contains both the files for the [default English version](https://github.com/AlexKuhnle/ShapeWorld/blob/master/shapeworld/realizers/dmrs/languages/english.json) as well as a [minimal specification](https://github.com/AlexKuhnle/ShapeWorld/blob/master/shapeworld/realizers/dmrs/languages/minimal.json) (for English) only containing the necessary components for very basic existential statements. This file is thought of as a starting point for other languages. First, run the following command line from the main ShapeWorld directory for your grammar (for instance, `[LANGUAGE]` being `english` and `[PATH_TO_ACE_CONFIG_TDL]` as `.../erg/ace/config.tdl`):

```bash
.../ShapeWorld$ shapeworld/realizers/dmrs/languages/integrate_language.sh [LANGUAGE] [PATH_TO_ACE_CONFIG_TDL]
```

This compiles the grammar with the version of ACE ShapeWorld is using, and copies the minimal specification file accordingly. After *'translating'* the English DMRS snippets in there, it should be possible to run the following command line to generate your first data:

```bash
.../ShapeWorld$ python generate.py -d [DATA_DIRECTORY] -t agreement -n multishape -l [LANGUAGE] -H
```

This should be sufficient to be able to extend the language file further, where the [full English specification file](https://github.com/AlexKuhnle/ShapeWorld/blob/master/shapeworld/realizers/dmrs/languages/english.json) then serves as guidance for what other entries can be included. If you encounter any problems, please let me know.


### Caption components

The following paragraphs describe the ShapeWorld-internal caption components and give examples from the default ERG-based English language specification file.
(Side note: Some of the terminology (for instance, *'noun'*) might be inspired by English, but this does not necessarily mean that the corresponding components are restricted in that way.)

#### Attributes

Attribute DMRS snippets are required to indicate the head *'modifier'* node with a `[attr]` identifier and the head *'noun'* node via `[type]`. For instance, the attribute *"red"* is defined as `[attr]:_red_a_sw e? =1=> [type]:node` for English. Note that both identifiers might apply to the same node, as is the case for the attribute *"square"* in English: `[attr,type]:_square_n_sw x?[pers=3]`. Furthermore, there are two special *'attribute'* entries, the *'empty'* unspecified type, in English *"shape"* given by `[attr,type]:_shape_n_sw x?[pers=3]`, and one turning a relation (see below) into an attribute, in English via a relative clause specified by `[attr]:pred e[ppi--] =1=> [type]:node`.

#### Entity types

Entity types are a conjunctive combination of attributes. Their DMRS representation is obtained as composition of the corresponding attribute DMRS snippets by unifying their nodes labeled as `[type]`. Consequently, there are no *'atomic'* type DMRS snippets.

#### Relations

Relation DMRS snippets are required to indicate the head relation node via `[rel]` and the head *'reference'* node with `[ref]`. An example from the specification for English is the relation *"to the left of something"*, which is defined as `[rel]:_to_p e? -2-> _left_n_of x[num=s] <-- _the_q; :_left_n_of <=1= _of_p e -2-> [ref]:node <-- _a_q`. There are again two special relation entries for turning an attribute or type into a relation. For English, they are defined as `[rel]:_be_v_id e? -2-> [ref]:_shape_n_sw x? <-- default_q` and `[rel]:_be_v_id e? -2-> [ref]:node <-- default_q`, respectively, both defining an *"is a"* pattern.

#### Existential

The special *existential* entry combines the restrictor type and the body relation, and just expresses that *"there is such an entity"*, in English defined as `_a_q --> [rstr]:pred x?[num=s] <-1- [body]:node`.

#### Quantifiers

...

#### Comparative quantifiers

...

#### Propositions

There is a special entry for each of the above components which turns it into the simplest form of a full proposition. Each of them is composed by fusing node identifier of the same name, so `[type]` with `[type]` etc. In English, for entity types this is a *"there is a"* statement given by `***[head]:_be_v_there e[ppi--] -1-> [type]:pred x? <-- _a_q`, and for the existential quantifier it is given by `***[head,body]:pred e[ppi--]`.



## Links and references

#### Tools

- [DMRS demo website](http://chimpanzee.ling.washington.edu/demophin/erg/)
- [MRS/EDS/DM demo website](http://erg.delph-in.net/logon)
- [pydmrs](https://github.com/delph-in/pydmrs), with a visualization tool for DMRS in XML format under `pydmrs/visualization/index.html`
- [PyDelphin](https://github.com/delph-in/pydelphin), by *Michael Goodman*

#### Parser/generator

- [Answer Constraint Engine](http://sweaglesw.org/linguistics/ace/), by *Woodley Packard*

#### Grammars

- [English Resource Grammar](http://www.delph-in.net/erg/), by *Dan Flickinger*
- [Zhong](http://moin.delph-in.net/ZhongTop) ([GitHub](https://github.com/delph-in/zhong)), by *Zhenzhen Fan, Sanghoun Song, and Francis Bond*
