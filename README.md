# Biomedical Concept Discovery

## License

See [here](LICENSE.md).

## Code of Conduct

This project adheres to the Contributor Covenant 
[code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md). 
By participating, you are expected to uphold this code. 
Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Project Satus: Unmaintained

This project is unmaintained and unsupported.

## Setup

First install either [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://docs.anaconda.com/anaconda/install/) on your system. We prefer Miniconda since it is more minimal and will take less time to install.

Next, create a conda environment and then activate it.

```bash
$  conda create -n concept_disc python=3.7
$  conda activate concept_disc
```

Install standard packages
```bash
$  conda install numpy scipy tqdm ipython cython
```

Install faiss
```bash
$  conda install faiss-cpu -c pytorch
```

Install and setup NLTK
```bash
$  conda install -c anaconda nltk; python -c "import nltk; nltk.download('punkt')"
```

Install [sent2vec](https://github.com/epfml/sent2vec).

Download [BioSentVec model](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin).

Install higra, scispacy, and scispacy models
```bash
$  pip install higra
$  pip install scispacy
$  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_craft_md-0.2.5.tar.gz
$  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_jnlpba_md-0.2.5.tar.gz
$  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_bc5cdr_md-0.2.5.tar.gz
$  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_bionlp13cg_md-0.2.5.tar.gz
```

### Example Usage

```bash
python src/discover_concepts.py --data_source 'pubmed_download' --pubmed_special_query 'single_cell_biology' --task 'concept_discovery' --biosentvec_path './bin/BioSentVec_PubMed_MIMICIII-bigram_d700.bin' --output_dir 'single_cell_biology_discovered_concepts'
```

### Example Output

```
================================================================================
Cluster phrases:
----------------
single-cell RNA (70) ; single-cell mRNA (7) ; single-cell DNA (3) ; single-cell SPAdes (1) ; Joint single-cell DNA (1) ; single-cell PA (1)

Closest concepts:
-----------------
        Primary Name: Finding of Lupus Erythematosus cell level Type: None (None)       Matching Synonym:Finding of Lupus Erythematosus cell level      Score:0.3510943055152893
        Primary Name: Glucose-6-phosphate dehydrogenase measurement, quantitative       Type: None (None)       Matching Synonym:Red cell G6PD level    Score:0.34502702951431274
        Primary Name: Set of peripheral neuroglial cells        Type: None (None)       Matching Synonym:Set of Schwann cells   Score:0.33851897716522217
        Primary Name: Absence of CD8+ T cells   Type: None (None)       Matching Synonym:Absence of CD8+ T cells        Score:0.33851897716522217
        Primary Name: Set of epinephric cells   Type: None (None)       Matching Synonym:Set of epinephric cells        Score:0.33851897716522217
        Primary Name: APUD Cells        Type: None (None)       Matching Synonym:Set of APUD cells      Score:0.33851897716522217
        Primary Name: Absence of B cells        Type: None (None)       Matching Synonym:Absence of B cells     Score:0.33851897716522217
        Primary Name: Cells     Type: None (None)       Matching Synonym:Set of cells   Score:0.33851897716522217
        Primary Name: Abnormality of T cells    Type: None (None)       Matching Synonym:Abnormality of T cells Score:0.33851897716522217
        Primary Name: Loss of Purkinje cells    Type: None (None)       Matching Synonym:Loss of Purkinje cells Score:0.33851897716522217

================================================================================
Cluster phrases:
----------------
constructs (71) ; REINDEER constructs (1) ; TALEN constructs (1)

Closest concepts:
-----------------
        Primary Name: Scaffold  Type: None (None)       Matching Synonym:scaffolds      Score:0.3803158402442932
        Primary Name: Tissue Scaffolds  Type: None (None)       Matching Synonym:Tissue scaffolds       Score:0.3803158402442932
        Primary Name: expression vector Type: None (None)       Matching Synonym:expression vectors     Score:0.3640291690826416
        Primary Name: Reporter  Type: None (None)       Matching Synonym:reporters      Score:0.3457525372505188
        Primary Name: SH3PXD2A gene     Type: None (None)       Matching Synonym:five SH3 domains       Score:0.3439686894416809
        Primary Name: Version   Type: None (None)       Matching Synonym:versions       Score:0.3217689096927643
        Primary Name: vineland-ii subdomains    Type: None (None)       Matching Synonym:vineland-ii subdomains Score:0.3171254098415375
        Primary Name: conceptual model  Type: None (None)       Matching Synonym:conceptual models      Score:0.3127778172492981
        Primary Name: proopiocortin fragments   Type: None (None)       Matching Synonym:proopiocortin fragments        Score:0.30793625116348267
        Primary Name: Carpal synostosis Type: None (None)       Matching Synonym:Carpal fusions Score:0.30436059832572937

================================================================================
Cluster phrases:
----------------
cell-free DNA (31) ; Plasma cell-free DNA (2) ; cell-free DNA NGS (1) ; cell-free RNA (1) ; Pathogen cell-free DNA (1) ; cell-free RNAs (1)

Closest concepts:
-----------------
        Primary Name: polyphenylalanine Type: None (None)       Matching Synonym:polyphenylalanine      Score:0.34550097584724426
        Primary Name: Cell Culture Techniques   Type: None (None)       Matching Synonym:cell culture   Score:0.2936530113220215
        Primary Name: embryo/fetus cell culture Type: None (None)       Matching Synonym:embryo/fetus cell culture      Score:0.2922312319278717
        Primary Name: inhibition of polyadenylate polymerase activity   Type: None (None)       Matching Synonym:inhibition of polyadenylate polymerase activity        Score:0.28492480516433716
        Primary Name: mononuclear cell procoagulant activity    Type: None (None)       Matching Synonym:mononuclear cell procoagulant activity Score:0.27952104806900024
        Primary Name: protein-synthesizing GTPase activity, initiation  Type: None (None)       Matching Synonym:protein-synthesizing GTPase activity, initiation       Score:0.27114439010620117
        Primary Name: inhibition of polyadenylic polymerase activity    Type: None (None)       Matching Synonym:inhibition of polyadenylic polymerase activity Score:0.2706105709075928
        Primary Name: cytoplasmic exosome       Type: None (None)       Matching Synonym:prokaryotic exosome multienzyme ribonuclease complex   Score:0.26286691427230835
        Primary Name: inhibition of poly(A) polymerase activity Type: None (None)       Matching Synonym:inhibition of poly(A) polymerase activity      Score:0.2625434994697571
        Primary Name: negative regulation of NAD+ ADP-ribosyltransferase activity       Type: None (None)       Matching Synonym:inhibition of poly(ADP-ribose)polymerase activity      Score:0.2625434994697571

================================================================================
```

## Security Issues

Please note: If you believe you have found a security issue, please responsibly disclose by contacting us at 
[security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
