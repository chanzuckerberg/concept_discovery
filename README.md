# Biomedical Concept Discovery

## Setup

NOTE: Concept discovery requires UMLS access. See `/home/ds-share/data2/users/rangell/entity_discovery/UMLS_preprocessing/AnntdData/`.

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

## Example Usage

