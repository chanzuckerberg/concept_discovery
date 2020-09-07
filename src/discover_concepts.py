import argparse
import csv
from collections import defaultdict
import glob
import json
import math
import os
import pickle
import sys
import time
from types import SimpleNamespace

import higra as hg
import faiss
from nltk.tokenize import word_tokenize
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.cluster.hierarchy import fcluster
import scispacy
import sent2vec
import spacy
from tqdm import tqdm, trange

from .eutils import PubmedSearch, PubmedDocFetch

from IPython import embed


def read_cord_data(data_dir):
    cord_uid_to_text = defaultdict(list)

    # open the file
    with open(os.path.join(data_dir, 'metadata.csv'), 'r') as f_in:
        reader = csv.DictReader(f_in)
        for row in tqdm(reader, desc='reading articles'):

            # access some metadata
            cord_uid = row['cord_uid']
            title = row['title']
            abstract = row['abstract']
            authors = row['authors'].split('; ')

            # access the full text (if available) for Intro
            introduction = []
            if row['pdf_json_files']:
                for json_path in row['pdf_json_files'].split('; '):
                    with open(os.path.join(data_dir, json_path)) as f_json:
                        full_text_dict = json.load(f_json)

                        # grab introduction section from *some* version of the full text
                        for paragraph_dict in full_text_dict['body_text']:
                            paragraph_text = paragraph_dict['text']
                            section_name = paragraph_dict['section']
                            if 'intro' in section_name.lower():
                                introduction.append(paragraph_text)

                        # stop searching other copies of full text if already got introduction
                        if introduction:
                            break

            # save for later usage
            cord_uid_to_text[cord_uid].append({
                'title': title,
                'abstract': abstract,
                'introduction': introduction
            })

    return cord_uid_to_text


def download_and_preprocess_pubmed(pmids):
    doc_fetcher = PubmedDocFetch()
    articles = doc_fetcher.fetch_documents(pmids)

    id_to_text = defaultdict(list)
    for article in articles:
        d = {}
        if article.title is not None:
            d['title'] = article.title
        if article.abstract is not None:
            d['abstract'] = article.abstract
        id_to_text[article.pmid].append(d)
    return id_to_text


def extract_mentions(id_to_text):
    #scispacy_model = spacy.load("en_core_sci_lg")
    scispacy_models = [
            spacy.load("en_ner_craft_md"),
            spacy.load("en_ner_jnlpba_md"),
            spacy.load("en_ner_bc5cdr_md"),
            spacy.load("en_ner_bionlp13cg_md")
    ]

    id_to_noun_based_phrases = defaultdict(list)
    id_to_other_based_phrases = defaultdict(list)
    for id, text in tqdm(id_to_text.items(),
                               desc='extracting mentions'):

        noun_based_phrases = []
        other_based_phrases = []
        for sub_text in text[0].values():
            if isinstance(sub_text, list):
                sub_text = ' '.join(sub_text)
            for model in scispacy_models:
                doc = model(sub_text)
                doc_tree = doc.to_json()
                ent_strings = [s.as_doc().text.strip()
                        for s in doc.ents]
                for ent, ent_string in zip(doc_tree['ents'], ent_strings):
                    start, end = ent['start'], ent['end']
                    pos_tags = []
                    for token in doc_tree['tokens']:
                        if token['start'] >= start and token['end'] <= end:
                            pos_tags.append(token['pos'])
                    if 'NOUN' in pos_tags or 'PROPN' in pos_tags:
                        noun_based_phrases.append(ent_string)
                    else:
                        other_based_phrases.append(ent_string)
        id_to_noun_based_phrases[id] = noun_based_phrases
        id_to_other_based_phrases[id] = other_based_phrases
    return id_to_noun_based_phrases, id_to_other_based_phrases


def get_phrase_reps_and_metadata(id_to_phrases, sent2vec_model):
    id2embed = {}
    phrase2id = {}
    id2phrase = {}
    doc2ids = defaultdict(set)
    phrase2docs = defaultdict(set)
    next_phrase_id = 0
    for doc_id, phrases in tqdm(id_to_phrases.items(),
                                desc='embedding phrases'):
        for ph in set(phrases):
            _ph = ' '.join(word_tokenize(ph))
            _emb = sent2vec_model.embed_sentence(_ph).reshape(-1,)
            # only include embedable phrases (0.0 is oov)
            if np.linalg.norm(_emb) != 0.0:
                phrase2docs[ph].add(doc_id)
                _id = None
                if ph in phrase2id.keys():
                    _id = phrase2id[ph]
                else:
                    _id = next_phrase_id
                    next_phrase_id += 1
                    phrase2id[ph] = _id
                    id2phrase[_id] = ph
                id2embed[_id] = _emb
                doc2ids[doc_id].add(_id)

    phrase_metadata = {
        'id2embed' : id2embed,
        'phrase2id' : phrase2id,
        'id2phrase' : id2phrase,
        'doc2ids' : doc2ids,
        'phrase2docs' : phrase2docs
    }
    phrase_metadata = SimpleNamespace(**phrase_metadata)
    return phrase_metadata


def build_coo_graph(ids, embeds, k=100):
    # find kNN
    X = np.vstack(embeds)
    X /= np.linalg.norm(embeds, axis=1)[:,np.newaxis]
    d = X.shape[1]
    n_cells = int(math.sqrt(X.shape[0]))
    n_probe = 50
    quantizer = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexIVFFlat(
            quantizer, d, n_cells, faiss.METRIC_INNER_PRODUCT
    )
    index.train(X)
    index.nprobe = n_probe
    index.add(X)
    D, I = index.search(X, k)

    # build sparse coo graph
    dim = max(ids) + 1
    v_remap_ids = np.vectorize(lambda i: ids[i])
    I = v_remap_ids(I)
    rows = np.tile(np.asarray(ids)[:,np.newaxis], (1, I.shape[1])).reshape(-1,)
    cols = I.reshape(-1,)
    data = D.reshape(-1,)
    not_self_loop_mask = (rows != cols)
    rows = rows[not_self_loop_mask]
    cols = cols[not_self_loop_mask]
    data = data[not_self_loop_mask]

    # add in fake edge
    rows = np.concatenate((rows, np.arange(dim-1), np.arange(1, dim)), axis=0)
    cols = np.concatenate((cols, np.arange(1, dim), np.arange(dim-1)), axis=0)
    data = np.concatenate((data, np.repeat(1e-12, 2*(dim-1))), axis=0)

    # make the graph symmetric
    _rows = np.concatenate((rows, cols), axis=0)
    _cols = np.concatenate((cols, rows), axis=0)
    _data = np.concatenate((data, data), axis=0)
    coo = csr_matrix((_data, (_rows, _cols)), shape=(dim, dim)).tocoo()

    return coo


def sparse_avg_hac(coo_pw_sim_mat):
    """Run hac on a coo sparse matrix of edges.
    :param coo_pw_sim_mat: N by N coo matrix w/ pairwise sim matrix
    :return: Z - linkage matrix, as in scipy linkage, other meta data from higra
    """
    ugraph, edge_weights = coo_2_hg(coo_pw_sim_mat)
    t, altitudes = hg.binary_partition_tree_average_linkage(ugraph, edge_weights)
    Z = hg.binary_hierarchy_to_scipy_linkage_matrix(t, altitudes=altitudes)
    return Z, t, altitudes, ugraph, edge_weights


def coo_2_hg(coo_mat):
    """Convert coo matrix to higra input format."""
    rows = coo_mat.row[coo_mat.row < coo_mat.col]
    cols = coo_mat.col[coo_mat.row < coo_mat.col]
    sims = coo_mat.data[coo_mat.row < coo_mat.col]
    dists = sims.max() - sims
    ugraph = hg.higram.UndirectedGraph(coo_mat.shape[0])
    ugraph.add_edges(rows.tolist(),cols.tolist())
    return ugraph, dists.astype(np.float32)


def cluster_phrases(phrase_metadata, threshold):
    ids, embeds = zip(*phrase_metadata.id2embed.items())
    print('Building sparse graph...')
    coo_pw_sim_mat = build_coo_graph(ids, embeds, k=100)
    print('Running avg HAC...')
    Z, _, _, _, _ = sparse_avg_hac(coo_pw_sim_mat)

    print('Generating flat clustering...')
    P = fcluster(Z, threshold)
    cluster_id2phrase = defaultdict(set)
    for i, _id in enumerate(phrase_metadata.id2embed.keys()):
        assert i == _id
        cluster_id2phrase[int(P[i])].add(phrase_metadata.id2phrase[_id])
    cluster_id2phrase_id = {
        cluster_id : set(map(lambda x : phrase_metadata.phrase2id[x], phrases))
            for cluster_id, phrases in cluster_id2phrase.items()
    }

    clustering_model = {
        'Z' : Z,
        'cluster_id2phrase' : cluster_id2phrase,
        'cluster_id2phrase_id' : cluster_id2phrase_id
    }
    clustering_model = SimpleNamespace(**clustering_model)
    return clustering_model


def embed_synonyms(sent2vec_model, umls_lexicon_dir):
    cuid2names = {}
    cuid2type = {}
    name_id2cuid = {}
    name_id2name = {}
    name_id2embed = {}

    next_name_id = 0
    for fname in glob.glob(os.path.join(umls_lexicon_dir, '*.tsv')):
        with open(fname, 'r') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            for row in tsv_reader:
                if row[0][0] == '#':
                    continue
                cuid = row[1].replace('UMLS:', '')
                cuid2names[cuid] = [row[0]] + row[-1].split('|') # the first one is the primary name
                cuid2type[cuid] = '{} ({})'.format('None', 'None') # might care about this more later
                for name in cuid2names[cuid]:
                    _name = ' '.join(word_tokenize(name))
                    _emb = sent2vec_model.embed_sentence(_name).reshape(-1,)
                    # only include embedable phrases (0.0 is oov)
                    if np.linalg.norm(_emb) != 0.0:
                        name_id2cuid[next_name_id] = cuid
                        name_id2name[next_name_id] = name
                        name_id2embed[next_name_id] = _emb
                        next_name_id += 1

    concept_metadata = {
        'cuid2names' : cuid2names,
        'cuid2type' : cuid2type,
        'name_id2cuid' : name_id2cuid,
        'name_id2name' : name_id2name,
        'name_id2embed' : name_id2embed,
    }
    concept_metadata = SimpleNamespace(**concept_metadata)
    return concept_metadata


def link_clusters(concept_metadata,
                  phrase_metadata,
                  clustering_model,
                  output_dir):

    # build knn index
    print('Finding closest synonyms to phrases...')
    PHRASE_SYNONYM_KNN_FILENAME = os.path.join(output_dir, 'phrase_synonym_knn.pkl')
    if not os.path.exists(PHRASE_SYNONYM_KNN_FILENAME):
        name_ids, name_embeds = zip(*concept_metadata.name_id2embed.items())
        X = np.vstack(name_embeds) # these are the concept synonym embeddings
        X /= np.linalg.norm(X, axis=1)[:,np.newaxis]
        Q = np.vstack(list(phrase_metadata.id2embed.values())) # these are the key phrase embeddings
        Q /= np.linalg.norm(Q, axis=1)[:,np.newaxis]
        k = 64
        d = X.shape[1]
        n_cells = 10000
        n_probe = 50
        quantizer = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexIVFFlat(
                quantizer, d, n_cells, faiss.METRIC_INNER_PRODUCT
        )
        index.train(X)
        index.nprobe = n_probe
        index.add(X)
        D, I = index.search(Q, k)

        v_name_id2cuid = np.vectorize(lambda x : concept_metadata.name_id2cuid[x])
        v_name_id2synonym = np.vectorize(lambda x : concept_metadata.name_id2name[x])
        knn_cuids = v_name_id2cuid(I)
        knn_synonyms = I

        phrase_synonym_knn_data = {
            'D' : D,
            'knn_cuids' : knn_cuids,
            'knn_synonyms' : knn_synonyms
        }
        phrase_synonym_knn_data = SimpleNamespace(**phrase_synonym_knn_data)
        with open(PHRASE_SYNONYM_KNN_FILENAME, 'wb') as f_out:
            pickle.dump(phrase_synonym_knn_data, f_out)
    else:
        with open(PHRASE_SYNONYM_KNN_FILENAME, 'rb') as f_in:
            phrase_synonym_knn_data = pickle.load(f_in)

    # for each cluster get top candidate concepts
    print('Determining cluster candidates...')
    cand_limit = 10
    phrase_id2candidates = {i : list(zip(phrase_synonym_knn_data.knn_cuids[i],
                                         phrase_synonym_knn_data.knn_synonyms[i],
                                         phrase_synonym_knn_data.D[i]))
            for i in range(phrase_synonym_knn_data.D.shape[0])}
    cluster_id2cand_w_scores = {}
    for cluster_id, phrase_ids in clustering_model.cluster_id2phrase_id.items():
        cluster_candidates = [x for cands in map(lambda phrase_id : phrase_id2candidates[phrase_id], phrase_ids) for x in cands]
        candidates2scores = defaultdict(list)
        for cuid, synonym_id, synonym_score in cluster_candidates:
            candidates2scores[cuid].append((concept_metadata.name_id2name[synonym_id], synonym_score))
        cand_w_scores = [(cand, max(synonym_score, key=lambda x : x[1])) for cand, synonym_score in candidates2scores.items()]
        cand_w_scores = sorted(cand_w_scores, key=lambda x : x[1][1], reverse=True)[:cand_limit]
        cluster_id2cand_w_scores[cluster_id] = cand_w_scores

    linked_cluster_metadata = {
        'cuid2names': concept_metadata.cuid2names,
        'cluster_id2phrase': clustering_model.cluster_id2phrase,
        'cluster_id2cand_w_scores': cluster_id2cand_w_scores
    }
    linked_cluster_metadata = SimpleNamespace(**linked_cluster_metadata)
    return linked_cluster_metadata


def write_top_concepts(args, phrase_metadata, clustering_model, output_filename):
    # write to output file
    print('Writing results to: {}...'.format(output_filename))
    with open(output_filename, 'w') as f:
        f.write(''.join(['=']*80) + '\n')
        _clusters = list(clustering_model.cluster_id2phrase.items())
        f_cluster2max_df = lambda y : max(list(map(lambda z : len(phrase_metadata.phrase2docs[z]), clustering_model.cluster_id2phrase[y])))
        _clusters = list(filter(lambda x : f_cluster2max_df(x[0]) < args.max_doc_freq, _clusters)) # BEST: 300
        _clusters.sort(key=lambda x : f_cluster2max_df(x[0]), reverse=True)
        for cluster_id, phrases in _clusters[:100]:
            phrase_counts = list(map(lambda x : len(phrase_metadata.phrase2docs[x]), phrases))
            f.write('{}\n'.format(' ; '.join(map(lambda x : '{} ({})'.format(x[0], x[1]), sorted(list(zip(phrases, phrase_counts)), key=lambda x : x[1], reverse=True)))))
            f.write(''.join(['=']*80) + '\n')
    print('Done.')


def write_discovered_concepts(args, phrase_metadata, concept_metadata, linked_cluster_metadata, output_filename):
    # write to output file
    print('Writing results to: {}...'.format(output_filename))
    with open(output_filename, 'w') as f:
        f.write(''.join(['=']*80) + '\n')
        _clusters = list(linked_cluster_metadata.cluster_id2phrase.items())

        # functions of clusters
        f_cluster2top_score = lambda y : max(list(tuple(zip(*list(tuple(zip(*linked_cluster_metadata.cluster_id2cand_w_scores[y]))[1])))[1]))
        f_cluster2max_df = lambda y : max(list(map(lambda z : len(phrase_metadata.phrase2docs[z]), linked_cluster_metadata.cluster_id2phrase[y])))
        f_cluster2sum_df = lambda y : sum(list(map(lambda z : len(phrase_metadata.phrase2docs[z]), linked_cluster_metadata.cluster_id2phrase[y])))

        # this sorts the clusters in ascending order of the score of the
        # maximum scoring entity from UMLS 2017AA divided by the
        # max of document frequencies of the phrases in the cluster
        _clusters = list(filter(lambda x : f_cluster2top_score(x[0]) < args.max_linking_score, _clusters)) # BEST: 0.4
        _clusters = list(filter(lambda x : f_cluster2max_df(x[0]) < args.max_doc_freq, _clusters)) # BEST: 300
        _clusters.sort(key=lambda x : f_cluster2top_score(x[0]) / f_cluster2max_df(x[0]))

        for cluster_id, phrases in _clusters[:100]:
            _cand_w_scores = linked_cluster_metadata.cluster_id2cand_w_scores[cluster_id]
            phrase_counts = list(map(lambda x : len(phrase_metadata.phrase2docs[x]), phrases))
            if max(phrase_counts) == 1:
                continue
            f.write('Cluster phrases:\n')
            f.write('----------------\n')
            f.write('{}\n\n'.format(' ; '.join(map(lambda x : '{} ({})'.format(x[0], x[1]), sorted(list(zip(phrases, phrase_counts)), key=lambda x : x[1], reverse=True)))))
            f.write('Closest concepts:\n')
            f.write('-----------------\n')
            for cuid, (top_scoring_synonym, score) in _cand_w_scores:
                f.write('\tPrimary Name: {}\tType: {}\tMatching Synonym:{}\tScore:{}\n'.format(
                        linked_cluster_metadata.cuid2names[cuid][0],
                        concept_metadata.cuid2type[cuid],
                        top_scoring_synonym,
                        score)
                )
            f.write('\n')
            f.write(''.join(['=']*80) + '\n')
    print('Done.')


def compute_and_cache(cache_path, fn_ptr, args):
    if not os.path.exists(cache_path):
        output = fn_ptr(*args)
        with open(cache_path, 'wb') as f:
            pickle.dump(output, f)
    else:
        with open(cache_path, 'rb') as f:
            output = pickle.load(f)
    return output


def get_and_check_args():
    # specify and gather cmdline args
    parser = argparse.ArgumentParser(description='Biomedical concept discovery pipeline')
    parser.add_argument('--data_source', type=str, choices=['pubmed_download', 'cord19'], required=True)
    parser.add_argument('--cord_data_path', type=str)
    parser.add_argument('--pubmed_special_query', type=str, choices=['latest_papers', 'single_cell_biology', 'primary_ciliary_dyskinesia'])
    parser.add_argument('--biosentvec_path', type=str, required=True)
    parser.add_argument('--task', type=str, choices=['concept_discovery', 'top_concepts'], required=True)
    parser.add_argument('--umls_lexicon_path', type=str, default='/home/ds-share/data2/users/rangell/entity_discovery/UMLS_preprocessing/AnntdData/')
    parser.add_argument('--clustering_threshold', type=float, default=0.6)
    parser.add_argument('--max_linking_score', type=float, default=0.4)
    parser.add_argument('--max_doc_freq', type=int, default=300)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # check validity of arguments
    assert args.data_source != 'cord19' or args.cord_data_path != None, \
            "Must provide `--cord_data_path` or choose another data source"
    assert args.task != 'concept_discovery' or args.umls_lexicon_path != None, \
            "Must provide `--umls_lexicon_path` or choose another task"

    return args


if __name__ == '__main__':
    args = get_and_check_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    ID_TO_TEXT_FILENAME = os.path.join(args.output_dir, 'id_to_text.pkl')
    ID_TO_PHRASES_FILENAME = os.path.join(args.output_dir, 'id_to_phrases.pkl')
    PHRASE_METADATA_FILENAME = os.path.join(args.output_dir, 'phrase_reps_metadata.pkl')
    CLUSTERING_MODEL_FILENAME = os.path.join(args.output_dir, 'clustering_model.pkl')
    CONCEPT_METADATA_FILENAME = os.path.join(args.output_dir, 'concept_metadata.pkl')
    LINKED_CLUSTER_DATA_FILENAME = os.path.join(args.output_dir, 'linked_cluster_data.pkl')

    # import and preprocess the documents
    print('Preprocessing documents...')
    if args.data_source == 'cord19':
        id_to_text = compute_and_cache(
                ID_TO_TEXT_FILENAME,
                read_cord_data,
                [args.cord_data_path]
        )
    elif args.data_source == 'pubmed_download':
        pubmed_search = PubmedSearch()
        if args.pubmed_special_query == 'latest_papers':
            pubmed_search.search('', create_date_range=('2020/07/01', '2020/08/30'), n_results=20000)
        elif args.pubmed_special_query == 'single_cell_biology':
            pubmed_search.search('single cell biology', create_date_range=('2018/01/01', '2020/08/30'), n_results=20000)
        elif args.pubmed_special_query == 'primary_ciliary_dyskinesia':
            pubmed_search.search('Primary Ciliary Dyskinesia', create_date_range=('2018/01/01', '2020/08/30'), n_results=20000)


        pmids = pubmed_search.get_search_response_dict()['response']['docids']
        id_to_text = compute_and_cache(
                ID_TO_TEXT_FILENAME,
                download_and_preprocess_pubmed,
                [pmids]
        )

    # create the final output filename
    output_filename = '.'.join([args.task,
                                str(args.clustering_threshold).replace('.', '_'),
                                str(args.max_linking_score).replace('.', '_'),
                                str(args.max_doc_freq),
                                'txt'])
    output_filename = os.path.join(args.output_dir, output_filename)

    # extract mentions
    print('Extracting mentions...')
    id_to_noun_based_phrases, id_to_other_based_phrases = compute_and_cache(
            ID_TO_PHRASES_FILENAME,
            extract_mentions,
            [id_to_text]
    )

    # create sent2vec model
    print('Loading BioSent2Vec model...')
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(args.biosentvec_path)

    # embed phrases and get metadata
    print('Preprocessing phrases...')
    phrase_metadata = compute_and_cache(
            PHRASE_METADATA_FILENAME,
            get_phrase_reps_and_metadata,
            [id_to_noun_based_phrases, sent2vec_model]
    )

    # cluster phrases
    print('Clustering phrases...')
    clustering_model = compute_and_cache(
            CLUSTERING_MODEL_FILENAME,
            cluster_phrases,
            [phrase_metadata, args.clustering_threshold]
    )

    if args.task == 'concept_discovery':
        # embed all concept synonyms and organize metdata
        print('Embedding all synonyms...')
        concept_metadata = compute_and_cache(
                CONCEPT_METADATA_FILENAME,
                embed_synonyms,
                [sent2vec_model, args.umls_lexicon_path]
        )

        # link clusters to concepts
        print('Linking clusters...')
        linked_cluster_metadata = compute_and_cache(
                LINKED_CLUSTER_DATA_FILENAME,
                link_clusters,
                [concept_metadata,
                 phrase_metadata,
                 clustering_model,
                 args.output_dir]
        )

        write_discovered_concepts(args, phrase_metadata, concept_metadata, linked_cluster_metadata, output_filename)
    elif args.task == 'top_concepts':
        write_top_concepts(args, phrase_metadata, clustering_model, output_filename)

