import csv
from collections import defaultdict
import glob
import json
import os
import pickle
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
import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
model = AutoModel.from_pretrained("allenai/biomed_roberta_base")

from IPython import embed


def collect_data(data_dir):
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


def extract_mentions(cord_uid_to_text):
    #scispacy_model = spacy.load("en_core_sci_lg")
    scispacy_models = [
            spacy.load("en_ner_craft_md"),
            spacy.load("en_ner_jnlpba_md"),
            spacy.load("en_ner_bc5cdr_md"),
            spacy.load("en_ner_bionlp13cg_md")
    ]  

    cord_uid_to_phrases = defaultdict(list)
    for cord_uid, text in tqdm(cord_uid_to_text.items(),
                               desc='extracting mentions'):
        phrases = []
        for sub_text in text[0].values():
            if isinstance(sub_text, list):
                sub_text = ' '.join(sub_text)
            for model in scispacy_models:
                doc = model(sub_text)
                tokens = [tok.text for tok in doc]
                phrases.extend([
                    (' '.join(tokens[:s.start]),
                     s.as_doc().text.strip(),
                     ' '.join(tokens[s.end:]))
                        for s in doc.ents
                ])
        cord_uid_to_phrases[cord_uid] = phrases
    return cord_uid_to_phrases


def get_phrase_reps_and_metadata_roberta(cord_uid_to_phrases, tokenizer, model):
    model.eval()

    MAX_LEN = 128

    id2embed = {}
    phrase2id = defaultdict(set)
    id2phrase = {}
    doc2ids = defaultdict(set)
    phrase2docs = defaultdict(set)
    next_phrase_id = 0
    for doc_id, phrases in tqdm(cord_uid_to_phrases.items(),
                                desc='embedding phrases'):
        for pre_text, ph, post_text in phrases:
            #_ph = ' '.join(word_tokenize(ph))
            #_emb = sent2vec_model.embed_sentence(_ph).reshape(-1,)

            with torch.no_grad():
                pre_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(pre_text))]).type(torch.int64)
                ph_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ph))]).type(torch.int64)
                post_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(post_text))]).type(torch.int64)

                half_len = (MAX_LEN - ph_ids.numel()) // 2
                if pre_ids.numel() < half_len and post_ids.numel() >= half_len:
                    post_ids = post_ids[:,:(2*half_len) - pre_ids.numel()]
                elif pre_ids.numel() >= half_len and post_ids.numel() < half_len:
                    pre_ids = pre_ids[:,-((2*half_len) - post_ids.numel()):]
                elif pre_ids.numel() >= half_len and post_ids.numel() >= half_len:
                    pre_ids = pre_ids[:,-(half_len):]
                    post_ids = post_ids[:,:half_len]

                start_index = pre_ids.numel()
                end_index = start_index + ph_ids.numel()

                try:
                    input_ids = torch.cat((pre_ids, ph_ids, post_ids), 1)
                    assert input_ids.numel() <= MAX_LEN
                    outputs = model(input_ids)
                    _emb = torch.mean(outputs[0].squeeze(0)[start_index:end_index, :], 0).numpy()
                except:
                    embed()
                    exit()

            # only include embedable phrases (0.0 is oov)
            if np.linalg.norm(_emb) != 0.0:
                phrase2docs[ph].add(doc_id)
                _id = next_phrase_id
                next_phrase_id += 1
                phrase2id[ph].add(_id)
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


def get_phrase_reps_and_metadata(cord_uid_to_phrases, sent2vec_model):
    id2embed = {}
    phrase2id = {}
    id2phrase = {}
    doc2ids = defaultdict(set)
    phrase2docs = defaultdict(set)
    next_phrase_id = 0
    for doc_id, phrases in tqdm(cord_uid_to_phrases.items(),
                                desc='embedding phrases'):
        for ph in phrases:
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
    n_cells = 10000
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


def cluster_phrases(phrase_metadata):
    ids, embeds = zip(*phrase_metadata.id2embed.items())
    print('Building sparse graph...')
    coo_pw_sim_mat = build_coo_graph(ids, embeds, k=100)
    print('Running avg HAC...')
    Z, t, altitudes, ugraph, edge_weights = sparse_avg_hac(coo_pw_sim_mat)

    clustering_model = {
        'Z' : Z,
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
    #for fname in glob.glob(os.path.join(umls_lexicon_dir, '*.json')):
    #    with open(fname, 'r') as f:
    #        type_dict = json.load(f)
    #    for cuid, concept_dict in type_dict['Concepts'].items():
    #        cuid2names[cuid] = concept_dict['names'] # the first one is the primary name
    #        cuid2type[cuid] = '{} ({})'.format(type_dict['TypeName'], type_dict['TypeID'])
    #        for name in concept_dict['names']:
    #            _name = ' '.join(word_tokenize(name))
    #            _emb = sent2vec_model.embed_sentence(_name).reshape(-1,)
    #            # only include embedable phrases (0.0 is oov)
    #            if np.linalg.norm(_emb) != 0.0:
    #                name_id2cuid[next_name_id] = cuid
    #                name_id2name[next_name_id] = name
    #                name_id2embed[next_name_id] = _emb
    #                next_name_id += 1
    
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


def embed_synonyms_roberta(tokenizer, model, umls_lexicon_dir):
    model.eval()

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
                    #_name = ' '.join(word_tokenize(name))
                    #_emb = sent2vec_model.embed_sentence(_name).reshape(-1,)
                    with torch.no_grad():
                        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(name))])
                        outputs = model(input_ids)
                        _emb = torch.mean(outputs[0].squeeze(0), 0).numpy()

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
                  threshold):
    # generate flat clustering
    print('Generating flat clustering...')
    P = fcluster(clustering_model.Z, threshold)
    cluster_id2phrase = defaultdict(set)
    for i, _id in enumerate(phrase_metadata.id2embed.keys()):
        assert i == _id
        cluster_id2phrase[int(P[i])].add(phrase_metadata.id2phrase[_id])
    cluster_id2phrase_id = {
        cluster_id : set(map(lambda x : phrase_metadata.phrase2id[x], phrases))
            for cluster_id, phrases in cluster_id2phrase.items()
    }

    # build knn index
    print('Finding closest synonyms to phrases...')
    PHRASE_SYNONYM_KNN_PATH = './bin/phrase_synonym_knn.pkl'
    if not os.path.exists(PHRASE_SYNONYM_KNN_PATH):
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
        with open(PHRASE_SYNONYM_KNN_PATH, 'wb') as f_out:
            pickle.dump(phrase_synonym_knn_data, f_out)
    else:
        with open(PHRASE_SYNONYM_KNN_PATH, 'rb') as f_in:
            phrase_synonym_knn_data = pickle.load(f_in)

    # for each cluster get top candidate concepts
    print('Determining cluster candidates...')
    cand_limit = 10
    phrase_id2candidates = {i : list(zip(phrase_synonym_knn_data.knn_cuids[i],
                                         phrase_synonym_knn_data.knn_synonyms[i],
                                         phrase_synonym_knn_data.D[i])) 
            for i in range(phrase_synonym_knn_data.D.shape[0])}
    cluster_id2cand_w_scores = {}
    for cluster_id, phrase_ids in cluster_id2phrase_id.items():
        cluster_candidates = [x for cands in map(lambda phrase_id : phrase_id2candidates[phrase_id], phrase_ids) for x in cands]
        candidates2scores = defaultdict(list)
        for cuid, synonym_id, synonym_score in cluster_candidates:
            candidates2scores[cuid].append((concept_metadata.name_id2name[synonym_id], synonym_score))
        cand_w_scores = [(cand, max(synonym_score, key=lambda x : x[1])) for cand, synonym_score in candidates2scores.items()]
        cand_w_scores = sorted(cand_w_scores, key=lambda x : x[1][1], reverse=True)[:cand_limit]
        cluster_id2cand_w_scores[cluster_id] = cand_w_scores

    linked_cluster_metadata = {
        'cuid2names': concept_metadata.cuid2names,
        'cluster_id2phrase': cluster_id2phrase,
        'cluster_id2cand_w_scores': cluster_id2cand_w_scores
    }
    linked_cluster_metadata = SimpleNamespace(**linked_cluster_metadata)
    return linked_cluster_metadata


def compute_and_cache(cache_path, fn_ptr, args):
    if not os.path.exists(cache_path):
        output = fn_ptr(*args)
        with open(cache_path, 'wb') as f:
            pickle.dump(output, f)
    else:
        with open(cache_path, 'rb') as f:
            output = pickle.load(f)
    return output


if __name__ == '__main__':

    if not os.path.isdir('./bin'):
        os.makedirs('./bin')

    #UMLS_LEXICON_DIR = '/home/ds-share/data2/users/rangell/lerac/coref_entity_linking/data/mm_st21pv_long_entities/umls_lexicons'
    UMLS_LEXICON_DIR = '/home/ds-share/data2/users/rangell/entity_discovery/UMLS_preprocessing/AnntdData/'
    SENT2VEC_MODEL_PATH = './bin/BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
    CLUSTERING_THRESHOLD = 0.6

    CORD_UID_TO_TEXT_PATH = './bin/cord_uid_to_text.pkl' 
    CORD_UID_TO_PHRASES_PATH = './bin/cord_uid_to_phrases.pkl' 
    PHRASE_METADATA_PATH = './bin/clustering_metadata.pkl'
    CLUSTERING_MODEL_PATH = './bin/clustering_model.pkl'
    CONCEPT_METADATA_PATH = './bin/concept_metadata.pkl'
    LINKED_CLUSTER_DATA_PATH = './bin/linked_cluster_data.pkl'
    DISCOVERED_ENTITIES_PATH = './bin/discovered_entities.txt'

    # import and preprocess the documents
    print('Preprocessing documents...')
    cord_uid_to_text = compute_and_cache(
            CORD_UID_TO_TEXT_PATH,
            collect_data,
            ['/iesl/canvas/rangell/entity_discovery/cord-19/2020-07-31/']
    )

    # extract mentions
    print('Extracting mentions...')
    cord_uid_to_phrases = compute_and_cache(
            CORD_UID_TO_PHRASES_PATH,
            extract_mentions,
            [cord_uid_to_text]
    )

    # create sent2vec model
    #print('Loading BioSent2Vec model...')
    #sent2vec_model = sent2vec.Sent2vecModel()
    #sent2vec_model.load_model(SENT2VEC_MODEL_PATH)

    roberta_tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
    roberta_model = AutoModel.from_pretrained("allenai/biomed_roberta_base")

    # embed phrases and get metadata
    print('Preprocessing phrases...')
    #phrase_metadata = compute_and_cache(
    #        PHRASE_METADATA_PATH,
    #        get_phrase_reps_and_metadata,
    #        [cord_uid_to_phrases, sent2vec_model]
    #)
    phrase_metadata = compute_and_cache(
            PHRASE_METADATA_PATH,
            get_phrase_reps_and_metadata_roberta,
            [cord_uid_to_phrases, roberta_tokenizer, roberta_model]
    )

    # cluster phrases
    print('Clustering phrases...')
    clustering_model = compute_and_cache(
            CLUSTERING_MODEL_PATH,
            cluster_phrases,
            [phrase_metadata]
    )

    # embed all concept synonyms and organize metdata
    print('Embedding all synonyms...')
    #concept_metadata = compute_and_cache(
    #        CONCEPT_METADATA_PATH,
    #        embed_synonyms,
    #        [sent2vec_model, UMLS_LEXICON_DIR]
    #)
    concept_metadata = compute_and_cache(
            CONCEPT_METADATA_PATH,
            embed_synonyms_roberta,
            [roberta_tokenizer, roberta_model, UMLS_LEXICON_DIR]
    )

    # link clusters to concepts
    print('Linking clusters...')
    linked_cluster_metadata = compute_and_cache(
            LINKED_CLUSTER_DATA_PATH,
            link_clusters,
            [concept_metadata,
             phrase_metadata,
             clustering_model,
             CLUSTERING_THRESHOLD]
    )

    # write to output file
    print('Writing results to: {}...'.format(DISCOVERED_ENTITIES_PATH))
    with open(DISCOVERED_ENTITIES_PATH, 'w') as f:
        f.write(''.join(['=']*80) + '\n')
        _clusters = list(linked_cluster_metadata.cluster_id2phrase.items())

        # functions of clusters
        f_cluster2top_score = lambda y : max(list(tuple(zip(*list(tuple(zip(*linked_cluster_metadata.cluster_id2cand_w_scores[y]))[1])))[1]))
        f_cluster2max_df = lambda y : max(list(map(lambda z : len(phrase_metadata.phrase2docs[z]), linked_cluster_metadata.cluster_id2phrase[y])))
        f_cluster2sum_df = lambda y : sum(list(map(lambda z : len(phrase_metadata.phrase2docs[z]), linked_cluster_metadata.cluster_id2phrase[y])))

        # this sorts the clusters in ascending order of the score of the
        # maximum scoring entity from UMLS 2017AA divided by the
        # max of document frequencies of the phrases in the cluster
        _clusters = list(filter(lambda x : f_cluster2top_score(x[0]) < 0.4, _clusters)) # BEST: 0.4
        _clusters = list(filter(lambda x : f_cluster2max_df(x[0]) < 300, _clusters)) # BEST: 300
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