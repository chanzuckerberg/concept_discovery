"""
Uses NCBI's Entrez "E-Utilities" web api to scrape Pubmed.
References:
    e-utils: https://www.ncbi.nlm.nih.gov/books/NBK25499/
    e-utils esearch detail: https://dataguide.nlm.nih.gov/eutilities/utilities.html#esearch
    searching pubmed: https://www.ncbi.nlm.nih.gov/books/NBK3827/
    "Best Match": https://www.nlm.nih.gov/pubs/techbull/jf17/jf17_pm_best_match_sort.html
                  https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2005343
"""

from datetime import datetime
import json
import os
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus
from urllib.request import urlopen
# noinspection PyPep8Naming
import xml.etree.ElementTree as ET

from .misc import pprint_xml
from .pubmed_dump import Article


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------


URL_PUBMED_SEARCH = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term="

URL_PUBMED_EFETCH = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id="


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

class PubmedSearch:
    """
    Wrapper around the e-utils 'esearch' api, to retrieve PMIDs for docs matching a Query.
    """

    def __init__(self):
        self.query = None
        self.pub_date_range = None
        self.create_date_range = None
        self.page_size = None
        self.sort_by = None

        self.execute_datetime = None
        self.search_url = None
        self.ret_code = None
        self.response = None
        self.reponse_root = None
        return

    def _save_search_request(self, search_text: str,
                             pub_date_range: Tuple[str, str],
                             create_date_range: Tuple[str, str],
                             n_results: int,
                             sort_by: str):
        self.query = search_text
        self.pub_date_range = pub_date_range
        self.create_date_range = create_date_range
        self.page_size = n_results
        self.sort_by = sort_by
        return

    def search(self, search_text: str,
               pub_date_range: Tuple[str, str] = None,
               create_date_range: Tuple[str, str] = None,
               n_results: int = 100,
               sort_by: str = "relevance") -> int:

        self._save_search_request(search_text, pub_date_range, create_date_range, n_results, sort_by)

        self.search_url = PubmedSearch.make_search_url(search_text,
                                                       pub_date_range=pub_date_range,
                                                       create_date_range=create_date_range,
                                                       n_results=n_results,
                                                       sort_by=sort_by)
        self.execute_datetime = datetime.now()

        with urlopen(self.search_url) as f:
            self.ret_code = f.getcode()
            self.response = f.read().decode('utf-8')

        self.reponse_root = ET.fromstring(self.response)
        return self.ret_code

    def get_warnings(self) -> List[str]:
        return get_xpath_text_list(self.reponse_root, "./WarningList", "OutputMessage")

    def get_translated_query(self):
        return get_xpath_text(self.reponse_root, "./QueryTranslation")

    def get_result_count(self) -> int:
        count_str = get_xpath_text(self.reponse_root, "./Count")
        if not count_str:
            return 0
        else:
            return int(count_str)

    def get_pmids(self) -> List[str]:
        return get_xpath_text_list(self.reponse_root, "./IdList", "Id")

    def get_search_response_dict(self) -> Dict[str, Dict[str, Any]]:
        pmids = self.get_pmids()
        sdict = dict(request=dict(source="pubmed",
                                  query=self.query,
                                  create_date_range=self.create_date_range,
                                  pub_date_range=self.pub_date_range,
                                  page_size=self.page_size,
                                  sort_by=self.sort_by),
                     response=dict(search_url=self.search_url,
                                   execute_datetime=str(self.execute_datetime),
                                   http_code=self.ret_code,
                                   translated_query=self.get_translated_query(),
                                   warnings=self.get_warnings() or None,
                                   result_size=self.get_result_count(),
                                   count=len(pmids),
                                   docids=pmids))
        return sdict

    @staticmethod
    def make_search_url(search_text: str,
                        pub_date_range: Tuple[str, str] = None,
                        create_date_range: Tuple[str, str] = None,
                        n_results: int = 100,
                        sort_by: str = "relevance"):
        """
        Converts `search_text` into a complete e-utils URL for retrieving the top `n_results`
        as sorted by relevance.

        :param search_text: The query text. May include AND. Examples:
            "Alzheimer's Disease"
            "Microfluidics AND single cell analysis"
        :param pub_date_range: Constraint on Paper Publication Date.
            (min, max) e.g. ("2019/12/01" , "2019/12/31")
            Date formats: YYYY/MM/DD, YYYY/MM, YYYY
        :param create_date_range: Constraint on Paper Creation Date (When paper enters the Pubmed DB).
            (min, max) e.g. ("2019/12/01" , "2019/12/31")
            Date formats: YYYY/MM/DD, YYYY/MM, YYYY
        :param n_results:
        :param sort_by: One of "relevance" (default) or "pub+date".
            Looks like "most+recent" is not supported.
        :return:
        """
        assert sort_by in [None, "relevance", "pub+date"]

        url = URL_PUBMED_SEARCH + quote_plus(search_text) + '&retmax=' + str(n_results)

        if create_date_range:
            assert len(create_date_range) == 2 and all(isinstance(d, str) for d in create_date_range)
            url += "&datetype=crdt&mindate=" + create_date_range[0] + "&maxdate=" + create_date_range[1]

        if pub_date_range:
            assert len(pub_date_range) == 2 and all(isinstance(d, str) for d in pub_date_range)
            # Doing the following is equivalent to using URL params 'datetype', 'mindate', 'maxdate' (see below)
            # search_text += f' AND ("{pub_date_range[0]}"[PDAT] : "{pub_date_range[1]}"[PDAT])'
            url += "&datetype=pdat&mindate=" + pub_date_range[0] + "&maxdate=" + pub_date_range[1]

        if sort_by:
            url += '&sort=' + sort_by

        return url
# /


class PubmedResults:
    def __init__(self, json_file: str, verbose: bool = True):
        """
        :param json_file: JSON dump of dict returned by `PubmedSearch.get_search_response_dict()`,
            e.g. as saved by `save_search_as_json()`.
        """
        self.file = json_file
        with open(os.path.expanduser(json_file)) as f:
            self.data_dict = json.load(f)

        if verbose:
            print("Nbr docs in", json_file, "=", len(self.get_pmids()))

        return

    def get_pmids(self):
        return self.data_dict["response"].get("docids")
# /


class PubmedDocFetch:
    """
    Interface to the e-utils 'efetch' api, to retrieve doc Title, Abstract
    """

    def __init__(self):
        self.requested_ids = None
        self.articles : List[Article] = []
        return

    def fetch_documents(self, pmids: List[str], verbose: bool = True):
        self.requested_ids = pmids
        self.articles = []

        # Batch request
        batch_sz = 200
        for s in range(0, len(pmids), batch_sz):
            if verbose:
                print("Retrieving ids", s, "-", min(s + batch_sz, len(pmids)) - 1, "...", flush=True)

            batch_pmids = pmids[s: s + batch_sz]
            root = self.fetch_docs_xml(batch_pmids)

            # We are ignoring any <PubmedBookArticle> entries
            for pxml in root.findall("./PubmedArticle"):
                article = Article.from_xml_root(pxml)
                self.articles.append(article)

        if verbose:
            print("Documents retrieved =", len(self.articles), "out of", len(pmids), flush=True)

        return self.articles

    @staticmethod
    def fetch_docs_xml(batch_pmids: List[str], verbose: bool = False):
        search_url = URL_PUBMED_EFETCH + ",".join(pmid.strip() for pmid in batch_pmids)
        if verbose:
            print("URL:", search_url)

        with urlopen(search_url) as f:
            ret_code = f.getcode()
            response = f.read().decode('utf-8')
            assert ret_code == 200, f"HTTP Return Code = {ret_code}.\n{response}"

        root = ET.fromstring(response)
        return root

# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def get_xpath_text(root, xpath):
    elem = root.find(xpath)
    if elem is None:
        return None
    else:
        return elem.text


def get_xpath_text_list(root, xpath, child_xpath):
    """
    Returns text values of child_xpath in order they occur
    """
    elem = root.find(xpath)
    if elem is None:
        return []
    else:
        return [child.text for child in elem.findall(child_xpath)]


def pp_if_notempty(msg, val):
    if val is not None and val != []:
        if isinstance(val, list):
            val = "; ".join(val)
        print(msg, val)
    return


def save_articles_with_mesh(pmids_file: str, output_file: str, max_count: int = 100):
    """
    Retrieve article data from PubMed for PMID's in `pmids_file`,
    and for each article that has MeSH headings, save it as XML to `output_file`.

    :param pmids_file: Path to file containing PMIDs, one per line
    :param output_file: Path to file
    :param max_count: Max nbr articles to be saved
    """
    with open(os.path.expanduser(pmids_file)) as f:
        pmids = [line.strip() for line in f.readlines()]

    pmfetch = PubmedDocFetch()
    articles = pmfetch.fetch_documents(pmids)

    if not articles:
        print("*** No articles returned!")
        return

    root = ET.Element("PubmedArticleSet")
    n = 0
    for article_ in articles:
        if article_.is_english and article_.abstract and article_.main_headings:
            axml = article_.to_xml(pubmed_format=True)
            root.append(axml)
            n += 1
            if n >= max_count:
                break

    with open(os.path.expanduser(output_file), "w") as outf:
        pprint_xml(root, file=outf)

    print()
    print(n, "articles saved to:", output_file)
    print()
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# Invoke as: python -m cr.pubmed.eutils  ...

if __name__ == '__main__':

    import argparse

    _argparser = argparse.ArgumentParser(
        description='Execute search on Pubmed.',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... save_articles_with_mesh  PMIDS_FILE  OUT_FILE
    _sub_cmd_parser = _subparsers.add_parser('save_articles_with_mesh',
                                             help="Retrieve and save articles with MeSH.")

    _sub_cmd_parser.add_argument('-n', '--max_nbr', type=int, default=100,
                                 help="Max nbr of articles to save.")
    _sub_cmd_parser.add_argument('pmids_file', type=str, help="File containing PMIDS, one per line")
    _sub_cmd_parser.add_argument('xml_file', type=str, help="File where article XML will be written")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print()

    if _args.subcmd == 'save_articles_with_mesh':

        save_articles_with_mesh(_args.pmids_file, _args.xml_file, max_count=_args.max_nbr)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
