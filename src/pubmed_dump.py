"""
Parse PubMed Dump

Ref:
https://www.nlm.nih.gov/databases/download/pubmed_medline.html
https://www.nlm.nih.gov/bsd/licensee/elements_alphabetical.html
https://www.nlm.nih.gov/bsd/licensee/elements_descriptions.html#medlinecitation

"""

from collections import defaultdict
from concurrent import futures
import glob
import gzip
import multiprocessing
import os
from pathlib import Path
import re
from threading import Thread
from typing import Dict, Generator, List, Optional, Sequence, Set, Union
# noinspection PyPep8Naming
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from .misc import PersistentObject


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

BASE_DIR = os.path.expanduser('~/Home/Projects/ConceptRecogn')

AB3P_DIR = os.path.join(BASE_DIR, 'Tools', 'Ab3P')
AB3P_CMD = './identify_abbr'

SPACES_PATT = re.compile(r'\s+')

SENTINEL = '_SENTINEL_'


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class PubmedDocument:
    def __init__(self, pmid: str, title: str = None, abstract: str = None, is_english: bool = True):
        self.pmid = pmid
        self.title = title
        self.abstract = abstract
        self.is_english = is_english
        return

    def get_text(self):
        txt = "\n".join([s for s in (self.title, self.abstract) if s])
        if not txt:
            txt = None
        return txt

    def __str__(self):
        return "pmid = {:s}\ntitle = {:s}\nabstract = {:s}".format(self.pmid, self.title, self.abstract)

    @classmethod
    def from_xml(cls, pubmed_article: ET.Element):
        assert pubmed_article.tag == "PubmedArticle"

        pmid = pubmed_article.findtext("./MedlineCitation/PMID")

        is_english = True

        title = extract_subelem_text(pubmed_article.find("./MedlineCitation/Article/ArticleTitle"))
        if not title or title == "Not Available":
            title = extract_subelem_text(pubmed_article.find("./MedlineCitation/Article/ARTICLETITLE"))
        if title:
            title = title.strip()
            if title.startswith("[") and title.endswith("]"):
                title = title.strip("[]")
                is_english = False
                if title.endswith("(author's transl)"):
                    title = title[:-len("(author's transl)")].strip()
        if title == "In Process Citation":
            title = None

        abstr = extract_subelem_text(pubmed_article.find("./MedlineCitation/Article/Abstract"))

        return cls(pmid, title, abstr, is_english)
# /


class LazyPubmedDocument:
    def __init__(self, pmid: str, article_xml: ET.Element, source: str = None):
        assert pmid is not None

        self.pmid = pmid.strip()
        self.article_xml = article_xml
        self.source = source

        self._title = None
        self._abstract = None
        self._is_english = None

        self._title_parsed = False
        return

    @property
    def title(self):
        self._parse_title_abstract()
        return self._title

    @property
    def abstract(self):
        self._parse_title_abstract()
        return self._abstract

    @property
    def is_english(self):
        self._parse_title_abstract()
        return self._is_english

    def get_text(self):
        txt = "\n".join([s for s in (self.title, self.abstract) if s])
        if not txt:
            txt = None
        return txt

    def get_mesh_headings_xml(self) -> List[ET.Element]:
        return self.article_xml.findall("./MedlineCitation/MeshHeadingList")

    def get_supplemental_mesh_xml(self) -> List[ET.Element]:
        """
        This info includes Supplemental Records on: Protocols, Diseases, Organisms
        """
        return self.article_xml.findall("./MedlineCitation/SupplMeshList")

    def get_chemicals_xml(self) -> List[ET.Element]:
        return self.article_xml.findall("./MedlineCitation/ChemicalList")

    def get_keywords_xml(self) -> List[ET.Element]:
        return self.article_xml.findall("./MedlineCitation/KeywordList")

    def _parse_title_abstract(self):
        if self._title_parsed:
            return

        is_english = True

        title = extract_subelem_text(self.article_xml.find("./MedlineCitation/Article/ArticleTitle"))
        if not title or title == "Not Available":
            title = extract_subelem_text(self.article_xml.find("./MedlineCitation/Article/ARTICLETITLE"))
        if title:
            title = title.strip()
            if title.startswith("[") and title.endswith("]"):
                title = title.strip("[]")
                is_english = False
                if title.endswith("(author's transl)"):
                    title = title[:-len("(author's transl)")].strip()
        if title == "In Process Citation":
            title = ""

        self._title = title
        self._is_english = is_english

        self._abstract = extract_subelem_text(self.article_xml.find("./MedlineCitation/Article/Abstract"))

        self._title_parsed = True
        return

    def to_xml(self) -> ET.Element:
        """
        Output format as parsed by `Article`
        """
        doc = ET.Element("Article", pmid=self.pmid)
        if self.source:
            doc.set("source", self.source)

        ET.SubElement(doc, "Title").text = self.title
        ET.SubElement(doc, "Abstract").text = self.abstract

        for children in [self.get_mesh_headings_xml(),
                         self.get_supplemental_mesh_xml(),
                         self.get_chemicals_xml(),
                         self.get_keywords_xml()]:
            if children:
                doc.extend(children)

        return doc

    def __str__(self):
        return "pmid = {:s}\ntitle = {:s}\nabstract = {:s}".format(self.pmid, self.title, self.abstract)

    @classmethod
    def from_pubmed_xml(cls, pubmed_article: ET, source: str = None):
        assert pubmed_article.tag == "PubmedArticle"

        pmid = pubmed_article.findtext("./MedlineCitation/PMID")

        return cls(pmid, pubmed_article, source=source)
# /


class PubmedDumpIndex(PersistentObject):

    def __init__(self):
        super().__init__()

        # Dir where all the dump files exist.
        #   dump_file_path = {base_dir}/{dump_file_name}
        self.base_dir = None

        # dump_file_name(str) -> List[pmid(str)]
        self.dumpfile_index = dict()

        # pmid(str) -> dump_file_name(str)
        self.docid_index = None

        return

    def get_dump_file(self, pmid: str) -> Optional[str]:
        """
        Returns absolute path (str) to file containing Doc with specified `pmid`,
        or None if not found.
        """
        if self.docid_index is None:
            self._build_docid_index()

        fname = self.docid_index.get(pmid)
        if fname is not None:
            return f"{self.base_dir}/{fname}"

        return

    def get_dump_files(self, pmids: Sequence[str]) -> Dict[str, str]:
        if self.docid_index is None:
            self._build_docid_index()

        pmid_file_dict = {pmid_ : self.get_dump_file(pmid_) for pmid_ in pmids}
        return pmid_file_dict

    def get_doc(self, pmid: str) -> Optional[LazyPubmedDocument]:
        dump_file = self.get_dump_file(pmid)
        if dump_file is None:
            return

        for doc in lazy_parse_dump_file(dump_file):
            if doc.pmid == pmid:
                return doc

        return

    def get_docs(self, pmids: Sequence[str]) -> Generator[LazyPubmedDocument, None, None]:
        """
        Generator yields LazyPubmedDocument for docs found for PMID in pmids.
        Order may be different. Only found docs are returned.
        """
        pmid_file_dict = self.get_dump_files(pmids)
        file_pmids = defaultdict(set)
        for pmid, fpath in pmid_file_dict.items():
            file_pmids[fpath].add(pmid)

        for dump_fpath, pmid_set in file_pmids.items():
            n_pmids = len(pmid_set)
            for doc in lazy_parse_dump_file(dump_fpath):
                if doc.pmid in pmid_set:
                    yield doc
                    n_pmids -= 1
                    if n_pmids == 0:
                        break

        return

    def _build_docid_index(self):
        self.docid_index = dict()
        for fpath, pmids in self.dumpfile_index.items():
            for pmid_ in pmids:
                self.docid_index[pmid_] = fpath

        return

    @staticmethod
    def build_save_index(pubmed_dump_files_or_patt: Union[str, List[str]],
                         output_file: str,
                         nprocs: int):
        """
        Run `nprocs` processes to build an index into `pubmed_dump_files`,
        and save it to `output_file`.

        :param pubmed_dump_files_or_patt: Glob pattern or list of paths containing Pubmed-Dump
            Assumes that all the files are in the same directory!

        :param output_file: Where each index will be saved, as a Pickle file (*.pkl)"
        :param nprocs:
        """
        print("PubmedDumpIndex.build_save_index:")
        print("   pubmed_dump_files_or_patt =", pubmed_dump_files_or_patt)
        print("   output_file =", output_file)
        print("   nprocs =", nprocs)

        output_file = os.path.expanduser(output_file)
        output_dir = os.path.dirname(output_file)
        output_dir = os.path.expanduser(output_dir)
        if not Path(output_dir).exists():
            print("Creating dir:", output_dir)
            Path(output_dir).mkdir()

        print('Starting {} processes ...'.format(nprocs), flush=True)

        m = multiprocessing.Manager()
        res_queue = m.Queue()

        # Using a process pool to start the sub-processes. Allows gathering return values.
        # With this method, Queue instance must be inherited by the sub-processes (e.g. as a global);
        # passing queue as an arg results in RuntimeError.
        with futures.ProcessPoolExecutor(max_workers=nprocs) as executor:
            results = executor.map(PubmedDumpIndex.build_index_procr,
                                   [pubmed_dump_files_or_patt] * nprocs,
                                   [res_queue] * nprocs,
                                   range(nprocs), [nprocs] * nprocs)

        pmindex = PubmedDumpIndex()

        # Put Queue consumer in a Thread
        t = Thread(target=pmindex._gather_file_docids, args=(nprocs, res_queue), daemon=False)
        t.start()
        # Join the consumer Thread until it is done
        t.join()

        # Get return values ... possible if processes started using ProcessPoolExecutor
        tot_docs_found = 0
        for (proc_nbr, docs_found) in results:
            print('... Sub-process {:d} found {:,d} docs'.format(proc_nbr, docs_found), flush=True)
            tot_docs_found += docs_found

        print('Total nbr docs written = {:,d}'.format(tot_docs_found))

        pmindex.save(output_file)
        return

    @staticmethod
    def build_index_procr(pubmed_dump_files_or_patt: Union[str, List[str]],
                          res_queue,
                          proc_nbr: int, nprocs: int):

        assert 0 <= proc_nbr < nprocs

        if isinstance(pubmed_dump_files_or_patt, List):
            pubmed_dump_files = [os.path.expanduser(f) for f in pubmed_dump_files_or_patt]
        else:
            pubmed_dump_files = glob.glob(os.path.expanduser(pubmed_dump_files_or_patt))

        # Ensure each process sees same ordering
        pubmed_dump_files = sorted(pubmed_dump_files)

        tot_docs_found = 0
        # Process every `nprocs`-th file starting at index `proc_nbr`
        for fi in range(proc_nbr, len(pubmed_dump_files), nprocs):
            file_pmids = []
            for doc in lazy_parse_dump_file(pubmed_dump_files[fi]):
                file_pmids.append(doc.pmid)

            res_queue.put(('add', proc_nbr, pubmed_dump_files[fi], file_pmids))
            tot_docs_found += len(file_pmids)

        res_queue.put((SENTINEL, proc_nbr))

        return proc_nbr, tot_docs_found

    def _gather_file_docids(self, nprocs: int, res_queue):
        n_dump_files_processed = 0
        while nprocs > 0:
            qry_data = res_queue.get()
            if qry_data[0] == SENTINEL:
                nprocs -= 1
                print('... Sub-process {} end recd.'.format(qry_data[1]), flush=True)
            else:
                n_dump_files_processed += 1
                _, proc_nbr, pubmed_dump_file, file_pmids = qry_data

                base_dir, file_name = os.path.split(pubmed_dump_file)
                if self.base_dir is None:
                    self.base_dir = base_dir
                self.dumpfile_index[file_name] = file_pmids

        print("Nbr dump files processed = {:,d}".format(n_dump_files_processed), flush=True)
        return
# /


# -----------------------------------------------------------------------------
#   Article - from PubMed or MeSH-Dump
# -----------------------------------------------------------------------------


class MeshHeading:
    def __init__(self, uid: str, name: str, is_major: bool):
        self.uid = uid
        self.name = name
        self.is_major = is_major
        return

    def __str__(self):
        return "{:s}: {:s}{:s}".format(self.uid, self.name, " *" if self.is_major else "")
# /


class SupplMeshName:
    def __init__(self, uid: str, name: str, suppl_type: str):
        self.uid = uid
        self.name = name
        self.suppl_type = suppl_type
        return

    def __str__(self):
        return "{:s}: {:s} [{:s}]".format(self.uid, self.name, self.suppl_type)
# /


class Qualifier(MeshHeading):
    def __init__(self, uid: str, name: str, is_major: bool):
        super().__init__(uid, name, is_major)
        return
# /


class MainHeading(MeshHeading):
    def __init__(self, uid: str, name: str, is_major: bool):
        super().__init__(uid, name, is_major)

        # Whether a Qualifier is marked as Major
        self.is_qualified_major: bool = False

        self.qualifiers: Set[Qualifier] = set()
        return

    def add_qualifier(self, qlfr: Qualifier):
        self.qualifiers.add(qlfr)
        if qlfr.is_major:
            self.is_qualified_major = True
        return

    def __str__(self):
        mystr = super().__str__()
        if self.qualifiers:
            mystr += " / " + ", ".join([str(qlfr) for qlfr in self.qualifiers])
        return mystr
# /


class Keyword:
    def __init__(self, name: str, is_major: bool):
        self.name = name
        self.is_major = is_major
        return

    def __str__(self):
        return "{:s}{:s}".format(self.name, " *" if self.is_major else "")
# /


class Article:
    def __init__(self, pmid: str, title: str, abstract: Optional[str]):
        self.pmid = pmid
        self.abstract = abstract

        self.is_english = True

        if title:
            title = title.strip()
            if title.startswith("[") and title.endswith("]"):
                title = title.strip("[]")
                self.is_english = False
                if title.endswith("(author's transl)"):
                    title = title[:-len("(author's transl)")].strip()

        if title == "In Process Citation":
            title = None

        self.title = title or ""

        self.main_headings: List[MainHeading] = []
        self.suppl_concept_records: List[SupplMeshName] = []
        self.keywords: List[Keyword] = []
        return

    def to_xml(self, pubmed_format: bool = False) -> ET.Element:
        """
        Get this article as an XML element.
        :param pubmed_format: Use XML format as returned by PubMed API ... PubmedArticle/MedlineCitation
        """
        def format_title():
            return escape(self.title if self.is_english else "[" + self.title + "]")

        def is_yn(flag: bool):
            return "Y" if flag else "N"

        if pubmed_format:
            root = ET.Element("PubmedArticle")
            medline = ET.SubElement(root, "MedlineCitation")
            ET.SubElement(medline, "PMID").text = self.pmid

            article = ET.SubElement(medline, "Article")
            ET.SubElement(article, "ArticleTitle").text = format_title()
            if self.abstract:
                ET.SubElement(article, "Abstract").text = escape(self.abstract)

            axml = medline

        else:
            root = ET.Element("Article", pmid=self.pmid)
            ET.SubElement(root, "Title").text = format_title()
            if self.abstract:
                ET.SubElement(root, "Abstract").text = escape(self.abstract)

            axml = root

        if self.main_headings:
            mhlist = ET.SubElement(axml, "MeshHeadingList")
            for mhdg in self.main_headings:
                mh_xml = ET.SubElement(mhlist, "MeshHeading")
                mh_descr = ET.SubElement(mh_xml, "DescriptorName", UI=mhdg.uid, MajorTopicYN=is_yn(mhdg.is_major))
                mh_descr.text = escape(mhdg.name)

                for qlfr in mhdg.qualifiers:
                    q_xml = ET.SubElement(mh_xml, "QualifierName", UI=mhdg.uid, MajorTopicYN=is_yn(mhdg.is_major))
                    q_xml.text = escape(qlfr.name)

        if self.suppl_concept_records:
            scr_list = ET.SubElement(axml, "SupplMeshList")
            for scr in self.suppl_concept_records:
                scr_xml = ET.SubElement(scr_list, "SupplMeshName", UI=scr.uid, Type=escape(scr.suppl_type))
                scr_xml.text = escape(scr.name)

        if self.keywords:
            kwd_list = ET.SubElement(axml, "KeywordList")
            for kwd in self.keywords:
                kwd_xml = ET.SubElement(kwd_list, "Keyword", MajorTopicYN=is_yn(kwd.is_major))
                kwd_xml.text = escape(kwd.name)

        return root

    def get_major_headings(self):
        return [hdg for hdg in self.main_headings if hdg.is_major or hdg.is_qualified_major]

    @staticmethod
    def from_xml_file(article_xml_file: str):
        tree = ET.parse(article_xml_file)
        return Article.from_xml_root(tree.getroot())

    # noinspection PyTypeChecker
    @staticmethod
    def from_xml_root(root: ET.Element):
        if root.tag == "Article":
            pmid = root.get('pmid')
            title = extract_subelem_text(root.find("./Title"))
            abstr = extract_subelem_text(root.find("./Abstract"))

        elif root.tag == "PubmedArticle":

            # All the tags of interest are under './MedlineCitation'
            root = root.find("./MedlineCitation")

            pmid = root.findtext("./PMID")

            title = extract_subelem_text(root.find("./Article/ArticleTitle"))
            if not title or title == "Not Available":
                title = extract_subelem_text(root.find("./Article/ARTICLETITLE"))

            abstr = extract_subelem_text(root.find("./Article/Abstract"))

        else:
            raise NotImplementedError(f"Cannot parse root.tag = {root.tag}. Should be one of: Article, PubmedArticle")

        article = Article(pmid, title, abstr)

        for mh_elem in root.findall("./MeshHeadingList/MeshHeading"):
            d_elem = mh_elem.find("./DescriptorName")
            main_hdg = MainHeading(d_elem.get("UI"), d_elem.text, d_elem.get("MajorTopicYN", "N") == "Y")
            article.main_headings.append(main_hdg)

            for q_elem in mh_elem.findall("./QualifierName"):
                main_hdg.add_qualifier(Qualifier(q_elem.get("UI"), q_elem.text, q_elem.get("MajorTopicYN", "N") == "Y"))

        for sm_elem in root.findall("./SupplMeshList/SupplMeshName"):
            scr = SupplMeshName(sm_elem.get("UI"), sm_elem.text, sm_elem.get("Type"))
            article.suppl_concept_records.append(scr)

        for kw_elem in root.findall("./KeywordList/Keyword"):
            kwd = Keyword(kw_elem.text, kw_elem.get("MajorTopicYN", "N") == "Y")
            article.keywords.append(kwd)

        return article
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def extract_subelem_text(xelem):
    """
    Extracts and combines text from sub-elements of `xelem`.

    :param xml.etree.ElementTree.Element xelem: xml.etree.ElementTree.Element.
    :return: str

    Special Cases
    -------------

        <title>GeneReviews<sup>®</sup></title>  =>  'GeneReviews ®'
        R<sub>0</sub>  => R0
        <i>text</i>  => text
        <b>text</b>  => text
        <u>text</u>  => text

    will be extracted as 'GeneReviews ®'.
    This is not strictly correct, but when tokenizing, will generate separate token for 'GeneReviews',
     which is desirable.
    """
    txt = None
    if xelem is not None:
        txt = ''
        for subelem in xelem.iter():
            if subelem.tag in ('abstract', 'title', 'p', 'sup', 'list-item'):
                if txt and not txt.endswith(' '):
                    txt += ' '
            elif subelem.tag == 'AbstractText':
                if txt and not txt.endswith('\n'):
                    txt += '\n'
                label = subelem.get("Label")
                if label and label.upper() != "UNLABELLED":
                    txt += label + ":\n"
            elif subelem.tag == "CopyrightInformation":
                continue

            if subelem.text:
                txt += subelem.text
                if subelem is not xelem and subelem.tag == 'title' and not txt.endswith(('. ', ': ')):
                    txt += ': '
            if subelem.tail:
                # Remove "\n" from subelem.tail
                txt += re.sub(r"\s+", " ", subelem.tail)

    if not txt:
        txt = None

    return clean_text(txt)


def clean_text(txt):
    if txt is not None:
        # Collapse multiple non-newline whitespaces to single BLANK
        txt = re.sub(r'((?!\n)\s)+', ' ', txt.strip())
        # Remove SPACE around newline
        txt = re.sub(r' ?\n ?', '\n', txt)
        # Collapse multiple newlines
        txt = re.sub(r'\n+', '\n', txt)
        # Remove SPACE preceding [,:.], IF there is also space after the punct.
        txt = re.sub(r' ([,:.]) ', r'\1 ', txt)
    return txt


def parse_dump_file(pubmed_dump_file: str) -> List[PubmedDocument]:
    is_gzipped = False
    open_fn = open
    if pubmed_dump_file.endswith(".gz"):
        is_gzipped = True
        open_fn = gzip.open

    with open_fn(pubmed_dump_file) as f:
        ftxt = f.read()
        if is_gzipped:
            # noinspection PyUnresolvedReferences
            ftxt = ftxt.decode("UTF-8")

    root = ET.fromstring(ftxt)

    pubmed_docs = []

    # Ignore elements "PubmedBookArticle"
    for doc_root in root.iterfind("./PubmedArticle"):
        doc = PubmedDocument.from_xml(doc_root)
        pubmed_docs.append(doc)

    return pubmed_docs


def lazy_parse_dump_file(pubmed_dump_file: str):
    """
    Generator for LazyPubmedDocument
    :param pubmed_dump_file:
    """
    is_gzipped = False
    open_fn = open
    if pubmed_dump_file.endswith(".gz"):
        is_gzipped = True
        open_fn = gzip.open

    with open_fn(pubmed_dump_file) as f:
        ftxt = f.read()
        if is_gzipped:
            # noinspection PyUnresolvedReferences
            ftxt = ftxt.decode("UTF-8")

    root = ET.fromstring(ftxt)

    # Ignore elements "PubmedBookArticle"
    for doc_root in root.iterfind("./PubmedArticle"):
        doc = LazyPubmedDocument.from_pubmed_xml(doc_root, source=pubmed_dump_file)
        yield doc

    return


def extract_from_pubmed_dump(pubmed_dump_file: str,
                             output_dir: str,
                             pmids_file: str = None,
                             max_docs: int = 0,
                             verbose=False):
    """
    Extracts Doc from PubMed dump, and writes it to `output_dir`.

    :param pubmed_dump_file:
    :param output_dir:
    :param pmids_file:
    :param max_docs:
    :param verbose:
    :return:
    """
    pmids = None
    if pmids_file is not None:
        with open(os.path.expanduser(pmids_file)) as f:
            pmids = set([line.strip() for line in f])

    output_dir = os.path.expanduser(output_dir)
    if not Path(output_dir).exists():
        print("Creating dir:", output_dir)
        Path(output_dir).mkdir()

    if verbose:
        print("Extracting from pubmed dump:", pubmed_dump_file, flush=True)

    n_docs = 0
    for doc in lazy_parse_dump_file(pubmed_dump_file):
        if pmids and doc.pmid not in pmids:
            continue

        doc_file = f"{output_dir}/{doc.pmid}.xml"
        ET.ElementTree(doc.to_xml()).write(doc_file, encoding="unicode", xml_declaration=True)

        if verbose:
            print("  ", doc.pmid, flush=True)

        n_docs += 1
        if 0 < max_docs <= n_docs:
            break

    return n_docs


def extract_proc_one(pubmed_dump_files_or_patt: Union[str, List[str]],
                     output_dir: str,
                     pmids_file: str,
                     proc_nbr: int,
                     nprocs: int):
    """
    Called from `extract_from_pubmed_dump_mp`, does the tasks for one process (`proc_nbr`) out of `nprocs` processes.

    :param pubmed_dump_files_or_patt:
    :param output_dir:
    :param pmids_file:
    :param proc_nbr: in range [0, nprocs - 1]
    :param nprocs: >= 1
    :return: proc_nbr, Nbr docs written
    """

    assert 0 <= proc_nbr < nprocs

    if isinstance(pubmed_dump_files_or_patt, List):
        pubmed_dump_files = [os.path.expanduser(f) for f in pubmed_dump_files_or_patt]
    else:
        print(f"extract_proc_one[{proc_nbr}]:  pubmed_dump_files_or_patt =", pubmed_dump_files_or_patt,
              flush=True)
        pubmed_dump_files = glob.glob(os.path.expanduser(pubmed_dump_files_or_patt))

    print("extract_proc_one[{}]:  nbr dump files = {:,d}".format(proc_nbr, len(pubmed_dump_files)), flush=True)

    # Ensure each process sees same ordering
    pubmed_dump_files = sorted(pubmed_dump_files)

    tot_docs_found = 0
    # Process every `nprocs`-th file starting at index `proc_nbr`
    for fi in range(proc_nbr, len(pubmed_dump_files), nprocs):
        tot_docs_found += extract_from_pubmed_dump(pubmed_dump_files[fi], output_dir, pmids_file, verbose=False)

    return proc_nbr, tot_docs_found


def extract_from_pubmed_dump_mp(pubmed_dump_files_or_patt: Union[str, List[str]],
                                output_dir: str,
                                pmids_file: str,
                                nprocs: int):
    """
    Run `nprocs` processes to extract docs of specified PMID.

    :param pubmed_dump_files_or_patt: Glob pattern or list of paths containing Pubmed-Dump
    :param output_dir: Where each doc will be written as a file: "{output_dir}/{pmid}.xml"
    :param pmids_file: One PMID per line
    :param nprocs:
    """
    print("extract_from_pubmed_dump_mp:")
    print("   pubmed_dump_files_or_patt =", pubmed_dump_files_or_patt)
    print("   output_dir =", output_dir)
    print("   pmids_file =", pmids_file)

    output_dir = os.path.expanduser(output_dir)
    if not Path(output_dir).exists():
        print("Creating dir:", output_dir)
        Path(output_dir).mkdir()

    print('Starting {} processes ...'.format(nprocs), flush=True)

    # Using a process pool to start the sub-processes. Allows gathering return values.
    # With this method, Queue instance must be inherited by the sub-processes (e.g. as a global);
    # passing queue as an arg results in RuntimeError.
    with futures.ProcessPoolExecutor(max_workers=nprocs) as executor:
        results = executor.map(extract_proc_one,
                               [pubmed_dump_files_or_patt] * nprocs,
                               [output_dir] * nprocs,
                               [pmids_file] * nprocs,
                               range(nprocs), [nprocs] * nprocs)

    # Get return values ... possible if processes started using ProcessPoolExecutor
    tot_docs_found = 0
    for (proc_nbr, docs_found) in results:
        print('... Subprocess {:d} found {:,d} docs'.format(proc_nbr, docs_found))
        tot_docs_found += docs_found

    print('Total nbr docs written = {:,d}'.format(tot_docs_found))
    return


def build_index(pubmed_dump_files_or_patt: Union[str, List[str]],
                output_file: str,
                nprocs: int):
    # Import class here so that load from pickle does not report errors
    # noinspection PyUnresolvedReferences
    from cr.pubmed.pubmed_dump import PubmedDumpIndex

    PubmedDumpIndex.build_save_index(pubmed_dump_files_or_patt, output_file, nprocs)
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# Invoke as: python -m pubmed_dump CMD ...

if __name__ == '__main__':

    import argparse
    from datetime import datetime

    from .misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='PubMed Dump Parser.',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... extract [-n NBR_PROCS]  DUMP_PATH_PATTERN  PMIDS_FILE  OUTPUT_DIR
    _sub_cmd_parser = _subparsers.add_parser('extract', help="Extract articles for specific PMIDs.")

    _sub_cmd_parser.add_argument('-n', '--nbr_procs', type=int, default=4,
                                 help="Nbr of sub-processes.")
    _sub_cmd_parser.add_argument('dump_path_pattern', type=str,
                                 help="Pattern for path to PubMed Dump files")
    _sub_cmd_parser.add_argument('pmids_file', type=str,
                                 help="Path to file containing PMIDs")
    _sub_cmd_parser.add_argument('output_dir', type=str,
                                 help="Output dir")

    # ... build_index [-n NBR_PROCS]  DUMP_PATH_PATTERN  PMIDS_FILE  OUTPUT_DIR
    _sub_cmd_parser = _subparsers.add_parser('build_index',
                                             help="Build and save PubmedDumpIndex.",
                                             description=("e.g.:  " +
                                                          "python -m pubmed_dump build_index -n 10 " +
                                                          "'../../PubMed/Data/D20191215/*.xml.gz' " +
                                                          "../../PubMed/Data/D20191215/pubmed_dump_index.pkl"))

    _sub_cmd_parser.add_argument('-n', '--nbr_procs', type=int, default=4,
                                 help="Nbr of sub-processes.")
    _sub_cmd_parser.add_argument('dump_path_pattern', type=str,
                                 help="Pattern for path to PubMed Dump files")
    _sub_cmd_parser.add_argument('output_file', type=str,
                                 help="Path to where PubmedDumpIndex will be written as a Pickle file")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print()
    print_cmd()

    if _args.subcmd == 'extract':

        extract_from_pubmed_dump_mp(_args.dump_path_pattern, _args.output_dir, _args.pmids_file, _args.nbr_procs)

    elif _args.subcmd == 'build_index':

        build_index(_args.dump_path_pattern, _args.output_file, _args.nbr_procs)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
