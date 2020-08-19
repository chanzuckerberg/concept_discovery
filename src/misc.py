"""
Misc utilities
"""

import contextlib
from datetime import datetime
import inspect
from io import StringIO, TextIOBase
import itertools
import json
import os.path
from pathlib import Path
import pickle
import psutil
import pwd
import resource
import sys
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union, Optional
import xml.etree.ElementTree as ET

import numpy as np


# ======================================================================================================
#   Classes
# ======================================================================================================


class AutoIncr:
    """
    Utility class implements an auto-incrementing 0-based counter.
    Use in defaultdict(AutoIncr().next) to implement an auto-incrementing
    key => sequential-int map.
    """
    def __init__(self, start=0):
        # Count of number of calls to incr(step=1)
        self.count = start

    def next(self, step=1):
        """Return next index, starting with `start`."""
        curr = self.count
        self.count += step
        return curr
# /


class Capturing(list):
    """
    Context manager to capture the stdout output of a func call.

    Usage:

    >>> with Capturing() as output:
    >>>     print 'hello world'
    >>>
    >>> print('displays on screen')
    >>>
    >>> with Capturing(output) as output:  # note the constructor argument
    >>>     print('hello world2')
    >>>
    >>> print('done')
    >>> print('output:', output)

    Output:

        displays on screen
        done
        output: ['hello world', 'hello world2']

    From: http://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
# /


class PersistentObject:
    def __init__(self):
        self._fpath = None
        super().__init__()

    @classmethod
    def load_from(cls, fpath, obj_name='', verbose=True):
        if verbose:
            if not obj_name:
                obj_name = cls.__name__
            print('Loading {} from: {} ...'.format(obj_name, fpath), end='', flush=True)

        with open(fpath, 'rb') as f:
            obj = pickle.load(f, fix_imports=False, encoding="UTF-8")

        obj._fpath = fpath

        if verbose:
            print(" completed.", flush=True)
        return obj

    def save(self, fpath, verbose=True):
        if verbose:
            print('Saving {} to: {} ...'.format(self.__class__.__name__, fpath), end='', flush=True)

        # prev_rec_limit = sys.getrecursionlimit()
        # sys.setrecursionlimit(32000)

        self._fpath = fpath

        with open(fpath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        # sys.setrecursionlimit(prev_rec_limit)

        if verbose:
            print(flush=True)
        return
# /


# ======================================================================================================
#   Functions on Lists, Arrays
# ======================================================================================================

def arg_first(predicate, iterable):
    """
    Index of First element in iterable that satisfies predicate,
        else None.

    NOTE: arg_first(lambda c: n < c, counts_cumsum) is MUCH SLOWER than
          ... np.searchsorted(counts_cumsum, n, side='right')
    """
    for i, elem in enumerate(iterable):
        if predicate(elem):
            return i
    return


def first(predicate, iterable):
    """First element in iterable that satisfies predicate."""
    return firstnot(lambda x: not predicate(x), iterable)


def firstnot(predicate, iterable):
    """First element in iterable that does not satisfy predicate."""
    try:
        return next(itertools.dropwhile(predicate, iterable))
    except StopIteration:
        return None


def list_without_nones(iterable):
    """List of all non-None elements"""
    return [e for e in iterable if e is not None]


def start_stop_index(seq: Sequence, keyfunc: Callable) -> Dict[Any, Tuple[int, int]]:
    """
    Returns {key: (start_idx, stop_idx)} for each unique key value in `seq`,
    s.t. all elements in seq[start_idx : stop_idx] have the key-val = key.
    Assumes that seq is sorted on key.

    :param seq: Sequence of elements, sorted on a key. Note that `None` should not be a valid key-value.
    :param keyfunc: Function that takes single argument, an element of seq, and returns the key-value.
    :return: dict
    """
    key_idx: Dict[Any, Tuple[int, int]] = dict()

    prev_key = None
    i, start_idx = 0, 0
    for i, element in enumerate(seq):
        key = keyfunc(element)
        if prev_key != key:
            if prev_key is not None:
                key_idx[prev_key] = (start_idx, i)
            start_idx = i
            prev_key = key

    key_idx[prev_key] = (start_idx, i + 1)
    return key_idx


def flatten(lst):
    """Generator: Flatten arbitrary list of atomic or list elements."""
    for x in lst:
        if isinstance(x, (list, tuple)):
            for e in flatten(x):
                yield e
        else:
            yield x


def sync_reshuffle_np(vecs):
    """Shuffles each array, in-place, synchronously"""
    state = np.random.get_state()
    for vec in vecs:
        np.random.shuffle(vec)
        np.random.set_state(state)
    return


def unique_in_order(vec):
    """
    Return unique elements in vec, in the order in which they first appeared in vec.
    Vec does not need to be sorted or grouped.
    :param vec: Vector
    :return: vector (1D array)
    """
    vec = np.asarray(vec)
    assert vec.ndim == 1
    _, idx = np.unique(vec, return_index=True)
    return vec[np.sort(idx)]


def collapse_identical_rows(mx):
    """
    Returns a Matrix (2D array) containing just the unique rows of `mx`, in sorted order.

    NOTE: Assumes identical rows are contiguous!!!

    :param numpy.ndarray mx: 2D array of shape (nr, nc)
    :return: numpy.ndarray -- a 2D array of shape (N, nc), where N is the number of unique rows in `mx`.
    """
    assert len(mx.shape) == 2
    unique_rows = []
    last_row = None
    for mx_row in mx:
        if last_row is None or not np.array_equal(mx_row, last_row):
            unique_rows.append(mx_row)
            last_row = mx_row
    unique_rows = np.array(unique_rows)
    return unique_rows


def count_repeats(numvec):
    """
    Count how often each value in a is repeated consecutively.
    Differs from numpy.unique in not looking for unique values and their total counts. Looks for consecutive repeats.

    `vec` can be reconstructed by:
        np.repeat(values, repeat_counts)
    which is functionally equivalent to:
        np.hstack([[v] * n for v, n in zip(values, repeat_counts)])

    :param numvec: numpy vector (or vector-like) of numbers
    :return: values, repeat_counts
    """
    numvec = np.asarray(numvec)
    assert(numvec.ndim == 1)
    w = np.where(np.hstack([[1], np.diff(numvec)]))[0]
    values = numvec[w]
    counts = np.diff(np.append(w, numvec.shape[0]))
    return values, counts


def cosine_similarity(x, y, mask=None, default=0.0):
    """
    Array of Cosine's of vectors along the last dimension.
    If x, y are 2D, then cos_xy[i] = cosine_similarity(x[i], y[i]).
    A RuntimeWarning is raised if any of the vectors have length zero (i.e. a vector of 0's),
    and corresponding result is `nan`.

    :param np.ndarray x:
    :param np.ndarray y: Must have same shape as x.
    :param np.ndarray mask: If provided, a bool array of same shape as x.shape[:-1]. False means return default value.
    :param float default: Default value to be returned when mask at that element is False.

    :return: scalar if x, y are vectors, else np.ndarray of ndim = x.ndim - 1
    """
    assert x.shape == y.shape
    assert mask is None or mask.shape == x.shape[:-1]

    x_dot_y = np.multiply(x, y).sum(axis=-1)
    x_magn_sq = np.square(x).sum(axis=-1)
    y_magn_sq = np.square(y).sum(axis=-1)
    if mask is None:
        cos_xy = x_dot_y / np.sqrt(np.multiply(x_magn_sq, y_magn_sq))
    else:
        # noinspection PyTypeChecker
        cos_xy = np.where(mask, x_dot_y / np.sqrt(np.multiply(x_magn_sq, y_magn_sq)), default)

    return cos_xy


# ======================================================================================================
#   Functions
# ======================================================================================================

def fn_name(fn):
    """Return str name of a function or method."""
    s = str(fn)
    if s.startswith('<function'):
        return 'fn:' + fn.__name__
    else:
        return ':'.join(s.split()[1:3])


def pp_funcargs(fn):
    arg_names = inspect.getfullargspec(fn).args
    print(fn_name(fn), "... args:")

    frame = inspect.currentframe()
    try:
        for i, name in enumerate(arg_names, start=1):
            if i == 1 and name == "self":
                continue
            print("   ", name, "=", frame.f_back.f_locals.get(name))

        print(flush=True)

    finally:
        # This is needed to ensure any reference cycles are deterministically removed as early as possible
        # see doc: https://docs.python.org/3/library/inspect.html#the-interpreter-stack
        del frame
    return


def get_from_module(identifier, module_params, module_name,
                    instantiate=False, kwargs=None):
    """
    [Copied from Keras] Convert identifier to function from specified module.

    :param identifier: Typically str.
    :param module_params:
    :param module_name:
    :param instantiate:
    :param kwargs:
    :return:
    """
    if isinstance(identifier, str):
        res = module_params.get(identifier)
        if not res:
            raise Exception('Invalid ' + str(module_name) + ': ' +
                            str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    elif type(identifier) is dict:
        name = identifier.pop('name')
        res = module_params.get(name)
        if res:
            return res(**identifier)
        else:
            raise Exception('Invalid ' + str(module_name) + ': ' +
                            str(identifier))
    return identifier


def print_cmd():
    import os
    import re

    module = os.path.relpath(sys.argv[0], ".")
    module = module.replace("/", ".")
    module = re.sub(r"\.py$", "", module)
    print("$>", "python -m", module, *sys.argv[1:])
    print()
    return


def show_memory_usage(label: str = None):
    if label is None:
        label = ""
    else:
        label = f"[{label}]  "

    print()
    print("{:s}psutil Memory usage = {:,d} Bytes".format(label, psutil.Process().memory_info().rss))
    memory_units = "kB" if sys.platform == "linux" else "Bytes"
    print("{:s}Peak memory used: self = {:,d} {:s}"
          .format(label, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, memory_units))
    print("{:s}Peak memory used: children = {:,d} {:s}"
          .format(label, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, memory_units))
    print(flush=True)
    return


def is_integer(x, min_value=0):
    """Convenience function to test that `x` is an int >= min_value"""
    return isinstance(x, (int, np.int, np.int8, np.int16, np.int32, np.int64)) \
           and x >= min_value


def pprint_xml(xml_elem, space='  ', level=0, is_last_child=True, file=sys.stdout):
    """
    pretty-print XML to stdout for tree rooted at `xml_elem`.
    Assumes SPACE is meaningless between tags, and between tag and text.
    So element.text and element.tail have SPACE stripped from both ends,
    and element.tail has SPACE then added at right for pretty-formatting.

    :param xml.etree.ElementTree.Element xml_elem: The tree rooted at this element will be printed.
    :param str space: The blank chars to use as single indent space at start of each line
    :param int level: Start indent level. leave this as default.
    :param is_last_child: for internal use, leave this as default.
    :param file: where to print
    """
    assert isinstance(level, int) and level >= 0

    indent = space * level

    print(indent, '<', xml_elem.tag, sep='', end='', file=file)
    if xml_elem.attrib:
        print(' ', ' '.join('{}="{}"'.format(k, v) for k, v in xml_elem.items()), sep='', end='', file=file)
    print('>', end='', file=file)

    my_text = xml_elem.text.strip() if xml_elem.text else None
    if my_text:
        print(my_text, sep='', end='', file=file)

    last_i = len(xml_elem) - 1
    for i, elem_ in enumerate(xml_elem):
        print(file=file)
        pprint_xml(elem_, space=space, level=level + 1, is_last_child=i == last_i, file=file)

    if len(xml_elem) > 0:
        print(indent, end='', file=file)
    print('</', xml_elem.tag, '>', sep='', end='', file=file)
    if is_last_child:
        print(file=file)

    my_tail = xml_elem.tail.strip() if xml_elem.tail else None
    if my_tail:
        print(indent, my_tail, sep='', end='\n', file=file)
        # if is_last_child:
        #     indent = space * (level - 1)
        #     print(indent, end='')

    return


def indent_xml(xml_elem, level=0, space=' '):
    """
    Adds LF and SPACEs to xml content s.t. ET.tostring() produces prettied output.
    Warning: Modifies the XML content !!!
    """
    indent = "\n" + level * space
    if len(xml_elem):
        # if not xml_elem.text or not xml_elem.text.strip():
        if not xml_elem.text:
            xml_elem.text = indent + space
        # if not xml_elem.tail or not xml_elem.tail.strip():
        if not xml_elem.tail:
            xml_elem.tail = indent
        elem_ = None
        for elem_ in xml_elem:
            indent_xml(elem_, level + 1, space)
        # Reduce the indent after last child
        if not elem_.tail or not elem_.tail.strip():
            elem_.tail = indent
    else:
        # if level and (not xml_elem.tail or not xml_elem.tail.strip()):
        if level and not xml_elem.tail:
            xml_elem.tail = indent
    return xml_elem


def pp_dict(d, msg=None, indent=4):
    if msg:
        print(msg, '-' * len(msg), sep='\n')

    for k, v in sorted(d.items()):
        if isinstance(v, Mapping):
            print('{:{indent}s}{}: {{'.format('', k, indent=indent))
            pp_dict(v, None, indent + 4)
            print('{:{indent}s}}}'.format('', indent=indent))
        else:
            if isinstance(v, (float, np.floating)):
                v = '{:.4f}'.format(v)
            print('{:{indent}s}{}: {}'.format('', k, v, indent=indent))
    return


def check_dir(dir_path, create_parents=True, verbose=True, msg=None):
    """
    Check if `dir_path` exists as a directory, and if not, create it.
    :param str dir_path:
    :param bool create_parents: If True, Then create all ancestors as needed.
    :param bool verbose:
    :param str msg: Message when creating dir
    :return:
    """
    path = Path(dir_path)
    if not path.exists():

        if verbose:
            print(msg or 'Creating dir:', dir_path)
            print(flush=True)

        path.mkdir(parents=create_parents)

    elif not path.is_dir():
        raise NotADirectoryError("Path exists but is not a dir: " + dir_path)

    return


def path_relative_to_home(path: str):
    if path.startswith("~"):
        return path

    # use os.path.realpath ?
    path = os.path.abspath(path)

    home_dir = os.environ.get("HOME")
    if not home_dir:
        user = os.environ.get("USER")
        if not user:
            return path

        home_dir = pwd.getpwnam(user).pw_dir

    path = os.path.relpath(path, home_dir)
    if path == ".":
        return "~"
    else:
        return "~/" + path


@contextlib.contextmanager
def smart_open(file, mode='r'):
    """
    When input is a path to a file, or an already opened file,
    use this so that only non-sys.std* files are closed.

    Usage:
    ```python
    with smart_open(myfile, 'w') as outf:
        ...
    ```

    :param file: Can be a sys.std* or a str
    :param mode: mode with which to open. Ignored if file is sys.std*
    :return:
    """
    if file in [sys.stdin, sys.stdout, sys.stderr] or isinstance(file, TextIOBase):
        fh = file
    else:
        fh = open(file, mode=mode)

    try:
        yield fh
    finally:
        if fh not in [sys.stdin, sys.stdout, sys.stderr]:
            fh.close()


def smart_open_filename(file: Union[str, TextIOBase]) -> str:
    # noinspection PyUnresolvedReferences
    return file if isinstance(file, str) else file.name


def read_json_opts(json_file, verbose=True):

    if isinstance(json_file, str):
        json_file = os.path.expanduser(json_file)

    with smart_open(json_file) as f:
        if verbose:
            print('Loading options from:', f.name, flush=True)

        opts = json.load(f)

    return opts


def write_json_opts(opts: Dict, json_file,
                    encoding: str = "UTF-8", ensure_ascii: bool = False, indent: int = 2, sort_keys: bool = True,
                    verbose: bool = True):
    if verbose:
        print('Writing to', json_file, '...', flush=True)

    with open(os.path.expanduser(json_file), 'w', encoding=encoding) as outfile:
        json.dump(opts, outfile, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys)

    return


@contextlib.contextmanager
def timed_exec(name: str, prefix: str = "-*- Time to", pre_msg: str = None, time_as_suffix_to_pre_msg=False, file=None):
    """
    Use this to print execution times of blocks.
    IF `time_as_suffix_to_pre_msg` AND `pre_msg` THEN
        run-time is added to end of line of `pre_msg`, and `name` is ignored

    >>> with timed_exec("running test", "Testing ..."):
    >>>     ...
    """
    if pre_msg is not None:
        print(pre_msg, file=file, end="" if time_as_suffix_to_pre_msg else "\n", flush=True)
    else:
        time_as_suffix_to_pre_msg = False

    t0 = datetime.now()
    yield
    if time_as_suffix_to_pre_msg:
        print("", datetime.now() - t0, file=file, flush=True)
    else:
        print(prefix, name, "=", datetime.now() - t0, file=file, flush=True)
    return


def query_yes_no(question, default="yes"):
    """
    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


# ======================================================================================================
#   Functions: Terminal Highlighting
# ======================================================================================================

ANSI_TERMINAL_FOREGROUND_COLORS = {
    'black':   '30',
    'red':     '31',
    'green':   '32',
    'yellow':  '33',
    'blue':    '34',
    'magenta': '35',
    'cyan':    '36',
    'white':   '97'
}

ANSI_TERMINAL_FORMATS = {
    'bold':       '1',
    'underlined': '4',
    'blinking':   '5',
    'reversed':   '7'
}


def terminal_highlighted(text, font_color='red', font_format='bold'):
    """Add control chars for highlighted printing on color terminal"""
    # return '\033[01;31m' + s + '\033[00m'
    clr_code = ANSI_TERMINAL_FOREGROUND_COLORS[font_color]
    fmt_code = ANSI_TERMINAL_FORMATS[font_format]
    return '\033[{};{}m{}\033[00m'.format(fmt_code, clr_code, text)


def highlight_occurrences(text, s):
    """
    Return text with all occurrences of `s` highlighted for terminals.

    :param text: str
    :param s: str
    :return: highlighted_text (int), nbr_occurrences (str)
    """
    assert len(s) > 0, 's cannot be empty string'
    if not text:
        return text, 0
    low_txt = text.casefold()
    low_s = s.casefold()
    occurrences = []
    start = 0
    while True:
        i = low_txt.find(low_s, start)
        if i < 0:
            break
        occurrences.append(i)
        start = i + 1

    s_len = len(s)
    for i in occurrences[::-1]:
        text = text[:i] + terminal_highlighted(text[i : i + s_len], font_color='green') + text[i+s_len:]

    return text, len(occurrences)


def print_hilit_occurrences(text, s):
    t, n = highlight_occurrences(text, s)
    print(terminal_highlighted('Nbr occurrences = {}'.format(n), font_color='green' if n else 'red'))
    if n:
        print(t)
    return


def highlight_spans(text, spans: Sequence[Tuple[int, int]], font_color='blue', font_format='bold'):
    txt_segments = []
    prev_ce = 0
    for cs, ce in sorted(spans):
        if cs > prev_ce:
            txt_segments.append(text[prev_ce : cs])
        txt_segments.append(terminal_highlighted(text[cs : ce], font_color, font_format))
        prev_ce = ce

    if prev_ce < len(text):
        txt_segments.append(text[prev_ce:])

    return "".join(txt_segments)


def highlight_spans_multicolor(text, format_spans_dict: Dict[Tuple[str, str], Sequence[Tuple[int, int]]]):
    """

    :param text:
    :param format_spans_dict:  (font_color, font_format) -> [(ch_start, ch_end), ...]
    :return:
    """
    # reverse dict
    spans_format = {tuple(span_): font_format
                    for font_format, span_seq in format_spans_dict.items() for span_ in span_seq}
    txt_segments = []
    prev_ce = 0
    for cs, ce in sorted(spans_format.keys()):
        if cs < prev_ce:
            continue
        elif cs > prev_ce:
            txt_segments.append(text[prev_ce : cs])

        font_color, font_format = spans_format[(cs, ce)]
        txt_segments.append(terminal_highlighted(text[cs : ce], font_color, font_format))
        prev_ce = ce

    if prev_ce < len(text):
        txt_segments.append(text[prev_ce:])

    return "".join(txt_segments)
