"""
Misc utilities
"""

import pickle
import sys


# ======================================================================================================
#   Classes
# ======================================================================================================


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

        self._fpath = fpath

        with open(fpath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        if verbose:
            print(flush=True)
        return
# /


# ======================================================================================================
#   Functions
# ======================================================================================================


def print_cmd():
    """
    Print the command used from Shell to run.
    Example usage:
    >>> if __name__ == '__main__':
    >>>
    >>>     print_cmd()
    >>>     ...
    """
    import os
    import re

    module = os.path.relpath(sys.argv[0], ".")
    module = module.replace("/", ".")
    module = re.sub(r"\.py$", "", module)
    print("$>", "python -m", module, *sys.argv[1:])
    print()
    return


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
