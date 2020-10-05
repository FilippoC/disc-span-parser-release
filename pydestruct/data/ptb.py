#from pyparsing import OneOrMore, nestedExpr
import pydestruct.head_percolation

class PTBTokenizer:
    def __init__(self, data):
        # with open(path, 'r') as file:
        #    self.data = file.read()
        self.data = data

        self._eof = False
        self._index = 0
        self.next()

    def next(self):
        self._token = ""

        try:
            # ignore white spaces
            while self.data[self._index] in [" ", "\t", "\n", "\r"]:
                self._index += 1

            if self.data[self._index] in ["(", ")"]:
                self._token = self.data[self._index]
                self._index += 1
            else:
                while self.data[self._index] not in [" ", "\t", "\n", "\r", "(", ")"]:
                    self._token += self.data[self._index]
                    self._index += 1
        except IndexError:
            if len(self._token) == 0:
                self._eof = True

    def eof(self):
        return self._eof

    def token(self):
        if self.eof():
            raise EOFError()
        else:
            return self._token


def read_ptb_list(tokenizer, fix_ptb=False):
    if tokenizer.token() != "(":
        raise SyntaxError("Expecting opening bracket, got: _%s_" % tokenizer.token())
    tokenizer.next()

    ret = list()
    while tokenizer.token() != ")":
        if tokenizer.token() == "(":
            ret.append(read_ptb_list(tokenizer, fix_ptb))
        else:
            ret.append(tokenizer.token())
            tokenizer.next()
    # consume closing parenthesis
    tokenizer.next()

    if len(ret) == 0:
        raise SyntaxError("Empty list!")
    if fix_ptb and type(ret[0]) == str and ret[0] == "PRT|ADVP":
        ret[0] = "PRT"
    return ret


def read_ptb_sentence(tokenizer, fix_ptb=False):
    if tokenizer.eof():
        raise SyntaxError("Cannot read sentence: EOF")
    if tokenizer.token() != "(":
        raise SyntaxError("Expecting opening bracket, got: _%s_" % tokenizer.token())
    tokenizer.next()

    roots = list()
    while tokenizer.token() != ")":
        roots.append(read_ptb_list(tokenizer, fix_ptb))
    tokenizer.next()

    # annotation bug in the train of the PTB
    if len(roots) == 2 and fix_ptb and roots[0][0] == "S" and roots[1][0] == ".":
        roots[0].append(roots[1])
        del roots[-1]
    if len(roots) != 1:
        raise SyntaxError("Sentence with several roots!")

    return roots[0]


# read the PTB, preprocess data and return it in a nice formace
def read_and_preprocess(path, fix_ptb=False, all_unlabeled_constituents=False, head_percolation_callback=None, use_dep_files=False):
    if head_percolation_callback is not None and use_dep_files:
        raise RuntimeError("Cannot use head percolation table and a dep file at the same time.")
    data = read(path, fix_ptb)
    data = [preprocess_all(node) for node in data]
    ret_data = [nested_expr_to_data(node, all_unlabeled_constituents=all_unlabeled_constituents) for node in data]
    if head_percolation_callback is not None:
        for i, node in enumerate(data):
            # note that all functional tags and morpological info have been remove...
            ret_data[i]["heads"] = pydestruct.head_percolation.extract_dependencies(node, head_percolation_callback)

    if use_dep_files:
        dep_data = read_dep_file(path[:path.rfind(".")] + ".dep")
        if len(dep_data) != len(ret_data):
            raise RuntimeError("Different number of sentences in tree and dep files")
        for i, heads in enumerate(dep_data):
            if len(ret_data[i]["words"]) != len(heads):
                raise RuntimeError("Sentences in tree and dep files are of different lengths!")
            ret_data[i]["heads"] = heads
    return ret_data

def read_dep_file(path):
    data = []
    new_need = True
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                new_need = True
            else:
                if new_need:
                    data.append(list())
                    new_need = False
                line = line.split()
                head = int(line[3]) - 1
                data[-1].append(head)
    return data


def read(path, fix_ptb=False):
    # this is way too slow
    #data = OneOrMore(nestedExpr(ignoreExpr=None)).parseFile(path).asList()
    # remove use outter list in the PTB format
    #for i in range(len(data)):
    #    if len(data[i]) == 1 and type(data[i][0]) == list:
    #        data[i] = data[i][0]
    #return data
    with open(path, "r") as file:
        text = file.read()
    tokenizer = PTBTokenizer(text)
    data = []
    while not tokenizer.eof():
        data.append(read_ptb_sentence(tokenizer, fix_ptb))
    return data


def nested_list_to_ptb(l, outer=True):
    return "("\
          + " ".join(
                    (nested_list_to_ptb(i, outer=False) if type(i) == list else str(i))
                    for i in l
                )\
          + ")"


# pre processing


def merge_unaries(node):
    if type(node[0]) != str:
        raise RuntimeError("First value must be a string")
    if len(node) == 2:
        if type(node[1]) == str:
            # this is a POS tag node
            return node
        elif len(node[1]) == 2 and type(node[1][1]) == str:
            # the child is actually a tag+token,
            # so we do not merge
            return node
        else:
            new_node = merge_unaries(node[1])
            return [node[0] + "_" + new_node[0]] + new_node[1:]
    else:
        return [node[0]] + [merge_unaries(child) for child in node[1:]]


def remove_function_and_label(node):
    if len(node) == 2 and type(node[1]) == str:
        # this is a POS tag node
        return node
    else:
        label = node[0]
        label = label.split("=")[0]
        label = label.split("-")[0] if label[0] != "-" else label
        return [label] + [remove_function_and_label(child) for child in node[1:]]


def remove_morphological_info(node):
    if len(node) == 2 and type(node[1]) == str:
        # this is a POS tag node
        label = node[0]
        label = label.split("##")[0]  # remove morphological labels
        label = label.split("-")[0]  # remove function tag
        return [label, node[1]]
    else:
        label = node[0]
        return [label] + [remove_morphological_info(child) for child in node[1:]]


def remove_empty(node):
    if len(node) == 2 and type(node[1]) == str:
        # this is a POS tag node
        if node[0] == "-NONE-":
            return []
        else:
            return node
    else:
        children = []
        for child in node[1:]:
            new_child = remove_empty(child)
            if len(new_child) > 0:
                children.append(new_child)
        if len(children) > 0:
            return [node[0]] + children
        else:
            return []


# default preoprocessing of the PTB
# we need to remove function label before merging unaries, otherwise result will be corrupted
def preprocess_ptb(node):
    return merge_unaries(remove_empty(remove_function_and_label(node)))


def preprocess_ftb(node):
    return merge_unaries(remove_empty(remove_function_and_label(remove_morphological_info(node))))


def preprocess_all(node):
    return preprocess_ftb(node)


def nested_expr_to_data(node, all_unlabeled_constituents=False):
    ret = {
        "words": list(),
        "tags": list(),
        "constituents": set(),
        "nested_expr": node
    }

    extract(node, ret["words"], ret["tags"], ret["constituents"])
    if all_unlabeled_constituents:
        words = list()
        cst = set()
        extract_all_unlabeled(node, words, cst)
        ret["all_unlabeled_constituents"] = cst
    return ret


def split_unary_chains(constituents, sep="_"):
    ret = set()
    for cst in constituents:
        for slabel in cst[0].split(sep):
            ret.add((slabel,) + cst[1:])
    return ret


def extract(node, words, tags, constituents):
    if len(node) == 0:
        raise IOError("Node without label")
    if len(node) == 1:
        raise RuntimeError(
            "Should not happen: leaf nodes must be parsed at the same time as the POS tag. Is there missing POS tags in the data?")
    else:
        if len(node) == 2 and type(node[1]) == str:
            # this is a POS tag node
            tags.append(node[0])
            words.append(node[1])

            return [len(tags) - 1]
        else:
            label = node[0]
            span = []
            for child in node[1:]:
                span = span + extract(child, words, tags, constituents)
            left = span[0]
            right = span[-1]
            constituents.add((label, left, right))

            return span


# this extract unlabeled constituents
# + all possible binarisation.
# this is useful as a constraint on the chart for
# computing the best dependency structure with constrained
# constituency structure
def extract_all_unlabeled(node, words, constituents):
    if len(node) == 0:
        raise IOError("Node without label")
    if len(node) == 1:
        raise RuntimeError(
            "Should not happen: leaf nodes must be parsed at the same time as the POS tag. Is there missing POS tags in the data?")
    else:
        if len(node) == 2 and type(node[1]) == str:
            # this is a POS tag node
            constituents.add((len(words), len(words)))
            words.append(node[1])

            return [len(words) - 1]
        else:
            label = node[0]
            span = []
            children = []
            for child in node[1:]:
                children.append(extract_all_unlabeled(child, words, constituents))
                span = span + children[-1]
            left = span[0]
            right = span[-1]
            constituents.add((left, right))

            if len(children) > 2:
                for i in range(len(children)):
                    for j in range(i + 1, len(children)):
                        left = children[i][0]
                        right = children[j][-1]
                        constituents.add((left, right))

            return span
