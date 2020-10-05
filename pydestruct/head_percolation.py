# head percolation table
from pyparsing import OneOrMore, nestedExpr


def all_left(label, child_labels):
    return 0


def all_right(label, child_labels):
    return len(child_labels) - 1


def ftb(label, children):
    for rule in ftb_rules:
        if label == rule[0] or rule[0] == "*":
            for subrule in rule[1:]:
                if subrule[0] == "first":
                    r = range(len(children))
                elif subrule[0] == "last":
                    r = range(len(children) - 1, 0 - 1, -1)
                else:
                    raise RuntimeError("Unknown direction: %s" % subrule[0])

                for i in r:
                    if children[i] in subrule[1:] or subrule[1:][0] == "*":
                        return i

            # print(label, children)
            # raise RuntimeError("Rule did not match anything!")

    # there is * rule so this should not happen
    raise RuntimeError("Unmatched label: %s" % label)


ftb_rules = OneOrMore(nestedExpr(ignoreExpr=None)).parseString("""(
    (S1 (first SENT) )
    (PONCT (last *) )
    (Sint (last VN) (last AP) (last NP) (last PP) (last VPinf)
    (last Ssub) (last VPpart) (last A ADJ ADJWH) (last ADV
    ADVWH) )
    (VPpart (first VPR VPP) (first VN) )
    (SENT (last VN) (last AP) (last NP) (last Srel) (last
    VPpart) (last AdP) (last I) (last Ssub) (last VPinf) (last PP)
    (last ADV ADVWH) )
    (COORD (first CS CC PONCT) )
    (AP (last A ADJ ADJWH) (last ET) (last VPP) (last ADV
    ADVWH) )
    (NP (first NPP PROREL PRO NC PROWH) (first NP)
    (first A ADJ ADJWH) (first AP) (first I) (first VPpart) (first
    ADV ADVWH) (first AdP) (first ET) (first DETWH DET)
    )
    (VPinf (first VN) (first VIMP VPR VS VINF V VPP) )
    (PP (first P) (first P+D) (first NP P+PRO) )
    (Ssub (last VN) (last AP) (last NP) (last PP) (last VPinf)
    (last Ssub) (last VPpart) (last A ADJ ADJWH) (last ADV
    ADVWH) )
    (VN (last VIMP VPR VS VINF V VPP) (last VPinf) )
    (Srel (last VN) (last AP) (last NP) )
    (AdP (last ADV ADVWH) )
    (* (first *) )
)""").asList()[0]


tables = {
    "first": all_left,
    "last": all_right,
    "ftb": ftb
}


def extract_dependencies(node, callback):
    words = list()
    head_dict = dict()

    root = extract_head(node, words, head_dict, callback)
    head_dict[root] = -1

    heads = [-2] * len(words)
    for mod, head in head_dict.items():
        heads[mod] = head

    if any(h == -2 for h in heads):
        raise RuntimeError("Error during lexicalization!")

    return heads


def extract_head(node, words, heads, callback):
    if len(node) == 0:
        raise IOError("Node without label")
    if len(node) == 1:
        raise RuntimeError("Should not happen: leaf nodes must be parsed at the same time as the POS tag. Is there missing POS tags in the data?")
    else:
        if len(node) == 2 and type(node[1]) == str:
            # this is a POS tag node
            words.append(node[1])

            return len(words) - 1
        else:
            label = node[0]
            child_heads = []
            child_labels = []
            for child in node[1:]:
                child_heads.append(extract_head(child, words, heads, callback))
                child_labels.append(child[0])

            if len(child_heads) == 1:
                return child_heads[0]
            else:
                head = child_heads[callback(label, child_labels)]
                for i, modifier in enumerate(child_heads):
                    heads[modifier] = head
                return head
