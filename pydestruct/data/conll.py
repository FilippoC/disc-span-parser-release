def read(path, format="conllx"):
    if not (format == "conllx" or format == "conllu"):
        raise RuntimeError("Only conllx and conllu form implemented for now")

    sentence = dict()
    with open(path) as h:
        need_new = True
        for line in h:
            line = line.strip()
            # skip newline
            if len(line) == 0:
                if not need_new:
                    yield sentence
                need_new = True
                continue
            # skip comments
            if line[0] == "#":
                continue
            else:
                if need_new:
                    sentence = dict({"tokens": list(), "multiwords": list(), "emptytokens": list()})
                    need_new = False
                values = line.split("\t")

                if values[0].find("-") >= 0:
                    begin, end = (int(i) for i in values[0].split("-"))
                    sentence["multiwords"].append({
                        "begin": begin,
                        "end": end,
                        "line": values
                    })

                elif values[0].find(".") >= 0:
                    begin, end = (int(i) for i in values[0].split("."))
                    sentence["emptytokens"].append({
                        "begin": begin,
                        "end": end,
                        "line": values
                    })

                elif format == "conllx":
                    sentence["tokens"].append({
                        "form": values[1],
                        "lemma": values[2],
                        "cpostag": values[3],
                        "postag": values[4],
                        "feats": values[5],
                        "head": int(values[6]),
                        "deprel": values[7],
                        "phead": values[8],
                        "pdephead": values[9]
                    })

                else:
                    deprel = values[7]
                    if deprel.find(":") >= 0:
                        deprel = deprel.split(":")[0]
                    sentence["tokens"].append({
                        "form": values[1],
                        "lemma": values[2],
                        "upos": values[3],
                        "xpos": values[4],
                        "feats": values[5],
                        "head": int(values[6]),
                        "deprel": deprel,
                        "deps": values[8],
                        "misc": values[9]
                    })

        if not need_new:
            yield sentence


def write(path, data_generator, deps_generator=None, format="conllx"):
    with open(path, "w") as f:
        for sentence in data_generator:
            if deps_generator is not None:
                sentence_deps = next(deps_generator)

            next_mw = 0
            next_et = 0
            for index, token in enumerate(sentence["tokens"], start=1):

                while next_mw < len(sentence["multiwords"]) and sentence["multiwords"][next_mw]["begin"] == index - 1:
                    f.write("\t".join(sentence["multiwords"][next_mw]["line"]))
                    f.write("\n")
                    next_mw += 1

                while next_et < len(sentence["emptytokens"]) and sentence["emptytokens"][next_et]["begin"] == index - 1:
                    f.write("\t".join(sentence["emptytokens"][next_et]["line"]))
                    f.write("\n")
                    next_et += 1

                if deps_generator is not None:
                    head, deprel = next(sentence_deps)
                    head = str(head)
                else:
                    head = str(token["head"])
                    deprel = token["deprel"]

                if format == "conllx":
                    f.write("\t".join((
                        str(index),
                        token["form"],
                        token["lemma"],
                        token["cpostag"],
                        token["postag"],
                        token["feats"],
                        head,
                        deprel,
                        token["phead"],
                        token["pdephead"]
                    )))
                else:
                    f.write("\t".join((
                        str(index),
                        token["form"],
                        token["lemma"],
                        token["upos"],
                        token["xpos"],
                        token["feats"],
                        head,
                        deprel,
                        token["deps"],
                        token["misc"]
                    )))
                # end of token
                f.write("\n")
            # end of sentence
            f.write("\n")

