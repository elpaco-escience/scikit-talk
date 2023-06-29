import pandas as pd
import os
import ConvoKitMeta
import json


from collections import defaultdict

class Corpus:
    def __init__(
        self,
        path
    ):
        self.content = pd.read_csv(path)
        self.meta = ConvoKitMeta(self, self.meta_index, "corpus")

    def to_json_pd(self):
        self.json = self.content.to_json

    def to_json(
        self,
        name,
        base_path
    )
        """
        Dumps the corpus and its metadata to disk. Optionally, set `force_version` to a desired integer version number,
        otherwise the version number is automatically incremented.
        :param name: name of corpus
        :param base_path: base directory to save corpus in (None to save to a default directory)
        """
        dir_name = name
        dir_name = os.path.join(base_path, dir_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)


        # dump corpus
        with open(os.path.join(dir_name, "corpus.json"), "w") as f:
            d_bin = defaultdict(list)
            meta_up = _dump_helper_bin(self.meta, d_bin)

            json.dump(meta_up, f)
            # for name, l_bin in d_bin.items():
            #     with open(os.path.join(dir_name, name + "-overall-bin.p"), "wb") as f_pk:
                    # pickle.dump(l_bin, f_pk)

        # # dump index
        # with open(os.path.join(dir_name, "index.json"), "w") as f:
        #     json.dump(
        #         self.meta_index.to_dict(
        #             exclude_vectors=exclude_vectors, force_version=force_version
        #         ),
        #         f,
        #     )

        # # dump vectors
        # if exclude_vectors is not None:
        #     vectors_to_dump = [v for v in self.vectors if v not in set(exclude_vectors)]
        # else:
        #     vectors_to_dump = self.vectors
        # for vector_name in vectors_to_dump:
        #     if vector_name in self._vector_matrices:
        #         self._vector_matrices[vector_name].dump(dir_name)
        #     else:
        #         src = os.path.join(self.corpus_dirpath, "vectors.{}.p".format(vector_name))
        #         dest = os.path.join(dir_name, "vectors.{}.p".format(vector_name))
        #         shutil.copy(src, dest)


def _dump_helper_bin(d: ConvoKitMeta, d_bin, fields_to_skip=None):  # object_idx
    """
    :param d: The ConvoKitMeta to encode
    :param d_bin: The dict of accumulated lists of binary attribs
    :return:
    """
    if fields_to_skip is None:
        fields_to_skip = []

    BIN_DELIM_L, BIN_DELIM_R = "<##bin{", "}&&@**>"

    obj_idx = d.index.get_index(d.obj_type)
    d_out = {}
    for k, v in d.items():
        if k in fields_to_skip:
            continue
        try:
            if len(obj_idx[k]) > 0 and obj_idx[k][0] == "bin":
                d_out[k] = "{}{}{}".format(BIN_DELIM_L, len(d_bin[k]), BIN_DELIM_R)
                d_bin[k].append(v)
            else:
                d_out[k] = v
        except KeyError:
            # fails silently (object has no such metadata that was indicated in metadata index)
            pass
    return d_out
