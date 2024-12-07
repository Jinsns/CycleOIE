# import sys
# sys.path.append("/Users/jinzhihong/Desktop/project/openie/CaRB")

# import logging
import re
from pprint import pprint
from pprint import pformat
from docopt import docopt

# Local imports
from test_scripts.CaRB.oie_readers.oieReader import OieReader
from test_scripts.CaRB.oie_readers.extraction import Extraction
import ipdb


class T5SepReader(OieReader):
    def __init__(self):
        self.name = 't5'

    def read(self, fn):
        d = {}
        with open(fn, encoding="utf-8") as fin:
            for line in list(fin)[1:]:
                text = line.split("\t")[0]
                # print("text: ", text)
                datas = line.split("\t")[1]
                datas = datas.split("<then>")
                for data in datas:
                    # print("data: ", data)
                    pattern = r"subject <is> (.*) <and> relation <is> (.*) <and> object <is> (.*)"
                    searchobj = re.search(pattern, data, re.M|re.I)
                    if searchobj != None:
                        subject = searchobj.group(1)
                        relation = searchobj.group(2)
                        object = searchobj.group(3)
                        curExtraction = Extraction(pred=relation, head_pred_index=-1, sent=text, confidence=0.1)
                        curExtraction.addArg(subject)
                        curExtraction.addArg(object)
                        d[text] = d.get(text, []) + [curExtraction]
        self.oie = d
        # print(d)


class T5JsonReader(OieReader):
    def __init__(self):
        self.name = 'T5'

    def read(self, fn):
        d = {}
        with open(fn, encoding="utf-8") as fin:
            for line in list(fin)[1:]:
                text = line.split("\t")[0]
                print("text: ", text)
                datas = line.split("\t")[1]
                datas = eval(datas)
                # datas = datas.split("</s></s>")
                for data in datas:
                    try:
                        subject = data["subject"]
                        relation = data["relation"]
                        object = data["object"]

                        curExtraction = Extraction(pred=relation, head_pred_index=-1, sent=text, confidence=0.4)
                        curExtraction.addArg(subject)
                        curExtraction.addArg(object)
                        d[text] = d.get(text, []) + [curExtraction]
                    except:
                        print("exception data: ", data)

        self.oie = d
        # print(d)


# benchie_sentences = []
# benchie_sentence_filepath = "/home/langchao/projects_jzh/cycle_training/test_scripts/benchie/data/sentences/sample300_en.txt"
# with open(benchie_sentence_filepath, mode='r', encoding="utf-8") as f:
#     benchie_sentences = f.readlines()



class BenchieResultReader():
    def __init__self(self):
        self.name = "BenchIEResult"

    def read(self, fn):
        benchie_sentences = []
        benchie_sentence_filepath = "/mnt/data/jinzhihong/projects/cycle/test_scripts/benchie/data/sentences/sample300_en.txt"
        with open(benchie_sentence_filepath, mode='r', encoding="utf-8") as f:
            benchie_sentences = f.readlines()

        d = {}
        with open(fn, encoding='utf-8') as fin:
            for line in fin:
                data = line.strip().split('\t')
                textid, arg1, rel, arg2 = data
                text = benchie_sentences[int(textid)-1].strip()
                curExtraction = Extraction(pred=rel, head_pred_index=-1, sent=text, confidence=float(1.0))
                curExtraction.addArg(arg1)
                curExtraction.addArg(arg2)
                d[text] = d.get(text, []) + [curExtraction]
        self.oie = d
        print(d)

    def normalizeConfidence(self):
        ''' Normalize confidence to resemble probabilities '''
        EPSILON = 1e-3

        confidences = [extraction.confidence for sent in self.oie for extraction in self.oie[sent]]
        maxConfidence = max(confidences)
        minConfidence = min(confidences)

        denom = maxConfidence - minConfidence + (2 * EPSILON)

        for sent, extractions in self.oie.items():
            for extraction in extractions:
                extraction.confidence = ((extraction.confidence - minConfidence) + EPSILON) / denom


if __name__ == "__main__":
    oie = T5SepReader()
    inp_fn = "my_outputs/dev_output.txt"
    out_fn = "my_outputs/dev_out_json"
    oie.read(inp_fn)
    oie.output_tabbed(out_fn)

    # logging.info("DONE")