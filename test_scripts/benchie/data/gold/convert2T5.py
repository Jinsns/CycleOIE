import csv
import json


def convert_from_sep(filein, fileout):
    text2data_dict = {}
    with open(filein, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for (i, line) in enumerate(list(reader)):
            sentence = line[0]
            relation = line[1]
            subject = line[2]
            object = line[3] if len(line) > 3 else "o"
            if len(line) > 4:
                print(f"{i} longer than 4")
                object += " ".join(line[4:])
            text2data_dict[sentence] = text2data_dict.get(sentence, [])
            text2data_dict[sentence].append([subject, relation, object])


    with open(fileout, mode='w', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for (text, datas) in text2data_dict.items():
            datas_str = ""
            for data in datas:
                data_str = f"subject<is>{data[0]}<and>relation<is>{data[1]}<and>object<is>{data[2]}<then>"
                datas_str += data_str
            writer.writerow([text, datas_str])


def convert_from_gold(filein, fileout):
    sentence_list = []
    with open(filein, mode='r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            if line.startswith("sent_id:"):
                sentence = line.split('\t')[-1]
                sentence_list.append(sentence)

    print(sentence_list)
    with open(fileout, mode='w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["source", "target"])
        for sentence in sentence_list:
            writer.writerow([sentence.replace("\n", ""), "placeholder"])



if __name__ == "__main__":
    # filein = "carb_explicit.txt"
    # fileout = "benchie_t5.tsv"
    # convert_from_sep(filein, fileout)
    filein = "2_annotators/benchie_gold_annotations_en.txt"
    fileout = "benchie300_t5.tsv"
    convert_from_gold(filein, fileout)
