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


def convert_from_json(filein, fileout):
    text2data_dict = {}
    with open(filein, mode='r', encoding="utf-8") as f:
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

    with open(fileout, mode="w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        dict_list = []
        for (text, datas) in text2data_dict.items():
            for data in datas:
                one_dict = {
                    "subject": data[0],
                    "relation": data[1],
                    "object": data[2]
                }
            dict_list.append(json.dumps(one_dict))
            writer.writerow([text, dict_list])







if __name__ == "__main__":
    type = "dev"
    filein = f"{type}.tsv"
    fileout = f"carb_{type}_t5.tsv"
    convert_from_sep(filein, fileout)