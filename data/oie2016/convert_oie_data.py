import csv


file_in = "oie2016_test"
file_out = "converted_oie2016_test"
sentence_list = []
with open(file_in, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in list(reader):
        sentence_list.append(row[0])

    sentence_list = set(sentence_list)

with open(file_out, mode='w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(["source", "target"])
    for sentence in sentence_list:
        writer.writerow([sentence, "target placeholder"])