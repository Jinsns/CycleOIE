import csv


imojie_trainset_filepath = "../imojie/train/imojie_t5_sft_data.tsv"
lsoie_trainset_filepath = "../lsoie/sep_clean_sro/lsoie_train.csv"

with open(imojie_trainset_filepath, mode='r', encoding="utf-8") as f:
    imojie_content = f.readlines()
    print(len(imojie_content))


with open(lsoie_trainset_filepath, mode='r', encoding='utf-8') as f:
    lsoie_content = f.readlines()[1:]
    print(len(lsoie_content))


total_content = imojie_content + lsoie_content
print(total_content[:10])


with open("imojie_lsoie.train", mode='w', encoding='utf-8', newline="") as f:
    f.writelines(total_content)


