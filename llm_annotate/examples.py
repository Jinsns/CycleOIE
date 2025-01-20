import csv
from tqdm import tqdm
import os, sys
import openai
# from openai import OpenAI

from tenacity import (retry, stop_after_attempt, wait_random_exponential)


openai.api_base = ""
openai.api_key = ""


def file2list(file_path):
    with open(file_path, "r", encoding="UTF-8") as f:
        sentence_list = f.readlines()
    return sentence_list


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def chat_openie(sentence_list, file_out):

    for sentence in tqdm(sentence_list):
        content = sentence.strip()
        system_content = """
            You are an expert in Information Extraction. Please extract triples from the given sentence. 
            Please write optional tokens in square brackets [ and ] .
            Link subject, predicate, and object in a triple with --> .
            Please write one triple in one line.
        """

        user1 = "He served as the first Prime Minister of Australia and became a founding justice of the High Court of Australia ."
        assistant1 = """
            He --> served as --> [the] [first] Prime Minister [of Australia]
            He --> served as [the] [first] Prime Minister of --> Australia
            He --> became --> [a] [founding] justice of [the] High Court [of Australia]
            He --> became [a] [founding] justice of --> [the] High Court [of Australia]
            He --> became [a] [founding] justice of [the] High Court of --> Australia
        """

        user2 = "Graner handcuffed him to the bars of a cell window and left him there , feet dangling off the floor , for nearly five hours ."
        assistant2 = """
            Graner --> handcuffed --> him
            Graner --> left him [there] for --> [nearly] five hours
            Graner --> handcuffed him to --> [the] bars [of a cell window]
            feet --> dangling off --> [the] floor
            Graner --> handcuffed him to [the] bars of --> [a] [cell] window
            Graner --> left --> him
            feet --> dangling off [the] floor for --> nearly five hours
        """

        user3 = "It deals with cases of fraud in relation to direct taxes and indirect taxes , tax credits , drug smuggling , and money laundering , cases involving United Nations trade sanctions , conflict diamonds and CITES ."
        assistant3 = """
            It --> deals with cases of fraud in [relation to] --> direct taxes
            It --> deals with cases of fraud in [relation to] --> indirect taxes
            It --> deals with cases of fraud in [relation to] --> tax credits
            It --> deals with cases of fraud in [relation to] --> drug smuggling
            It --> deals with cases of fraud in [relation to] --> money laundering
            It --> deals with cases involving --> United Nations trade sanctions
            It --> deals with cases involving --> conflict diamonds
            It --> deals with cases involving --> CITES
            It --> deals --> with cases of fraud in [relation to] money laundering
            It --> deals with cases of --> fraud in [relation to] direct taxes
            cases --> involving --> United Nations trade sanctions
            cases --> involving --> conflict diamonds
            cases --> involving --> CITES
        """

        user4 = "Because of this association , St. Michael was considered to be the patron saint of colonial Maryland , and as such was honored by the river being named for him ."
        assistant4 = """
            St. Michael --> was considered [to be] --> [the] patron saint [of colonial Maryland]
            St. Michael --> was considered [to be] [the] patron saint --> of colonial Maryland
            St. Michael --> was honored by --> [the] river being named for him
            [the] river --> [being] named for --> him
            St. Michael --> was considered [to be] [the] patron saint [of colonial Maryland] Because of --> [this] association
        """

        user5 = "The show was designed to appear as if the viewer was channel surfing through a multi-channel wasteland , happening upon spoof adverts , short sketches , and recurring show elements ."
        assistant5 = """
            [The] viewer --> was happening upon --> spoof adverts
            [The] viewer --> was happening upon --> short sketches
            [The] viewer --> was happening upon --> recurring show elements
            [the] viewer --> was channel surfing through --> [a] [multi-channel] wasteland
            [The] show --> was designed to appear as if [the] viewer was channel surfing through --> [a] multi-channel wasteland
            [The] show --> was designed to appear as if [the] viewer was happening upon --> spoof adverts
            [The] show --> was designed to appear as if [the] viewer was happening upon --> short sketches
            [The] show --> was designed to appear as if [the] viewer was happening upon --> recurring show elements
        """

        user6 = sentence




        # print("user input: \n", user_content)

        response = completion_with_backoff(
            # model="text-davinci-003",
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},

                {"role": "user", "content": user1},
                {"role": "user", "content": assistant1},

                {"role": "user", "content": user2},
                {"role": "user", "content": assistant2},

                {"role": "user", "content": user3},
                {"role": "user", "content": assistant3},

                {"role": "user", "content": user4},
                {"role": "user", "content": assistant4},

                {"role": "user", "content": user5},
                {"role": "user", "content": assistant5},

                {"role": "user", "content": user6}  # new sentence

            ],
        )
        completion = response['choices'][0]["message"]["content"]
        # start = completion.find("{")
        # end = completion.rfind("}")
        with open(file_out, "a", encoding="UTF-8") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([content, completion])
            print(f"!!!!!!\n{sentence} : {completion}")




if __name__ == "__main__":
    # test benchie zh
    file_in = "/Users/jzh/Desktop/研/projects/tcpoie_20240801/lsoie_wiki_train_sample_5000"
    file_out = "/Users/jzh/Desktop/研/projects/tcpoie_20240801/lsoie_wiki_train_sample_5000_annotations"


    with open(file_out, mode='w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source", "target"])

    sentence_list = file2list(file_in)
    chat_openie(sentence_list, file_out)