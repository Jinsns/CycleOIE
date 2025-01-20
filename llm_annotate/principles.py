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
        system_content = "System You are an expert annotator in Information Extraction."
        user_content_head = ''''Following the guidelines, extract information from the given sentence below.
Guidelines:
(1) Each triple should represent two entities or concepts, and the verb-mediated relation between them. For
example, from the input sentence "Michael Jordan, who is a former basketball player, was born in Brooklyn.",
there are three entities and concepts— Michael Jordan, former basketball player and Brooklyn—which are
related as follows: {"subject": "Michael Jordan", "relation": "is", "object": "former basketball player"} and
{"subject": "Michael Jordan", "relation": "was born in", "object": "Brooklyn"}.
(2) When possible, if an extraction is in passive voice, the annotator should place its active voice equivalent
into the appropriate fact synset. For instance, consider the sentence "The ball was kicked by John."; then,
the fact synset should contain the following triples: {"subject": "[The] ball", "relation": "was kicked by",
"object": "John"} {"subject": "John", "relation": "kicked", "object": "[The] ball"}
(3) Extractions that indicate attribution of another core piece of information should be placed in separate fact
synset, because they indicate a separate piece of information with separate predicate. For example, the core
information of the sentence "Conspiracy theorists say that Barack Obama was born in Kenya." is that Barack
Obama was born in Kenya. the triple {"subject": "Barack Obama", "relation": "[was] born in", "object":
"Kenya"}—in one fact synset, and the triples indicating attribution— {"subject": "Conspiracy theorists",
"relation": "say that", "object": "Barack Obama was born in Kenya"}—in another.
(4) The annotator should not extract incomplete clauses, i.e., triples that lack crucial piece of information.
Suppose there is the input sentence "He was honored by the river being named after him". The following
triple should not be manually extracted: ("He"; "was honored by"; "[the] river"), but the following triples
should be: {"subject": "He", "relation": "was honored by [the] river being named after", "object": "him"} and
{"subject":"[the] river", "relation": "being named after", "object": "him"}.
(5) The annotator should not allow for conjunctive phrases to form an argument (i.e., subject or object). Such
arguments should be placed into separate extractions (and in separate fact synsets). Consider the sentence
"Michael Jordan and Scottie Pippen played for Chicago Bulls.". The annotator should manually extract the
following triples: {"subject":"M. Jordan", "relation": "played for", "object": "Chicago Bulls"} {"subject":"S.
Pippen", "relation": "played for", "object": "Chicago Bulls"} The annotator should not, however, extract
{"subject": "M. J. and S. P.", "relation": "played for", "object": "Chicago Bulls"}.
(6)We focus on explicit extractions, which means that every word in the extracted triple must be present in
the original input sentence. Therefore, implicit extractions—i.e., extractions that contain inferred information
with words not found in the sentence—are not considered. One example implicit extraction is {"subject":
"Michael Jordan", "relation": "be", "object": "Prof."} from the input sentence "Prof. Michael Jordan lives
in USA.", where the triple infers that Michael Jordan is professor without being explicitly indicated in the
sentence (i.e., the word "be" is not present in the input sentence, it is inferred)'''

        user_content_tail = (
            "Don’t miss any fact.\n"
            'Format these facts into compact JSON format {"subject": "", "relation": "", "object": ""}.\n'
            '1 compact JSON in 1 line.\n'
            "Don’t generate any token outside the most outer bracket{} of JSON.\n"
        )

        user_content = user_content_head + 'Given sentence: "' + content + '"\n' + user_content_tail



        # print("user input: \n", user_content)

        response = completion_with_backoff(
            # model="text-davinci-003",
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
        )
        completion = response['choices'][0]["message"]["content"]
        start = completion.find("{")
        end = completion.rfind("}")
        with open(file_out, "a", encoding="UTF-8") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([content, completion[start:end+1]])
            print(f"!!!!!!\n{sentence} : {completion[start:end+1]}")




if __name__ == "__main__":
    # test benchie zh
    file_in = "data/lsoie/sample/lsoie_wiki_train_sample_2147"
    file_out = "data/lsoie/sample/lsoie_wiki_train_sample_2147_principle_annotations"


    with open(file_out, mode='w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source", "target"])

    sentence_list = file2list(file_in)
    chat_openie(sentence_list, file_out)