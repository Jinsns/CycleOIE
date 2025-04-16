import os
import csv
import argparse

import torch
from torch.nn import CrossEntropyLoss
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import re
import numpy as np
import random
import math
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from datasets import load_metric
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

from test_scripts.CaRB.carb import carb_eval

date = "1207"
preference = "principles"

text2data_filepath = f"./data/lsoie/examples/lsoie-g-1.0.tsv"
text2data_validation_file = "./data/carb/sep/carb_dev_t5.csv"
# text2data_test_file = text2data_validation_file

data2text_filepath = text2data_filepath
data2text_validation_file = text2data_validation_file
# data2text_test_file = data2text_validation_file

carb_test_file = "./data/carb/sep/carb_test_t5.csv"
benchie_test_file = "./data/benchie/benchie300_t5.tsv"

per_gpu_train_batch_size = 4
per_gpu_eval_batch_size = 4
gradient_accumulation_steps = 8
data2text_learning_rate = 1e-5
text2data_learning_rate = 1e-5
scheduler_type = "linear"
warmup_steps = 0
num_epochs = 100
max_input_length = 96
min_output_length = 3
max_output_length = 96
num_beams = 4

selection_metric = "loss"
delta = 0.001

output_dir = f"./cycle_weights/{date}_lsoie_validation_{preference}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

patience = 100

save_epochs = 5

# cycle train without sft bootstrap
# t5_model_path = "google/flan-t5-base"
t5_model_path = "sft_weights/flan-t5-base"
data2text_model = t5_model_path
text2data_model = t5_model_path
text2data_tokenizer = t5_model_path


# do_train = True
do_train = True
do_generate = True
do_eval = True
do_test = False

device = torch.device('cuda:1')

tokenizer = T5Tokenizer.from_pretrained(text2data_tokenizer)
new_words = ['<is>', '<and>', '<then>']
tokenizer.add_tokens(new_words)
model_text2data = T5ForConditionalGeneration.from_pretrained(text2data_model)
model_text2data.resize_token_embeddings(len(tokenizer)) # if train from flan-base-t5
model_text2data.to(device)

model_data2text = T5ForConditionalGeneration.from_pretrained(data2text_model)
model_data2text.resize_token_embeddings(len(tokenizer)) # if train from flan-base-t5
model_data2text.to(device)


def tokenize_val_content(sample, input_type="text"):
    if input_type == "text":
        return tokenizer(sample['source'], padding='max_length', truncation=True, max_length=max_input_length)
    else:
        return tokenizer(sample['target'], padding='max_length', truncation=True, max_length=max_input_length)


if do_train:
    text = load_dataset('csv', data_files={"train": text2data_filepath}, delimiter='\t')
    tokenized_text = text.map(lambda x: tokenize_val_content(x, input_type="text"), batched=True)
    tokenized_text.set_format('torch', ['attention_mask', 'input_ids'], output_all_columns=True)
    text_dataloader = DataLoader(tokenized_text['train'], shuffle=True, batch_size=per_gpu_train_batch_size)

    triplets = load_dataset('csv', data_files={"train": data2text_filepath}, delimiter='\t')  # to change
    tokenized_triplets = triplets.map(lambda x: tokenize_val_content(x, input_type="data"), batched=True)
    tokenized_triplets.set_format('torch', ['attention_mask', 'input_ids'], output_all_columns=True)
    triplets_dataloader = DataLoader(tokenized_triplets['train'], shuffle=True, batch_size=per_gpu_train_batch_size)

    optimizer_text2data = AdamW(list(model_text2data.parameters()), lr=text2data_learning_rate)
    optimizer_data2text = AdamW(list(model_data2text.parameters()), lr=data2text_learning_rate)

    num_text_training_steps = num_epochs * len(text_dataloader)
    num_data_training_steps = num_epochs * len(triplets_dataloader)

    lr_scheduler_text2data = get_scheduler(
        scheduler_type,
        optimizer=optimizer_text2data,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_text_training_steps
    )

    lr_scheduler_data2text = get_scheduler(
        scheduler_type,
        optimizer=optimizer_data2text,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_data_training_steps
    )

if do_eval:
    if text2data_validation_file != None:
        text2triplets_val = load_dataset('csv', data_files={'dev': text2data_validation_file}, delimiter='\t')
        tokenized_text2triplets_val = text2triplets_val.map(lambda x: tokenize_val_content(x, input_type="text"),
                                                            batched=True)
        tokenized_text2triplets_val.set_format('torch', ['attention_mask', 'input_ids'], output_all_columns=True)
        text2triplets_val_dataloader = DataLoader(tokenized_text2triplets_val['dev'], shuffle=False,
                                                  batch_size=per_gpu_eval_batch_size)

    if data2text_validation_file != None:
        triplets2text_val = load_dataset('csv', data_files={'dev': data2text_validation_file}, delimiter='\t')
        tokenized_triplets2text_val = triplets2text_val.map(lambda x: tokenize_val_content(x, input_type="text"),
                                                            batched=True)
        tokenized_triplets2text_val.set_format('torch', ['attention_mask', 'input_ids'], output_all_columns=True)
        triplets2text_val_dataloader = DataLoader(tokenized_triplets2text_val['dev'], shuffle=False,
                                                  batch_size=per_gpu_eval_batch_size)

    if carb_test_file != None:
        carb_text2data_test = load_dataset('csv', data_files={'test': carb_test_file}, delimiter='\t')
        tokenized_carb_text2data_test = carb_text2data_test.map(lambda x: tokenize_val_content(x, input_type="text"),
                                                                batched=True)
        tokenized_carb_text2data_test.set_format('torch', ['attention_mask', 'input_ids'], output_all_columns=True)
        carb_text2data_test_dataloader = DataLoader(tokenized_carb_text2data_test['test'], shuffle=False,
                                                    batch_size=per_gpu_eval_batch_size)

    if benchie_test_file != None:
        benchie_text2data_test = load_dataset('csv', data_files={'test': benchie_test_file}, delimiter='\t')
        tokenized_benchie_text2data_test = benchie_text2data_test.map(
            lambda x: tokenize_val_content(x, input_type="text"), batched=True)
        tokenized_benchie_text2data_test.set_format('torch', ['attention_mask', 'input_ids'], output_all_columns=True)
        benchie_text2data_test_dataloader = DataLoader(tokenized_benchie_text2data_test['test'], shuffle=False,
                                                       batch_size=per_gpu_eval_batch_size)



def train_one_direction(model1, model2, data_loader, num_training_steps, optimizer, lr_scheduler, inter_type):
    """
    model1 = model_data2text
    model2 = model_text2data
    data_loader = triplets_dataloader
    optimizer = optimizer_text2data
    lr_scheduler = lr_scheduler_text2data
    inter_type = "text"
    """

    model1.eval()
    model2.train()

    optimizer.zero_grad()

    progress_bar = tqdm(range(math.ceil((len(data_loader)) / gradient_accumulation_steps)))

    step = 0
    batch_loss = 0
    total_loss = 0

    for batch in data_loader:
        # column 'source' is text
        # column 'target' is data
        # so
        # when text2data, input is 'source', output is 'target'
        # when data2text, input is 'target', output is 'source'
        if inter_type == "text":  # data2text2data
            batch.pop('source')
            raw_input = batch.pop("target")  # raw_input is data
        elif inter_type == "data":  # text2data2text
            batch.pop('target')
            raw_input = batch.pop("source")  # raw_input is text

        model1_input = {k: v.to(device) for k, v in batch.items()}

        # generate intermediate outputs
        with torch.no_grad():
            intermediate_outputs = model1.generate(**model1_input, max_length=max_output_length, num_beams=num_beams)

        decoded_intermediate_outputs = tokenizer.batch_decode(intermediate_outputs, skip_special_tokens=True)
        # prepare labels and inputs
        tokenized_intermediate_outputs = tokenizer(decoded_intermediate_outputs, return_tensors='pt', padding=True,
                                                   truncation=True, max_length=max_input_length)
        model2_label = raw_input
        processed_labels = \
        tokenizer(model2_label, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)[
            'input_ids']
        processed_labels[processed_labels == 0] = -100
        tokenized_intermediate_outputs['labels'] = processed_labels
        model2_input = {k: v.to(device) for k, v in tokenized_intermediate_outputs.items()}
        # final outputs and loss calculation
        outputs = model2(**model2_input)

        loss = None
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), model2_input['labels'].view(-1))
        loss = loss.mean()
        loss = loss / gradient_accumulation_steps
        loss.backward()
        batch_loss += loss.item()
        total_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            progress_bar.set_description("Train Batch Loss: %f" % (batch_loss))
            batch_loss = 0
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        step += 1

    return total_loss



def eval_model(model, data_loader, test_mode=False):
    model.eval()
    progress_bar = tqdm(range(math.ceil((len(data_loader)) / gradient_accumulation_steps)))
    step = 0
    batch_loss = 0
    total_loss = 0
    source_texts = []
    generated_texts = []
    target_texts = []
    for batch in data_loader:
        # source -> text
        raw_input = batch.pop('source')
        model_label = batch.pop('target')
        model_input = {k: v.to(device) for k, v in batch.items()}
        processed_labels = \
        tokenizer(model_label, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)[
            'input_ids']
        processed_labels[processed_labels == 0] = -100
        model_input['labels'] = processed_labels.to(device)
        with torch.no_grad():
            outputs = model(**model_input)

        if do_generate:
            del model_input['labels']
            with torch.no_grad():
                generated_outputs = model.generate(**model_input, min_length=min_output_length,
                                                   max_length=max_output_length, num_beams=num_beams,
                                                   early_stopping=True)
            decoded_outputs = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
            generated_texts += decoded_outputs
            target_texts += model_label
            source_texts += raw_input
        loss = outputs.loss
        total_loss += loss.item()
        loss = loss / gradient_accumulation_steps
        batch_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            progress_bar.set_description("Eval Batch Loss: %f" % (batch_loss))
            batch_loss = 0
            progress_bar.update(1)

        step += 1

    resulting_metrics = {'loss': total_loss / len(data_loader)}
    return resulting_metrics, generated_texts, source_texts


# ## Training and validating scripts
if do_train:
    epoch_progress_bar = tqdm(range(num_epochs))
    text2data_best = 1000000
    text2data_best_metrics = None
    text2data_patience = 0
    data2text_best = 1000000
    data2text_best_metrics = None
    data2text_patience = 0
    for epoch in range(num_epochs):
        epoch_progress_bar.set_description("Cycle/Epoch %d: " % (epoch))
        print("\nTraining: data-text-data direction")
        total_text2data_loss = train_one_direction(model_data2text, model_text2data,
                                                                          triplets_dataloader, num_data_training_steps,
                                                                          optimizer_text2data, lr_scheduler_text2data,
                                                                          'text')
        print('model_text2data - train total loss: %.4f ' % (total_text2data_loss))

        if do_eval and text2data_validation_file != None:
            text2data_dev_metrics, generated_texts, source_texts = eval_model(model_text2data, text2triplets_val_dataloader)
            total_text2data_dev_loss = text2data_dev_metrics['loss']
            text2data_selection_metric = text2data_dev_metrics[selection_metric]

            print('model_text2data - dev total loss: %.4f' % (total_text2data_dev_loss))
            print(text2data_dev_metrics)
            if text2data_patience <= patience and text2data_best - text2data_selection_metric >= delta:
                text2data_best = text2data_selection_metric
                text2data_best_metrics = text2data_dev_metrics
                text2data_patience = 0
                model_text2data.save_pretrained(os.path.join(output_dir, 'text2data-best'))
                tokenizer.save_pretrained(os.path.join(output_dir, 'text2data-best'))
                print("data2text-best saved: epoch/cycle %d" % (epoch))
            else:
                text2data_patience += 1

            # generate for testing with carb.py
            carb_text2data_test_metrics, carb_generated_texts, carb_source_texts = eval_model(model_text2data, carb_text2data_test_dataloader)
            # generation into generation/temp_generation
            temp_generation_dir = "generation/temp_generation"
            if not os.path.exists(temp_generation_dir):
                os.makedirs(temp_generation_dir)
            temp_generation_filepath = os.path.join(temp_generation_dir,
                                                    f'carb_text2data.generations.{date}{preference}_preferred')
            zippedlist = zip(carb_source_texts, carb_generated_texts)
            # print(zippedlist)
            with open(temp_generation_filepath, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(zippedlist)
            # compute auc, precision, recall, f1 then write to eval_log
            # dataset_type should be "dev" or "test", carb has a dev set and a test set
            auc, optimal_f1_point = carb_eval(epoch=epoch, eval_loss=total_text2data_dev_loss, benchmark="carb",
                                              dataset_type="test",
                                              cycle_dataset_name=text2data_filepath.split('/')[-1].split('.csv')[0],
                                              generation_filepath=temp_generation_filepath)

            benchie_text2data_test_metrics, benchie_generated_texts, benchie_source_texts = eval_model(model_text2data,
                                                                                                       benchie_text2data_test_dataloader)
            temp_generation_filepath = os.path.join(temp_generation_dir,
                                                    f'benchie_text2data.generations.{date}{preference}_preferred')
            zippedlist = zip(benchie_source_texts, benchie_generated_texts)
            with open(temp_generation_filepath, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(zippedlist)
            # dataset_type should be "dev" or "test", benchie only has test set
            auc, optimal_f1_point = carb_eval(epoch=epoch, eval_loss=total_text2data_dev_loss, benchmark="benchie",
                                              dataset_type="test",
                                              cycle_dataset_name=text2data_filepath.split('/')[-1].split('.csv')[0],
                                              generation_filepath=temp_generation_filepath)

        if epoch % save_epochs == 0:
            model_text2data.save_pretrained(os.path.join(output_dir, 'text2data-' + str(epoch)))
            print('text2data-' + str(epoch) + ' saved')

        print("\nTraining: text-data-text direction")
        total_data2text_loss = train_one_direction(model_text2data, model_data2text,
                                                                          text_dataloader, num_text_training_steps,
                                                                          optimizer_data2text, lr_scheduler_data2text,
                                                                          'data')
        print('model_data2text - train total loss: %.4f' % (total_data2text_loss))

        if do_eval and data2text_validation_file != None:
            data2text_dev_metrics, _, _ = eval_model(model_data2text, triplets2text_val_dataloader)
            total_data2text_dev_loss = data2text_dev_metrics['loss']
            if selection_metric == 'loss':
                data2text_selection_metric = data2text_dev_metrics[selection_metric]
            else:
                data2text_selection_metric = 1 - data2text_dev_metrics[selection_metric]
            print('model_data2text - dev total loss: %.4f' % (total_data2text_dev_loss))
            print(data2text_dev_metrics)

            if data2text_patience <= patience and data2text_best - data2text_selection_metric >= delta:
                data2text_best = data2text_selection_metric
                data2text_best_metrics = data2text_dev_metrics
                data2text_patience = 0
                model_data2text.save_pretrained(os.path.join(output_dir, 'data2text-best'))
                tokenizer.save_pretrained(os.path.join(output_dir, 'data2text-best'))
                print("data2text-best saved: epoch/cycle %d" % (epoch))
            else:
                data2text_patience += 1

        if epoch % save_epochs == 0:
            model_data2text.save_pretrained(os.path.join(output_dir, 'data2text-' + str(epoch)))
            tokenizer.save_pretrained(os.path.join(output_dir, 'data2text-' + str(epoch)))
            print('data2text-' + str(epoch) + ' saved')

        epoch_progress_bar.update(1)
        if text2data_patience > patience and data2text_patience > patience:
            print("Both models exceed the patience, training terminated")
            break

    print("\nTraining completed")
    if do_eval:
        print("\nBest data2text model:")
        print(data2text_best_metrics)
        del model_data2text

        print("\nBest text2data model:")
        print(text2data_best_metrics)
        del model_text2data




