import os.path as osp
import random
import json, yaml
import copy

import numpy as np
from scipy import signal
import soundfile as sf

import torch
import torchaudio
from torch.utils.data import Dataset
import whisper
from slam_llm.utils.compute_utils import calculate_output_length_1d
from slam_llm.datasets.speech_dataset import classify_en_cn

from funasr.tokenizer.g2p_tokenizer import G2PTokenizer



class Phoneme2TextDatasetJsonl(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer

        # self.data_list = contents
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt = dataset_config.get("prompt", None)
        self.prompt = "You are a speech recognition model. Transcribe English-Chinese code switch according to pronunciation into text without any punctuation marks. "

     
        self.prompt_template = "USER: {}\n ASSISTANT:"
        self.answer_template = "{}"
        self.fix_length_text = dataset_config.get("fix_length_text", -1) # for Q-former
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.input_type = dataset_config.get("input_type", None)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]"

        symbol_table_path = "/ssd/zhuang/code/FunASR/funasr/tokenizer/g2p_tokenizer_utils/vocab.json"
        self.phonme_tokenizer = G2PTokenizer(symbol_table_path)
      
        self.split = split
        self.data_list = []
        if split == "train":
            with open(dataset_config.train_data_path, encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)
        else:
            with open(dataset_config.val_data_path, encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)


    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):
        return data_dict["target_len"] if "target_len" in data_dict else 0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_dict = self.data_list[index]
        target = data_dict.get("target", None)
        task = data_dict.get("prompt", "Phoneme2Text")
        key = data_dict.get("key", None)

        text = classify_en_cn(target)
        output = ''
        regular_target = ''
        for language, txt in text:
            regular_target += txt
            phone, _ = self.phonme_tokenizer.tokenize(txt, text_language=language)
            if output != '':
                output += '|'
            output += phone

        instruct = output.replace('|', '') + ". "
        target = regular_target

        prompt = "The pronunciation is uncertain, but it could be " + instruct + self.prompt
        prompt = self.prompt_template.format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)

        # if self.input_type == "raw":
        #     instruct_length = 0
        #     prompt = instruct + prompt
        #     prompt_ids = self.tokenizer.encode(prompt)
        #     prompt_length = len(prompt_ids)
        # elif self.input_type == "features":
        #     instruct_ids = self.instruct_tokenizer.encode(instruct)
        #     instruct_length = len(instruct_ids)
        instruct_ids = None

        if self.fix_length_text > 0: # for Q-former
            instruct_length = self.fix_length_text
        instruct_length = 0
        instruct_pseudo = torch.full((instruct_length,), -1) # placeholder

        if self.inference_mode:
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
            example_ids = torch.cat((instruct_pseudo, prompt_ids))  # [audio,prompt]
            example_mask = example_ids.ge(-1)  # [True,True]

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "instruct_ids": instruct_ids if self.input_type == "features" else None,
                "instruct_length": instruct_length,
            }

        answer = self.answer_template.format(target)
        example = prompt + answer
        example_ids = self.tokenizer.encode(example)  # [prompt,answer]
        example_ids.append(self.tokenizer.eos_token_id)  # [prompt,answer,eos]
        example_ids = torch.tensor(example_ids, dtype=torch.int64)
        example_ids = torch.cat((instruct_pseudo, example_ids))  # [instruct,prompt,answer,eos]

        labels_ids = copy.deepcopy(example_ids)  # [instruct,prompt,answer,eos]
        labels_ids[:instruct_length + prompt_length] = -1  # [-1,-1,answer,eos]
        example_mask = example_ids.ge(-1)  # [True,True,True,True]

        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        example_ids[~example_mask] = 0  # [instruct,prompt,answer,eos]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "instruct_ids": instruct_ids if self.input_type == "features" else None,
            "instruct_length": instruct_length,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None
        input_ids_max_length = max([s['input_ids'].shape[0] for s in samples])
        input_ids = torch.stack([self.pad(s['input_ids'], input_ids_max_length, self.tokenizer.pad_token_id)
                                 for s in samples])
        attention_mask = torch.stack([self.pad(s['attention_mask'], input_ids_max_length, False)
                                      for s in samples])
        if self.input_type == "raw":
            instruct_max_length = 0
            instruct_ids = None
        elif self.input_type == "features":
            instruct_max_length = max([s['instruct_ids'].shape[0] for s in samples])
            instruct_ids = torch.stack([self.pad(s['instruct_ids'], instruct_max_length, self.instruct_tokenizer.pad_token_id)
                                  for s in samples])
            instruct_mask = torch.zeros(len(samples), instruct_max_length)
            for line, sample in enumerate(samples):
                instruct_mask[line, :sample['instruct_length']] = 1
    
        modality_mask = torch.zeros_like(attention_mask)
        for line, sample in enumerate(samples):
            modality_mask[line, :sample['instruct_length']] = 1

        if self.inference_mode:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "instruct_ids": instruct_ids if self.input_type == "features" else None,
                "instruct_mask": instruct_mask if self.input_type == "features" else None,
                "modality_mask": modality_mask,
            }

        labels = torch.stack([self.pad(s['labels'], input_ids_max_length, self.IGNORE_INDEX)
                              for s in samples])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "instruct_ids": instruct_ids if self.input_type == "features" else None,
            "instruct_mask": instruct_mask if self.input_type == "features" else None,
            "modality_mask": modality_mask
        }



def get_phoneme2text_dataset(dataset_config, tokenizer, split):
    dataset = Phoneme2TextDatasetJsonl(dataset_config, tokenizer, split)

    return dataset
