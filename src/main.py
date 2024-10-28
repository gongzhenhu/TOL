import os
import argparse
import logging
import random
from typing import List, Union
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from functools import partial
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor

from transformers import AdamW, T5Tokenizer
from t5 import MyT5ForConditionalGeneration
from t5_score import MyT5ForConditionalGenerationScore
from transformers import get_linear_schedule_with_warmup

import pickle
from data_utils import ABSADataset
from data_utils_single import ABSADataset_Single
from eval_utils import compute_scores
from const import *
from itertools import chain
from data_utils import read_line_examples_from_file
from eval_utils import compute_scores, extract_spans_para

# configure logging at the root level of Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

# configure logging on module level, redirect to file
logger = logging.getLogger("pytorch_lightning.core")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def init_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../data/", type=str)
    parser.add_argument("--task", default="unified", choices=['asqp', 'acos', 'aste', 'tasd', 'unified'])
    parser.add_argument("--dataset", default='seed5', type=str, help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='../pretrained/t5-base', type=str, help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir",  default='../outputs', type=str, help="Output directory")
    parser.add_argument("--load_ckpt_name", default=None, type=str, help="load ckpt path")


    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_inference", default=True, help="Whether to run inference with trained checkpoints")

    parser.add_argument("--max_seq_length", default=200, type=int)
    parser.add_argument("--train_batch_size", default=2, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_train_epochs", default=2, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=25, help="random seed for initialization")

    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--beam_size", default=1, type=int)
    parser.add_argument("--save_top_k", default=1, type=int)
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    parser.add_argument("--target_type", default="rank", choices=["rank", "rand", "heuristic"], type=str)
    parser.add_argument("--lowercase", action='store_true')
    parser.add_argument("--multi_task", action='store_true')
    parser.add_argument("--constrained_decode",action="store_true", help='constrained decoding when evaluating')
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--data_ratio", default=1.0, type=float, help="low resource data ratio")
    parser.add_argument("--single_view_type", default="heuristic", choices=["rank", "rand", "heuristic"], type=str)
    parser.add_argument("--eval_data_split", default='test', choices=["test", "dev"], type=str)

    parser.add_argument("--order", default='sortindex', choices=["random", "sortalpha", "mvp", "sortwordnum", "mvp1", 'none', "sortindex1", "sortindex"], type=str)
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--adaptive_order", default='None', choices=['None', "ada"], type=str)
    parser.add_argument("--num_pretrained_epochs", default=1, type=int)
    parser.add_argument("--pretrained_warmup_steps", default=0.0, type=float)
    parser.add_argument("--pretrained_learning_rate", default=1e-4, type=float)

    args = parser.parse_args()


    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    return args

class T5FineTuner(pl.LightningModule):

    def __init__(self, config, tfm_model, tokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=['tfm_model'])
        self.config = config
        self.model = tfm_model
        self.tokenizer = tokenizer

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        
    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(input_ids=batch["source_ids"],
                       attention_mask=batch["source_mask"],
                       labels=lm_labels,
                       decoder_attention_mask=batch['target_mask'])

        loss = outputs[0]
        return loss
    
    def evaluate(self, batch, stage=None):

        outs = self.model.generate(
                input_ids=batch['source_ids'],
                attention_mask=batch['source_mask'],
                max_length=self.config.max_seq_length,
                num_beams=self.config.beam_size,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                prefix_allowed_tokens_fn=partial(
                    self.prefix_allowed_tokens_fn, self.config.task, self.config.dataset,
                    batch['source_ids']) if args.constrained_decode else None,
            )


        dec = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in outs.sequences
        ]
        target = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]

        loss = self._step(batch)

        return {f"{stage}_loss": loss, f'{stage}_dec': dec, f'{stage}_target': target}

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def validation_epoch_end(self, outputs):
        dec = list(chain(*[x['val_dec'] for x in outputs]))
        tar = list(chain(*[x['val_target'] for x in outputs]))
        assert len(dec) == len(tar)
        scores, _, _ = compute_scores(dec, tar, verbose=False)
        f1 = torch.tensor(scores[0]['f1'], dtype=torch.float64)
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_f1", f1, prog_bar=True)
        self.log("avg_val_loss", avg_loss, prog_bar=True)
        return {"val_f1":f1, 'avg_val_loss':avg_loss}

    
    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                        lr=self.config.learning_rate,
                        eps=self.config.adam_epsilon)
        scheduler = {
            "scheduler":
            get_linear_schedule_with_warmup(optimizer,
                                            **self.config.lr_scheduler_init),
            "interval":
            "step",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self, model=None):
        print("load training data.")
        train_dataset = ABSADataset(tokenizer=self.tokenizer,
                                    model=model,
                                    task_name=self.config.task,
                                    data_name=self.config.dataset,
                                    data_type="train",
                                    args=self.config)
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=2)

        return dataloader

    def val_dataloader(self, model=None):
        val_dataset = ABSADataset(tokenizer=self.tokenizer,
                                  model=model,
                                    task_name=self.config.task,
                                data_name=self.config.dataset,
                                data_type="dev",
                                args=self.config)
        return DataLoader(val_dataset,
                        batch_size=self.config.eval_batch_size,
                        num_workers=2)
    
    def test_dataloader(self, model=None):
        test_dataset = ABSADataset(tokenizer=self.tokenizer,
                                  model=model,
                                    task_name=self.config.task,
                                data_name=self.config.dataset,
                                data_type="test",
                                args=self.config)
        return DataLoader(test_dataset,
                        batch_size=self.config.eval_batch_size,
                        num_workers=2)

    def train_dataloader_single(self):
        print("load training data.")
        train_dataset = ABSADataset_Single(tokenizer=self.tokenizer,
                                    task_name=self.config.task,
                                    data_name=self.config.dataset,
                                    data_type="train",
                                    args=self.config)

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=2)

        return dataloader



    def prefix_allowed_tokens_fn(self, task, data_name, source_ids, batch_id,
                                input_ids):
        """
        Constrained Decoding
        # ids = self.tokenizer("text", return_tensors='pt')['input_ids'].tolist()[0]
        """
        if not os.path.exists('./force_tokens.json'):
            dic = {"cate_tokens":{}, "all_tokens":{}, "sentiment_tokens":{}, 'special_tokens':[]}
            for task in force_words.keys():
                dic["all_tokens"][task] = {}
                for dataset in force_words[task].keys():
                    cur_list = force_words[task][dataset]
                    tokenize_res = []
                    for w in cur_list:
                        tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0])
                    dic["all_tokens"][task][dataset] = tokenize_res
            for k,v in cate_list.items():
                tokenize_res = []
                for w in v:
                    tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0]) 
                dic["cate_tokens"][k] = tokenize_res
            sp_tokenize_res = []
            for sp in ['great', 'ok', 'bad']:
                sp_tokenize_res.extend(self.tokenizer(sp, return_tensors='pt')['input_ids'].tolist()[0])
            for task in force_words.keys():
                dic['sentiment_tokens'][task] = sp_tokenize_res
            dic['sentiment_tokens'] = sp_tokenize_res
            special_tokens_tokenize_res = []
            for w in ['[O','[A','[S','[C','[SS']:
                special_tokens_tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0]) 
            special_tokens_tokenize_res = [r for r in special_tokens_tokenize_res if r != 784]
            dic['special_tokens'] = special_tokens_tokenize_res
            import json
            with open("force_tokens.json", 'w') as f:
                json.dump(dic, f, indent=4)

        to_id = {
            'OT': [667],
            'AT': [188],
            'SP': [134],
            'AC': [254],
            'SS': [4256],
            'EP': [8569],
            '[': [784],
            ']': [908],
            'it': [34],
            'null': [206,195]
        }

        left_brace_index = (input_ids == to_id['['][0]).nonzero()
        right_brace_index = (input_ids == to_id[']'][0]).nonzero()
        num_left_brace = len(left_brace_index)
        num_right_brace = len(right_brace_index)
        last_right_brace_pos = right_brace_index[-1][
            0] if right_brace_index.nelement() > 0 else -1
        last_left_brace_pos = left_brace_index[-1][
            0] if left_brace_index.nelement() > 0 else -1
        cur_id = input_ids[-1]

        if cur_id in to_id['[']:
            return force_tokens['special_tokens']
        elif cur_id in to_id['AT'] + to_id['OT'] + to_id['EP'] + to_id['SP'] + to_id['AC']:  
            return to_id[']']  
        elif cur_id in to_id['SS']:  
            return to_id['EP'] 

        # get cur_term
        if last_left_brace_pos == -1:
            return to_id['['] + [1]   # start of sentence: [
        elif (last_left_brace_pos != -1 and last_right_brace_pos == -1) \
            or last_left_brace_pos > last_right_brace_pos:
            return to_id[']']  # ]
        else:
            cur_term = input_ids[last_left_brace_pos + 1]

        ret = []
        if cur_term in to_id['SP']:  # SP
            ret = force_tokens['sentiment_tokens'][task]
        elif cur_term in to_id['AT']:  # AT
            force_list = source_ids[batch_id].tolist()
            if task != 'aste':  
                force_list.extend(to_id['it'] + [1])  
            ret = force_list  
        elif cur_term in to_id['SS']:
            ret = [3] + to_id[']'] + [1]
        elif cur_term in to_id['AC']:  # AC
            ret = force_tokens['cate_tokens'][data_name]
        elif cur_term in to_id['OT']:  # OT
            force_list = source_ids[batch_id].tolist()
            if task == "acos":
                force_list.extend(to_id['null'])  # null
            ret = force_list
        else:
            raise ValueError(cur_term)

        if num_left_brace == num_right_brace:
            ret = set(ret)
            ret.discard(to_id[']'][0]) # remove ]
            for w in force_tokens['special_tokens']:
                ret.discard(w)
            ret = list(ret)
        elif num_left_brace > num_right_brace:
            ret += to_id[']'] 
        else:
            raise ValueError
        ret.extend(to_id['['] + [1]) # add [
        return ret
    

def evaluate(model, task, data, data_type):
    """
    Compute scores given the predictions and gold labels
    """
    tasks, datas, sents, _ = read_line_examples_from_file(
        f'../data/{task}/{data}/{data_type}.txt', task, data, lowercase=False)

    outputs, targets, probs = [], [], []

    dataset = ABSADataset(model.tokenizer,
                            task_name=task,
                            model=None,
                            data_name=data,
                            data_type=data_type,
                            args=args)
    data_loader = DataLoader(dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
    device = torch.device('cuda:0')
    model.model.to(device)
    model.model.eval()

    for batch in tqdm(data_loader):
        # beam search
        outs = model.model.generate(
            input_ids=batch['source_ids'].to(device),
            attention_mask=batch['source_mask'].to(device),
            max_length=args.max_seq_length,
            num_beams=args.beam_size,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            prefix_allowed_tokens_fn=partial(
                model.prefix_allowed_tokens_fn, task, data,
                batch['source_ids']) if args.constrained_decode else None,
        )

        dec = [
            model.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in outs.sequences
        ]
        target = [
            model.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]
        outputs.extend(dec)
        targets.extend(target)

    labels_counts = Counter([len(l.split('[SSEP]')) for l in outputs])
    print("pred labels count", labels_counts)

    scores, all_labels, all_preds = compute_scores(outputs,
                                                   targets,
                                                   verbose=True)
    return scores

from pytorch_lightning.callbacks import Callback

class SaveLastTwoEpochsCallback(Callback):
    def __init__(self, dirpath):
        super().__init__()
        self.dirpath = dirpath
        self.last_two_checkpoints = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        checkpoint_path = os.path.join(self.dirpath, f"final_pretrained_model_{epoch:d}.ckpt")

        torch.save(trainer.model.model.state_dict(), checkpoint_path)

        self.last_two_checkpoints.append(checkpoint_path)

        if len(self.last_two_checkpoints) > 2:
            oldest_checkpoint = self.last_two_checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)
                print(f"Deleted old checkpoint: {oldest_checkpoint}")

def single_train_function(args):
    print("\n", "=" * 30, f"NEW SINGLE EXP: {args.task} on {args.dataset}", "=" * 30, "\n")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)


    print("\n****** Conduct Training ******")

    tfm_model = MyT5ForConditionalGeneration.from_pretrained(args.model_name_or_path, local_files_only=True if args.model_name_or_path != "t5-base" else False)
    model = T5FineTuner(args, tfm_model, tokenizer)

    # load data
    train_loader = model.train_dataloader_single()
    # config optimizer
    t_total = ((len(train_loader.dataset) //
                (args.train_batch_size * max(1, args.n_gpu))) //
                args.gradient_accumulation_steps *
                float(args.num_pretrained_epochs))
    
    args.lr_scheduler_init = {
        "num_warmup_steps": args.pretrained_warmup_steps,
        "num_training_steps": t_total
    }

    save_last_two_callback = SaveLastTwoEpochsCallback(dirpath=args.output_dir)
    train_params = dict(
        accelerator="gpu",
        devices=1,
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        max_epochs=args.num_pretrained_epochs,
        check_val_every_n_epoch=10000,   
        callbacks=[save_last_two_callback]
    )    
    trainer = pl.Trainer(**train_params)
    trainer.fit(model, train_dataloaders=model.train_dataloader_single(), val_dataloaders=None)
    print("Finish training and saving the model!")
    checkpoint_1 = torch.load(f'{args.output_dir}/final_pretrained_model_{args.num_pretrained_epochs-1}.ckpt')
    checkpoint_2 = torch.load(f'{args.output_dir}/final_pretrained_model_{args.num_pretrained_epochs-2}.ckpt')
    return checkpoint_1, checkpoint_2

    

def train_function(args):

    set_seed(args.seed)
    order_model = None

    if args.do_train:
        if args.adaptive_order != "None" or args.train_order != "None":
            checkpoint_single_1, checkpoint_single_2 = single_train_function(args)
            order_model_1 = MyT5ForConditionalGenerationScore.from_pretrained(args.model_name_or_path, local_files_only=True if args.model_name_or_path != "t5-base" else False)
            order_model_2 = MyT5ForConditionalGenerationScore.from_pretrained(args.model_name_or_path, local_files_only=True if args.model_name_or_path != "t5-base" else False)
            order_model_1.load_state_dict(checkpoint_single_1)
            order_model_2.load_state_dict(checkpoint_single_2)

        print("\n", "=" * 30, f"NEW EXP: {args.task} on {args.dataset}", "=" * 30, "\n")
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

        set_seed(args.seed)

        print("\n****** Conduct Training ******")
        tfm_model = MyT5ForConditionalGeneration.from_pretrained(args.model_name_or_path, local_files_only=True if args.model_name_or_path != "t5-base" else False)
        if args.pretrained:
            tfm_model.load_state_dict(checkpoint_single_1) 
        model = T5FineTuner(args, tfm_model, tokenizer)
        
        # torch.cuda.empty_cache()
        # load data
        train_loader = model.train_dataloader()
        # config optimizer
        t_total = ((len(train_loader.dataset) //
                    (args.train_batch_size * max(1, args.n_gpu))) //
                   args.gradient_accumulation_steps *
                   float(args.num_train_epochs))
        args.lr_scheduler_init = {
            "num_warmup_steps": args.warmup_steps,
            "num_training_steps": t_total
        }

        train_params = dict(
            accelerator="gpu",
            devices=1,
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            check_val_every_n_epoch=2,
        )    
        trainer = pl.Trainer(**train_params)
        trainer.fit(model, train_dataloaders=model.train_dataloader((order_model_1, order_model_2)), val_dataloaders=model.val_dataloader())

        model.model.save_pretrained(os.path.join(args.output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
        print("Finish training and saving the model!")

    if args.do_inference:
        print("\n****** Conduct inference on trained checkpoint ******")

        model_path = os.path.join(args.output_dir, "final")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        tfm_model = MyT5ForConditionalGeneration.from_pretrained(model_path)
        model = T5FineTuner(args, tfm_model, tokenizer)

        log_file_path = os.path.join(args.output_dir, "result.txt")
        with open(log_file_path, "a+") as f:
            config_str = f"seed: {args.seed}, beam: {args.beam_size}, constrained: {args.constrained_decode}\n"
            print(config_str)
            f.write(config_str)
            scores = evaluate(model,
                                  args.task,
                                  args.dataset,
                                  data_type=args.eval_data_split)
            for i in range(4):
                    exp_results = "datatype: {} class:{} precision: {:.2f} recall: {:.2f} F1 = {:.2f}".format(
                    args.eval_data_split, i, scores[i]['precision'], scores[i]['recall'], scores[i]['f1'])
                    print(exp_results)
                    f.write(exp_results + "\n")
            f.write("\n")
            f.flush()

    return scores

if __name__ == "__main__":
    
    args = init_args()
    train_function(args)
    