from torch.utils.data import Dataset
from itertools import permutations
import random
from const import *

def read_line_examples_from_file(data_path,
                                 task_name,
                                 data_name,
                                 lowercase,
                                 silence=True):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    tasks, datas = [], []
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if lowercase:
                line = line.lower()
            if "unified" in task_name:
                _task, _data, line = line.split("\t")
                tasks.append(_task)
                datas.append(_data)
            else:
                tasks.append(task_name)
                datas.append(data_name)
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return tasks, datas, sents, labels

def get_orders(task, data, args, sents, labels):

    if args.single_view_type == 'rank':
        orders = optim_orders_all[task][data]
    elif args.single_view_type == 'rand':
        orders = [random.Random(args.seed).choice(
            optim_orders_all[task][data])]
    elif args.single_view_type == "heuristic":
        orders = heuristic_orders[task]
    return orders

def parse_aste_tuple(_tuple, sent):
    if isinstance(_tuple[0], str):
        res = _tuple
    elif isinstance(_tuple[0], list):
        # parse at
        start_idx = _tuple[0][0]
        end_idx = _tuple[0][-1] if len(_tuple[0]) > 1 else start_idx
        at = ' '.join(sent[start_idx:end_idx + 1])

        # parse ot
        start_idx = _tuple[1][0]
        end_idx = _tuple[1][-1] if len(_tuple[1]) > 1 else start_idx
        ot = ' '.join(sent[start_idx:end_idx + 1])
        res = [at, ot, _tuple[2]]
    else:
        print(_tuple)
        raise NotImplementedError
    return res

def get_task_tuple(_tuple, task):
    if task == "aste":
        at, ot, sp = _tuple
        ac = None
    elif task == "tasd":
        at, ac, sp = _tuple
        ot = None
    elif task in ["asqp", "acos"]:
        at, ac, sp, ot = _tuple
    else:
        raise NotImplementedError

    if sp:
        sp = sentword2opinion[sp.lower()] if sp in sentword2opinion \
            else senttag2opinion[sp.lower()]  # 'POS' -> 'good'
    if at and at.lower() == 'null':  # for implicit aspect term
        at = 'it'

    return at, ac, sp, ot

def add_prompt(sent, task, data_name, args):
    if args.multi_task:
        # add task and data prefix
        sent = [task, ":", data_name, ":"] + sent
    return sent

def get_para_targets_single(sents, labels, data_name, data_type, task, args):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    new_sents = []

    optim_orders = get_orders(task, data_name, args, sents, labels)[:1]

    for i in range(len(sents)):
        label = labels[i]
        cur_sent = sents[i]
        cur_sent_str = " ".join(cur_sent)

        # ASTE: parse at & ot
        if task == 'aste':
            assert len(label[0]) == 3
            parsed_label = []
            for _tuple in label:
                parsed_tuple = parse_aste_tuple(_tuple, sents[i])
                parsed_label.append(parsed_tuple)
            label = parsed_label
            

        quad_list = []
        for _tuple in label:
            at, ac, sp, ot = get_task_tuple(_tuple, task)
            element_dict = {"[A]": at, "[O]": ot, "[C]": ac, "[S]": sp}
            token_end = 3

            element_list = []
            for key in optim_orders[0].split(" "):
                element_list.append("{} {}".format(key, element_dict[key]))

            x = permutations(element_list)
            permute_object = {}
            for each in x:
                order = []
                content = []
                for e in each:
                    order.append(e[0:token_end])
                    content.append(e[token_end:])
                order_name = " ".join(order)
                content = " ".join(content)
                permute_object[order_name] = [content, " ".join(each)]

            quad_list.append(permute_object)

        for o in optim_orders:
            tar = []
            for each_q in quad_list:
                tar.append(each_q[o][1])

            # add prompt
            new_sent = add_prompt(cur_sent, task, data_name, args)
            for each_label in tar:
                new_sents.append(new_sent)
                targets.append(each_label)

    return new_sents, targets

def get_transformed_io_single(data_path, data_name, task_name, data_type, args):
    """
    The main function to transform input & target according to the task
    """
    tasks, datas, sents, labels = read_line_examples_from_file(
        data_path, task_name, data_name, args.lowercase)

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]

    # low resource
    if args.data_ratio != 1.0:
        num_sample = int(len(inputs) * args.data_ratio)
        sample_indices = random.sample(list(range(0, len(inputs))), num_sample)
        sample_inputs = [inputs[i] for i in sample_indices]
        sample_labels = [labels[i] for i in sample_indices]
        inputs, labels = sample_inputs, sample_labels
        print(
            f"Low resource: {args.data_ratio}, total train examples = {num_sample}")
        if num_sample <= 20:
            print("Labels:", sample_labels)

    new_inputs, targets = get_para_targets_single(inputs, labels, data_name, data_type, args.task, args)
        
    print(len(inputs), len(new_inputs), len(targets))
    return new_inputs, targets

def get_transformed_io_unified_single(data_path, data_name, task_name, data_type, args):
    """
    The main function to transform input & target according to the task
    """
    tasks, datas, sents, labels = read_line_examples_from_file(
        data_path, task_name, data_name, args.lowercase)
    sents = [s.copy() for s in sents]
    new_inputs, targets = [], []
    for task, data, sent, label in zip(tasks, datas, sents, labels):
        new_input, target = get_para_targets_single([sent], [label], data, data_type, task, args)
        new_inputs.extend(new_input)
        targets.extend(target)

    print("Ori sent size:", len(sents))
    print("Input size:", len(new_inputs), len(targets))
    print("Examples:")
    print(new_inputs[:10])
    print(targets[:10])

    return new_inputs, targets

class ABSADataset_Single:

    def __init__(self, tokenizer, task_name, data_name, data_type, args):
        
        self.tokenizer = tokenizer
        self.data_path = f'{args.data_path}/{task_name}/{data_name}/{data_type}.txt'
        self.max_len = args.max_seq_length
        self.task_name = task_name
        self.data_name = data_name
        self.data_type = data_type
        self.args = args

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):

        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze
        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }

    def _build_examples(self):
        
        if self.args.multi_task:
            inputs, targets = get_transformed_io_unified_single(self.data_path,
                                             self.data_name,
                                             self.task_name,
                                             self.data_type,
                                             self.args)
        else:
            inputs, targets = get_transformed_io_single(self.data_path,
                                    self.data_name,
                                    self.task_name,
                                    self.data_type,
                                    self.args)
        for i in range(len(inputs)):
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus([input], 
                                                               max_length=self.max_len,
                                                               padding='max_length',
                                                               truncation=True,
                                                               return_tensors='pt')
            tokenized_target = self.tokenizer.batch_encode_plus([target],
                                                                max_length=self.max_len,
                                                                padding="max_length",
                                                                truncation=True,
                                                                return_tensors="pt",
                                                                add_special_tokens=False)
            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target) 
