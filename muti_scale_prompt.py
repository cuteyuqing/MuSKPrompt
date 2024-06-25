import json
import random
from datetime import datetime
from time import sleep
import logging
import argparse
from tqdm import tqdm
import csv
import os
import numpy as np
import torch
import pickle
import copy
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from utils.dataset import *
from utils.template import *
from utils.memory import Memory_bank
import time



os.environ["TOKENIZERS_PARALLELISM"] = "false"  
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="KNN Prompting.")
    parser.add_argument(
        "--llm_dir",
        type=str,
        default='./llm/gpt2-xl',
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default='./data/',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='snli',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n_train_shot",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--n_demo_shot",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--n_prompt_icl",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--cmd_p",
        type=int,
        default=-1,
        help='central moment discrepancy p, -1 means kl divergence',
    )
    parser.add_argument(
        "--use_class_center",
        type=int,
        default=1,
        help='whether to use class center for cmd',
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help='class-wise weight for score,0-1',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='device to run the model',
    )
    
    parser.add_argument(
        "--more_icl",
        action='store_true',
        help='indicating that all training sets were used as icl prompt',
    )
    parser.add_argument(
        "--more_memory",
        action='store_true',
        help='',
    )
    
    parser.add_argument(
        "--t",
        type=float,
        default=1e-8,
        help='temperature value for score, 0-1, making the difference in scores closer together more pronounced',
    )
    parser.add_argument(
        "--use_knn",
        type=int,
        default=1,
        help='whether to use knn for inference',
    )
    parser.add_argument(
        "--prompt_level",
        type=int,
        default=2,
        help='The number of layers of prompt.',
    )   
    parser.add_argument(
        "--voteorsum",
        type=int,
        default=0,
        help='Calculation of the final prediction, which defaults to sum.',
    )   
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./output',
    )
    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def llm_gen(model, prompt, tokenizer, max_context_len):
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt", padding=True).to(device=model.device)
    if inputs['input_ids'].shape[1] > max_context_len:
        inputs['input_ids'] = inputs['input_ids'][:, -max_context_len:]
        inputs['attention_mask'] = inputs['attention_mask'][:, -max_context_len:]
    with torch.no_grad():
        logits = model.forward(input_ids=inputs['input_ids'],
                               attention_mask=inputs['attention_mask'],
                               return_dict=True).logits.detach().cpu()
    gen_logits = logits[:, -1, :]

    return gen_logits


def main():    
    args = parse_args()
    np.random.seed(args.seed)
    if not os.path.exists(os.path.join('./log/ours',args.dataset)):
        os.mkdir(os.path.join('./log/ours',args.dataset))
    args.n_prompt_icl = args.n_train_shot - args.n_demo_shot
    if args.n_prompt_icl <= 0:
        raise Exception("Num. of demonstration must be set smaller than num. of training.")

    args.knn = min(args.knn, args.n_prompt_icl)  
    print(args)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_dir, use_fast=False)
    # set pad token ids for batched inference cus gpt2 does not have one
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_config = AutoConfig.from_pretrained(args.llm_dir)
    model = AutoModelForCausalLM.from_pretrained(args.llm_dir)
    model.to(device)
    model.eval()

    if 'gpt2' in args.llm_dir:
        # max_context_len = 1024
        max_context_len = 1022
    else:
        max_context_len = 2048

    # prepare dataset
    if args.dataset == 'sst2':
        AutoDataset = SST2Dataset
    elif args.dataset == 'yelpp':
        AutoDataset = YELPPDataset
    elif args.dataset == 'mrpc':
        AutoDataset = MRPCDataset
    elif args.dataset == 'snli':
        AutoDataset = SNLIDataset
    elif args.dataset == 'subj':
        AutoDataset = SUBJDataset
    elif args.dataset == 'agnews':
        AutoDataset = AGNEWSDataset
    elif args.dataset == 'cb':
        AutoDataset = CBDataset
    elif args.dataset == 'cr':
        AutoDataset = CRDataset
    elif args.dataset == 'dbpedia':
        AutoDataset = DBPEDIADataset
    elif args.dataset == 'mpqa':
        AutoDataset = MPQADataset
    elif args.dataset == 'mr':
        AutoDataset = MRDataset
    elif args.dataset == 'rte':
        AutoDataset = RTEDataset
    elif args.dataset == 'trec':
        AutoDataset = TRECDataset

    datadir = os.path.join(args.data_dir, args.dataset)
    
    dev_data = AutoDataset(datadir, mode='dev')
    anchor_data = AutoDataset(datadir, mode='train')
    anchor_data.subsamplebyshot(args.n_train_shot, args.seed)
    anchor_labels = []
    dev_labels = [dev_data.label2id[ins['label']] for ins in dev_data.data]
    test_pred = []
    test_score = []
    logits_output = []
    start_time = time.time()
    

    n_demo_shot = [1<<i for i in range(args.prompt_level)]
    if(args.prompt_level>=1):
        args.prompt_level = 3
        n_demo_shot = [1<<i for i in range(args.prompt_level)]
    else:
        n_demo_shot = [args.n_demo_shot]
    
    
    for args.n_demo_shot in n_demo_shot:
        print('*************************************')
        print('The number of examples in this layer is {}'.format(args.n_demo_shot))
        
        logits_tmp = []
        args.n_prompt_icl = args.n_train_shot - args.n_demo_shot
        if(args.n_prompt_icl<=0):
            raise Exception("num. of demonstration must be set smaller than num. of training.")

        train_data = copy.deepcopy(anchor_data)

    
        train_data.subsamplebyshot(args.n_demo_shot, 77)
        prompt_prefix = make_prompt(train_data, args.dataset, mode='train')
        ################
        print('train_data.data', len(train_data.data))
        print('anchor_data.data', len(anchor_data.data))
        label2id = dev_data.label2id
        logger.info(f"===== build anchor store of {anchor_data.__len__()} anchor examples =====")
        
        anchor_store = Memory_bank(K=anchor_data.__len__(),
                                dim=model_config.vocab_size,
                                knn=args.knn,
                                n_class=len(label2id))
        label_tmp = []     
        for ins in tqdm(anchor_data.data, total=anchor_data.__len__()):
            labels = label2id[ins['label']]
            prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
            # variables gen_logits are only on cpu devices
            # gen_logits.shape torch.FloatTensor[1, 50257]
            gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
            label_tmp.append(labels)
            logits_tmp = gen_logits.numpy() if len(logits_tmp) == 0 else np.concatenate((logits_tmp, gen_logits.numpy()), axis=0)
            anchor_store.enqueue(torch.softmax(gen_logits, dim=-1), torch.tensor(labels))       
        anchor_labels = np.expand_dims(np.asarray(label_tmp), axis=0) if len(anchor_labels) == 0 else np.concatenate((anchor_labels, np.expand_dims(np.asarray(label_tmp), axis=0)), axis=0)
        # calculate class level knowledge after storing all instance level knowledge
        anchor_store.compute_class_center()  
        
        logger.info(f"===== eval on {dev_data.__len__()} dev examples =====")
        dev_pred = [] # [256]
        dev_score = [] # [256, n_class]
        for ins in tqdm(dev_data.data, total=dev_data.__len__()):
            prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
            gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
            logits_tmp = np.concatenate((logits_tmp, gen_logits.numpy()), axis=0)
            list1, list2 = anchor_store.knn_infer(torch.softmax(gen_logits, dim=-1), args)
            dev_pred.extend(list1)
            if(not args.use_knn):
                dev_score.append(list2)
        # logits_output.shape [prompt_level, len(anchor_shot)+256, 50257]
        logits_output = np.expand_dims(logits_tmp, axis=0) if len(logits_output) == 0 else np.concatenate((logits_output, np.expand_dims(logits_tmp,axis=0)), axis=0)
        dev_pred = np.asarray(dev_pred)
        dev_score = np.asarray(dev_score)
        test_pred = np.expand_dims(dev_pred, axis=0) if len(test_pred) == 0 else np.concatenate((test_pred, np.expand_dims(dev_pred, axis=0)), axis=0)
        test_score = np.expand_dims(dev_score, axis=0) if len(test_score) == 0 else np.concatenate((test_score, np.expand_dims(dev_score, axis=0)), axis=0)
    
    test_pred = test_pred.reshape(args.prompt_level,-1)
    print('test_pred shape', test_pred.shape) 
    print('test_score shape', test_score.shape) 
    dev_pred = []
    dev_score = 0
    dev_score = np.sum(test_score, axis=0) 
    if(args.voteorsum):
        print('sum............................')
        dev_pred = np.argmin(dev_score, axis=1).tolist() 
    else:
        print('vote............................')
        test_pred = np.transpose(test_pred, axes=(1,0))
        
        for row in test_pred: 
            counts = np.bincount(row)
            most_class = np.argmax(counts)
            dev_pred.append(most_class)
    print('dev_score shape:', dev_score.shape)

    end_time = time.time()
    print('Done. Total time: {} (mins)'.format((end_time - start_time) / 60))


    dev_correct = [1 if dev_labels[i] == dev_pred[i] else 0 for i in range(len(dev_labels))]
    acc = sum(dev_correct) / len(dev_labels)
    logger.info(f"Acc: {acc}")
    


    save_results_file = os.path.join(args.output_dir, 'muti_scale_prompt.csv'.format(args.dataset))
    csv_exists = os.path.isfile(save_results_file)
    with open(save_results_file, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not csv_exists:
            csvwriter.writerow(['dataset', 'llm', 'n_train_shot', 'seed', 'knn', 'alpha',
                                'use_class_center', 'voteorsum', 'prompt_level', 'acc'])
        csvwriter.writerow([args.dataset,
                            args.llm_dir,
                            args.n_train_shot,
                            args.seed,
                            args.knn,
                            args.alpha,
                            args.use_class_center,
                            args.voteorsum,
                            args.prompt_level,
                            acc])
    # Write logits_output to pickle once
    # Visualization of experimental preparations.
    
    # if(not os.path.exists(os.path.join(args.output_dir, 'logits_result', args.dataset))):
    #     os.makedirs(os.path.join(args.output_dir, 'logits_result', args.dataset))
    # logits_file = os.path.join(args.output_dir, 'logits_result', args.dataset, 'logits_seed{}_k{}_dev{}.pkl'.format(args.seed, args.n_train_shot, len(dev_labels)))
    # with open(logits_file, 'wb') as file:
    #     pickle.dump(logits_output, file) # [prompt_level, len(anchor_shot)+256, 50257]
    
    # labels_file = os.path.join(args.output_dir, 'logits_result', args.dataset, 'labels_seed{}_k{}_dev{}.pkl'.format(args.seed, args.n_train_shot, len(dev_labels)))
    # with open(labels_file, 'wb') as file:
    #     dev_labels = np.asarray(dev_labels) # [256]
    #     dev_labels = np.tile(dev_labels, (args.prompt_level, 1)) # [prompt_level, 256]
    #     pickle.dump(np.concatenate((anchor_labels,dev_labels),axis=1), file) # [prompt_level, len(anchor_shot*n_class)+256]
    # pred_file = os.path.join(args.output_dir, 'logits_result', args.dataset, 'pred_seed{}_k{}_dev{}.pkl'.format(args.seed, args.n_train_shot, len(dev_labels[0])))
    # with open(pred_file, 'wb') as file:
    #     pickle.dump(np.concatenate((anchor_labels,test_pred),axis=1), file) # [prompt_level, len(anchor_shot*n_class)+256]
    

if __name__ == "__main__":
    main()
