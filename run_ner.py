#!python
# coding=utf-8
'''
:Description  : To be filled
:Version      : 1.0
:Author       : LilNeo
:Date         : 2022-03-15 21:58:20
:LastEditors  : wy
:LastEditTime : 2022-03-23 14:27:01
:FilePath     : /nernen/run_ner.py
:Copyright 2022 LilNeo, All Rights Reserved. 
'''
import argparse
import glob
import os
import time
import torch

from transformers import WEIGHTS_NAME
from utils import init_logger, logger, seed_everything
from dataset import load_and_cache_examples
from train_eval_test import train, evaluate, predict
import logging
from CONSTANT import MODEL_CLASSES, DATASET, processors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters

    # parser.add_argument("--save_weights", action="store_true")
    # parser.add_argument("--save_pred_result", action="store_true")
    # parser.add_argument("--weights", default="./weights", type=str)

    parser.add_argument("--dataset", choices=["ncbi", "cdr"], default="ncbi", type=str)
    parser.add_argument("--task_name", choices=["ner", "nen"], default="ner", type=str)
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.", )
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: ")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.", )

    # Other parameters
    parser.add_argument('--markup', default='bio', type=str,
                        choices=['bios', 'bio'])
    parser.add_argument('--loss_type', default='ce', type=str,
                        choices=['lsr', 'focal', 'ce'])
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--eval_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--crf_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for crf and linear layer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )

    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number", )
    parser.add_argument("--predict_checkpoints",type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()
    print("cur path：" + str(os.getcwd()))

    if not os.path.exists(args.output_dir): # './outputs'
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '/{}'.format(args.dataset) # './outputs/cdr'
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '/{}'.format(args.task_name) # './outputs/cdr/ner'
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '/{}'.format(args.model_type) # './outputs/cdr/ner/bert'
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H：%M：%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.dataset}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu") # cpu
    args.n_gpu = torch.cuda.device_count() # 0
    args.device = device # cpu
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)
    # Set seed
    seed_everything(args.seed)

    # Prepare NER task
    args.dataset = args.dataset.lower() # 'cdr'
    if args.dataset not in processors: # from processors.ner_seq import ner_processors as processors
        raise ValueError("Task not found: %s" % args.dataset)
    processor_class = processors[args.dataset] # <class 'data_process.CDRProcessor'>
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config_class: BertConfig
    # model_class: BertSoftmaxForNer 或 BertSoftmaxForNen
    # tokenizer_class: BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    DICT_DATASET = DATASET[args.dataset]["to"] # {'train': 'CDR/train.txt', 'dev': 'CDR/dev.txt', 'test': 'CDR/test.txt', 'zs_test': 'CDR/zs_test.txt'}
    data_processor = processor_class(tokenizer, DICT_DATASET, args.train_max_seq_length)
    # TODO
    if args.task_name == 'ner': # NER任务
        label_list = data_processor.get_ner_labels() # ['X', 'B-Chemical', 'O', 'B-Disease', 'I-Chemical', 'I-Disease']
    else: # NEN任务
        label_list = data_processor.get_nen_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)} # {0: 'X', 1: 'B-Chemical', 2: 'O', 3: 'B-Disease', 4: 'I-Chemical', 5: 'I-Disease'}
    args.label2id = {label: i for i, label in enumerate(label_list)} # {'X': 0, 'B-Chemical': 1, 'O': 2, 'B-Disease': 3, 'I-Chemical': 4, 'I-Disease': 5}
    num_labels = len(label_list) # 6
    args.model_type = args.model_type.lower() # 'bert'
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    config.loss_type = args.loss_type # 'ce'
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        cached_train_dataset_root = './data/dataset_cache/' # './data/dataset_cache/'
        if not os.path.exists(cached_train_dataset_root):
            os.mkdir(cached_train_dataset_root)
        cached_train_dataset_root = cached_train_dataset_root + '{}'.format(args.dataset) # './data/dataset_cache/cdr'
        if not os.path.exists(cached_train_dataset_root):
            os.mkdir(cached_train_dataset_root)
        train_dataset = load_and_cache_examples(data_processor, cached_train_dataset_root, data_type='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            predict(args, model, tokenizer, prefix=prefix)
