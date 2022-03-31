# from processors.ner_seq import convert_examples_to_features
# from processors.ner_seq import ner_processors as processors
from dataset import collate_fn
from metrics.ner_metrics import SeqEntityScore
from torch.utils.data import DataLoader
import os
import torch
from utils import logger, seed_everything, ProgressBar, get_entities
from transformers import get_linear_schedule_with_warmup, AdamW
from dataset import load_and_cache_examples
import numpy as np
from CONSTANT import DATASET, processors

def train(args, train_dataset, model, tokenizer):
	""" Train the model """
	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) # 以args.per_gpu_train_batch_size=24为例  args.train_batch_size = 24 * 1 = 24
	# train_sampler = RandomSampler(train_dataset)
	# train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)
	train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
	# cdr数据集len(train_dataset): 5819
	# 以batch_size = 24为例，len(train_dataloader)= 5819/24 ≈ 243
	if args.max_steps > 0: # args.max_steps: -1
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs # 243/1*3 = 729

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay
		},
		{
			"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
			"weight_decay": 0.0
		}
	]
	args.warmup_steps = int(t_total * args.warmup_proportion) # 729 * 0.1 = 72
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
	# Check if saved optimizer or scheduler states exist
	if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
			os.path.join(args.model_name_or_path, "scheduler.pt")):
		# Load in optimizer and scheduler states
		optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))) # './data/model_data/biobert-base-cased-v1.1\\optimizer.pt'
		scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))) # './data/model_data/biobert-base-cased-v1.1\\scheduler.pt'
	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
				args.train_batch_size * args.gradient_accumulation_steps * 1 # 24
				)
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)
	global_step = 0
	steps_trained_in_current_epoch = 0
	# Check if continuing training from a checkpoint
	if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
		# set global_step to gobal_step of last saved checkpoint from model path
		global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
		epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
		steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
		logger.info("  Continuing training from checkpoint, will skip to saved global_step")
		logger.info("  Continuing training from epoch %d", epochs_trained)
		logger.info("  Continuing training from global step %d", global_step)
		logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
	pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=int(args.num_train_epochs))
	if args.save_steps == -1 and args.logging_steps == -1:
		args.logging_steps = len(train_dataloader) # 243
		args.save_steps = len(train_dataloader) # 243
	for epoch in range(int(args.num_train_epochs)):
		pbar.reset()
		pbar.epoch_start(current_epoch=epoch)
		for step, batch in enumerate(train_dataloader):
			# Skip past any already trained steps if resuming training
			# batch :长为4的列表
			# [
			# 	{
			# 		'input_ids': tensor([[  101,  1821, 23032,  ...,     0,     0,     0],
			#         					 [  101,  6593,   131,  ...,     0,     0,     0],
			#         					 ...,
			#         					 [  101,  1142,  9249,  ...,     0,     0,     0]]), # torch.Size(24, 128)
			# 		'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
			#         						  [0, 0, 0,  ..., 0, 0, 0],
			#         						  ...,
			#        							  [0, 0, 0,  ..., 0, 0, 0]]), # torch.Size(24, 128)
			# 		'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
			#         						  [1, 1, 1,  ..., 0, 0, 0],
			#        					          ...,
			#         						  [1, 1, 1,  ..., 0, 0, 0]]) # torch.Size(24, 128)
			# 	},
			# 	tensor([[2, 3, 5,  ..., 0, 0, 0],
			#         	[2, 2, 2,  ..., 0, 0, 0],
			#         	...,
			#         	[2, 2, 2,  ..., 0, 0, 0]]), # torch.Size(24, 128)
			# 	tensor([[  2, 536, 536,  ...,   0,   0,   0],
			# 			[  2,   2,   2,  ...,   0,   0,   0],
			# 			...,
			# 			[  2,   2,   2,  ...,   0,   0,   0]]), # torch.Size(24, 128)
			# 	tensor([65, 38, 55, 38, 59, 22, 11, 16, 11, 38,  8, 39, 16, 39, 15, 18, 49, 23, 34, 48,  7, 77, 58, 64]) # torch.Size(24)
			# ]
			batch = collate_fn(batch)
			# [
			# 	{
			# 		'input_ids': tensor([[101, 1821, 23032, ..., 0, 0, 0],
			# 					   		 [101, 6593, 131, ..., 0, 0, 0],
			# 					   		 ...,
			# 					   	     [101, 1142, 9249, ..., 0, 0, 0]]), # torch.Size(24, 77)
			# 	    'token_type_ids': tensor([[0, 0, 0, ..., 0, 0, 0],
			# 								  [0, 0, 0, ..., 0, 0, 0],
			# 								  ...,
			# 								  [0, 0, 0, ..., 0, 0, 0]]), # torch.Size(24, 77)
			#		'attention_mask': tensor([[1, 1, 1, ..., 0, 0, 0],
			# 								  [1, 1, 1, ..., 0, 0, 0],
			# 								  ...,
			# 								  [1, 1, 1, ..., 0, 0, 0]]) # torch.Size(24, 77)
			# 	},
			# 	tensor([[2, 3, 5, ..., 0, 0, 0],
			# 			[2, 2, 2, ..., 0, 0, 0],
			# 			...,
			# 			[2, 2, 2, ..., 0, 0, 0]]), # torch.Size(24, 77)
			# 	tensor([[2, 536, 536, ..., 0, 0, 0],
			# 		 	[2, 2, 2, ..., 0, 0, 0],
			# 		 	...,
			# 		 	[2, 2, 2, ..., 0, 0, 0]]), # torch.Size(24, 77)
			#  	tensor([65, 38, 55, 38, 59, 22, 11, 16, 11, 38, 8, 39, 16, 39, 15, 18, 49, 23, 34, 48, 7, 77, 58, 64]) # torch.Size(24)
			# ]
			if steps_trained_in_current_epoch > 0:
				steps_trained_in_current_epoch -= 1
				continue
			model.train()
			batch[0]['input_ids'] = batch[0]['input_ids'].to(args.device)
			batch[0]['token_type_ids'] = batch[0]['token_type_ids'].to(args.device)
			batch[0]['attention_mask'] = batch[0]['attention_mask'].to(args.device)
			batch[1] = batch[1].to(args.device)
			batch[2] = batch[2].to(args.device)
			batch[3] = batch[3].to(args.device)
			batch = tuple(t for t in batch)
			inputs, ner, nen, lens = batch
			if args.task_name == 'ner':
				labels = ner
			elif args.task_name == 'nen':
				labels = nen
			outputs = model(labels=labels, **inputs)
			# (
			# 	tensor(1.9131, device='cuda:0', grad_fn= < NllLossBackward0 >),
			# 	一个大小为torch.Size([24, 58, num_labels])的tensor
			# )
			loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
			if args.gradient_accumulation_steps > 1: # 跳过
				loss = loss / args.gradient_accumulation_steps
			loss.backward()
			pbar(step, {'loss': loss.item()})
			tr_loss += loss.item()
			if (step + 1) % args.gradient_accumulation_steps == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
				global_step += 1
				if args.logging_steps > 0 and global_step % args.logging_steps == 0: # 每args.logging_steps个steps评估一下
					# Log metrics
					print(" ")
					# Only evaluate when single GPU otherwise metrics may not average well
					evaluate(args, model, tokenizer)

				if args.save_steps > 0 and global_step % args.save_steps == 0: # 每args.save_steps个steps保存一下checkpoint
					# Save model checkpoint
					output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step)) # 如./outputs/cdr/ner/bert/checkpoint-486
					if not os.path.exists(output_dir):
						os.makedirs(output_dir)
					# Take care of distributed/parallel training
					model_to_save = (model.module if hasattr(model, "module") else model) # hasattr(model, "module"): False
					model_to_save.save_pretrained(output_dir)
					# ./outputs/cdr/ner/bert/checkpoint-486/config.json
					# ./outputs/cdr/ner/bert/checkpoint-486/pytorch_model.bin
					torch.save(args, os.path.join(output_dir, "training_args.bin")) # ./outputs/cdr/ner/bert/checkpoint-486/training_args.bin
					tokenizer.save_vocabulary(output_dir) # ./outputs/cdr/ner/bert/checkpoint-486/vocab.txt
					logger.info("Saving model checkpoint to %s", output_dir)
					torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
					# ./outputs/cdr/ner/bert/checkpoint-486/optimizer.pt
					torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
					# ./outputs/cdr/ner/bert/checkpoint-486/scheduler.pt
					logger.info("Saving optimizer and scheduler states to %s", output_dir)
		logger.info("\n")
		if 'cuda' in str(args.device):
			torch.cuda.empty_cache()
	return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
	metric = SeqEntityScore(args.id2label, markup=args.markup) # <metrics.ner_metrics.SeqEntityScore 对象>
	eval_output_dir = args.output_dir # './outputs/cdr/ner/bert'
	if not os.path.exists(eval_output_dir):
		os.makedirs(eval_output_dir)

	cached_eval_dataset_root = './data/dataset_cache/'  # './data/dataset_cache/'
	if not os.path.exists(cached_eval_dataset_root):
		os.mkdir(cached_eval_dataset_root)
	cached_eval_dataset_root = cached_eval_dataset_root + '{}'.format(args.dataset)  # './data/dataset_cache/cdr'
	if not os.path.exists(cached_eval_dataset_root):
		os.mkdir(cached_eval_dataset_root)
	DICT_DATASET = DATASET[args.dataset]["to"] # {'train': 'CDR/train.txt', 'dev': 'CDR/dev.txt', 'test': 'CDR/test.txt', 'zs_test': 'CDR/zs_test.txt'}
	processor_class = processors[args.dataset]  # <class 'data_process.Processor'>
	data_processor = processor_class(tokenizer, DICT_DATASET, args.eval_max_seq_length)
	eval_dataset = load_and_cache_examples(data_processor, cached_eval_dataset_root, data_type='eval')

	args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
	# Note that DistributedSampler samples randomly
	# eval_sampler = SequentialSampler(eval_dataset)
	eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)
	# Eval!
	logger.info("***** Running evaluation %s *****", prefix)
	logger.info("  Num examples = %d", len(eval_dataset))
	logger.info("  Batch size = %d", args.eval_batch_size)
	eval_loss = 0.0
	nb_eval_steps = 0
	pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
	for step, batch in enumerate(eval_dataloader):
		batch = collate_fn(batch)
		model.eval()

		batch[0]['input_ids'] = batch[0]['input_ids'].to(args.device)
		batch[0]['token_type_ids'] = batch[0]['token_type_ids'].to(args.device)
		batch[0]['attention_mask'] = batch[0]['attention_mask'].to(args.device)
		batch[1] = batch[1].to(args.device)
		batch[2] = batch[2].to(args.device)
		batch[3] = batch[3].to(args.device)
		batch = tuple(t for t in batch)

		with torch.no_grad():
			inputs, ner, nen, lens = batch
			if args.task_name == 'ner':
				labels = ner
			elif args.task_name == 'nen':
				labels = nen
			outputs = model(labels=labels, **inputs)
		tmp_eval_loss, logits = outputs[:2]
		eval_loss += tmp_eval_loss.item() # 0.0876813605427742
		nb_eval_steps += 1
		preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
		# 预测的token标签，大小为(16, 55)的列表
		# [[2, 3, 5, 5, 5, ..., 2, 2, 2],
		#  [2, 2, 2, 2, 2, ..., 2, 2, 3],
		#  ...,
		#  [2, 2, 2, 1, 4, ..., 2, 2, 2]]
		out_label_ids = labels.cpu().numpy().tolist() # TODO
		# 真实的token标签，大小为(16, 55)的列表
		# [[2, 3, 5, 5, 5, ..., 0, 0, 0],
		#  [2, 2, 2, 2, 2, ..., 0, 0, 0],
		#  ...,
		#  [2, 2, 2, 1, 4, ..., 0, 0, 0]]
		input_lens = lens.cpu().numpy().tolist()
		# 有效长度，长为16的列表：[22, 35, 46, 17, 26, 28, 25, 39, 12, 23, 27, 23, 33, 31, 55, 39]
		for i, label in enumerate(out_label_ids):
			# 如label：[2, 3, 5, 5, 5, ..., 0, 0, 0]  #长55
			temp_1 = []
			temp_2 = []
			for j, m in enumerate(label): # 如j:0, m:2
				if j == 0: # 跳过第一个标签，CLS标签
					continue
				elif j == input_lens[i] - 1: # 遇到最后一个有效标签SEP时，表示遍历完一句话，忽略这个SEP标签，并metric.update，结束循环
					metric.update(pred_paths=[temp_2], label_paths=[temp_1])
					# pred_paths：[[3, 5, 5, 5, 5, 5, 5, 5, 5, 2, 1, 4, 4, 4, 3, 2, 2, 2, 2, 2]]
					# label_paths：[['B-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'O', 'B-Chemical', 'I-Chemical', 'I-Chemical', 'I-Chemical', 'B-Disease', 'O', 'O', 'O', 'O', 'O']]
					break
				else: # 对每个标签,从id转成标签标识符，并添加到temp_1、temp_2
					temp_1.append(args.id2label[out_label_ids[i][j]])
					# out_label_ids[i][j]：3
					# args.id2label[out_label_ids[i][j]]：'B-Disease'
					temp_2.append(preds[i][j])
					# preds[i][j]：3
			# temp_1:
			# [] ->
			# ['B-Disease'] ->
			# ['B-Disease', 'I-Disease'] ->
			# ... ->
			# ['B-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'O', 'B-Chemical', 'I-Chemical', 'I-Chemical', 'I-Chemical', 'B-Disease', 'O', 'O', 'O', 'O', 'O']

			# temp_2:
			# [] ->
			# [3] ->
			# [3, 5] ->
			# ... ->
			# [3, 5, 5, 5, 5, 5, 5, 5, 5, 2, 1, 4, 4, 4, 3, 2, 2, 2, 2, 2]
		pbar(step)
	logger.info("\n")
	eval_loss = eval_loss / nb_eval_steps # 0.10104974798442846
	eval_info, entity_info = metric.result()
	# eval_info: {'acc': 0.8639983013058711, 'recall': 0.8563611491108071, 'f1': 0.8601627734911742} 总体的P、R、f1
	# entity_info: {'Disease': {'acc': 0.7941, 'recall': 0.7819, 'f1': 0.788}, 'Chemical': {'acc': 0.9183, 'recall': 0.9149, 'f1': 0.9166}} 各个类别的P、R、f1
	results = {f'{key}': value for key, value in eval_info.items()}
	results['loss'] = eval_loss
	# results:
	# {'acc': 0.8639983013058711, 'recall': 0.8563611491108071, 'f1': 0.8601627734911742, 'loss': 0.10104974798442846}
	logger.info("***** Eval results %s *****", prefix)
	info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
	logger.info(info)
	logger.info("***** Entity results %s *****", prefix)
	for key in sorted(entity_info.keys()):
		logger.info("******* %s results ********" % key)
		info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
		logger.info(info)
	return results

import json
def predict(args, model, tokenizer, prefix=""):
	pred_output_dir = args.output_dir
	if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
		os.makedirs(pred_output_dir)

	test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='test')
	# Note that DistributedSampler samples randomly
	# test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
	test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
	# Eval!
	logger.info("***** Running prediction %s *****", prefix)
	logger.info("  Num examples = %d", len(test_dataset))
	logger.info("  Batch size = %d", 1)

	results = []
	output_submit_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
	pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
	for step, batch in enumerate(test_dataloader):
		model.eval()
		batch = tuple(t.to(args.device) for t in batch)
		with torch.no_grad():
			inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
			if args.model_type != "distilbert":
				# XLM and RoBERTa don"t use segment_ids
				inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
			outputs = model(**inputs)
		logits = outputs[0]
		preds = logits.detach().cpu().numpy()
		preds = np.argmax(preds, axis=2).tolist()
		preds = preds[0][1:-1]  # [CLS]XXXX[SEP]
		tags = [args.id2label[x] for x in preds]
		label_entities = get_entities(preds, args.id2label, args.markup)
		json_d = {}
		json_d['id'] = step
		json_d['tag_seq'] = " ".join(tags)
		json_d['entities'] = label_entities
		results.append(json_d)
		pbar(step)
	logger.info("\n")
	with open(output_submit_file, "w") as writer:
		for record in results:
			writer.write(json.dumps(record) + '\n')


# def load_and_cache_examples(args, task, tokenizer, data_type='train'):
# 	if args.local_rank not in [-1, 0] and not evaluate:
# 		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
# 	processor = processors[task]()
# 	# Load data features from cache or dataset file
# 	cached_features_file = os.path.join(args.data_dir, 'cached_soft-{}_{}_{}_{}'.format(
# 		data_type,
# 		list(filter(None, args.model_name_or_path.split('/'))).pop(),
# 		str(args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length),
# 		str(task)))
# 	if os.path.exists(cached_features_file) and not args.overwrite_cache:
# 		logger.info("Loading features from cached file %s", cached_features_file)
# 		features = torch.load(cached_features_file)
# 	else:
# 		logger.info("Creating features from dataset file at %s", args.data_dir)
# 		label_list = processor.get_labels()
# 		if data_type == 'train':
# 			examples = processor.get_train_examples(args.data_dir)
# 		elif data_type == 'dev':
# 			examples = processor.get_dev_examples(args.data_dir)
# 		else:
# 			examples = processor.get_test_examples(args.data_dir)
# 		features = convert_examples_to_features(examples=examples,
# 												tokenizer=tokenizer,
# 												label_list=label_list,
# 												max_seq_length=args.train_max_seq_length if data_type == 'train' \
# 													else args.eval_max_seq_length,
# 												cls_token_at_end=bool(args.model_type in ["xlnet"]),
# 												pad_on_left=bool(args.model_type in ['xlnet']),
# 												cls_token=tokenizer.cls_token,
# 												cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
# 												sep_token=tokenizer.sep_token,
# 												# pad on the left for xlnet
# 												pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
# 												pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
# 												)
# 		if args.local_rank in [-1, 0]:
# 			logger.info("Saving features into cached file %s", cached_features_file)
# 			torch.save(features, cached_features_file)
# 	if args.local_rank == 0 and not evaluate:
# 		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
# 	# Convert to Tensors and build dataset
# 	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
# 	all_input_masegment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
# # 	all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
# # 	all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
# # 	dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
# # 	return datasetk = torch.tensor([f.input_mask for f in features], dtype=torch.long)
# 	all_s


