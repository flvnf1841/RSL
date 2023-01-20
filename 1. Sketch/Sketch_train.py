import os
import wandb
import logging
import random
import numpy as np
import torch


from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

from lee_sketch_utils import load_and_cache_examples, delete_model
from lee_evaluation import evaluate
from lee_loss import cal_loss

def get_lr(optimizer : transformers.optimization) -> float:
    """
    Get learning_rate.
    Args:
        optimizer (transformers.optimization).
    Return:
        Learning_Rate (float).
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def trainer(args, model, tokenizer, pad_token_label_id, logger):
    """Train the model"""
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode="train_ex300", logger=logger)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader)//args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule ( linear warmup and decay )
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Check if saved optimizer or scheduler state exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and \
        os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps *
        (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0

        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained,
                            int(args.num_train_epochs),
                            desc="Epoch",
                            disable=args.local_rank not in [-1, 0])

    evaluation_loss = 100000000000
    f1_0 = -100000000000

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1,0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)

            outputs = model(**inputs)
            labels = batch[3]
            loss = cal_loss(labels=labels,
                            logits=outputs[0],
                            attention_mask=inputs['attention_mask'],
                            uni_w=args.uni_w,
                            ske_w=args.ske_w)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to averate on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            wandb.log({"Train loss": loss})
            loss.backward()
            tr_loss += loss.item()
            if (step+1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad() # optimizer.zero_grad()와 쓰임이 거의 같다고 보면 됨
                global_step += 1
                logger.info("global_step:%d, step_tr_loss:%.4f", global_step, loss.item())
                wandb.log({"learning_rate": get_lr(optimizer)})
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    #Log metrics
                    if(args.local_rank == -1 and args.evaluate_during_training):
                        results, _ = evaluate(args, model, tokenizer, pad_token_label_id, logger, mode="dev")
                        for key, value in results.items():
                            logger.info("global_step:%d,eval_key:%s,eval_value:%s",global_step, str(key), "\n" + str(value))

                        if results['f1_0'] > f1_0 or results['eval_loss'] < evaluation_loss:#체크필요

                            f1_0 = results['f1_0']
                            evaluation_loss = results['eval_loss']
                            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step)) #lee/sketch_model

                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)


                            model_to_save = (model.module if hasattr(model, "module") else model)
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))

                            logger.info("Saving best model till now checkpoint to %s", output_dir)

                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(),os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

                            if not output_dir.endswith('checkpoint-1000'):
                                delete_model(args.output_dir)

                        # tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        # tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logger.info("global_step:%d, lr:%.4f", global_step, scheduler.get_lr()[0])
                        logger.info("global_step:%d, loss:%.4f", global_step, (tr_loss - logging_loss) / args.logging_steps)
                        logging_loss = tr_loss


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        epochs_trained += 1

        wandb.log({"Epoch": epochs_trained})

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break


    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    if args.wandb: wandb.finish()
    return global_step, tr_loss / global_step

def inferencer(args, model, tokenizer, pad_token_label_id, logger):
    pass
