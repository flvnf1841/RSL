import os
import argparse

def parse_args(mode='train'):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="lee/data/",
        type=str,
        help="The input data dir.",
    )
    parser.add_argument(
        "--file_paths",
        default={
            'train_skeletons_path': "lee/data/train_skeletons.json",
            'train_ex300_skeletons_path': "lee/data/train_skeletons_exp_none.json",
            'dev_skeletons_path': "lee/data/dev_skeletons.json",
            'dev_exp300_skeletons_path': "lee/data/dev_skeletons_exp_none.json",
            'test_skeletons_path': "lee/data/test_skeletons_wo_exp.json",
            'test_exp300_skeletons_path': "lee/data/test_skeletons_exp_none.json"
        },
        type=dict,
        help="The skeletons files paths.",
    )
    parser.add_argument(
        "--log_path",
        default="lee/log/",
        type=str,
        help="The training log path.",
    )
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="Model type bert",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="Path to pre-trained bert model or shortcut name",
    )
    parser.add_argument(
        "--output_dir",
        default="lee/sketch_model/",
        type=str,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=300,  # 300으로 하니까 counterfactual ending이 짤리는 경우가 있음
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train",
                        action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict_on_test",
                        action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--keep_accents",
                        action="store_const",
                        const=True,
                        help="Set this flag if model is trained with accents.")
    parser.add_argument(
        "--strip_accents",
        action="store_const",
        const=True,
        help="Set this flag if model is trained without accents.")
    parser.add_argument("--use_fast",
                        action="store_const",
                        const=True,
                        help="Set this flag to use fast tokenization.")
    parser.add_argument("--per_gpu_train_batch_size",
                        # default=8, original
                        default=8,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        # default=8,
                        default=8,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.001,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=5,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps",
                        default=2000,
                        type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps",
                        type=int,
                        default=1000,  # 500
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=1000,  # 500
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help=
        "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir",
                        default=True,
                        action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache",
                        action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--unique_flag",
                        type=str,
                        default="",
                        help="type",
                        required=True)  # 2080 5050 8020
    parser.add_argument("--uni_w",
                        type=float,
                        default=0.5,
                        help="unique(causal) words weight",
                        required=True)
    parser.add_argument("--ske_w",
                        type=float,
                        default=0.5,
                        help="skeleton(background) words weight",
                        required=True)
    parser.add_argument("--pred_dir_path",
                        type=str,
                        default="lee/output/sketch_pred_results/",
                        help="predicted skeletons results file path")
    # #-- 수정. 파서 아규먼트에 등록이 안되어 있어 따로 추가함. sketch_label_process에서 사용
    parser.add_argument("--input_dropout_rate",
                        type=float,
                        default=0.8,
                        help="dropout_rate")
    parser.add_argument('--wandb', default=True, action='store_true')
    parser.add_argument('--name', default='seq300_uni0.2_batch8_drop0.8_ep5_decay0.001', type=str)

    args = parser.parse_args()
    return args