import torch
import logging
import argparse


UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
USR_idx = 4
SYS_idx = 5
CLS_idx = 6

def get_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="ED")
    parser.add_argument("--comet_file", type=str, default="data/Comet")
    parser.add_argument("--max_num_dialog", type=int, default=9)
    parser.add_argument("--concept_num", type=int, default=3)
    parser.add_argument("--total_concept_num", type=int, default=10, help='the maximum number of external concepts injection for a sentence.')
    parser.add_argument("--cs_num", type=int, default=5)
    parser.add_argument("--emb_file", type=str)
    parser.add_argument("--save_path", type=str, default="save/test")
    parser.add_argument("--model_path", type=str, default="save/test")
    parser.add_argument("--save_path_dataset", type=str, default="save/")

    # Model
    parser.add_argument("--UNK_idx", type=int, default=0)
    parser.add_argument("--PAD_idx", type=int, default=1)
    parser.add_argument("--EOS_idx", type=int, default=2)
    parser.add_argument("--SOS_idx", type=int, default=3)
    parser.add_argument("--USR_idx", type=int, default=4)
    parser.add_argument("--SYS_idx", type=int, default=5)
    parser.add_argument("--CLS_idx", type=int, default=6)
    parser.add_argument("--KG_idx", type=int, default=7)
    parser.add_argument("--SEP_idx", type=int, default=8)
    
    # cs relation
    parser.add_argument("--self_loop", type=int, default=2)
    parser.add_argument("--contain", type=int, default=3)
    parser.add_argument("--temporary", type=int, default=4)
    parser.add_argument("--intent_idx", type=int, default=5)
    parser.add_argument("--need_idx", type=int, default=6)
    parser.add_argument("--want_idx", type=int, default=7)
    parser.add_argument("--effect_idx", type=int, default=8)
    parser.add_argument("--react_idx", type=int, default=9)
    parser.add_argument("--relation_num", type=int, default=10)
    
    # Train/Test
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--hidden_dim", type=int, default=300)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--max_grad_norm", type=float, default=2.0)
    parser.add_argument("--beam_size", type=int, default=5)
    
    parser.add_argument("--pointer_gen", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--basic_learner", default=True, action="store_true")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--softmax", default=True, action="store_true")
    parser.add_argument("--mean_query", action="store_true")
    parser.add_argument("--schedule", type=float, default=10000)
    
    ## transformer
    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--filter", type=int, default=300)
    
    ## GraphTransformer
    parser.add_argument("--graph_layer_num", type=int, default=1)
    parser.add_argument("--graph_ffn_emb_dim", type=int, default=300)
    parser.add_argument("--graph_num_heads", type=int, default=2)
    
    # Other
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model", type=str, default="case")
    parser.add_argument("--cuda", default=True, action="store_true")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--split_data_seed", type=int, default=13)
    
    parser.add_argument("--pretrain", default=True, action="store_true")
    parser.add_argument("--pretrain_epoch", type=int, default=4)
    parser.add_argument("--woStrategy", default=True, action="store_true")
    parser.add_argument("--model_file_path", type=str, default="save/test/")
    parser.add_argument("--warmup", type=int, default=12000)
    parser.add_argument("--fine_weight", type=float, default=0.2)
    parser.add_argument("--coarse_weight", type=float, default=1.0)
    
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--large_decoder", action="store_true")
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--is_coverage", action="store_true")
    parser.add_argument("--use_oov_emb", action="store_true")
    parser.add_argument("--pretrain_emb", default=True, action="store_true")
    parser.add_argument("--weight_sharing", action="store_true")
    parser.add_argument("--label_smoothing", default=True, action="store_true")
    parser.add_argument("--noam", default=True, action="store_true")
    parser.add_argument("--universal", action="store_true")
    parser.add_argument("--act", action="store_true")
    parser.add_argument("--act_loss_weight", type=float, default=0.001)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.1, help='dropout')
    
    args = parser.parse_args()
    cuda_id = "cuda:" + str(args.gpu)
    args.device = torch.device(cuda_id) if torch.cuda.is_available() else 'cpu'
    args.emb_file = args.emb_file or "vectors/glove.6B.{}d.txt".format(str(args.emb_dim))
    print_opts(args)
    
    return args

def print_opts(opts):
    """Prints the values of all command-line arguments."""
    print("=" * 80)
    print("Opts".center(80))
    print("-" * 80)
    for key in opts.__dict__:
        if key == "device":
            continue
        if opts.__dict__[key]:
            print("{:>30}: {:<30}".format(key, opts.__dict__[key]).center(80))
    print("=" * 80)

config = get_args()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M"
)
collect_stats = False
