import argparse
import logging
from utils import set_color, load_config_files, show_args_info, setup_global_seed, \
    check_output_path, data_partition, data_preparation
import time
from trainer import MyTrainer
import torch
from model import create_model_diffu, Att_Diffuse_model 
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def str2bool(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def main():
    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument('--dataset', default='UB', type=str)
    parser.add_argument('--config_files', type=str, default='./config/', help='config yaml files')
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument('--log_file', default='log/', help='log dir path')

    # model args
    parser.add_argument("--no", type=int, default=2, help="model/process idenfier, e.g., 1, 2, 3...")
    parser.add_argument("--dropout", type=float, default=0.2, help="hidden dropout p for embedding layer and FFN")

    ## Self-Attention
    parser.add_argument("--n_heads", type=int, default=2, help="num of attention heads in SAL")
    parser.add_argument("--n_layers", type=int, default=2, help="num of attention heads in SAL")
    parser.add_argument("--hidden_act", type=str, default="relu", help="/")  # gelu relu
    
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--inner_size", type=int, default=512) 
    
    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of optimizer")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size of training phase")
    parser.add_argument("--seed", type=int, default=2025, help="global random seed for CUDA and pytorch")
    parser.add_argument("--loss_type", type=str, default="CE", help="loss function of model")
    parser.add_argument("--tensorboard_on", type=str2bool, default=False,
                       help="whether to launch the tensorboard and record the scalar to analyse the training phase")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    
    
    # Diffusion args 
    parser.add_argument('--emb_dropout', type=float, default=0.2, help='Dropout of item embedding')
    parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='Diffusion for t generation')
    parser.add_argument('--diffusion_steps', type=int, default=32, help='Diffusion step')  # 32
    parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='uncertainty weight')
    parser.add_argument('--noise_schedule', default='trunc_lin', help='Beta generation')  ## cosine, linear, trunc_cos, trunc_lin, pw_lin, sqrt
    parser.add_argument('--rescale_timesteps', default=True, help='rescal timesteps')
    
    parser.add_argument("--behw", type=int, default=0, help="behavior weight noise, e.g., 1, 2, 3...")
    # parser.add_argument("--tag_weight", type=float, default=1.0, help="target behavior weight noise, e.g., 1, 2, 3...")
    parser.add_argument("--global_weight", type=float, default=0.2, help="feature loss, e.g., 0.1, 0.2, ..., 0.5")

    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    
    # TODO logging
    if not os.path.exists(args.log_file):
        os.makedirs(args.log_file)
    if not os.path.exists(args.log_file + args.dataset):
        os.makedirs(args.log_file + args.dataset)
    
    file_name_prefix = 'seed_%d_k=5,7,9_n_layers_%d_n_heads_%d_t_%d_dropout_%.2f_emb_dropout_%.2f_no_%d_behw_%d_global_weight_%.2f' % (args.seed, args.n_layers, args.n_heads, args.diffusion_steps, args.dropout, args.emb_dropout, args.no, args.behw, args.global_weight)
    
    logging.basicConfig(level=logging.INFO, filename=args.log_file + args.dataset + '/' + file_name_prefix + '.log', filemode='w')
    logger = logging.getLogger(__name__)
    print(set_color('Using CUDA: ' + str(args.cuda_condition) + '\n', 'green'))

    config_dict = load_config_files(args.config_files, args)
    show_args_info(argparse.Namespace(**config_dict))
    # logger.info(argparse.Namespace(**config_dict))
    for key,value in config_dict.items():
        logger.info(f'{key}: {value}')

    setup_global_seed(config_dict['seed'])
    check_output_path(config_dict['output_dir'])

    data = data_partition(config_dict['dataset'])
    train_dataloader, valid_dataloader, test_dataloader, config_dict = data_preparation(config_dict, data)


    # Diffusion model
    args_new = argparse.Namespace(**config_dict)
    args_new.behavior_num = config_dict['beha_num']
    args_new.max_len  = config_dict['max_seq_length']
    diffu_rec = create_model_diffu(args_new)
    model = Att_Diffuse_model(diffu_rec, args_new, config_dict)
    
    # trainer
    trainer = MyTrainer(model, train_dataloader, valid_dataloader, test_dataloader, config_dict, logger, args_new)

    if config_dict['do_eval']:
        trainer.load()
        _, test_info = trainer.test(record=False)
        print(set_color(f'\nFinal Test Metrics: ' +
                        test_info + '\n', 'pink'))
        
        logger.info(set_color(f'\nFinal Test Metrics: ' +
                        test_info + '\n', 'pink'))
    else:
        trainer.train()


if __name__ == '__main__':
    main()