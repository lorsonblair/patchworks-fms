import argparse
import os
#import logging
import torch
import random
import numpy as np
from exp_main import ExpMain

from utils.tools import *
#import multiprocessing as mp

#import warnings
#warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser("Huggingface's PatchTST")

    # experiments
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed for reproducibility')
    parser.add_argument("--mode", type=str, default="fp", help="precision: fp, int, or ang")
    parser.add_argument("--nbits", type=int, default=32, help="number of precision bits") 
    parser.add_argument("--itr", type=int, default=1, help="number of iterations")

    # data
    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='location of model checkpoints')
    parser.add_argument("--logs", type=str, default="./logs", help="base log directory")
    parser.add_argument("--resultpath", type=str, default="./results", help="location of files")
    #parser.add_argument("--data_type", type=str, default="custom", help="dataset type")
    parser.add_argument("--root_path", type=str, default="./dataset/", help="root path of the data file")
    parser.add_argument("--dataset", type=str, default="weather.csv", help="dataset")
    parser.add_argument("--data_name", type=str, default="Weather", help="dataset type")
    parser.add_argument("--overwrite", action="store_true", default=True, help="overwrite output_dir")
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, \
                            S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, \
                            b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument("--num_workers", type=int, default=10, help="number of cores in cpu")

    # model config
    parser.add_argument("--model", type=str, default="patchtst", help="model type")
    parser.add_argument("--run_id", type=str, default="", help="model identifier")
    parser.add_argument("--context_length", type=int, default=512, help="context length of the input sequence, i.e., lookback window")
    parser.add_argument("--prediction_length", type=int, default=24, help="prediction horizon")
    parser.add_argument('--label_length', type=int, default=48, help='start token length')
    parser.add_argument("--patch_length", type=int, default=16, help="patch length")
    parser.add_argument("--patch_stride", type=int, default=8, help="stride")
    parser.add_argument("--e_layers", type=int, default=3, help="number of hidden layers")
    parser.add_argument("--num_input_channels", type=int, default=7, help="number of input channels.")
    parser.add_argument("--d_model", type=int, default=128, help="transformer layers' dimension")
    parser.add_argument("--n_heads", type=int, default=16, help="number of attention heads")
    parser.add_argument("--d_ff", type=int, default=256, help="dimension of the ffn layer in the Transformer encoder")
    parser.add_argument("--norm_type", type=str, default="batchnorm", help="normalization at each transformer layers. batchnorm or layernorm")
    parser.add_argument("--head_dropout", type=float, default=0.0, help="dropout probability in the linear head.")
    parser.add_argument("--positional_encoding_type", type=str, default="random", help="Positional encodings. Options random and sincos.")
    parser.add_argument("--scaling", type=str, default="std", help="scale input targets: mean, std, or None")
    parser.add_argument("--weight_init", type=str, default="he", help="weight initialization strategy")
    parser.add_argument("--init_std", type=float, default=0.02, help="std of weight initialization")
    parser.add_argument("--loss", type=str, default="mse", help="the loss function")


    # patchtst
    parser.add_argument("--attention_dropout", type=float, default=0.0, help="dropout probability in the attension layer.")
    parser.add_argument("--positional_dropout", type=float, default=0.0, help="dropout probability in the positional embedding layer.")
    parser.add_argument("--path_dropout", type=float, default=0.0, help="dropout path in the residual block.")
    parser.add_argument("--ff_dropout", type=float, default=0.0, help="dropout probability in the positional ffn")
    parser.add_argument("--pre_norm", action="store_true", default=False, 
                        help="apply normalization before self-attention (True) or after residual block (False)")
    parser.add_argument("--mask_input", action="store_true", default=False, help="mask input")
    parser.add_argument("--mask_type", type=str, default="random", 
                        help="Masking ratio applied to mask the input data during random pretraining")
    parser.add_argument("--random_mask_ratio", type=float, default=0.0, help="percentage of mask")
    parser.add_argument("--pooling", type=str, default=None, help="pooling type")
    
    
    # tinytimemixer
    parser.add_argument("--dropout", type=float, default=0.0, help="The dropout probability the `TinyTimeMixer` backbone.")
    parser.add_argument("--gated_attn", action="store_true", default=False, 
                        help="Enable gated attention. If true, self_attn = False.")
    parser.add_argument("--self_attn", action="store_true", default=False,
                        help="""enable Tiny self attention across patches. This can be enabled when the output 
                             of Vanilla TinyTimeMixer with gated attention is not satisfactory. Enabling 
                             this leads to explicit pair-wise attention and modelling across patches.""")
    parser.add_argument("--use_pe", action="store_true", default=False, 
                        help="""Enable the use of positional embedding for the tiny self-attention layers. 
                              Works only when `self_attn` is set to `True`.""")
    parser.add_argument("--use_decoder", action="store_true", default=False, help="Use decoder")
    parser.add_argument("--d_layers", type=int, default=8, help="Number of decoder layers")
    parser.add_argument("--d_d_model", type=int, default=16, help="Hidden feature size of the decoder") 

    # training and optimization
    parser.add_argument("--do_train", type=int, default=1, help="perform training")
    parser.add_argument("--bf16", type=int, default=0, help="use bfloat 16")
    parser.add_argument("--optim", type=str, default="adamw", help="optimizer to use")
    parser.add_argument("--scheduler", type=str, default="calr", help="scheduler for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--scale_lr", type=int, default=0, help="whether to scale the learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay coefficient")
    parser.add_argument("--amsgrad", type=int, default=0, help="use amsgrad version of Adam")
    parser.add_argument("--max_grad_norm", type=float, default="inf", help="max grad norm for gradient clipping.")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, 
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/", help="output directory")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="log directory")
    parser.add_argument("--log_strategy", type=str, default="epoch", help="logging strategy")
    parser.add_argument("--train_epochs", type=int, default=100, help="train epochs")
    #parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument("--do_eval", action="store_true", default=True, help="do evaluation, i.e., validation")
    parser.add_argument("--eval_strategy", type=str, default="epoch", help="evaluation strategy")
    parser.add_argument("--batch_size", type=int, default=128, help="per gpu batch size for training and evaluation")
    parser.add_argument("--save_limit", type=int, default=2, help="limit for number of saves")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="options: 'no', 'epoch', 'steps', 'best': save best model")
    parser.add_argument("--load_best_model", action="store_true", default=True, help="load best model at the end of training")
    parser.add_argument("--metric", type=str, default="eval_loss", help="metric for early stopping")
    parser.add_argument("--greater", action="store_true", default=False, help="for loss")
    parser.add_argument("--label_names", type=str, default="future_values", help="label names for list")
    #parser.add_argument("--num_cycles", type=float, default=0.5, help="The number of waves in the cosine schedule")
    #parser.add_argument("--warmup_steps", type=int, default=-1, help="The number of warmup steps for cosine schedule")

    # early stopping callback
    parser.add_argument("--early_stopping", action="store_true", default=False, help="enable early stopping callback")
    parser.add_argument("--patience", type=int, default=-1, help="early stopping patience. -1 means patience not set. early stopping not enabled")
    parser.add_argument("--threshold", type=float, default=1e-8, help="minimum improvement")

    # quantization
    parser.add_argument("--init_ckpt", type=str, default="", help="qat training checkpoint")
    parser.add_argument("--qlast", action="store_true", default=False, help="quantize the last layer")
    #parser.add_argument("--qskip_layer_name", nargs="*", default=["model.encoder.embedder.input_embedding"], help="")
    #parser.add_argument("--recipe", type=str, default="none", help= "qat recipe")
    parser.add_argument('--nbits_w', type=int, default=32, help='weight precision')
    parser.add_argument('--nbits_a', type=int, default=32, help='activation precision')
    parser.add_argument('--nbits_w_alt', type=int, default=None, help='weight precision for normally skipped layers')
    parser.add_argument('--nbits_a_alt', type=int, default=None, help='activation precision for normally skipped layers')
    parser.add_argument('--nbits_w_head', type=int, default=32, help='weight precision for linear head')
    parser.add_argument('--nbits_a_head', type=int, default=32, help='activation precision for linear head')
    parser.add_argument('--nbits_bmm1', type=int, default=32, help='weight precision for bmm1')
    parser.add_argument('--nbits_bmm2', type=int, default=32, help='weight precision for bmm2')
    parser.add_argument('--qw_mode', type=str, default='sawb+', help='weight quantization mode, e.g., lpuq, sawb(+), dorefa')
    parser.add_argument('--qa_mode', type=str, default='pact+', help='activation quantization mode, e.g., pact, lpuq, lsq, or qil')
    #parser.add_argument('--qa_qkv_mode', type=str, default='pact+', help='activation quantization mode for qkv layer')
    #parser.add_argument('--qw_qkv_mode', type=str, default='sawb+', help='weight quantization mode for qkv layer')
    parser.add_argument('--bmm1_qm1_mode', type=str, default='pact', help='activation quantization, e.g., pact, lpuq, lsq, or qil')
    parser.add_argument('--bmm1_qm2_mode', type=str, default='pact', help='activation quantization, e.g., pact, lpuq, lsq, or qil')
    parser.add_argument('--bmm2_qm1_mode', type=str, default='pact', help='activation quantization, e.g., pact, lpuq, lsq, or qil')
    parser.add_argument('--bmm2_qm2_mode', type=str, default='pact', help='activation quantization, e.g., pact, lpuq, lsq, or qil')
    parser.add_argument('--w_reg', type=int, default=0, help='perform weight kurtosis regularization')
    parser.add_argument('--a_reg', type=int, default=0, help='perform activation kurtosis regularization')
    parser.add_argument('--bmm1_reg', type=int, default=0, help='activation kurtosis regularization')
    parser.add_argument('--bmm2_reg', type=int, default=0, help='activation kurtosis regularization')
    parser.add_argument('--pact_a_lr', type=float, default=1e-4, help='pact learning rate')
    parser.add_argument('--pact_a_decay', type=float, default=0.0, help='clip val for activation decay')
    parser.add_argument('--align_zero', action='store_true', help='set align_zero flags in W and A quantizers to True')
    parser.add_argument('--qmodel_calibration', type=int, default=5, help='Num of batches for Qmodel calibration')
    #parser.add_argument('--qmodel_calibration_new', default=5, type=int, help='new method for calibration')
    parser.add_argument('--qkvsync', action='store_true', help='synchronize clip vals for QKV layers')
    parser.add_argument('--clip_val_asst_percentile', nargs='+', type=float, default=(0.1, 99.9), help='percentile for clip_val initialization')
    parser.add_argument('--plotsvg', action='store_true', default=False, help='save computation graph (graphviz/pygraphviz)')
    
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # initialize ExpMain object, i.e., set experiments
    exp = ExpMain(args)

    prc = f"{args.mode}{args.nbits}"
        
    # Set multiprocessing start method
    #mp.set_start_method('spawn', force=True)
    
    for itr in range(args.itr):
        
        ext = "_bf16" if args.bf16 else ""
        args.run_id = f"{args.run_id}{ext}"

        if args.mode == "int":
                    
            if args.init_ckpt != "none":
                setting_ckpt = f"{args.model}_{args.model_id}_{itr}"
            
        # experiments identifier
        #args.run_id = f"{args.run_id}_rs{args.random_seed}"   
        
        
        # for checkpoints and trainer stuff
        ckpt_dir = f"{args.checkpoints}/{args.model}/{prc}/{args.init_ckpt}/{args.run_id}"

        # cleanup ckpt_dir name
        if "//" in ckpt_dir:
            ckpt_dir = ckpt_dir.replace("//", "/")

                
        # TODO: fix logging mechanism 
        # for logging
        '''
        log_dir = f"{args.logs}/{prc}/{args.model}/{init_ckpt}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/{args.run_id}.log"
        os.path.join(log_dir, log_file)
 
        logging.basicConfig(
            filename=log_file,
            format="%(message)s",
            level=logging.INFO,
        ) 
        logger = logging.getLogger(__name__)
        '''
        # scale learning rate if required
        if args.scale_lr:
            args.learning_rate = args.learning_rate * math.sqrt(2 * args.batch_size * 1)

        # print args for each run
        printArgs(args)

        if args.do_train:
            
            print(f">>>>>>>>>> Start training : {args.run_id} >>>>>>>>>>")
           
            misc_time, \
            best_eval_loss, \
            crsp_train_loss, \
            epoch_best_loss = exp.train(args.run_id, ckpt_dir) 

            print("\nTraining completed >>>>>>>>>>")
            print(f"Misc time: {misc_time:.3f}")
            print(f"Best eval loss: {best_eval_loss:.7f}")
            print(f"Corresponding train loss: {crsp_train_loss}")
            print(f"Epoch of best loss: {epoch_best_loss}")
                                
        print(f"\n>>>>>>>>>> Testing : {args.run_id} >>>>>>>>>>")
        misc_time, test_results = exp.test(ckpt_dir=ckpt_dir, test = not args.do_train)
        print(f"Testing completed >>>>>>>>>>")
        print(f"Misc time: {misc_time:.3f}")
        print(f"{test_results}")

        torch.cuda.empty_cache()
        

if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)
    main()
