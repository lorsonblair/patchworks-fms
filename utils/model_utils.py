import torch.nn as nn

from transformers import (
    PatchTSTConfig,
    PatchTSTForPrediction,
)

from transformers.models.patchtst.modeling_patchtst import (
    PatchTSTPreTrainedModel,
    PatchTSTPositionalEncoding,
    PatchTSTBatchNorm,
)

from tsfm_public import (
    TinyTimeMixerConfig,
    TinyTimeMixerModel,
    TinyTimeMixerForPrediction,
)

"""
Patched loss_function property that doesn't warn about missing loss_type.
This is necessary for FMS Model Optimizer's qmodel_prep compatibility.
"""
def patched_loss_function(self):
    if not hasattr(self, '_loss_function'):
        # Default to MSE for regression/forecasting tasks
        self._loss_function = nn.MSELoss()
    
    return self._loss_function


"""
Class to enable PyTorch default initialization of nn.Linear layers.
"""
class CustomPatchTSTForPrediction(PatchTSTForPrediction):
    def __init__(self, config):

        original_init = PatchTSTForPrediction._init_weights

        PatchTSTPreTrainedModel._init_weights = self._custom_init_weights

        # init with custom weights
        super().__init__(config)

        # Restore original for other instances
        #PatchTSTForPrediction._init_weights = original_init

    def _custom_init_weights(self, module):

        if isinstance(module, PatchTSTPositionalEncoding):

            # get the number of patches
            num_patches = (
                max(self.config.context_length, self.config.patch_length) - self.config.patch_length
            ) // self.config.patch_stride + 1
            # initialize cls_token
            if self.config.use_cls_token:
                nn.init.normal_(module.cls_token, std=0.02)
                num_patches += 1
            # initialize positional encoding
            module.position_enc = module._init_pe(self.config, num_patches)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, PatchTSTBatchNorm):
            module.batchnorm.bias.data.zero_()
            module.batchnorm.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            # use PyTorch defaults for nn.Linear layers
            pass


"""
Gets the model with desired config for training and inference.
Currently supports patchtst and tinytimemixer.
"""
def get_model(args):
    
    if args.model == "patchtst":
        config = PatchTSTConfig(
            do_mask_input = args.mask_input,
            num_input_channels=args.num_input_channels,
            context_length=args.context_length,
            patch_length=args.patch_length,
            patch_stride=args.patch_stride,
            prediction_length=args.prediction_length,
            d_model=args.d_model,
            ffn_dim=args.d_ff,
            num_attention_heads=args.n_heads,
            num_hidden_layers=args.e_layers,
            positional_dropout=args.positional_dropout,
            attention_dropout=args.attention_dropout,
            path_dropout=args.path_dropout,
            ff_dropout=args.ff_dropout,
            head_dropout=args.head_dropout,
            pooling_type=args.pooling,
            positional_encoding_type=args.positional_encoding_type,
            init_std=args.init_std,
            scaling=args.scaling,
            loss=args.loss,
            pre_norm=args.pre_norm,
            norm_type=args.norm_type,
            share_embedding=False,
        )

        if args.weight_init == "hf":
            model = PatchTSTForPrediction(config=config)
        elif args.weight_init == "he":
            model = CustomPatchTSTForPrediction(config=config)
        else:
            raise ValueError(f"{args.weight_init} not recognized. Use either 'hf' or 'he'")

    elif args.model == "ttmx":
        
        post_init = (
            True 
            if args.weight_init == "hf" 
            else False
            if args.weight_init == "he"
            else (
                print(f"ValueError: {args.weight_init} not recognized. Use either 'hf' or 'he'")
            )
        )

        config = TinyTimeMixerConfig(
            context_length=args.context_length,
            patch_length=args.patch_length,
            patch_stride=args.patch_stride,
            num_input_channel=args.num_input_channels,
            d_model=args.d_model,
            prediction_length=args.prediction_length,
            num_layers=args.e_layers,
            dropout=args.dropout,
            gated_attn=args.gated_attn,
            norm_mlp=args.norm_type,
            self_attn=args.self_attn,
            self_attn_heads=args.n_heads,
            use_positional_encoding=args.use_pe,
            positional_encoding_type=args.positional_encoding_type,
            scaling=args.scaling,
            loss=args.loss,
            init_std=args.init_std,
            post_init=post_init,
            head_dropout=args.head_dropout,
            use_decoder=args.use_decoder,
            decoder_num_layers=args.d_layers,
            decoder_d_model=args.d_d_model,
        )
        
        model = TinyTimeMixerForPrediction(config=config)

    return model
