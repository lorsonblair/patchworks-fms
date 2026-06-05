import os
import time
import math
import torch
#from data_provider.data_provider_hug import *
from data_provider.data_factory import data_provider

import transformers
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from utils.model_utils import (
    patched_loss_function,
    get_model
)

import torch.optim as optim
from lion_pytorch import Lion

from utils.tools import *
#from torch.utils.tensorboard import SummaryWriter

# Apply patch to prevent loss_function from being triggered in hugging face
transformers.modeling_utils.PreTrainedModel.loss_function = property(patched_loss_function)

# Class for experiments
class ExpMain:
    def __init__(self, args):
        
        self.args = args
        #self.device = self._acquire_device()

        # set random seed for reproducibility
        transformers.set_seed(args.random_seed)
        
        # get model
        self.model = get_model(args).to(self._acquire_device())
        
        # quantization specifics
        if self.args.mode == "int":
            # import quantization library
            from fms_mo import (
                qconfig_init, 
                qmodel_prep
            )
            from fms_mo.utils.utils import patch_torch_bmm 

            self.qconfig_init = qconfig_init
            self.qmodel_prep = qmodel_prep
            self.patch_torch_bmm = patch_torch_bmm
            
            # initialize qconfig
            self.qcfg = self.qconfig_init(args=self.args)
        
        #super().__init__()

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda')
            #print(f'Use GPU: cuda: {torch.cuda.current_device()}')
        else:
            device = torch.device('cpu')
            #print('Use CPU')
        
        return device

    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    

    def _get_optimizer(self):
        # using the original Adam optimizer
        if self.args.optim == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                amsgrad=self.args.amsgrad,
            )
        elif self.args.optim == "lion":
            optimizer = Lion(
                self.model.parameters(),
                lr = self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        
        return optimizer
    
    
    def _get_scheduler(self, optimizer, train_steps):

        #total_train_steps = train_steps * self.args.train_epochs
        ''' 
        if self.args.scheduler == "oclr":
            scheduler = optim.lr_scheduler.OneCycleLR(
                            optimizer=optimizer,
                            epochs=self.args.train_epochs,
                            steps_per_epoch=train_steps,
                            pct_start=self.args.pct_start,
                            max_lr=self.args.learning_rate
                        )
        '''
        if self.args.scheduler == "calr":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer=optimizer,
                            T_max=self.args.train_epochs * train_steps, 
                            eta_min=0
                        )

        elif self.args.scheduler == "lmlr":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                            optimizer=optimizer,
                            num_warmup_steps=self.args.warmup_steps,
                            num_training_steps=self.args.train_epochs * train_steps,
                            num_cycles=self.args.num_cycles,
                        )
        
        else:
            print(f"ERROR: '{self.args.scheduler}' is not supported. Use either 'oclr', 'calr', 'lmlr'")
            exit(1)
        
        return scheduler
        

    def train(self, run_id, ckpt_dir, init_ckpt_dir="", cleanup=True):
      
        #for name, module in self.model.named_modules():
        #    print(f"{name}")
        #exit() 
        misc_time = time.time()
        output_dir = f"{ckpt_dir}/output/"
        logging_dir = f"{ckpt_dir}/logs/"
        
        # scale learning rate to account for gradient accumulation
        if self.args.grad_accumulation_steps > 1:
            self.args.learning_rate = (
                self.args.learning_rate * math.sqrt(2 * self.args.batchsize * self.args.grad_accumulation_steps)
            
        )

        # use bfloat 16 precision
        #bf16 = True if self.args.nbits == 16 else False

        # prepare for training
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=self.args.overwrite,
            learning_rate=self.args.learning_rate,
            max_grad_norm=self.args.max_grad_norm,
            num_train_epochs=self.args.train_epochs,
            do_eval=self.args.do_eval,
            eval_strategy=self.args.eval_strategy,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.grad_accumulation_steps,
            dataloader_num_workers=self.args.num_workers,
            save_strategy=(self.args.save_strategy),
            logging_strategy=self.args.log_strategy,
            save_total_limit=self.args.save_limit,
            load_best_model_at_end=self.args.load_best_model, 
            metric_for_best_model=self.args.metric,
            greater_is_better=self.args.greater,
            label_names=[self.args.label_names],
            logging_dir=logging_dir,
            seed=self.args.random_seed,
            bf16=self.args.bf16,
            bf16_full_eval=self.args.bf16
        )
                       
        # datasets and preprocessing 
        train_dataset, train_dataloader = self._get_data("train")
        eval_dataset, _ = self._get_data("eval")
               
        #train_steps = math.ceil(len(train_dataset) / self.args.batch_size)
        train_steps = len(train_dataloader)
        
        # get optimizer and scheduler for trainer
        optimizer = self._get_optimizer()
                
        # for quantization
        if self.args.mode == "int":
            #self.qcfg = self.qconfig_init(args=self.args)                       
            #base_name = f"{result_path}"

            if self.args.init_ckpt != "none": 

                print(f"... performing QAT from a trained {self.args.trained_ckpt} model ...")
                self.model.load_state_dict(torch.load(ckpt_dir))
            else:
                print(f"... Performing QAT from scratch ...")

            #if self.args.qmodel_calibration > 0:
            calib_data = [next(iter(train_dataloader)) for _ in range(self.qcfg["qmodel_calibration"])]
            
            # save qat graph
            graph_dir = f"{ckpt_dir}/qgraph"
            os.makedirs(graph_dir, exist_ok=True)
            
            self.qmodel_prep(
                self.model, 
                calib_data, 
                self.qcfg,
                optimizer=optimizer,
                save_fname=f"{graph_dir}/qgraph.pt",
                qlast=self.args.qlast,
            )

        # get scheduler 
        scheduler = self._get_scheduler(optimizer, train_steps)
        print(f"\nOptimizer: {optimizer.__class__.__name__}")
        print(f"Scheduler: {scheduler.__class__.__name__}\n")

        # create callbacks if needed
        callbacks = []
        
        if self.args.early_stopping:
            if self.args.patience > 0:
                early_stopping_callback = EarlyStoppingCallback(
                    early_stopping_patience = self.args.patience,
                    early_stopping_threshold = self.args.threshold,
                )
                callbacks.append(early_stopping_callback)
            else:
                print("ERROR: patience not set; must be > 0")
                exit(1)
        
        # trainer setup 
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=(optimizer, scheduler),
            callbacks=callbacks,
        )
        
        misc_time = time.time() - misc_time

        # perform training
        if self.args.mode == "int":
            with self.patch_torch_bmm(self.qcfg):
                self.trainer.train()
        else:
            self.trainer.train()
        
        _misc_time = time.time()
        # saved best model after training
        best_model_path = f"{ckpt_dir}/model"
        os.makedirs(best_model_path, exist_ok=True)
        self.model.save_pretrained(
            save_directory=best_model_path,
            state_dict=self.model.state_dict(),
        )

        # saved pt using torch
        best_model_path_ = f"{best_model_path}/model.pt"
        torch.save(
            self.model.state_dict(), 
            best_model_path_
        )
        
        # reload model into self.model
        #self.model = PatchTSTForPrediction.from_pretrained(best_model_path)
                
        # cleanup to remove checkpoints
        if cleanup:
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

        # get best eval loss and train loss
        log_history = self.trainer.state.log_history
        best_eval_loss, crsp_train_loss, epoch_best_loss = get_train_metrics(log_history)
        
        misc_time += time.time() - _misc_time

        return misc_time, best_eval_loss, crsp_train_loss, epoch_best_loss # get_train_metrics(log_history)


    def test(self, ckpt_dir="", test=0):
        
        misc_time = time.time()

        test_dataset, test_dataloader = self._get_data("test")
                
        if test:
            best_model_path = f"{ckpt_dir}/model"
            best_model_path_ = f"{best_model_path}/model.pt"
            #self.model = PatchTSTForPrediction.from_pretrained(best_model_path)
                        
            output_dir = f"{ckpt_dir}/temp" 
            # prepare for training
            training_args = TrainingArguments(
                output_dir=output_dir,
                do_train=self.args.do_train,
                num_train_epochs=2,#self.qcfg["qmodel_calibration_new"],
                do_eval=self.args.do_eval,
                per_device_eval_batch_size=self.args.batch_size,
                report_to="none",
                metric_for_best_model=self.args.metric,
                eval_strategy=self.args.eval_strategy,
                label_names=[self.args.label_names],
            )
            
            # create trainer with loaded model
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                eval_dataset=test_dataset,
            )

            if self.args.mode == "int":
                self.qcfg["qmodel_calibration"] = 1
                self.qcfg = self.qconfig_init(args=self.args)
                calib_data = [next(iter(test_dataloader)) for _ in range(self.qcfg["qmodel_calibration"])]
                
                self.qmodel_prep(
                    self.model,
                    calib_data,
                    self.qcfg,
                    save_fname=f"{output_dir}/qgraph.pt",
                    qlast=self.args.qlast,
                )

                # cleanup temp training params
                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)
 
            #self.model.from_pretrained(
            #    pretrained_model_name_or_path=best_model_path,
            #    local_files_only=True,
            #    use_safetensors=True
            #)
            self.model.load_state_dict(torch.load(best_model_path_))
            
        
        misc_time = time.time() - misc_time

        # perform inference
        if self.args.mode == "int":
            with self.patch_torch_bmm(self.qcfg):
                results = self.trainer.evaluate(test_dataset)
        else:
            results = self.trainer.evaluate(test_dataset)

        return misc_time, results
