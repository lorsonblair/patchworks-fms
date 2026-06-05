if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/int2" ]; then
    mkdir ./logs/int2
fi

seq_len=512
model_name=PatchTST

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

e_layers=3
n_heads=16
d_model=128
d_ff=256

nbits_w=2
nbits_a=2

random_seed=2024

for pred_len in  96 #192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --mode 'qat' \
      --is_training 1 \
      --trained_ckpt 'none' \
      --nbits_w $nbits_w \
      --nbits_a $nbits_a \
      --nbits_w_head 2 \
      --nbits_a_head 2 \
      --qw_mode 'sawb+' \
      --qa_mode 'pact+' \
      --bmm1_qm1_mode 'pact+' \
      --bmm1_qm2_mode 'pact+' \
      --bmm2_qm1_mode 'pact+' \
      --bmm2_qm2_mode 'pact+' \
      --nbits_bmm1 $nbits_a \
      --nbits_bmm2 8 \
      --pact_a_lr 0.0001 \
      --pact_a_decay 0.00005 \
      --align_zero \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers $e_layers \
      --n_heads $n_heads \
      --d_model $d_model \
      --d_ff $d_ff \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --attn_dropout 0\
      --patch_len 16 \
      --stride 8\
      --des 'Exp' \
      --train_epochs 100 \
      --patience 20 \
      --itr 1 \
      --batch_size 512 \
      --learning_rate 0.0001 \
      >logs/int$nbits_w/exp/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_W'$nbits_w'A'$nbits_a.log 
done
