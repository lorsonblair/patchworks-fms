if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/fp32" ]; then
    mkdir .logs/LongForecasting/fp32
fi

seq_len=512
model_name=PatchTST

e_layers=3
n_heads=16
d_model=128
d_ff=256

root_path_name=./dataset/
data_path_name=bandwidth-72milc36_bw_v2.csv
model_id_name=Bandwidth-72milc36
data_name=custom

random_seed=2024
for pred_len in 96 #336 #128 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 216 \
      --e_layers $e_layers \
      --n_heads $n_heads \
      --d_model $d_model \
      --d_ff $d_ff \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      >logs/LongForecasting/fp32/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_el'$e_layers'_nh'$n_heads'_dm'$d_model'_df'$d_ff.log 
done
