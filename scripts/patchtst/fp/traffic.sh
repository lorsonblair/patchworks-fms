seq_len=512
model=patchtst
nbits=32
prc=fp$nbits

root_path=./dataset/
dataset=traffic.csv
data_name=traffic

if [ ! -d "./logs" ]; then
     mkdir ./logs
fi

if [ ! -d "./logs/$model" ]; then
    mkdir ./logs/$model
fi

if [ ! -d "./logs/$model/$prc" ]; then
    mkdir ./logs/$model/$prc
fi

mgn=inf
eph=90
seed=42

for pred_len in 96 #192 336 720
do
    run_id=$model'_'$data_name'_'$seq_len'_'$pred_len'_'$prc'_pe-rnd_gn-'$mgn'_ps02_pt02_ff02_ep'$eph'_rs'$seed'_embed-0'

    python -u runexp.py \
      --run_id $run_id \
      --random_seed $seed \
      --mode "fp" \
      --do_train 1 \
      --model $model \
      --root_path $root_path \
      --dataset $dataset \
      --features M \
      --context_length $seq_len \
      --prediction_length $pred_len \
      --num_input_channels 862 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --positional_dropout 0.2\
      --ff_dropout 0.2\
      --path_dropout 0.2\
      --train_epochs $eph\
      --batch_size 24 \
      --learning_rate 1e-4 \
      --scale_lr 1 \
      >logs/$model/$prc/$run_id.log 
done
