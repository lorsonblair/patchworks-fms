seq_len=512
model=patchtst
nbits=16
prc=fp$nbits

root_path=./dataset/
dataset=electricity.csv
data_name=electricity

itr=1

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

for pred_len in 192 336 720
do
    run_id=$model'_'$data_name'_'$seq_len'_'$pred_len'_'$prc'_win-he_pe-rnd_gn-'$mgn'_ps01_pt02_ff02_ep'$eph'_rs'$seed 

    python -u runexp.py \
      --run_id $run_id \
      --random_seed $seed \
      --mode fp \
      --nbits $nbits \
      --do_train 1 \
      --model $model \
      --bf16 1 \
      --root_path $root_path \
      --dataset $dataset \
      --data_name $data_name \
      --features M \
      --context_length $seq_len \
      --prediction_length $pred_len \
      --num_input_channels 321 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --positional_dropout 0.1\
      --ff_dropout 0.2\
      --path_dropout 0.2\
      --train_epochs $eph \
      --batch_size 64 \
      --learning_rate 1e-4 \
      --max_grad_norm $mgn \
      >logs/$model/$prc/$run_id.log
done
