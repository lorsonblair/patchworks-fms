seq_len=512
model=ttmx
nbits=32
prc=fp$nbits

root_path=./dataset/
dataset=electricity.csv
data_name=electricity

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
ephs=90
seed=42


for pred_len in 96 192 336 720
do
    run_id=$model'_'$data_name'_'$seq_len'_'$pred_len'_'$prc'_win-he_pe-rnd_gated_dc_gn-'$mgn'_dr02_hd02_ep'$ephs'_rs'$seed

    python -u runexp.py \
      --random_seed $seed \
      --run_id $run_id \
      --mode fp \
      --do_train 1 \
      --model $model \
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
      --dropout 0.2\
      --head_dropout 0.2\
      --gated_attn \
      --use_decoder \
      --train_epochs $ephs \
      --batch_size 64 \
      --num_workers 5 \
      --learning_rate 1e-4 \
      --weight_decay 0 \
      --max_grad_norm $mgn \
      >logs/$model/$prc/$run_id.log
done
