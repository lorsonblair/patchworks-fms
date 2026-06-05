seq_len=512
model=patchtst
init_ckpt=none

root_path=./dataset/
dataset=traffic.csv
data_name=traffic

epochs=90
amg=0
nbits=4
prc=int$nbits

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/$model" ]; then
    mkdir ./logs/$model
fi

if [ ! -d "./logs/$model/$prc" ]; then
    mkdir ./logs/$model/$prc
fi

if [ ! -d "./logs/$model/$prc/$init_ckpt" ]; then
    mkdir ./logs/$model/$prc/$init_ckpt
fi

eph=90
mgn=0.1
amg=0
qmode=df
wreg=0
areg=0

seed=42

for pred_len in 96
do
    run_id=$model'_'$data_name'_'$seq_len'_'$pred_len'_'$prc'_'$qmode'_win-he_pe-rnd_gn-01_ps02_pt02_ff02_ep'$eph'_rs'$seed'_scale-lr'

    python -u runexp.py \
      --run_id $run_id \
      --model $model \
      --random_seed $seed \
      --mode int \
      --init_ckpt $init_ckpt \
      --qlast \
      --nbits $nbits \
      --do_train 1 \
      --root_path $root_path \
      --dataset $dataset \
      --data_name $data_name \
      --num_workers 5 \
      --nbits_w $nbits \
      --nbits_a $nbits \
      --nbits_w_head $nbits \
      --nbits_a_head $nbits \
      --w_reg $wreg \
      --a_reg $areg \
      --nbits_bmm1 $nbits \
      --nbits_bmm2 8 \
      --align_zero \
      --context_length $seq_len \
      --prediction_length $pred_len \
      --num_input_channels 862 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --positional_dropout 0.2\
      --path_dropout 0.2\
      --ff_dropout 0.2\
      --train_epochs $eph \
      --batch_size 24 \
      --learning_rate 1e-4 \
      --scale_lr 1 \
      --max_grad_norm $mgn \
      >logs/$model/$prc/$init_ckpt/$run_id.log
done
