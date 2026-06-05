model=ttmx
seq_len=512
init_ckpt=none

root_path=./dataset/
dataset=electricity.csv
data_name=electricity

nbits=2
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
mgn=inf
qmode=reg-wa
wreg=1
areg=1
seed=42

for pred_len in 96 #336 720
do
    run_id=$model'_'$data_name'_'$seq_len'_'$pred_len'_'$prc'_'$qmode'_win-he_pe-rnd_gated_dc_gn-'$mgn'_dr02_hd02_ep'$eph'_rs'$seed

    python -u runexp.py \
      --run_id $run_id \
      --model $model \
      --random_seed $seed \
      --mode int \
      --qlast \
      --init_ckpt $init_ckpt \
      --nbits $nbits \
      --do_train 1 \
      --root_path $root_path \
      --dataset $dataset \
      --data_name $data_name \
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
      --num_input_channels 321 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --gated_attn \
      --use_decoder \
      --dropout 0.2 \
      --head_dropout 0.2 \
      --train_epochs $eph\
      --batch_size 64 \
      --learning_rate 1e-4 \
      --max_grad_norm $mgn \
      >logs/$model/$prc/$init_ckpt/$run_id.log
done
