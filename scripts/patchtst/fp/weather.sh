seq_len=512
model=patchtst
nbits=16
prc=fp$nbits

root_path=./dataset/
dataset=weather.csv
data_name=weather

lr=1e-4
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

epochs=50
mgn=inf
seed=42

for pred_len in 192 336 720 
do
    run_id=$model'_'$data_name'_'$seq_len'_'$pred_len'_'$prc'_win-he_pe-rnd_gn-'$mgn'_ps02_pt02_ff02_rs'$seed
     
    python -u runexp.py \
      --random_seed $seed \
      --run_id $run_id \
      --mode 'fp' \
      --nbits $nbits \
      --bf16 1 \
      --do_train 1 \
      --model $model \
      --root_path $root_path \
      --dataset $dataset \
      --data_name $data_name \
      --features M \
      --context_length $seq_len \
      --prediction_length $pred_len \
      --num_input_channels 21 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --positional_dropout 0.2 \
      --path_dropout 0.2 \
      --ff_dropout 0.2 \
      --positional_encoding random \
      --train_epochs $epochs \
      --batch_size 512 \
      --learning_rate 1e-4 \
      --max_grad_norm $mgn \
      >logs/$model/$prc/$run_id'_bf16'.log
done
