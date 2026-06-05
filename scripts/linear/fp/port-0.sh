# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/DLinear" ]; then
    mkdir ./logs/LongForecasting/DLinear
fi

seq_len=336
model_name=DLinear

root_path_name=./dataset/
data_path_name=milc-port-0-ts-sorted.csv
model_id_name=Port0
#data_name=custom

for pred_len in 192 #96 128 336 720
do
  python3 -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 --batch_size 128 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
