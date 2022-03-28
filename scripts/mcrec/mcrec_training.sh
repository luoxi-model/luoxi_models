echo "begin"
time=$(date "+%Y%m%d%H")
echo ${time}

model_prefix=o # c, o, mc
model_name=${model_prefix}rec
save=./${model_name}_${time}
data=./dataset/${model_prefix}rec_training.txt,./dataset/${model_prefix}rec_test.txt

echo ${data}

nohup python -m torch.distributed.launch --master_port 28489 train_hub/mcrec.py \
       --save-interval=1000 \
       --optimizer=adam \
       --lr=0.001 \
       --weight-decay=0.0 \
       --eval-interval=100 \
       --save=${save} \
       --tables=${data} \
       --log-interval=100 \
       --batch-size=512 \
       --model_type=${model_name} \
       >${model_name}_${time}.log 2>&1 &
