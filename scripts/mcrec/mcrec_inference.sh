echo "begin"
time="2022032305" # you need to specify your log file suffix
echo ${time}

rm -rf tmp

model_prefix=o # c, o, mc
model_name=${model_prefix}rec
load=./${model_name}_${time}
data=./dataset/${model_prefix}rec_training.txt,./dataset/${model_prefix}rec_test.txt
outputs=./${model_name}_${time}/inference_result.txt

echo ${data}

nohup python -m torch.distributed.launch --master_port 28489 train_hub/mcrec.py \
       --save-interval=1000 \
       --optimizer=adam \
       --lr=0.001 \
       --weight-decay=0.0 \
       --eval-interval=100 \
       --task-type=inference \
       --load=${load} \
       --outputs=${outputs} \
       --tables=${data} \
       --log-interval=100 \
       --batch-size=512 \
       --model_type=${model_name} \
       >${model_name}_${time}_inference.log 2>&1 &
