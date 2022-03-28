echo "begin"
time="2022032305" # you need to specify your log file suffix
echo ${time}

rm -rf tmp

model_prefix=o # o or mc
model_name=${model_prefix}rec
load=./${model_name}_${time}
data=./dataset/${model_prefix}rec_training.txt,./dataset/${model_prefix}rec_test.txt
onnx_output=./${model_name}_${time}/output.onnx

echo ${data}

nohup python -m torch.distributed.launch --master_port 28489 train_hub/mcrec.py \
       --save-interval=1000 \
       --optimizer=adam \
       --lr=0.001 \
       --weight-decay=0.0 \
       --eval-interval=100 \
       --task-type=onnx_export \
       --load=${load} \
       --onnx_export_path=${onnx_output} \
       --tables=${data} \
       --log-interval=100 \
       --batch-size=512 \
       --model_type=${model_name} \
       >${model_name}_${time}_export.log 2>&1 &
