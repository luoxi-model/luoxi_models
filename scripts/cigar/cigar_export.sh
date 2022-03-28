echo "begin"
time=$(date "+%Y%m%d%H")
echo ${time}

model_name=CIGAR
save=/home/fay.cyf/fay.cyf/cigar
data=/home/fay.cyf/fay.cyf/cigar/cigar_alimama_test_10000.txt,/home/fay.cyf/fay.cyf/cigar/cigar_alimama_test_10000.txt
#onnx_export_path=/mnt2/yyang/cigar_onnx/save/
load=/home/fay.cyf/fay.cyf/cigar
load_model_name=CIGAR_2022032315
load_model_path=${load}/${load_model_name}/1000

echo ${data}

python -m torch.distributed.launch --master_port 29493 train_hub/cigar.py \
       --save-interval=10 \
       --optimizer=adam \
       --lr=0.001 \
       --weight-decay=0.0 \
       --column_len=29 \
       --eval-interval=10 \
       --save=${save}/${load_model_name} \
       --tables=${data} \
       --log-interval=10 \
       --batch-size=512 \
       --num-epochs=2 \
       --model=${model_name} \
       --task-type=onnx_export\
       --load=${load_model_path}