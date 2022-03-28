echo "begin"
time=$(date "+%Y%m%d%H")
echo ${time}
if [ -d "tmp/" ];then
  rm -rf tmp
fi

model_name=CIGAR
load=/home/fay.cyf/fay.cyf/cigar/CIGAR_2022032315
data=/home/fay.cyf/fay.cyf/cigar/cigar_alimama_train.txt,/home/fay.cyf/fay.cyf/cigar/cigar_alimama_test_10000.txt
outputs=/home/fay.cyf/fay.cyf/cigar/CIGAR_2022032315/result.txt

echo ${data}

nohup  python -m torch.distributed.launch --master_port 28486 train_hub/cigar.py \
       --save-interval=10000 \
       --lr=0.001 \
       --optimizer=adam \
       --weight-decay=0.0 \
       --column_len=29 \
       --eval-interval=100 \
       --task-type=inference \
       --load=${load} \
       --outputs=${outputs} \
       --tables=${data} \
       --log-interval=100 \
       --batch-size=512 \
       --model=${model_name} \
       >${model_name}_${time}_infer.log 2>&1 &