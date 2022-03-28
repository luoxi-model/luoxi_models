echo "begin"
time=$(date "+%Y%m%d%H")
echo ${time}
if [ -d "tmp/" ];then
  rm -rf tmp
fi

model_name=CIGAR
save=/home/fay.cyf/fay.cyf/cigar/${model_name}_${time}
data=/home/fay.cyf/fay.cyf/cigar/cigar_alimama_train.txt,/home/fay.cyf/fay.cyf/cigar/cigar_alimama_test_10000.txt


echo ${data}

nohup  python -m torch.distributed.launch --master_port 20848 train_hub/cigar.py \
       --save-interval=1000 \
       --optimizer=adam \
       --lr=0.001 \
       --weight-decay=0.0 \
       --column_len=29 \
       --eval-interval=100 \
       --save=${save} \
       --tables=${data} \
       --log-interval=100 \
       --batch-size=512 \
       --model=${model_name} \
       >${model_name}_${time}.log 2>&1 &