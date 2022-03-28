echo "begin"
time=$(date "+%Y%m%d%H")
echo ${time}
model_name=GGCN
load=/GGCN_2022031401/
data=/mnt2/pytorch/GGCN/ppi,/mnt2/pytorch/GGCN/ppi
rm -rf tmp
outputs=/GGCN_2022031401/result.txt

echo ${data}


  python -m torch.distributed.launch --master_port 2849 train_hub/GGCN.py \
       --save-interval=200 \
       --optimizer=adam \
       --lr=0.001 \
       --weight-decay=0.0 \
       --column_len=29 \
       --task-type=inference \
       --eval-interval=100 \
       --load=${load} \
       --outputs=${outputs} \
       --tables=${data} \
       --log-interval=50 \
       --batch-size=1 \
       --model=${model_name} \
       >${model_name}_${time}.log 2>&1 &

