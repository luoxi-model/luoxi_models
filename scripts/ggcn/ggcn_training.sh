echo "begin"
time=$(date "+%Y%m%d%H")
echo ${time}
model_name=GGCN
save=/${model_name}_${time}
data=/mnt2/pytorch/GGCN/ppi,/mnt2/pytorch/GGCN/ppi

python -m torch.distributed.launch --master_port 2849 train_hub/GGCN.py \
       --save-interval=200 \
       --optimizer=adam \
       --lr=0.001 \
       --weight-decay=0.0 \
       --column_len=29 \
       --eval-interval=100 \
       --save=${save} \
       --tables=${data} \
       --log-interval=50 \
       --batch-size=1 \
       --model=${model_name} \
       >${model_name}_${time}.log 2>&1 &

echo ${data}
