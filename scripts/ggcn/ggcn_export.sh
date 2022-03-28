echo "begin"
time=$(date "+%Y%m%d%H")
echo ${time}

model_name=GGCN


echo ${data}

model_name=GGCN
save=/${model_name}_${time}
data=/mnt2/pytorch/GGCN/ppi,/mnt2/pytorch/GGCN/ppi

onnx_export_path=/mnt2/pytorch/save/
onnx_model_name=onnx_${model_name}_${time}.onnx
load_model_path=/GGCN_2022031401/3200/



python -m torch.distributed.launch --master_port 2819 train_hub/GGCN.py \
       --save-interval=200 \
       --optimizer=adam \
       --lr=0.001 \
       --weight-decay=0.0 \
       --column_len=29 \
       --eval-interval=100 \
       --save=${save} \
       --onnx_export_path=${onnx_export_path} \
       --onnx_model_name=${onnx_model_name} \
       --tables=${data} \
       --log-interval=50 \
       --batch-size=1 \
       --model=${model_name}\
      --task-type=onnx_export\
       --load_model_path=${load_model_path}
