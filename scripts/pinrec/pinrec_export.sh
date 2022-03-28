echo "begin"

cd ../../

if [ -d "tmp/" ];then
  rm -rf tmp
fi

dataset=""
train_file="${dataset}/train.txt"
test_file="${dataset}/test.txt"
data="${train_file},${test_file}"

load=`pwd`/data/output/pinrec/result
save=`pwd`/data/output/pinrec/result
output=`pwd`/data/output/pinrec/result/rerank.txt

onnx_export_path=`pwd`/onnx_test/

ARCH_CONF_FILE=`pwd`/scripts/pinrec/movielens_conf.json

python -m torch.distributed.launch --master_port 12350 train_hub/pinrec.py \
       --task-type=inference \
       --batch-size=512 \
       --num-epochs=1 \
       --train-iters=40000 \
       --log-interval=100 \
       --save-interval=10000 \
       --lr=0.001 \
       --eval-interval=40001 \
       --load=${load} \
       --save=${save} \
       --tables=${data} \
       --model=pinrec \
       --group_num=5 \
       --arch_config=${ARCH_CONF_FILE} \
       --stage_switch_epoch=2 \
       --optimizer=adam \
       --outputs=${output} \
       --task-type=onnx_export \
       --onnx_export_path=${onnx_export_path}