echo "begin"

cd ../../

if [ -d "tmp/" ];then
  rm -rf tmp
fi

save=`pwd`/data/output/pin/result

dataset=""
train_file="${dataset}/train.txt"
test_file="${dataset}/test.txt"
data="${train_file},${test_file}"

ARCH_CONF_FILE=`pwd`/scripts/pinrec/movielens_conf.json

python -m torch.distributed.launch --master_port 35213 train_hub/pinrec.py \
       --batch-size=128 \
       --num-epochs=12 \
       --clip-grad=0.0 \
       --train-iters=40000 \
       --log-interval=100 \
       --save-interval=9708 \
       --lr=0.001 \
       --eval-interval=40001 \
       --save=${save} \
       --tables=${data} \
       --model=pinrec \
       --group_num=5 \
       --arch_config=${ARCH_CONF_FILE} \
       --stage_switch_epoch=2 \
       --optimizer=adam \
       --backward-step-contains-in-forward-step