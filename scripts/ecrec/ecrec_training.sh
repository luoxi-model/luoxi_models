echo "begin"
if [ -d "tmp/" ];then
  rm -rf tmp
fi

save=/mnt2/songxiao/ecrec/test
#infer=/mnt2/songxiao/ecrec/test/infer_result.txt
#infer_table=/mnt2/songxiao/memory/book/book_10w_test_new.txt
data=/mnt2/songxiao/memory/book/book_10w_train_new.txt,/mnt2/songxiao/memory/book/book_10w_test_new.txt


python train_hub/ecrec.py \
       --save-interval=200 \
       --lr=0.0001 \
       --eval-interval=200 \
       --eval-iters=135 \
       --save=${save} \
       --tables=${data} \
       --task-type=train \
       --num-epochs=2 \
       --weight-decay=0.01 \
       --batch-size=128 \
       --optimizer='adam' \
       --find-unused-parameters
#       --load=${save}
