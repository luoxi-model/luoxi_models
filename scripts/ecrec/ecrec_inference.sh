echo "begin"
if [ -d "tmp/" ];then
  rm -rf tmp
fi

save=/mnt2/songxiao/ecrec/test
infer=/mnt2/songxiao/ecrec/test/infer_result.txt
infer_table=/mnt2/songxiao/memory/book/book_10w_test_new.txt
data=/mnt2/songxiao/memory/book/book_10w_train_new.txt,/mnt2/songxiao/memory/book/book_10w_test_new.txt


python train_hub/ecrec.py \
       --eval-iters=135 \
       --save=${save} \
       --task-type=inference \
       --outputs=${infer} \
       --infer_table=${infer_table} \
       --batch-size=128 \
       --load=${save}
