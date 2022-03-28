echo "begin"
if [ -d "tmp/" ];then
  rm -rf tmp
fi

save=/mnt2/songxiao/ecrec/test
#infer=/mnt2/songxiao/ecrec/test/infer_result.txt
#infer_table=/mnt2/songxiao/memory/book/book_10w_test_new.txt
#data=/mnt2/songxiao/memory/book/book_10w_train_new.txt,/mnt2/songxiao/memory/book/book_10w_test_new.txt



python train_hub/ecrec.py \
       --task-type=onnx_export\
       --onnx_export_path=${save}\
       --load=${save}
