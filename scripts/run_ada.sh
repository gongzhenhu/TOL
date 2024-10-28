export CUDA_VISIBLE_DEVICES=0

cd src

for TASK in aste; do
    for DATA in laptop14; do
            for SEED in 10 20 30 40 50 60 70 80 90 100; do
                    for num_pretrained_epochs in 5; do
                        OUT_DIR="../outputs/$TASK/${DATA}/$SEED"
                        mkdir -p $OUT_DIR
                        python main.py \
                            --data_path "../data/" \
                            --dataset $DATA \
                            --model_name_or_path ../pretrained/t5-base \
                            --output_dir $OUT_DIR \
                            --num_train_epochs 20 \
                            --task $TASK \
                            --seed $SEED \
                            --train_batch_size 16 \
                            --gradient_accumulation_steps 1 \
                            --learning_rate 1e-4 \
                            --lowercase \
                            --eval_batch_size 64 \
                            --constrained_decode \
                            --do_train \
                            --adaptive_order ada\
                            --num_pretrained_epochs $num_pretrained_epochs\
                            | tee ${OUT_DIR}/train_${order}_${SEED}_${SVP_TYPE}.log \
                            2> ${OUT_DIR}/train_${order}_${SEED}_${SVP_TYPE}.err
            done
        done
    done
done
