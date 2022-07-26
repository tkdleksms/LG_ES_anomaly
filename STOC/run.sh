#!/bin/bash

# START=1
# END=67

# for i in $(seq $START $END);
# do
#     data_name='real_'$i.csv
#     echo $data_name
#     python main.py  --trend_learning=True \
#                     --training=True \
#                     --shuffle=False \
#                     --root_path='../yahoo_S5/A1Benchmark/' \
#                     --model='STOC' \
#                     --experiment_name='STOC_test' \
#                     --data_name $data_name \
#                     --patience=30 \
#                     --epochs=500 \
#                     --batch_size=128
# done


START=1
END=1
for i in $(seq $START $END);
do
    data_name='synthetic_'$i.csv
    echo $data_name
    python main.py  --trend_learning=False \
                    --training=True \
                    --shuffle=False \
                    --root_path='../yahoo_S5/A2Benchmark/' \
                    --model='STOC' \
                    --experiment_name='STOC_test' \
                    --data_name $data_name \
                    --patience=300 \
                    --epochs=30 \
                    --batch_size=128
done

# START=1
# END=100
# for i in $(seq $START $END);
# do
#     data_name='A3Benchmark-TS'$i.csv
#     echo $data_name
#     python main.py  --trend_learning=True \
#                 --training=True \
#                 --shuffle=False \
#                 --root_path='../yahoo_S5/A3Benchmark/' \
#                 --model='STOC' \
#                 --experiment_name='STOC_test' \
#                 --data_name $data_name \
#                 --patience=300 \
#                 --epochs=500 \
#                 --batch_size=128
# done

# START=1
# END=100
# for i in $(seq $START $END);
# do
#     data_name='A4Benchmark-TS'$i.csv
#     echo $data_name
#     python main.py  --trend_learning=True \
#                 --training=True \
#                 --shuffle=False \
#                 --root_path='../yahoo_S5/A4Benchmark/' \
#                 --model='STOC' \
#                 --experiment_name='STOC_test' \
#                 --data_name $data_name \
#                 --patience=300 \
#                 --epochs=500 \
#                 --batch_size=128
# done
