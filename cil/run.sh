LOG_DIR="./logs"
CUDA_VISIBLE_DEVICES=2 /usr/bin/time -f "Time elapsed: %E" python main.py \
    --config-path configs \
    --config-name  CIFAR_100_FedDuet_Incremental_10_iid.yaml\
    dataset_root="../data/" \
    class_order="class_orders/cifar100.yaml" \
    # > $LOG_DIR/CIFAR_100_FedDuet_Incremental_10_iid.log 2>&1 &
