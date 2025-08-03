LOG_DIR="./logs"
#7.8 AAAI第一天

7.17
More Clients
##CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_40.yaml
#CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Other/iid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_40.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_40.txt 2>&1 &
#Tiny_ImageNet_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_10.yaml
#CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Other/iid \
#    --config-name Tiny_ImageNet_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_10.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_10.txt 2>&1 &
##Tiny_ImageNet_100_10Prompt_Pool_Incremental_40_sta_01_cross_10_client_10.yaml
#CUDA_VISIBLE_DEVICES=7 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Other/iid \
#    --config-name Tiny_ImageNet_100_10Prompt_Pool_Incremental_40_sta_01_cross_10_client_10.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_100_10Prompt_Pool_Incremental_40_sta_01_cross_10_client_10.txt 2>&1 &
##
###Tiny_ImageNet_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_20.yaml
##CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_10_niid.yaml
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Other/niid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_10_niid.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_10_niid.txt 2>&1 &
###
#####CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_20_niid.yaml
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Other/niid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_20_niid.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_20_niid.txt 2>&1 &
##
####CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_40_niid.yaml
#CUDA_VISIBLE_DEVICES=5 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Other/niid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_40_niid.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_40_niid.txt 2>&1 &
#
#FedKNOW
#Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress.yaml
#CUDA_VISIBLE_DEVICES=3 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedKNOW/Other/iid \
#    --config-name Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress.txt 2>&1 &
##Tiny_ImageNet_10FedKNOW_Incremental_40_no_compress.yaml
#CUDA_VISIBLE_DEVICES=3 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedKNOW/Other/iid \
#    --config-name Tiny_ImageNet_10FedKNOW_Incremental_40_no_compress.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedKNOW_Incremental_40_no_compress.txt 2>&1 &
#
#
##10_1_FedKNOW_Incremental10_no_compress_client_20.yaml
#CUDA_VISIBLE_DEVICES=2 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedKNOW/Other/iid \
#    --config-name 10_1_FedKNOW_Incremental10_no_compress_client_20.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/10_1_FedKNOW_Incremental10_no_compress_client_20.txt 2>&1 &



##10_1_FedKNOW_Incremental10_no_compress_client_40.yaml
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedKNOW/Other/iid \
#    --config-name 10_1_FedKNOW_Incremental10_no_compress_client_40.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/10_1_FedKNOW_Incremental10_no_compress_client_40.txt 2>&1 &
##
#
##10_1_FedKNOW_Incremental10_no_compress_client_40.yaml
#
#CUDA_VISIBLE_DEVICES=2 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedKNOW/Other/niid \
#    --config-name 10_1_FedKNOW_Incremental10_no_compress_client_40.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/10_1_FedKNOW_Incremental10_no_compress_client_40_niid.txt 2>&1 &

#
##
###CIFAR_100_10Prompt_Pool_Incremental_20_no_aux_loss_client_5_niid_05.yaml
#CUDA_VISIBLE_DEVICES=3 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Other \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_no_aux_loss_client_5_niid_05.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_no_aux_loss_client_5_niid_05.txt 2>&1 &



DomainNet
#10_1_MoAFCL_no_compress.yaml
#CUDA_VISIBLE_DEVICES=7 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/MoAFCL/DomainNet \
#    --config-name 10_1_MoAFCL_no_compress.yaml\
#    dataset_root="../data/" \
#    > $LOG_DIR/10_1_MoAFCL_no_compress.txt 2>&1 &




#20clients
#10_1_FedWEIT_Incremental10_no_compress_client20.yaml
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedWEIT/iid \
#    --config-name 10_1_FedWEIT_Incremental10_no_compress_client20.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/10_1_FedWEIT_Incremental10_no_compress_client20.txt 2>&1 &
#
##10_1_FedCLIP_Incremental10_no_compress_client20.yaml
#CUDA_VISIBLE_DEVICES=2 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedCLIP/iid \
#    --config-name 10_1_FedCLIP_Incremental10_no_compress_client20.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/10_1_FedCLIP_Incremental10_no_compress_client20.txt 2>&1 &
#
##10_1_FedCPrompt_Incremental10_no_compress_client20.yaml
#CUDA_VISIBLE_DEVICES=3 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedCPrompt/iid \
#    --config-name 10_1_FedCPrompt_Incremental10_no_compress_client20.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/10_1_FedCPrompt_Incremental10_no_compress_client20.txt 2>&1 &

#10_1_MoAFCL_Incremental10_no_compress_client20.yaml
#TODO
#CUDA_VISIBLE_DEVICES=4 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/MoAFCL/iid \
#    --config-name 10_1_MoAFCL_Incremental10_no_compress_client20.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/10_1_MoAFCL_Incremental10_no_compress_client20.txt 2>&1 &

##CIFAR_100_10Prompt_Pool_Incremental_10_no_aux_loss_client_20.yaml
#CUDA_VISIBLE_DEVICES=5 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Other/iid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_10_no_aux_loss_client_20.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_10_no_aux_loss_client_20.txt 2>&1 &
#
##CIFAR_100_10Prompt_Pool_Incremental_10_sta_01_cross_10_client_20.yaml
#CUDA_VISIBLE_DEVICES=6 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Other/iid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_10_sta_01_cross_10_client_20.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_10_sta_01_cross_10_client_20.txt 2>&1 &



#Tiny_ImageNet_100_10_Only_Prompt_Incremental_40_sta_01_cross_10_client_5.yaml
#CUDA_VISIBLE_DEVICES=2 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/niid \
#    --config-name Tiny_ImageNet_100_10_Only_Prompt_Incremental_40_sta_01_cross_10_client_5.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_100_10_Only_Prompt_Incremental_40_sta_01_cross_10_client_5_niid.txt 2>&1 &
#
#
##Tiny_ImageNet_100_10_Only_MoE_Incremental_40_sta_01_cross_10_client_5.yaml
#CUDA_VISIBLE_DEVICES=3 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/niid \
#    --config-name Tiny_ImageNet_100_10_Only_MoE_Incremental_40_sta_01_cross_10_client_5.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_100_10_Only_MoE_Incremental_40_sta_01_cross_10_client_5_niid.txt 2>&1 &



#FLOPS -- Prompt Pool
#CUDA_VISIBLE_DEVICES=4 /usr/bin/time -f "Time elapsed: %E" python analyze_flops.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/iid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5.yaml\

#FLOPS moafcl
#CUDA_VISIBLE_DEVICES=4 /usr/bin/time -f "Time elapsed: %E" python analyze_flops_moafcl.py \
#    --config-path configs/708/MoAFCL/iid \
#    --config-name 10_1_MoAFCL_Incremental20_no_compress.yaml\


#CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared128.yaml
#CUDA_VISIBLE_DEVICES=4 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/iid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared128.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared128.txt 2>&1 &
#
##CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared128.yaml
#CUDA_VISIBLE_DEVICES=4 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/niid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared128.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared128_niid.txt 2>&1 &


#CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared256.yaml
#CUDA_VISIBLE_DEVICES=5 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/iid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared256.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared256.txt 2>&1 &
#
##CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared256.yaml
#CUDA_VISIBLE_DEVICES=5 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/niid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared256.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared256_niid.txt 2>&1 &


#CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared128_drop04.yaml
#CUDA_VISIBLE_DEVICES=6 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/iid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared128_drop04.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared128_drop04.txt 2>&1 &
#
##CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared128_drop04.yaml
#CUDA_VISIBLE_DEVICES=6 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/niid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared128_drop04.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared128_drop04_niid.txt 2>&1 &



#CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared256_drop04.yaml
#CUDA_VISIBLE_DEVICES=7 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/iid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared256_drop04.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared256_drop04.txt 2>&1 &
#
##CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared256_drop04.yaml
#CUDA_VISIBLE_DEVICES=7 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/niid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared256_drop04.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_shared256_drop04_niid.txt 2>&1 &

#CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_drop04.yaml
#CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/iid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_drop04.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_drop04.txt 2>&1 &
#
##CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_drop04.yaml
#CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/niid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_drop04.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_drop04_niid.txt 2>&1 &
#
##Tiny_ImageNet_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_drop04.yaml
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/iid \
#    --config-name Tiny_ImageNet_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_drop04.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_drop04.txt 2>&1 &
#
##Tiny_ImageNet_10Prompt_Pool_Incremental_40_sta_01_cross_10_client_5_drop04.yaml
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/iid \
#    --config-name Tiny_ImageNet_10Prompt_Pool_Incremental_40_sta_01_cross_10_client_5_drop04.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10Prompt_Pool_Incremental_40_sta_01_cross_10_client_5_drop04.txt 2>&1 &
#
##Tiny_ImageNet_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_drop04.yaml
#CUDA_VISIBLE_DEVICES=2 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Ablation/niid \
#    --config-name Tiny_ImageNet_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_drop04.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_5_drop04_niid.txt 2>&1 &




#7.29
#10_1_FedCLIP_Incremental10_no_compress_client10.yaml
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedCLIP/iid \
#    --config-name 10_1_FedCLIP_Incremental10_no_compress_client10.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/10_1_FedCLIP_Incremental10_no_compress_client10.txt 2>&1 &
#
##10_1_FedCLIP_Incremental10_no_compress_client15.yaml
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedCLIP/iid \
#    --config-name 10_1_FedCLIP_Incremental10_no_compress_client15.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/10_1_FedCLIP_Incremental10_no_compress_client15.txt 2>&1 &
#
##10_1_FedWEIT_Incremental10_no_compress_client10.yaml
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedWEIT/iid \
#    --config-name 10_1_FedWEIT_Incremental10_no_compress_client10.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/10_1_FedWEIT_Incremental10_no_compress_client10.txt 2>&1 &
#
##10_1_FedWEIT_Incremental10_no_compress_client15.yaml
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedWEIT/iid \
#    --config-name 10_1_FedWEIT_Incremental10_no_compress_client15.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/10_1_FedWEIT_Incremental10_no_compress_client15.txt 2>&1 &

#Tiny_ImageNet_10FedCLIP_Incremental_20_no_compress_client10.yaml
#CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedCLIP/iid \
#    --config-name Tiny_ImageNet_10FedCLIP_Incremental_20_no_compress_client10.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedCLIP_Incremental_20_no_compress_client10.txt 2>&1 &
#
##Tiny_ImageNet_10FedCLIP_Incremental_20_no_compress_client15.yaml
#CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedCLIP/iid \
#    --config-name Tiny_ImageNet_10FedCLIP_Incremental_20_no_compress_client15.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedCLIP_Incremental_20_no_compress_client15.txt 2>&1 &
#
##Tiny_ImageNet_10FedCLIP_Incremental_20_no_compress_client20.yaml
#CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedCLIP/iid \
#    --config-name Tiny_ImageNet_10FedCLIP_Incremental_20_no_compress_client20.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedCLIP_Incremental_20_no_compress_client20.txt 2>&1 &
#
#
#
##Tiny_ImageNet_10FedCPrompt_Incremental_20_no_compress.yaml
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedCPrompt/iid \
#    --config-name Tiny_ImageNet_10FedCPrompt_Incremental_20_no_compress.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedCPrompt_Incremental_20_no_compress.txt 2>&1 &
#
##Tiny_ImageNet_10FedCPrompt_Incremental_20_no_compress_client10.yaml
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedCPrompt/iid \
#    --config-name Tiny_ImageNet_10FedCPrompt_Incremental_20_no_compress_client10.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedCPrompt_Incremental_20_no_compress_client10.txt 2>&1 &
##Tiny_ImageNet_10FedCPrompt_Incremental_20_no_compress_client15.yaml
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedCPrompt/iid \
#    --config-name Tiny_ImageNet_10FedCPrompt_Incremental_20_no_compress_client15.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedCPrompt_Incremental_20_no_compress_client15.txt 2>&1 &
#
##Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress.yaml
#CUDA_VISIBLE_DEVICES=2 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedKNOW/iid \
#    --config-name Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress.txt 2>&1 &
##Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress_client10.yaml
#CUDA_VISIBLE_DEVICES=2 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedKNOW/iid \
#    --config-name Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress_client10.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress_client10.txt 2>&1 &
##Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress.yaml
#CUDA_VISIBLE_DEVICES=7 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedKNOW/iid \
#    --config-name Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress.txt 2>&1 &
##Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress_client15.yaml
#CUDA_VISIBLE_DEVICES=7 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedKNOW/iid \
#    --config-name Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress_client15.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedKNOW_Incremental_20_no_compress_client15.txt 2>&1 &
#
##10_1_FedKNOW_Incremental10_no_compress_client_15.yaml
#CUDA_VISIBLE_DEVICES=3 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedKNOW/Other/iid \
#    --config-name 10_1_FedKNOW_Incremental10_no_compress_client_15.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/10_1_FedKNOW_Incremental10_no_compress_client_15.txt 2>&1 &
##10_1_FedKNOW_Incremental10_no_compress_client_25.yaml
#CUDA_VISIBLE_DEVICES=3 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedKNOW/Other/iid \
#    --config-name 10_1_FedKNOW_Incremental10_no_compress_client_25.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/10_1_FedKNOW_Incremental10_no_compress_client_25.txt 2>&1 &
#
##Tiny_ImageNet_10FedWEIT_Incremental_20_no_compress_client20.yaml
#CUDA_VISIBLE_DEVICES=3 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedWEIT/iid \
#    --config-name Tiny_ImageNet_10FedWEIT_Incremental_20_no_compress_client20.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedWEIT_Incremental_20_no_compress_client20.txt 2>&1 &
#
##Tiny_ImageNet_10FedWEIT_Incremental_20_no_compress_client10.yaml
#CUDA_VISIBLE_DEVICES=3 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedWEIT/iid \
#    --config-name Tiny_ImageNet_10FedWEIT_Incremental_20_no_compress_client10.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedWEIT_Incremental_20_no_compress_client10.txt 2>&1 &
#
##Tiny_ImageNet_10FedWEIT_Incremental_20_no_compress_client15.yaml
#CUDA_VISIBLE_DEVICES=3 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/FedWEIT/iid \
#    --config-name Tiny_ImageNet_10FedWEIT_Incremental_20_no_compress_client15.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10FedWEIT_Incremental_20_no_compress_client15.txt 2>&1 &
#
#
##Tiny_ImageNet_10MoAFCL_Incremental_20_no_compress_client20.yaml
#CUDA_VISIBLE_DEVICES=4 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/MoAFCL/iid \
#    --config-name Tiny_ImageNet_10MoAFCL_Incremental_20_no_compress_client20.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10MoAFCL_Incremental_20_no_compress_client20.txt 2>&1 &
##Tiny_ImageNet_10MoAFCL_Incremental_20_no_compress_client10.yaml
#CUDA_VISIBLE_DEVICES=4 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/MoAFCL/iid \
#    --config-name Tiny_ImageNet_10MoAFCL_Incremental_20_no_compress_client10.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10MoAFCL_Incremental_20_no_compress_client10.txt 2>&1 &
#
##Tiny_ImageNet_10MoAFCL_Incremental_20_no_compress_client15.yaml
#CUDA_VISIBLE_DEVICES=4 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/MoAFCL/iid \
#    --config-name Tiny_ImageNet_10MoAFCL_Incremental_20_no_compress_client15.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_10MoAFCL_Incremental_20_no_compress_client15.txt 2>&1 &
#
##Tiny_ImageNet_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_20.yaml
#CUDA_VISIBLE_DEVICES=5 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Other/iid \
#    --config-name Tiny_ImageNet_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_20.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_20.txt 2>&1 &
#
##Tiny_ImageNet_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_10.yaml
#CUDA_VISIBLE_DEVICES=5 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Other/iid \
#    --config-name Tiny_ImageNet_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_10.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_10.txt 2>&1 &
#
##Tiny_ImageNet_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_15.yaml
#CUDA_VISIBLE_DEVICES=6 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Other/iid \
#    --config-name Tiny_ImageNet_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_15.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/tinyimagenet.yaml" \
#    > $LOG_DIR/Tiny_ImageNet_100_10Prompt_Pool_Incremental_20_sta_01_cross_10_client_15.txt 2>&1 &
#
#
#CIFAR_100_10Prompt_Pool_Incremental_10_sta_01_cross_10_client_15.yaml
#CUDA_VISIBLE_DEVICES=6 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Other/iid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_10_sta_01_cross_10_client_15.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_10_sta_01_cross_10_client_15.txt 2>&1 &
#
#
##CIFAR_100_10Prompt_Pool_Incremental_10_sta_01_cross_10_client_10.yaml
#CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f "Time elapsed: %E" python main.py \
#    --config-path configs/708/Prompt_pool_fedavg/Other/iid \
#    --config-name CIFAR_100_10Prompt_Pool_Incremental_10_sta_01_cross_10_client_10.yaml\
#    dataset_root="../data/" \
#    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_10_sta_01_cross_10_client_10.txt 2>&1 &
#
##CIFAR_100_10Prompt_Pool_Incremental_10_sta_01_cross_10_client_20.yaml

CUDA_VISIBLE_DEVICES=2 /usr/bin/time -f "Time elapsed: %E" python main.py \
    --config-path configs/708/Prompt_pool_fedavg/Other/iid \
    --config-name CIFAR_100_10Prompt_Pool_Incremental_10_sta_01_cross_10_client_20.yaml\
    dataset_root="../data/" \
    class_order="class_orders/cifar100.yaml" \
#    > $LOG_DIR/CIFAR_100_10Prompt_Pool_Incremental_10_sta_01_cross_10_client_20.txt 2>&1 &


