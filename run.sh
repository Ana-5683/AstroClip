#astroclip
set -e
CUDA_VISIBLE_DEVICES=2,6 torchrun --nproc_per_node=2 --master_port=29503 astroclip/trainer.py fit -c configs/astroclip_qi.yaml

#astrodino
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29503 astroclip/astrodino/trainer.py -c astroclip/astrodino/config.yaml --run-name 66

#photoencoder
set -e
CUDA_VISIBLE_DEVICES=2,4,5,6 python astroclip/photoencoder/trainer.py --run_name 01

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 astroclip/astrodino/trainer.py -c astroclip/astrodino/config_m.yaml --run_name 37

