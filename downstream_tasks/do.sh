









# dino
set -e
DSET=dr16q_v4
NAME=c48
MODEL=astrodino
CKPT=astrodino_67
CUDA_VISIBLE_DEVICES=2 python downstream_tasks/get_embedding.py --dset $DSET --name $NAME --model $MODEL --ckpt $CKPT && CUDA_VISIBLE_DEVICES=2 python downstream_tasks/redshift.py --model $MODEL --ckpt $CKPT

# clip
set -e
DSET=dr16q_v4
NAME=c48
MODEL=astroclip_image
CKPT=astroclip_44
CUDA_VISIBLE_DEVICES=0 python downstream_tasks/get_embedding.py --model $MODEL --dset $DSET --name $NAME --ckpt $CKPT
CUDA_VISIBLE_DEVICES=0 python downstream_tasks/redshift.py --model $MODEL --ckpt $CKPT

# clip ip(错误的示例，需修改)
set -e
MODEL=astroclip_ip
CKPT=astroclip_62
CUDA_VISIBLE_DEVICES=2 python downstream_tasks/get_embedding_ip.py --model $MODEL --ckpt $CKPT &&
CUDA_VISIBLE_DEVICES=2 python downstream_tasks/redshift.py --model $MODEL --ckpt $CKPT &&



