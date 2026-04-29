









# dino
set -e
CUDA_VISIBLE_DEVICES=2 python downstream_tasks/get_embedding.py --dset data_g3_z --name c48_pe --model astrodino --ckpt astrodino_63 && CUDA_VISIBLE_DEVICES=2 python downstream_tasks/redshift.py --model astrodino --ckpt astrodino_63

# clip
set -e
CUDA_VISIBLE_DEVICES=0 python downstream_tasks/get_embedding.py --model astroclip_image  --dset data_g3_z --name c48_pe --ckpt astroclip_44
CUDA_VISIBLE_DEVICES=0 python downstream_tasks/redshift.py --model astroclip_image --ckpt astroclip_44

# clip ip
set -e
CUDA_VISIBLE_DEVICES=2 python downstream_tasks/get_embedding_ip.py --model astroclip_ip --ckpt astroclip_62 &&
CUDA_VISIBLE_DEVICES=2 python downstream_tasks/redshift.py --model astroclip_ip --ckpt astroclip_62 &&



