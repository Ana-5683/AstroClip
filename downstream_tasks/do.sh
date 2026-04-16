









# dino
set -e
CUDA_VISIBLE_DEVICES=2 python downstream_tasks/get_embedding.py --dset data_g3_z --name c48_pe --model astrodino --ckpt astrodino_63 && CUDA_VISIBLE_DEVICES=2 python downstream_tasks/redshift.py --model astrodino --ckpt astrodino_63
CUDA_VISIBLE_DEVICES=4 python downstream_tasks/get_embedding.py --dset data_g3_z --name c48_pe --model astrodino --ckpt astrodino_64 && CUDA_VISIBLE_DEVICES=4 python downstream_tasks/redshift.py --model astrodino --ckpt astrodino_64

set -e
CUDA_VISIBLE_DEVICES=0 python downstream_tasks/get_embedding.py --dset data_g3_z --name c48_pe --model astrodino --ckpt astrodino_66 && CUDA_VISIBLE_DEVICES=0 python downstream_tasks/redshift.py --model astrodino --ckpt astrodino_66



# clip
set -e
CUDA_VISIBLE_DEVICES=0 python downstream_tasks/get_embedding.py --model astroclip_image  --dset data_g3_z --name c48_pe --ckpt astroclip_44
CUDA_VISIBLE_DEVICES=0 python downstream_tasks/redshift.py --model astroclip_image --ckpt astroclip_44

CUDA_VISIBLE_DEVICES=0 python downstream_tasks/get_embedding.py --model astroclip_spectrum  --dset data_g3_z --name c48_pe --ckpt astroclip_44
CUDA_VISIBLE_DEVICES=0 python downstream_tasks/redshift.py --model astroclip_spectrum --ckpt astroclip_44

# clip ip
set -e
CUDA_VISIBLE_DEVICES=2 python downstream_tasks/get_embedding_ip.py --model astroclip_ip --ckpt astroclip_62 &&
CUDA_VISIBLE_DEVICES=2 python downstream_tasks/redshift.py --model astroclip_ip --ckpt astroclip_62 &&
CUDA_VISIBLE_DEVICES=2 python downstream_tasks/get_embedding.py --model astroclip_spectrum --ckpt astroclip_62 &&
CUDA_VISIBLE_DEVICES=2 python downstream_tasks/redshift.py --model astroclip_spectrum --ckpt astroclip_62


#67W
set -e
CUDA_VISIBLE_DEVICES=3 python downstream_tasks/data67W/get_embedding.py --model astrodino --ckpt astrodino_31 --output_dir ../pretrained/embeddings/astrodino_31
CUDA_VISIBLE_DEVICES=3 python downstream_tasks/data67W/redshift.py --model astrodino --pretrained_dir ../pretrained/embeddings/astrodino_31 --output_dir ../pretrained/results/astrodino_31


