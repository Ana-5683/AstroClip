set -e
CUDA_VISIBLE_DEVICES=0 python astroclip/astrophoto/downstream/embed_photometry.py --ckpt_path ../pretrained/astrophoto_03.ckpt --output_dir ../pretrained/embeddings/astrophoto_03 --embedding_dim 256
CUDA_VISIBLE_DEVICES=0 python astroclip/astrophoto/downstream/evaluate_redshift.py --embedding_dir ../pretrained/embeddings/astrophoto_03 --output_dir ../pretrained/results/astrophoto_03