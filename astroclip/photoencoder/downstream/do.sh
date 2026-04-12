set -e
CUDA_VISIBLE_DEVICES=4 python astroclip/photoencoder/downstream/embed_photometry.py --ckpt photoencoder_00
CUDA_VISIBLE_DEVICES=4 python astroclip/photoencoder/downstream/evaluate_redshift.py --model_name photometry --ckpt photoencoder_00