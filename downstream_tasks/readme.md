## preprocess_data运行

``` bash
python downstream_tasks/preprocess_data.py --data_dir ../data/data_g3_z
```

## get_embedding运行
``` bash
CUDA_VISIBLE_DEVICES=7 python downstream_tasks/get_embedding.py --model astrodino --ckpt astrodino_02 --output_dir ../pretrained/embeddings/astrodino_02
```
## redshift运行

``` bash
CUDA_VISIBLE_DEVICES=7 python downstream_tasks/redshift.py --model astroclip_spectrum --pretrained_dir ../pretrained/embeddings/astrodino_04 --output_dir ../pretrained/results/astrodino_04
```