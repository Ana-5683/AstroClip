set -e
python mean_std_photo.py --dset_dir data_sd_z15
python mean_std_photo.py --dset_dir data_mine_z15

set -e
python mean_std_image.py --dset_dir data_sd_z15
python mean_std_image.py --dset_dir data_mine_z15 --batch_size 64

set -e
python analyze_parallel.py --dset_dir data_mine_z15 --num_workers 2
python analyze_parallel.py --dset_dir data_sd_z15 --num_workers 2


