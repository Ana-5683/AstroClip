# preprocess_data运行
把mean和std换成对应数据集,另外dr16q_v4的测光数据并未做处理
``` bash
python downstream_tasks/preprocess_data.py --dset dr16q_v4 --name c48 --size 48
```

# 阅读do.sh看如何执行get_embedding.py和redshift.py