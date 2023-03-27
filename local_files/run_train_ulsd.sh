# 默认的数据位置在/dataset下, dataset_name为其下的文件夹名称，分为train和test两个文件夹
# ulsd_20230316_6 -> train, test
cd ../
python train.py --dataset_name ulsd_20230316 --order 6 --model_name ulsd_origin.pkl --gpu 2 --save_checkpoint
