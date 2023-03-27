# ulsd模型
1.采用fisheye-distortion项目里的方法将pinhole图像以及边沿线信息进行鱼眼图像的转换
2.convert_2_ulsd_data_evaluate.py将1中的数据转为ULSD里的数据格式,用于evaluation,
其中保存的image没有提早resize，而gt是提早resize好的,脚本里暂时不考虑'spherical'
3.convert_2_ulsd_data_train.py将1中的数据转为ULSD里的数据格式,train,
其中保存的image没有提早resize，而gt是提早resize好的,脚本里暂时不考虑'spherical'
# 跑通流程，但是效果不好，估计跟学习率之类的训练计划有关，暂时不处理

# 添加lld模型
1. convert_2_lld_data_train.py用于生成lld的数据格式
2. dataset/wireframe_dset_curve.py 用于加载数据

