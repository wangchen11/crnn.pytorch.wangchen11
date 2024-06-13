# CRNN

序言
--------

原始项目[crnn.pytorch](https://github.com/meijieru/crnn.pytorch )运行遇到了许多兼容性问题，在经过近一周的踩坑后终于能够正常训练了。

## 软件版本

Python 3.12.3

## 快速开始

1. 安装依赖

```shell
python3 00_setup.py
```

2. 生成预训练数据

```shell
python3 01_gendb_for_pretrain.py
```

3. 开始预训练

```shell
python3 02_train_pretrain.py
```

4. 检验预训练成果

```shell
python3 03_val_last_pretrain.py
```

## 使用CUDA加速训练

1. 首先跑通上述流程
2. 查看电脑显卡的CUDA版本
3. 通过官网 https://pytorch.org/get-started/locally/ 查询安装命令
4. 卸载 torch cpu版本`pip3 install torch`
5. 输入官网上查询的命令重新安装torch gpu版本

## 使用CUDA踩的坑

安装完成torch gpu版本后运行，运行提示fbgemm.dll加载失败，据说需手动VC_redist。X64 - 14.29.30153 下载页面 https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170

安装VC_redist后重启问题依旧

尝试 https://discuss.pytorch.org/t/failed-to-import-pytorch-fbgemm-dll-or-one-of-its-dependencies-is-missing/201969

运行依赖检查工具后检查fbgemm.dll的依赖发现缺少 libomp140.x86_64.dll

重新安装Visual Studio Community 2022依然不行

重新安装Visual Studio Enterprise 2022依然不行

手动下载 libomp140.x86_64.dll 后将其放入当前工程根目录（执行python命令的目录）问题解决

https://www.dllme.com/dll/files/libomp140_x86_64/037e19ea9ef9df624ddd817c6801014e/download

## 参考项目

[crnn.pytorch](https://github.com/meijieru/crnn.pytorch )





