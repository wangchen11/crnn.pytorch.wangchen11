import os
from urllib.parse import urlparse  

# 豆瓣镜像 http://pypi.douban.com/simple/
# 清华镜像 https://pypi.tuna.tsinghua.edu.cn/simple
# 中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/
# 华中理工大学：http://pypi.hustunique.com/
# 山东理工大学：http://pypi.sdutlinux.org/ 
# 阿里云 "https://mirrors.aliyun.com/pypi/simple/"

pipMirror     = "https://pypi.tuna.tsinghua.edu.cn/simple"
pipMirrorHost = urlparse(pipMirror).netloc
pipInstallPrefix = f"pip install -i {pipMirror} --trusted-host {pipMirrorHost} "

# os.system("pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/")
os.system(pipInstallPrefix + "torch torchvision torchaudio matplotlib pandas")

# 如果要安装pytorch GPU版本，见官网 https://pytorch.org/get-started/locally/
# 这里是我的笔记本需要执行的安装指令，如果已经安装了cpu版本则需要先卸载。
# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
# 安装完成后运行，fbgemm.dll加载失败，据说需手动VC_redist。X64 - 14.29.30153 下载页面 https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170
# 安装VC_redist后重启问题依旧
# 尝试 https://discuss.pytorch.org/t/failed-to-import-pytorch-fbgemm-dll-or-one-of-its-dependencies-is-missing/201969
# 运行依赖检查工具后检查fbgemm.dll的依赖发现缺少 libomp140.x86_64.dll
# 重新安装Visual Studio Community 2022依然不行
# 重新安装Visual Studio Enterprise 2022依然不行
# 手动下载 libomp140.x86_64.dll 后将其放入当前工程根目录（执行python命令的目录）问题解决
# https://www.dllme.com/dll/files/libomp140_x86_64/037e19ea9ef9df624ddd817c6801014e/download