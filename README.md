# pytorch_bert
pytorch and bert









基于bert的gpu环境搭建
摘要：本文件记录搭建docker容器使用gpu进行基于bert的模型训练过程
步骤：
1.服务器环境准备
    注意：cuda10必须使用nvidia-docker2，docker容器才可使用主机gpu
    centos7 安装cuda nvidia 驱动 docker
     a、nvidia驱动官网
  https://www.nvidia.com/Download/index.aspx?spm=a2c63.p38356.879954.7.2c524526OCooRk&lang=cn
      
       选择驱动要根据实际的GPU卡型号选，可以参考阿里云上的文章：
          https://www.alibabacloud.com/help/zh/doc-detail/108502.htm
          
           cuda驱动下载网址
           https://developer.nvidia.com/cuda-downloads
          
           比如下载10.0 10.1 相对高版本的驱动程序
          
            cudnn也选择较高版本，例如10.0

           cudnn下载地址
           https://developer.nvidia.com/cudnn

          b、安装cuda驱动
             yum -y install gcc kernel-devel "kernel-devel-uname-r == $(uname -r)" dkms

           sh cuda_10.0.130_410.48_linux.run

           Do you accept the previously read EULA?
           accept/decline/quit: accept

               Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 396.37?
               (y)es/(n)o/(q)uit: y

              Do you want to install the OpenGL libraries?
              (y)es/(n)o/(q)uit [ default is yes ]: y

             Do you want to run nvidia-xconfig?
            This will update the system X configuration file so that the NVIDIA X driver
            is used. The pre-existing X configuration file will be backed up.
            This option should not be used on systems that require a custom
            X configuration, such as systems with multiple GPU vendors.
           (y)es/(n)o/(q)uit [ default is no ]:

           Install the CUDA 10.0 Toolkit?
           (y)es/(n)o/(q)uit: y

           Enter Toolkit Location
          [ default is /usr/local/cuda-10.0 ]:

          Do you want to install a symbolic link at /usr/local/cuda?
          (y)es/(n)o/(q)uit: y

          Install the CUDA 10.0 Samples?
          (y)es/(n)o/(q)uit: y

         Enter CUDA Samples Location
        [ default is /root ]:

          编辑环境变量

         vi /etc/profile
         export PATH=$PATH:/usr/local/cuda/bin
         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

         source /etc/profile

c、安装cudnn

tar -zxf cudnn-10.0-linux-x64-v7.5.0.56.tgz

cd cuda10

cp include/cudnn.h /usr/local/cuda-10.0/include/

cp lib64/libcudnn* /usr/local/cuda-10.0/lib64/

chmod a+r /usr/local/cuda-10.0/include/cudnn.h /usr/local/cuda-10.0/lib64/libcudnn*

d、安装nvidia驱动
sh NVIDIA-Linux-x86_64-440.64.00.run

nvidia-smi

e、安装docker19
unzip docker19.zip 
cd docker/
yum localinstall  *.rpm

f、下载安装nvidia-docker2，这里一定要是版本2，才能使用主机gpu，cuda版本是10.0
利用外网机器centos系统，通过阿里云的源进行下载，下载的时候指定downlodonly参数

下载docker19、nvidia-docker2

wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.rep
yum install epel-release
yum install yum-utils device-mapper-persistent-data lvm2
yum-config-manager \
    --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo
mkdir container-selinux
yum install --downloadonly --downloaddir=container-selinux container-selinux
mkdir docker-ce 
yum install docker-ce-19.03.6 docker-ce-cli-19.03.6 containerd.io --downloadonly --downloaddir=docker-ce
mkdir nvidia-docker2
yum install nvidia-docker2 --downloadonly —downloaddir=nvidia-docker2

Nvidia-docker卸载： docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo yum remove nvidia-docker
Sudo yum install -y nvidia-docker2



2.去英伟达官网下载镜像，包含cuda：10.2，cudnn7，ubuntu18.04，python3,
runtime表示只有运行环境，如果需要开发环境，用dev
https://hub.docker.com/r/nvidia/cuda

docker pull  nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04



3.DOCKERFILE
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04-python3
RUN  mkdir -p /code
WORKDIR  /code
COPY  requirements.txt  requirements.txt
RUN   pip3  install -r  requirements.txt -i  http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
COPY docker/docker-entrypoint.sh /docker-entrypoint.sh
copy bearbert bearbert
copy config config
copy pytorch_pretrained_bert pytorch_pretrained_bert
RUN   apt-get update && apt-get install -y vim && apt-get install -y poppler-utils
EXPOSE 12127
CMD [“/bin/bash","/docker-entrypoint.sh"]

注：这里网不好去build docker的时候，大文件可以先下载下来到镜像里边先安装提交

4.相关命令
import sys
sys.path.append('/code')

Head -10 file1 > file2
file 文件名

docker run -d -t --runtime=nvidia  -p 12128:12128 bearbert:v2
docker commit  -m "run with gpus param" 9692 bearbert:v2
PYTHONIOENCODING=utf-8 python3 main_server.py 

curl “http://localhost:12128/bearbert/train?trainId=20191011001&traindataPath=/code/data/train.data10.csv&series=test”

service docker start
service docker stop

5.容器内cuda可用测试
import torch
torch.cuda.current_device()
torch.cuda.is_available()


