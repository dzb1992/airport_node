FROM python:3.6
LABEL version="v1"

#将Docker-test目录下的代码添加到镜像中的code文件夹（两个目录参数中间有空格分开）
COPY . /Airport_Node
# 设置code文件夹是工作目录
WORKDIR /Airport_Node

RUN pip install -r requirements.txt -i https://pypi.douban.com/simple

CMD ["python","predict_grid.py"]


