FROM python:3.8.8

WORKDIR /app

ADD train.py ./train.py
ADD requirements.txt ./requirements.txt
ADD test_set ./test_set
ADD train_set ./train_set
ADD mid_opus ./mid_opus

RUN pip install -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple/ --no-dependencies

ENTRYPOINT [ "python", "train.py" ]
