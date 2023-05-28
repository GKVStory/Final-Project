FROM python:3.8.8

WORKDIR /app

ADD preprocess.py ./preprocess.py
ADD result.csv ./result.csv
ADD opus ./opus
ADD requirements.txt ./requirements.txt

RUN pip install -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple/ --no-dependencies

ENTRYPOINT [ "python", "preprocess.py" ]