FROM python:3.8.8

WORKDIR /app

ADD serve.py ./serve.py
ADD requirements.txt ./requirements.txt
ADD my_opus ./my_opus
ADD text.txt ./text.txt

RUN pip install -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple/ --no-dependencies

ENTRYPOINT [ "python", "serve.py"]