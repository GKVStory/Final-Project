FROM python:3.8.8

WORKDIR /app

ADD test.py ./test.py
ADD requirements.txt ./requirements.txt
ADD my_opus ./my_opus

RUN pip install -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple/ --no-dependencies

ENTRYPOINT [ "python", "test.py" ]