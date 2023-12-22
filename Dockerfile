FROM python:3.10-slim
WORKDIR workspace
COPY . .
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
CMD ["python3","val.py"]