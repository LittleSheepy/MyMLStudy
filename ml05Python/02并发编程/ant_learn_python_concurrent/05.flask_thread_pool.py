import flask
import json
import time
from concurrent.futures import ThreadPoolExecutor

app = flask.Flask(__name__)
pool = ThreadPoolExecutor()

def read_file():
    time.sleep(0.1)
    return "file result"


def read_db():
    time.sleep(0.2)
    return "db result"


def read_api():
    time.sleep(0.3)
    return "api result"


@app.route("/")
@app.route("/short")
def index():
    result_file = pool.submit(read_file)
    result_db = pool.submit(read_db)
    result_api = pool.submit(read_api)

    return json.dumps({
        "result_file": result_file.result(),
        "result_db": result_db.result(),
        "result_api": result_api.result(),
    })

@app.route("/long")
def index1():
    result_file = read_file()
    result_db = read_db()
    result_api = read_api()

    return json.dumps({
        "result_file": result_file,
        "result_db": result_db,
        "result_api": result_api,
    })

if __name__ == '__main__':
    app.run()


"""
创建文件 curl-format.txt
内容：
time_total: %{time_total}\n

在文件目录执行：
curl -w "@curl-format.txt" -o NUL -s http://127.0.0.1:5000/
curl -w "@curl-format.txt" -o NUL -s http://127.0.0.1:5000/long
"""