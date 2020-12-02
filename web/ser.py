
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, parse_config_file, options
from tornado.web import Application, RequestHandler

define('port', type=int, default=8888, multiple=False)
#parse_config_file('config')


class IndexHandler(RequestHandler):
    def get(self, *args, **kwargs):
        html = '''
        <form method=post action=/login enctype=multipart/form-data>
            <p>
                用户名：<input type=text name=uname>
            </p>
            <p>
                密码：<input type=password name=upwd>
            </p>
            <p>
                <input type=submit value=提交>
            </p>
        </form>
        '''
        self.write(html)
        msg = self.get_arguments('msg')
        if msg:
            self.write('用户名或密码错误')

    def post(self, *args, **kwargs):
        pass


class LoginHandler(RequestHandler):
    def get(self, *args, **kwargs):
        pass

    def post(self, *args, **kwargs):
        uname = self.get_arguments('uname')[0]
        upwd = self.get_arguments('upwd')[0]
        if uname == 'abc' and upwd == '123':
            self.redirect('/python')  # 页面跳转
        else:
            self.redirect('/?msg=false')


class PythonHandler(RequestHandler):
    def get(self, *args, **kwargs):
        self.write('登陆成功')

    def post(self, *args, **kwargs):
        pass

url_list = [('/', IndexHandler),
            ('/login', LoginHandler),
            ('/python', PythonHandler)]

app = Application(url_list)
server = HTTPServer(app)
server.listen(options.port)
IOLoop.current().start()