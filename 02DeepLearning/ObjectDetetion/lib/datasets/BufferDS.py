import threading

class BufferDS:
    def __init__(self, ds, ds_func, size=3, batch_size=1):
        self.size = size
        self.ds = ds
        self.ds_func = ds_func          # data 处理成input数据, 包括读图， bbox_encode
        self.batch_size = batch_size

        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()
        self.buffer = []
        lock = threading.RLock()
        self.has_data = threading.Condition(lock)
        self.has_pos = threading.Condition(lock)

    def read(self):
        while True:
            data = self.ds.next_batch(self.batch_size)
            data = self.ds_func(data)
            self.put(data)

    def get(self):
        """Get data from this buffer.
           If the buffer is empty then the current thread is blocked until there is at least one data available.
        """
        with self.has_data:
            while len(self.buffer) == 0:
                self.has_data.wait()
            result = self.buffer[0]
            del self.buffer[0]
            self.has_pos.notify_all()
        return result

    def put(self, data):
        """Put the data into this buffer. The current thread is blocked if the buffer is full.
        """
        with self.has_pos:
            while len(self.buffer) >= self.size:
                self.has_pos.wait()
            self.buffer.append(data)
            self.has_data.notify_all()
            print("dataDS put", len(self.buffer))


if __name__ == '__main__':
    import time
    buffer = BufferDS(10)
    def get():
        for _ in range(1000):
            print(buffer.get())
            time.sleep(0.1)
    def put():
        for i in range(1000):
            print("put", i)
            buffer.put(i)
            time.sleep(0.01)

    th_put = threading.Thread(target=put, daemon=True)
    th_get = threading.Thread(target=get, daemon=True)
    th_put.start()
    th_get.start()
    th_put.join()
    th_get.join()
