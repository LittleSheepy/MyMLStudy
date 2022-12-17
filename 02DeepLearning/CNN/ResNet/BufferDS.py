import threading
class Buffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.pos = 0
        lock = threading.RLock()
        self.has_data = threading.Condition(lock)
        self.has_pos = threading.Condition(lock)

    def get_size(self):
        return self.size
    def get(self):
        """
        Get data from this buffer.
        If the buffer is empty then the current thread is blocked until there is at least one data available.
        :return:
        """
        with self.has_data:
            while len(self.buffer) == 0:
                self.has_data.wait()
            result = self.buffer[0]
            del self.buffer[0]
            self.has_pos.notify_all()
        return result
    def put(self, data):
        """
        Put data into buffer.
        the current thread is blocked if thw buffer is full.
        :param data:
        :return:
        """
        with self.has_pos:
            while len(self.buffer) >= self.size:
                self.has_pos.wait()
            self.buffer.append(data)
            self.has_data.notify_all()

class BufferDS:
    def __init__(self, buffer_size, ds, batch_size):
        # ds.num_examples, ds.next_batch(batch_size)
        self.ds = ds
        self.batch_size = batch_size
        self.buffer = Buffer(buffer_size)
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    @property
    def num_examples(self):
        return self.ds.num_examples

    def next_batch(self, batch_size):
        return self.buffer.get()

    def read(self):
        while True:
            data = self.ds.next_batch(self.batch_size)
            self.buffer.put(data)

def test_Buffer():
    import time
    buffer = Buffer(10)
    def get():
        for i in range(10000):
            print("buffer.get()",i)
            print(buffer.get())
            # time.sleep(0.01)
    def put():
        for i in range(10000):
            print("buffer.put()",i)
            buffer.put(i)

    th_put = threading.Thread(target=put, daemon=True)
    th_get = threading.Thread(target=get, daemon=True)
    th_get.start()
    th_put.start()
    th_get.join()
    th_put.join()

def test_BufferDS():
    pass
if __name__ == '__main__':
    test_Buffer()