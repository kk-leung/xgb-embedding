from time import time


class Timer:
    def __init__(self):
        self.cur = time()

    def tic(self):
        self.cur = time()

    def toc(self, msg, reset=False):
        print("[{:8.2f}] {}".format(time() - self.cur, msg))
        if reset:
            self.cur = time()
