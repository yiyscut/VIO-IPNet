import time

class TicToc:
    def __init__(self):
        self.tic()

    def tic(self):
        self.start = time.time()

    def toc(self):
        end = time.time()
        elapsed_seconds = (end - self.start) * 1000  # Convert to milliseconds
        return elapsed_seconds
