import time

class Timer:
    def __init__(self):
        self.reset()

    def reset(self, restart=False):
        self.begin = None
        self.total = 0
        if restart:
            self.start()

    def start(self):
        if self.begin is not None:
            raise RuntimeError("Timer is already running")
        self.begin = time.time()

    def stop(self):
        if self.begin is None:
            raise RuntimeError("Timer is not running")

        end = time.time()
        self.total += end - self.begin
        self.begin = None

    def seconds(self):
        if self.begin is not None:
            end = time.time()
            return self.total + (end - self.begin)
        else:
            return self.total

    def minutes(self):
        return self.seconds() / 60


class Timers:
    def __init__(self):
        self.total = Timer()

        self.epoch = Timer()
        self.epoch_network = Timer()
        self.epoch_backward = Timer()
        self.epoch_parser = Timer()

        self.dev = Timer()
        self.dev_network = Timer()
        self.dev_parser = Timer()
