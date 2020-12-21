
class Logger:
    def __init__(self, log_file, log_type='w+', to_print=True):
        self.log_file = log_file
        self.log_type = log_type
        self.cache = []
        self.to_print = to_print

    def cache_in(self, inline, to_print=True):
        self.cache.append(inline)
        if self.to_print and to_print:
            print(inline)

    def log(self, log):
        with open(self.log_file, self.log_type) as f:
            f.writelines(log + '\n')
            print(log)

    def cache_write(self):
        with open(self.log_file, self.log_type) as f:
            for log in self.cache:
                f.writelines(log + '\n')