import os


class textlog:

    def __init__(self, filename, fields):
        self.filename = filename
        if os.path.exists(filename):
            self.file = open(filename, "a")
        else:
            self.file = open(filename, "w")
            self.file.write(",".join(fields) + "\n")
        self.fields = fields

    def write(self, fields):
        message = [str(v) for k, v in fields.items()]
        self.file.write(','.join(message) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()
