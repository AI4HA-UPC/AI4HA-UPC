class textlog:

    def __init__(self, filename, fields):
        self.filename = filename
        self.file = open(filename, "w")
        self.fields = fields
        self.write(", ".join(fields))

    def write(self, fields):
    	message = [v for v in fields]
    	self.file.write(', '.join(message) + "\n")

    def close(self):
        self.file.close()

