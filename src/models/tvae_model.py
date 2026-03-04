from sdv.single_table import TVAESynthesizer

class TVAEModel:
    def __init__(self, metadata, epochs=50):
        self.model = TVAESynthesizer(metadata, epochs=epochs)

    def train(self, real_data):
        self.model.fit(real_data)

    def generate(self, num_rows):
        return self.model.sample(num_rows=num_rows)