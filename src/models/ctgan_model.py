from sdv.single_table import CTGANSynthesizer

class CTGANModel:
    def __init__(self, metadata, epochs=50):
        self.model = CTGANSynthesizer(metadata, epochs=epochs)

    def train(self, real_data):
        self.model.fit(real_data)

    def generate(self, num_rows):
        return self.model.sample(num_rows=num_rows)