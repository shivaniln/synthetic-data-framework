from sdv.single_table import GaussianCopulaSynthesizer

class CopulaModel:
    def __init__(self, metadata):
        self.model = GaussianCopulaSynthesizer(metadata)

    def train(self, real_data):
        self.model.fit(real_data)

    def generate(self, num_rows):
        return self.model.sample(num_rows=num_rows)