import pandas as pd
from sdv.metadata import Metadata

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.metadata = None

    def load_and_clean(self):
        print(f" Reading: {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        
        # Basic cleaning
        self.df.dropna(how='all', axis=0, inplace=True)
        self.df.dropna(how='all', axis=1, inplace=True)
        
        # SDV 1.x Metadata Detection
        self.metadata = Metadata.detect_from_dataframe(
            data=self.df,
            table_name='main_table'
        )
        
        return self.df, self.metadata