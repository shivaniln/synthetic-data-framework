import pandas as pd
import numpy as np
from sdv.metadata import Metadata

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.metadata = None

    def load_and_clean(self):
        # Load and drop 100% empty rows/columns
        self.df = pd.read_csv(self.file_path)
        self.df.dropna(how='all', axis=0, inplace=True)
        self.df.dropna(how='all', axis=1, inplace=True)

        for col in self.df.columns:
            # Categorical/Object Handling
            if self.df[col].dtype == 'object' or self.df[col].dtype.name == 'category':
                if not self.df[col].mode().empty:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                else:
                    self.df[col] = self.df[col].fillna("Unknown")
                
                # Sanitize strings to prevent GAN category duplication
                self.df[col] = self.df[col].astype(str).str.strip()
            
            # Numerical Handling
            elif np.issubdtype(self.df[col].dtype, np.number):
                self.df[col] = self.df[col].fillna(self.df[col].median())

        # Metadata detection for the SDV framework
        self.metadata = Metadata.detect_from_dataframe(
            data=self.df,
            table_name='input_table'
        )
        
        return self.df, self.metadata