import os
import pandas as pd

from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

class Data:
    def __init__(self, args):
        self.args = args
        self.timestamp_column = "date"
        self.id_columns = []
        self.data_raw = pd.read_csv(
            os.path.join(args.root_path, args.dataset),
            parse_dates=[self.timestamp_column],
        )
        self.forecast_columns = list(self.data_raw.columns[1:])

    def get_data(self, tag):
    
        #timestamp_column = "date"
        #id_columns = []

        idx = (
            0
            if tag == "train"
            else 1
            if tag == "vali"
            else 2
            if tag == "test"
            else (
                print(f"Error: {tag} not recognized. Use one of train, vali, or test!"),
                exit(1)
            )
        )

        #data = pd.read_csv(
        #    os.path.join(args.root_path, args.data_path),
        #    parse_dates=[timestamp_column],
        #)
        #forecast_columns = list(data_raw.columns[1:])

        # get split
        num_train = int(len(self.data_raw) * 0.7)
        num_test = int(len(self.data_raw) * 0.2)
        num_vali = len(self.data_raw) - num_train - num_test
        border1s = [
            0,
            num_train - self.args.context_length,
            len(self.data_raw) - num_test - self.args.context_length,
        ]
        border2s = [
            num_train, 
            num_train + num_vali, 
            len(self.data_raw)
        ]

        start_index = border1s[idx]  # None indicates beginning of dataset
        end_index = border2s[idx]

        # we shift the start of the evaluation period back by context length so that
        # the first evaluation timestamp is immediately following the training data
        #vali_start_index = border1s[1]
        #vali_end_index = border2s[1]

        #test_start_index = border1s[2]
        #test_end_index = border2s[2]

        data = select_by_index(
            self.data_raw,
            id_columns=self.id_columns,
            start_index=start_index,
            end_index=end_index,
        )
        #print(tag, len(data))
        return data#, vali_data, test_data
   
   
    def get_dataset(self, tag, data, tsp):
        dataset = ForecastDFDataset(
            tsp.preprocess(data),
            id_columns=self.id_columns,
            target_columns=self.forecast_columns,
            context_length=self.args.context_length,
            prediction_length=self.args.prediction_length,
        )
        print(f"{tag}: {len(dataset)}")
        return dataset

    def get_tsp(self):
        tsp = TimeSeriesPreprocessor(
            timestamp_column=self.timestamp_column,
            id_columns=self.id_columns,
            target_columns=self.forecast_columns,
            scaling=True,
        )
        
        return tsp 
