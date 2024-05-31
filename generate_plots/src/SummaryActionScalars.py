import os
import pandas as pd
from tbparse import SummaryReader

class SummaryActionScalars:
    def __init__(self, log_dir, run):
        self.log_dir = log_dir
        self.run = run
        self.df = self._load_df()
        
        self.files_in_directory = os.listdir(os.path.join(self.log_dir, self.run))
        self.action_scalars = [action_scalar for action_scalar in self.files_in_directory if any(tag in action_scalar for tag in self.df.tag.unique())]
        self.global_scalars = [tag for tag in self.df.tag.unique() if tag not in [action_scalar[:action_scalar.rfind('_')] for action_scalar in self.action_scalars]]
        
    def _load_df(self):
        reader = SummaryReader(self.log_dir + '/' + self.run)
        return reader.scalars

    def tf_dir_2_df(self, scalar, action):
        reader = SummaryReader(os.path.join(self.log_dir, self.run, scalar + '_' + str(action)))
        return reader.scalars

    def generate_summary_action_scalars(self):
        tag_to_value_dict = {}

        for action_scalar in self.action_scalars:
            parts = action_scalar.split('_')
            key = '_'.join(parts[:-1])
            value = int(parts[-1])

            if key in tag_to_value_dict:
                if value not in tag_to_value_dict[key]:
                    tag_to_value_dict[key][value] = self.tf_dir_2_df(key, value)
            else:
                tag_to_value_dict[key] = {value: self.tf_dir_2_df(key, value)}

        return tag_to_value_dict

    def generate_summary_global_scalars(self):
        tag_to_value_dict = {}

        for global_scalar in self.global_scalars:
            tag_to_value_dict[global_scalar] = self.df[self.df.tag==global_scalar].drop(columns=['tag'])

        return tag_to_value_dict   