import ast
from pathlib import Path
from typing import NamedTuple

import click
import numpy as np
import pandas as pd
import wfdb
from biosppy.signals import ecg
from wfdb.processing import resample_sig

"""
The five classes include normal (N), supraventricular (S), ventricular (V), fusion (F) and beats
of unknown etiology (Q)
"""
# TODO work on this
MIT_LABEL_CORRESPONDENCY = {
    0: ["NORM", "IRBBB", "CRBBB", "CLBBB", "ILBBB", "PACE"],
    1: ["SVARR", "SVTAC", "PSVT", "AFLT", "AFIB"],
    2: ["BIGU", "TRIGU"],
    3: ["FUSION"],
    4: ["OTHERS"],
}


class PTBXLDataset:
    DATA_FOLDER_PATH = Path("../../artifacts/ecg/data/")
    PROCESSED_DATA_FOLDER_PATH = Path("../../artifacts/ecg/data/processed/ptbxl")
    PTB_XL_ROOT_PATH = (
        DATA_FOLDER_PATH
        / "ptbxl/ptbxl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    )
    PTB_ROOT_PATH = DATA_FOLDER_PATH / "ptb-diagnostic-ecg-database-1.0.0"
    MIT_BIH_ROOT_PATH = DATA_FOLDER_PATH / "mit-bih-arrhythmia-database-1.0.0"
    CHAPMAN_ROOT_PATH = DATA_FOLDER_PATH / "chapman"
    ICBEB_ROOT_PATH = DATA_FOLDER_PATH / "ICBEB_2018"

    class Dataset(NamedTuple):
        name: str
        path: Path

    PTB_XL = Dataset("PTB-XL", PTB_XL_ROOT_PATH)
    PTB = Dataset("PTB", PTB_ROOT_PATH)
    MIT_BIH = Dataset("MIT-BIH", MIT_BIH_ROOT_PATH)
    CHAPMAN = Dataset("CHAPMAN", CHAPMAN_ROOT_PATH)
    ICBEB = Dataset("ICBEB", ICBEB_ROOT_PATH)

    DATASETS = {
        PTB_XL.name: PTB_XL.path,
        PTB.name: PTB.path,
        MIT_BIH.name: MIT_BIH.path,
        CHAPMAN.name: CHAPMAN.path,
        ICBEB.name: ICBEB.path,
    }
    MIT_BIH_PREPROCESS_FREQUENCY = 125
    VALIDATION_FOLD = 9
    TEST_FOLD = 10

    PTBXL_RHYTHM_CLASS_LABELS = {
        "SBRAD": 0,
        "SR": 1,
        "AFIB": 2,
        "STACH": 3,
        "AFLT": 4,
        "SARRH": 5,
        "SVTAC": 6,
    }

    def __init__(
        self,
        sampling_rate=500,
        ptb_xl_file="ptbxl_database.csv",
        scp_file="scp_statements.csv",
        metadata_file_path: Path = None,
        data_file_path: Path = None,
        scp_statements_file_path: Path = None,
    ):
        self.files_sampling_rate = sampling_rate
        self.metadata = (
            pd.read_csv(self.PTB_XL_ROOT_PATH / ptb_xl_file, index_col="ecg_id")
            if metadata_file_path is None
            else pd.read_csv(metadata_file_path, index_col="ecg_id")
        )
        self.metadata.scp_codes = self.metadata.scp_codes.apply(
            lambda x: ast.literal_eval(x)
        )
        self.data = (
            self._load_ptb_xl_signal_data(
                self.metadata, sampling_rate, self.PTB_XL_ROOT_PATH
            )
            if data_file_path is None
            else self._load_ptb_xl_signal_data(
                self.metadata, sampling_rate, data_file_path
            )
        )
        self.agg_df = (
            pd.read_csv(self.PTB_XL_ROOT_PATH / scp_file, index_col=0)
            if scp_statements_file_path is None
            else pd.read_csv(scp_statements_file_path, index_col=0)
        )

    def return_dict_datasets(self):
        return self.DATASETS

    def aggregate_diagnostic(
        self, y_dic, aggregation_df: pd.DataFrame, subclasses=False
    ):
        tmp = []
        diagnostic_column = "diagnostic_subclass" if subclasses else "diagnostic_class"
        for key in y_dic.items():
            if key in aggregation_df.index:
                tmp.append(aggregation_df.loc[key, diagnostic_column])
        return list(set(tmp))

    def aggregate_form_rhythm_all(self, y_dic, aggregation_df: pd.DataFrame):
        tmp = []
        for key in y_dic.keys():
            if key in aggregation_df.index:
                tmp.append(key)
        return list(set(tmp))

    def binary_classes(self, x):
        if len(x) > 0:
            sample = x[0]
            if sample == "NORM":
                label = 0
            else:
                label = 1
        else:
            label = -1

        return label

    def paper_heartbeat_extract(self, x, y, signal_sampling_rate, max_length=187):
        individual_signal = []
        (rpeaks,) = ecg.hamilton_segmenter(x, signal_sampling_rate)
        (rpeaks,) = ecg.correct_rpeaks(
            signal=x, rpeaks=rpeaks, sampling_rate=signal_sampling_rate, tol=0.1
        )
        if len(rpeaks) > 1:
            rr_time_intervals = np.diff(rpeaks)
            nominal_hb_period = np.median(rr_time_intervals)
            signal_part = int(np.ceil(1.2 * nominal_hb_period))
            for rp in rpeaks:
                rr_interval = x[rp : (rp + signal_part)]
                if len(rr_interval) <= max_length:
                    padded_signal = np.pad(
                        rr_interval, (0, (max_length - len(rr_interval))), "constant"
                    )
                    signal_and_label = [padded_signal, y]
                    individual_signal.append(signal_and_label)
            if individual_signal:
                individual_signal.pop(-1)
        return individual_signal

    def extract_all_signals(self, data, labels):
        all_signals = []
        for x, y in zip(data, labels):
            sample_rr_intervals = self.paper_heartbeat_extract(
                x, y, self.MIT_BIH_PREPROCESS_FREQUENCY
            )
            if sample_rr_intervals:
                all_signals.append(sample_rr_intervals)
        return all_signals

    def normalize_data(self):
        data = np.apply_along_axis(
            resample_sig,
            1,
            self.data,
            self.files_sampling_rate,
            self.MIT_BIH_PREPROCESS_FREQUENCY,
        )
        data = data[
            :, 0, :, :
        ].squeeze()  # remove locations from peak finding algorithm
        data = np.apply_along_axis(
            lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), 1, data
        )
        return data

    def create_PTB_XL_rhythm_dataset(
        self,
        specific_lead: int = None,
        mit_bih_preprocessing=False,
        multiple_labels=False,
        CHAPMAN_classes=False,
    ):
        if (
            self.files_sampling_rate < self.MIT_BIH_PREPROCESS_FREQUENCY
            and mit_bih_preprocessing
        ):
            print("=" * 30)
            print(
                "The original frequency should be 500 in order to downsample to 125 hz. "
                "Please use file_sampling_rate=500"
            )
        else:
            dataset_file_name = (
                f"ptb_xl_dataset_rhythm_original_frequency_{self.files_sampling_rate}"
            )
            agg_df = self.agg_df[self.agg_df.rhythm == 1]
            metadata = self.metadata.copy()
            metadata["label"] = metadata.scp_codes.apply(
                self.aggregate_form_rhythm_all, args=(agg_df,)
            )

            metadata = metadata.reset_index()
            idx_obs_to_drop = metadata[metadata.label.map(len) == 0].index
            mask = np.ones(len(self.data))
            mask[idx_obs_to_drop] = False
            mask = mask.astype(bool)
            metadata = metadata.drop(idx_obs_to_drop)
            metadata = metadata.reset_index(drop=True)
            self.data = self.data[mask]

            if not multiple_labels:
                metadata.label = metadata.label.apply(lambda x: x[0])
                dataset_file_name = dataset_file_name + "_single_label"

            self.metadata = metadata

            if CHAPMAN_classes:
                dataset_file_name = dataset_file_name + "_chapman_classes"
            self.create_datasets(
                dataset_file_name, mit_bih_preprocessing, specific_lead, CHAPMAN_classes
            )

    def create_PTB_XL_form_dataset(
        self, specific_lead: int = None, mit_bih_preprocessing=False
    ):
        if (
            self.files_sampling_rate < self.MIT_BIH_PREPROCESS_FREQUENCY
            and mit_bih_preprocessing
        ):
            print("=" * 30)
            print(
                "The original frequency should be 500 in order to downsample to 125 hz. "
                "Please use file_sampling_rate=500"
            )
        else:
            dataset_file_name = (
                f"ptb_xl_dataset_form_original_frequency_{self.files_sampling_rate}"
            )
            agg_df = self.agg_df[self.agg_df.form == 1]
            metadata = self.metadata.copy()
            metadata["label"] = self.metadata.scp_codes.apply(
                self.aggregate_form_rhythm_all, args=(agg_df,)
            )
            self.create_datasets(
                dataset_file_name, mit_bih_preprocessing, specific_lead
            )

    def create_PTB_XL_all_dataset(
        self, specific_lead: int = None, mit_bih_preprocessing=False
    ):
        if (
            self.files_sampling_rate < self.MIT_BIH_PREPROCESS_FREQUENCY
            and mit_bih_preprocessing
        ):
            print("=" * 30)
            print(
                "The original frequency should be 500 in order to downsample to 125 hz. "
                "Please use file_sampling_rate=500"
            )
        else:
            dataset_file_name = f"ptb_xl_dataset_all_classes_original_frequency_{self.files_sampling_rate}"
            metadata = self.metadata.copy()
            metadata["label"] = metadata.scp_codes.apply(
                self.aggregate_form_rhythm_all, args=(self.agg_df,)
            )
            self.create_datasets(
                dataset_file_name, mit_bih_preprocessing, specific_lead
            )

    def create_PTB_XL_diagnostic_dataset(
        self,
        diagnostic_subclasses: bool = False,
        specific_lead: int = None,
        mit_bih_preprocessing: bool = False,
    ):
        if (
            self.files_sampling_rate < self.MIT_BIH_PREPROCESS_FREQUENCY
            and mit_bih_preprocessing
        ):
            print("=" * 30)
            print(
                "The original frequency should be 500 in order to downsample to 125 hz. "
                "Please use file_sampling_rate=500"
            )
        else:
            dataset_file_name = (
                f"ptb_xl_dataset_original_frequency_{self.files_sampling_rate}"
            )
            metadata = self.metadata.copy()
            agg_df = self.agg_df.copy()

            agg_df = agg_df[agg_df.diagnostic == 1]
            if diagnostic_subclasses:
                metadata["label"] = metadata.scp_codes.apply(
                    self.aggregate_diagnostic, args=(agg_df, diagnostic_subclasses)
                )
                dataset_file_name = dataset_file_name + "_diagnostic_subclass"
            else:
                metadata["label"] = metadata.scp_codes.apply(
                    self.aggregate_diagnostic, args=(agg_df,)
                )
                dataset_file_name = dataset_file_name + "_diagnostic_superclass"

            self.create_datasets(
                dataset_file_name, mit_bih_preprocessing, specific_lead
            )

    def create_datasets(
        self,
        dataset_file_name,
        mit_bih_preprocessing,
        specific_lead,
        CHAPMAN_classes=False,
    ):
        self.data = self.normalize_data() if mit_bih_preprocessing else self.data
        if specific_lead:  # TODO refactor with new changes
            data = self.data[:, :, specific_lead]
            self.create_folds_and_save_files(
                dataset_file_name + f"_lead_{specific_lead}",
                data,
                self.metadata,
                mit_bih_preprocessing,
                CHAPMAN_classes,
            )
        else:
            self.create_folds_and_save_files(
                dataset_file_name + f"_all_leads",
                mit_bih_preprocessing,
                CHAPMAN_classes,
            )

        self.metadata.to_csv(
            self.PROCESSED_DATA_FOLDER_PATH / (dataset_file_name + "_metadata.csv"),
            header=False,
            index=False,
        )

    def create_folds_and_save_files(
        self, dataset_file_name, mit_bih_style=False, CHAPMAN_classes=False
    ):
        # Split data into train and test
        # X
        train_folds = ~self.metadata.strat_fold.isin(
            [self.VALIDATION_FOLD, self.TEST_FOLD]
        )
        val_folds = self.metadata.strat_fold == self.VALIDATION_FOLD
        test_folds = self.metadata.strat_fold == self.TEST_FOLD
        X_train = self.data[train_folds]
        X_meta_train = self.metadata[train_folds]
        X_val = self.data[val_folds]
        X_meta_val = self.metadata[val_folds]
        X_test = self.data[test_folds]
        X_meta_test = self.metadata[test_folds]
        # Y
        y_train = self.metadata[train_folds]["label"]
        y_val = self.metadata[val_folds]["label"]
        y_test = self.metadata[test_folds]["label"]
        if mit_bih_style:
            train_signals = self.extract_all_signals(X_train, y_train)
            val_signals = self.extract_all_signals(X_val, y_val)
            test_signals = self.extract_all_signals(X_test, y_test)
            for signals, fold_name in zip(
                [train_signals, val_signals, test_signals],
                ["train", "validation", "test"],
            ):
                dataset = self.dataset_splitted_signals(signals)
                dataset.to_csv(
                    self.PROCESSED_DATA_FOLDER_PATH
                    / (dataset_file_name + f"_{fold_name}_mit-bih_preprocess.csv"),
                    header=False,
                    index=False,
                )
        else:
            for x, y, fold_name in zip(
                [(X_train, X_meta_train), (X_val, X_meta_val), (X_test, X_meta_test)],
                [y_train, y_val, y_test],
                ["train", "validation", "test"],
            ):
                y = np.expand_dims(y, axis=1)
                repetitions = x[0].shape[1]
                x_2d = x[0].reshape(len(y) * repetitions, x[0].shape[-1])
                dataset = pd.DataFrame(
                    np.concatenate(
                        [x_2d, np.expand_dims(np.repeat(y, repetitions), -1)], axis=-1
                    )
                )
                dataset = pd.concat(
                    (
                        x[1].loc[x[1].index.repeat(repetitions)].reset_index(drop=True),
                        dataset,
                    ),
                    axis=1,
                )
                if CHAPMAN_classes:
                    dataset.iloc[:, -1] = dataset.iloc[:, -1].apply(
                        lambda t: int(self.PTBXL_RHYTHM_CLASS_LABELS.get(t, -1))
                    )
                    dataset = dataset[dataset.iloc[:, -1] != -1]

                dataset.rename(columns={0: "lead", 1001: "class"})
                dataset.to_csv(
                    self.PROCESSED_DATA_FOLDER_PATH
                    / (dataset_file_name + f"_{fold_name}.csv"),
                    header=True,
                    index=False,
                )

    @staticmethod
    def _load_ptb_xl_signal_data(df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path / f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path / f) for f in df.filename_hr]
        signal_data = np.array(
            [
                np.c_[
                    np.array(meta["sig_name"]),
                    signal.T,
                ]
                for signal, meta in data
            ]
        )
        return signal_data

    @staticmethod
    def dataset_splitted_signals(signals):
        dataset = pd.DataFrame(
            [rr_interval for individual in signals for rr_interval in individual]
        )
        dataset = dataset[
            dataset.iloc[:, -1] != -1
        ]  # remove no label signals, they have -1 value, TODO CHECK THIS
        y = dataset[1].apply(lambda x: x if len(x) != 0 else np.nan)  # Remove no label
        x = pd.DataFrame.from_records(dataset[0])
        dataset = pd.concat([x, y], axis=1)
        dataset = dataset[~dataset.iloc[:, -1].isna()]
        dataset.iloc[:, -1].apply(lambda x: ast.literal_eval(x))
        return dataset


@click.command()
@click.option(
    "--dataset",
    default="PTB-XL",
    help="PTB-XL, PTB, MIT-BIH, CHAPMAN, ICBEB2018, ICBEB2019",
    required=True,
)
def main():
    pass


SAMPLING_RATE = 100

if __name__ == "__main__":
    ptb_xl_data = PTBXLDataset(sampling_rate=SAMPLING_RATE)
    # TODO work on generating dataset with all leads
    ptb_xl_data.create_PTB_XL_rhythm_dataset(
        mit_bih_preprocessing=False, multiple_labels=False, CHAPMAN_classes=True
    )
    # ptb_xl_data.create_PTB_XL_rhythm_dataset(mit_bih_preprocessing=True)
    # ptb_xl_data.create_PTB_XL_form_dataset(mit_bih_preprocessing=False)
    # ptb_xl_data.create_PTB_XL_form_dataset(mit_bih_preprocessing=True)
    # ptb_xl_data.create_PTB_XL_diagnostic_dataset(diagnostic_subclasses=False, mit_bih_preprocessing=False)
    # ptb_xl_data.create_PTB_XL_diagnostic_dataset(diagnostic_subclasses=False, mit_bih_preprocessing=True)
    # ptb_xl_data.create_PTB_XL_diagnostic_dataset(diagnostic_subclasses=True, mit_bih_preprocessing=False)
    # ptb_xl_data.create_PTB_XL_diagnostic_dataset(diagnostic_subclasses=True, mit_bih_preprocessing=True)
    # ptb_xl_data.create_PTB_XL_all_dataset(mit_bih_preprocessing=False)
    # ptb_xl_data.create_PTB_XL_all_dataset(mit_bih_preprocessing=True)

"""
Index(['description', 'diagnostic', 'form', 'rhythm', 'diagnostic_class',
       'diagnostic_subclass', 'Statement Category',
       'SCP-ECG Statement Description', 'AHA code', 'aECG REFID', 'CDISC Code',
       'DICOM Code'],
      dtype='object')

"""

"""
N	Normal beat
S	Supraventricular premature beat
V	Premature ventricular contraction
F	Fusion of ventricular and normal beat
Q	Unclassifiable beat
"""
