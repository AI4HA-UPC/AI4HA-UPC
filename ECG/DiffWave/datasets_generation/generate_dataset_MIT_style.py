import ast
from pathlib import Path
from typing import NamedTuple
import numpy as np
import pandas as pd
import wfdb
from biosppy.signals import ecg
from matplotlib import pyplot as plt
from wfdb.processing import resample_sig
from dataclasses import dataclass
import click
from sklearn.preprocessing import MultiLabelBinarizer

SAMPLING_RATE = 500

"""
The five classes include normal (N), supraventricular (S), ventricular (V), fusion (F) and beats
of unknown etiology (Q)
"""
# TODO work on the correspondence with MIT-BIH
MIT_LABEL_CORRESPONDENCY = {
    0: ["NORM", "IRBBB", "CRBBB", "CLBBB", "ILBBB", "PACE"],
    1: ["SVARR", "SVTAC", "PSVT", "AFLT", "AFIB"],
    2: ["BIGU", "TRIGU"],
    3: ["FUSION"],
    4: ["OTHERS"],
}


class PTBXLDataset:
    DATA_FOLDER_PATH = Path("../../artifacts/ecg/data/")
    PTB_XL_ROOT_PATH = (
        DATA_FOLDER_PATH
        / "ptbxl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    )
    PTB_ROOT_PATH = DATA_FOLDER_PATH / "ptb-diagnostic-ecg-database-1.0.0"
    MIT_BIH_ROOT_PATH = DATA_FOLDER_PATH / "mit-bih-arrhythmia-database-1.0.0"
    CHAPMAN_ROOT_PATH = DATA_FOLDER_PATH / "chapman"
    ICBEB_ROOT_PATH = DATA_FOLDER_PATH / "ICBEB_2018"

    class Dataset(NamedTuple):
        name: str
        path: Path

    class PTB_XL_DATASET_LABELS(NamedTuple):
        label_type: str

    PTB_XL_LABELS_ALL = PTB_XL_DATASET_LABELS("all")
    PTB_XL_LABELS_DIAGNOSTIC = PTB_XL_DATASET_LABELS("diagnostic")
    PTB_XL_LABELS_FORM = PTB_XL_DATASET_LABELS("form")
    PTB_XL_LABELS_RHYTHM = PTB_XL_DATASET_LABELS("rhythm")

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

    def return_dict_datasets(self):
        return self.DATASETS

    def aggregate_diagnostic(self, y_dic, aggregation_df: pd.DataFrame):
        tmp = []
        for key in y_dic.keys():
            if key in aggregation_df.index:
                tmp.append(aggregation_df.loc[key].diagnostic_class)
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
        rr_time_intervals = np.diff(rpeaks)
        nominal_hb_period = np.median(rr_time_intervals)
        signal_part = int(np.ceil(1.2 * nominal_hb_period))
        for rp in rpeaks:
            rr_interval = x[rp : (rp + signal_part)]
            if len(rr_interval) <= max_length:
                padded_signal = np.pad(
                    rr_interval, (0, (max_length - len(rr_interval))), "constant"
                )
                signal_and_label = np.append(padded_signal, y)
                individual_signal.append(signal_and_label)
        if individual_signal:
            individual_signal.pop(-1)
        return individual_signal

    def extract_all_signals(self, data, labels, new_freq):
        all_signals = []
        for x, y in zip(data, labels):
            sample_rr_intervals = self.paper_heartbeat_extract(x, y, new_freq)
            if sample_rr_intervals:
                all_signals.append(sample_rr_intervals)
        return all_signals

    def create_PTB_XL_dataset(
        self,
        task_labels,
        diagnostic_superclasses: bool = False,
        diagnostic_subclasses: bool = False,
        binary_classes: bool = True,
    ):
        metadata = pd.read_csv(
            self.PTB_XL_ROOT_PATH / "ptbxl_database.csv", index_col="ecg_id"
        )
        metadata.scp_codes = metadata.scp_codes.apply(lambda x: ast.literal_eval(x))
        data = self._load_ptb_xl_signal_data(
            metadata, SAMPLING_RATE, self.PTB_XL_ROOT_PATH
        )
        agg_df = pd.read_csv(self.PTB_XL_ROOT_PATH / "scp_statements.csv", index_col=0)

        if task_labels == self.PTB_XL_LABELS_FORM.label_type:
            agg_df = agg_df[agg_df.form == 1]
        elif task_labels == self.PTB_XL_LABELS_RHYTHM.label_type:
            agg_df = agg_df[agg_df.rhythm == 1]
        else:
            agg_df = agg_df[agg_df.diagnostic == 1]
            if diagnostic_superclasses:
                metadata["diagnostic_class"] = metadata.scp_codes.apply(
                    self.aggregate_diagnostic, agg_df
                )
            elif diagnostic_subclasses:
                metadata["diagnostic_class"] = metadata.scp_codes.apply(
                    self.aggregate_diagnostic, agg_df
                )
        if binary_classes:
            metadata["labels"] = metadata.diagnostic_class.apply(self.binary_classes)
        else:
            mlb = MultiLabelBinarizer().fit(metadata.diagnostic_class)
            metadata["labels"] = (
                MultiLabelBinarizer().fit_transform(metadata.diagnostic_class).tolist()
            )

        num_patients = data.shape[0]
        common_lead = 1
        new_freq = 125
        new_X = np.apply_along_axis(resample_sig, 1, data, SAMPLING_RATE, new_freq)
        new_X = new_X[:, 0, :, :].squeeze()  # remove locations
        num_samples_per_patient = new_X.shape[1]
        lead_two_X = new_X[:, :, common_lead]
        lead_two_X = np.apply_along_axis(
            lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), 1, lead_two_X
        )

        # Split data into train and test
        test_fold = 10
        # Train
        X_train = lead_two_X[np.where(metadata.strat_fold != test_fold)]
        y_train = metadata[(metadata.strat_fold != test_fold)].labels_real
        # Test
        X_test = lead_two_X[np.where(metadata.strat_fold == test_fold)]
        y_test = metadata[metadata.strat_fold == test_fold].labels_real

        train_signals = extract_all_signals(X_train, y_train)
        test_signals = extract_all_signals(X_test, y_test)

        dataset = pd.DataFrame(
            [rr_interval for individual in train_signals for rr_interval in individual]
        )
        print(dataset.iloc[:, -1].value_counts())
        dataset = dataset[
            dataset.iloc[:, -1] != -1
        ]  # remove no label signals, they have -1 value
        print(len(dataset))
        print(dataset.iloc[:, -1].value_counts())
        dataset.to_csv(
            self.DATA_FOLDER_PATH / "ptb_xl_dataset_normal_abnormal_train.csv",
            header=False,
            index=False,
        )

        dataset = pd.DataFrame(
            [rr_interval for individual in test_signals for rr_interval in individual]
        )
        print(dataset.iloc[:, -1].value_counts())
        dataset = dataset[
            dataset.iloc[:, -1] != -1
        ]  # remove no label signals, they have -1 value
        print(len(dataset))
        print(dataset.iloc[:, -1].value_counts())
        dataset.to_csv(
            self.DATA_FOLDER_PATH / "ptb_xl_dataset_normal_abnormal_test.csv",
            header=False,
            index=False,
        )

    def _load_ptb_xl_signal_data(self, df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path / f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path / f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data


@click.command()
@click.option(
    "--dataset",
    default="PTB-XL",
    help="PTB-XL, PTB, MIT-BIH, CHAPMAN, ICBEB2018, ICBEB2019",
    required=True,
)
def main():
    pass


if __name__ == "__main__":

    ptb_xl_data = PTBXLDataset()
    task = PTBXLDataset.PTB_XL_LABELS_DIAGNOSTIC
    ptb_xl_data.create_PTB_XL_dataset(task, True)

    for i in train_signals[:50]:
        plt.plot(i[0][:-1])
        plt.title(f"{'normal' if i[-1][-1] == 0 else 'abnormal'}")
        plt.show()
        plt.close()

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
