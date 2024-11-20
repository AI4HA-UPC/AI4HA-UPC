import ast
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np
import pandas as pd
import wfdb
from wfdb.processing import resample_sig
from tqdm import tqdm


class ChapmanDataset:
    DATA_FOLDER_PATH = Path("../../artifacts/ecg/data/")
    PROCESSED_DATA_FOLDER_PATH = Path("../../artifacts/ecg/data/processed/chapman")
    CHAPMAN_DATA_ROOT_PATH = DATA_FOLDER_PATH / "chapman/WFDBRecords"

    CHAPMAN_ROOT_PATH = DATA_FOLDER_PATH / "chapman"

    class Dataset(NamedTuple):
        name: str
        path: Path

    CHAPMAN = Dataset("CHAPMAN", CHAPMAN_ROOT_PATH)

    DATASETS = {
        CHAPMAN.name: CHAPMAN.path,
    }

    PREPROCESS_FREQUENCY = 125
    # TODO buscar esto en el paper
    VALIDATION_FOLD = 9
    TEST_FOLD = 10

    CHANNELS = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]
    CODES_CORRECTIONS = {
        55827005: 164873001,
        445118002: 164909002,
        81898007: 75532003,
    }

    NUM_OBSERVATIONS_PART = 20000

    CHAPMAN_TO_PTBXL_LABELING = {
        "SB": "SBRAD",
        "SR": "SR",
        "AFIB": "AFIB",
        "ST": "STACH",
        "AF": "AFLT",
        "SI": "SARRH",
        "SVT": "SVTAC",
    }

    PTBXL_CLASS_LABELS = {
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
        sampling_rate: int = 500,
        separated_leads=True,
        conditions_file: str = "ConditionNames_SNOMED-CT.csv",
    ):
        self.files_sampling_rate = sampling_rate
        self.conditions_df = pd.read_csv(
            self.CHAPMAN_ROOT_PATH / conditions_file, index_col=0
        )
        self.separated_leads = separated_leads

    def preprocess_chapman_files(self):
        df = pd.DataFrame()
        data_part = 0
        not_found_codes = {"code": []}

        for file in self.CHAPMAN_DATA_ROOT_PATH.rglob("*.mat"):
            file_name = file.stem
            file_folder_path = file.parent
            file_path_and_stem = file_folder_path / file_name
            for num_chan, channel in enumerate(self.CHANNELS):
                data = {}
                try:
                    record = wfdb.io.rdrecord(file_path_and_stem, channels=[num_chan])
                except Exception as e:
                    print(
                        f"An error has occurred reading file {file_name}. Error: {e}."
                    )
                    continue
                try:
                    data["age"] = record.comments[0].replace(" ", "").split(":")[-1]
                    data["sex"] = record.comments[1].replace(" ", "").split(":")[-1]
                    data["lead"] = channel
                    data["id"] = file.stem
                    data["signal"] = []
                    if record.fs != self.files_sampling_rate:
                        record.p_signal = np.expand_dims(
                            resample_sig(
                                record.p_signal.squeeze(),
                                record.fs,
                                self.files_sampling_rate,
                            )[0],
                            axis=1,
                        )
                        record.fs = self.files_sampling_rate
                    for datum in record.p_signal.flat:
                        if not np.isnan(datum):
                            data["signal"].append(datum)
                        else:
                            data["signal"].append(None)
                    data["signal"] = [
                        datum if datum is not np.isnan(datum) else None
                        for datum in record.p_signal.flat
                    ]
                    diagnosis = list(
                        map(
                            lambda x: int(x),
                            record.comments[2].replace(" ", "")[3:].split(","),
                        )
                    )
                    aux_diagnosis = []
                    if 427393009 in diagnosis:
                        print("found")
                    for element in diagnosis:
                        if self.CODES_CORRECTIONS.get(element):
                            element = self.CODES_CORRECTIONS.get(element)
                        element_row = self.conditions_df[
                            self.conditions_df.Snomed_CT == element
                        ]
                        if not element_row.empty:
                            aux_diagnosis.append(element_row.index[0])
                        else:
                            print(f"Code not found {element}")
                            if element not in not_found_codes["code"]:
                                not_found_codes["code"].append(element)
                                df_not_codes = pd.Series(not_found_codes)
                                df_not_codes.to_csv(
                                    "chapman_snomed_codes_not_found.csv"
                                )

                    data["diagnosis"] = aux_diagnosis
                except Exception as e:
                    print(f"An error has occurred processing the data. Error: {e}")
                    continue
                aux = pd.Series(data).to_frame().T
                aux = pd.concat([aux, aux.signal.apply(pd.Series)], axis=1)
                aux = aux.drop("signal", axis=1)
                col_to_move = aux.pop("diagnosis")
                aux.insert(len(aux.columns), "diagnosis", col_to_move)
                df = pd.concat((df, aux))
                if len(df) == self.NUM_OBSERVATIONS_PART:
                    df.to_csv(
                        self.PROCESSED_DATA_FOLDER_PATH
                        / f"chapman_data_freq_{self.files_sampling_rate}_part_{data_part}.csv"
                    )
                    data_part = data_part + 1
                    df = pd.DataFrame()
        df_not_codes = pd.Series(not_found_codes)
        df_not_codes.to_csv(
            self.PROCESSED_DATA_FOLDER_PATH / "chapman_snomed_codes_not_found.csv"
        )
        print("==== End of dataset creation ====")

    def files_by_leads(self, group=False):
        if group:
            files = self.PROCESSED_DATA_FOLDER_PATH.rglob(
                f"chapman_data_freq_{self.files_sampling_rate}_part*"
            )
            df_all_leads = pd.DataFrame()
            for file in files:
                df = pd.read_csv(file, index_col=0)
                df.reset_index(drop=True, inplace=True)
                df_all_leads = pd.concat((df_all_leads, df), axis=0)
            df_all_leads.to_csv(
                self.PROCESSED_DATA_FOLDER_PATH
                / f"chapman_data_freq_{self.files_sampling_rate}_all_leads.csv"
            )
        else:
            for ch in self.CHANNELS:
                files = self.PROCESSED_DATA_FOLDER_PATH.rglob(
                    f"chapman_data_freq_{self.files_sampling_rate}_part*"
                )
                df_lead = pd.DataFrame()
                for file in files:
                    df = pd.read_csv(file, index_col=0)
                    df.reset_index(drop=True, inplace=True)
                    df = df[df.lead == ch]
                    df_lead = pd.concat((df_lead, df), axis=0)
                df_lead.to_csv(
                    self.PROCESSED_DATA_FOLDER_PATH
                    / f"chapman_data_freq_{self.files_sampling_rate}_lead_{ch}.csv"
                )

    def ptb_compatible(self):
        if self.separated_leads:
            files = [
                self.PROCESSED_DATA_FOLDER_PATH
                / f"chapman_data_freq_{self.files_sampling_rate}_lead_{ch}.csv"
                for ch in self.CHANNELS
            ]
        else:
            files = self.PROCESSED_DATA_FOLDER_PATH.rglob(
                f"chapman_data_freq_{self.files_sampling_rate}_all_leads*.csv"
            )
        for file in tqdm(files):
            df = pd.read_csv(file, index_col=0)
            df = df.reset_index(drop=True)
            df.diagnosis = df.diagnosis.apply(ast.literal_eval)
            df.diagnosis = df.diagnosis.apply(
                lambda x: [
                    self.CHAPMAN_TO_PTBXL_LABELING[ch_label]
                    for ch_label in x
                    if self.CHAPMAN_TO_PTBXL_LABELING.get(ch_label)
                ]
            )
            df.diagnosis = df.diagnosis.apply(sorted)
            df.diagnosis = df.diagnosis.apply(
                lambda x: [self.PTBXL_CLASS_LABELS[ptbxl_label] for ptbxl_label in x]
            )
            df.diagnosis = df.diagnosis.apply(lambda x: x[0] if len(x) > 0 else None)
            df = df[df.diagnosis.notna()]
            df.diagnosis = df.diagnosis.apply(lambda x: int(x))
            df.to_csv(file.parent / (file.stem + f"_ptbxl_labels.csv"), index=False)

    def create_val_test_subsets(
        self,
        ratio: float = 0.2,
        predefined_indices: Tuple[pd.Series, pd.Series] = (None, None),
    ) -> None:
        if self.separated_leads:
            files = [
                self.PROCESSED_DATA_FOLDER_PATH
                / f"chapman_data_freq_{self.files_sampling_rate}_lead_{ch}_ptbxl_labels.csv"
                for ch in self.CHANNELS
            ]
        else:
            files = self.PROCESSED_DATA_FOLDER_PATH.rglob(
                f"chapman_data_freq_{self.files_sampling_rate}_all_leads_ptbxl_labels.csv"
            )
        for file in tqdm(files):
            data = pd.read_csv(file)
            predefined_indices_empty = all(
                [element is None for element in predefined_indices]
            )
            if not predefined_indices_empty:
                val_indices = predefined_indices[0]
                test_indices = predefined_indices[1]
                val_set = data.iloc[val_indices, :]
                test_set = data.iloc[test_indices, :]
                train_set = data.drop(val_indices).drop(test_indices)
            else:
                freqs = (
                    data.groupby(["id", "diagnosis"], as_index=False)["diagnosis"]
                    .median()
                    .groupby("diagnosis")["diagnosis"]
                    .transform("count")
                )
                ids_data = pd.DataFrame({"id": data["id"].unique()})
                num_unique_samples = len(ids_data)
                num_samples_train = num_unique_samples - int(
                    np.ceil(num_unique_samples * ratio)
                )
                num_samples_val_test = num_unique_samples - num_samples_train
                val_test_ids = ids_data.sample(
                    num_samples_val_test, replace=False, weights=freqs
                )["id"]
                val_test_set = data.set_index("id").loc[val_test_ids].reset_index()
                train_set = data.set_index("id").drop(val_test_ids).reset_index()
                assert (len(val_test_set) + len(train_set)) == len(data)
                val_test_freqs = (
                    data.set_index("id")
                    .loc[val_test_ids]
                    .reset_index()
                    .groupby(["diagnosis"], as_index=False)["diagnosis"]
                    .transform("count")["diagnosis"]
                )
                num_samples_val = len(val_test_ids) - int(
                    np.ceil(len(val_test_ids) * 0.5)
                )
                val_set_ids = val_test_ids.sample(
                    num_samples_val, weights=val_test_freqs
                )
                test_set = (
                    data.set_index("id")
                    .loc[val_test_ids]
                    .drop(val_set_ids)
                    .reset_index()
                )
                val_set = data.set_index("id").loc[val_set_ids].reset_index()
                test_set_ids = test_set.id
                predefined_indices = (val_set_ids, test_set_ids)
                assert not val_test_set.id.isin(train_set.id).all()
                assert not val_set_ids.isin(test_set_ids).all()
                val_set_ids.reset_index(drop=True).to_csv(
                    "chapman_validation_indices.csv"
                )
                test_set_ids.reset_index(drop=True).to_csv("chapman_test_indices.csv")
            assert not val_set.id.isin(test_set.id).all()
            assert (len(train_set) + len(val_set) + len(test_set)) == len(data)
            train_set.to_csv(file.parent / (file.stem + f"_train.csv"), index=False)
            val_set.to_csv(file.parent / (file.stem + f"_validation.csv"), index=False)
            test_set.to_csv(file.parent / (file.stem + f"_test.csv"), index=False)


SAMPLING_RATE = 100

if __name__ == "__main__":
    ch_dataset = ChapmanDataset(sampling_rate=SAMPLING_RATE, separated_leads=False)
    # ch_dataset.preprocess_chapman_files()
    # ch_dataset.files_by_leads(group=True)
    # ch_dataset.ptb_compatible()
    ch_dataset.create_val_test_subsets()  # TODO de momento solo funciona con el conjunto de TODOS los leads
    # TODO Los csv contienen nans en age y en algunos leads. Fix en el preprocesamiento de los loaders
    exit(0)
