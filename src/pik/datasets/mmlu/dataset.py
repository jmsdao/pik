from typing import Union, Iterable
import pandas as pd
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets


SUBSETS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
    'clinical_knowledge', 'college_biology', 'college_chemistry',
    'college_computer_science', 'college_mathematics', 'college_medicine',
    'college_physics', 'computer_security', 'conceptual_physics', 'econometrics',
    'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'global_facts', 'high_school_biology', 'high_school_chemistry',
    'high_school_computer_science', 'high_school_european_history',
    'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics',
    'high_school_microeconomics', 'high_school_physics', 'high_school_psychology',
    'high_school_statistics', 'high_school_us_history', 'high_school_world_history',
    'human_aging', 'human_sexuality', 'international_law', 'jurisprudence',
    'logical_fallacies', 'machine_learning', 'management', 'marketing',
    'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
    'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
    'professional_law', 'professional_medicine', 'professional_psychology',
    'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
    'virology', 'world_religions',
]
CHOICE_LABELS = ["(A)", "(B)", "(C)", "(D)"]
TEMPLATE = """{}

Choices:
{}
{}
{}
{}"""


class MMLUDataset(Dataset):
    """Creates a PyTorch Dataset for the MMLU dataset.

    See: https://huggingface.co/datasets/tasksource/mmlu
    """
    def __init__(self, choice_labels=CHOICE_LABELS, template=TEMPLATE):
        self.choice_labels = choice_labels
        self.template = template

        data_dict = {
            name: load_dataset("tasksource/mmlu", name=name) for name in SUBSETS
        }

        # Concat all subsets
        datasets = []
        for subset in data_dict.keys():
            datasets.append(data_dict[subset]["test"])  # type: ignore
            datasets.append(data_dict[subset]["validation"])  # type: ignore
        self.dataset = concatenate_datasets(datasets)

        # Map qid to subset name for self.dataset
        subset_names = []
        for subset in data_dict.keys():
            num_rows = data_dict[subset]["test"].num_rows  # type: ignore
            num_rows += data_dict[subset]["validation"].num_rows  # type: ignore
            subset_names.extend(num_rows * [subset])
        self.subset_indices = pd.DataFrame({"subset": subset_names})
        self.subset_indices["qid"] = self.subset_indices.index
        self.subset_indices = self.subset_indices[["qid", "subset"]]

        # Concat all dev splits for prompting set
        prompt_sets = []
        for subset in data_dict.keys():
            prompt_sets.append(data_dict[subset]["dev"])  # type: ignore
        self.prompting = concatenate_datasets(prompt_sets)

        # Map qid to subset name for self.prompting
        subset_names = []
        for subset in data_dict.keys():
            num_rows = data_dict[subset]["dev"].num_rows  # type: ignore
            subset_names.extend(num_rows * [subset])
        self.prompting_indices = pd.DataFrame({"subset": subset_names})
        self.prompting_indices["qid"] = self.prompting_indices.index
        self.prompting_indices = self.prompting_indices[["qid", "subset"]]

    def __len__(self) -> int:
        """Returns the number of rows in the dataset."""
        return self.dataset.num_rows

    def __getitem__(
        self, key: Union[int, Iterable[int], slice]
    ) -> Union[tuple[str, str], tuple[list[str], list[str]]]:
        """
        Returns a tuple containing:
            question (str | list[str])
            answer (str | list[str])
        """
        datasubset = self.dataset[key]

        if isinstance(key, int):
            choices = [
                f"{lab} {ch}"
                for lab, ch in zip(self.choice_labels, datasubset["choices"])
            ]
            question = self.template.format(datasubset["question"], *choices)
            answer = self.choice_labels[datasubset["answer"]]
        else:
            question, answer = [], []
            for q, c, a in zip(
                datasubset["question"], datasubset["choices"], datasubset["answer"]
            ):
                choices = [f"{lab} {ch}" for lab, ch in zip(self.choice_labels, c)]
                question.append(self.template.format(q, *choices))
                answer.append(self.choice_labels[a])

        return (question, answer)  # type: ignore
