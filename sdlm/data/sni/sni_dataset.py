# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Natural Instruction V2 Dataset."""


import json
import os
import random

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@article{wang2022benchmarking,
  title={Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks},
  author={Wang, Yizhong and Mishra, Swaroop and Alipoormolabashi, Pegah and Kordi, Yeganeh and others},
  journal={arXiv preprint arXiv:2204.07705},
  year={2022}
}
"""

_DESCRIPTION = """
Natural-Instructions v2 is a benchmark of 1,600+ diverse language tasks and their expert-written instructions.
It covers 70+ distinct task types, such as tagging, in-filling, and rewriting.
These tasks are collected with contributions of NLP practitioners in the community and
through an iterative peer review process to ensure their quality.
"""

_URL = "https://instructions.apps.allenai.org/"
_VERSION = "2.6"
_RELEASE_URL = f"https://api.github.com/repos/allenai/natural-instructions/zipball/v{_VERSION}"


class NIConfig(datasets.BuilderConfig):
    def __init__(
        self,
        split_subdir="splits/default/",
        task_subdir="tasks/",
        max_num_instances_per_task=None,
        max_num_instances_per_eval_task=None,
        seed=42,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.split_subdir: str = split_subdir
        self.task_subdir: str = task_subdir
        self.seed: int = seed
        self.max_num_instances_per_task: int = max_num_instances_per_task
        self.max_num_instances_per_eval_task: int = (
            max_num_instances_per_eval_task or max_num_instances_per_task
        )


class NaturalInstructions(datasets.GeneratorBasedBuilder):
    """NaturalInstructions Dataset."""

    VERSION = datasets.Version(_VERSION + ".0")
    BUILDER_CONFIG_CLASS = NIConfig
    BUILDER_CONFIGS = [
        NIConfig(name="default", description="Default config for NaturalInstructions V2")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            version=self.VERSION,
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),  # instance_id
                    "Task": datasets.Value("string"),
                    "Contributors": datasets.Value("string"),
                    "Source": [datasets.Value("string")],
                    "URL": [datasets.Value("string")],
                    "Categories": [datasets.Value("string")],
                    "Reasoning": [datasets.Value("string")],
                    "Definition": [datasets.Value("string")],
                    "Positive Examples": [
                        {
                            "input": datasets.Value("string"),
                            "output": datasets.Value("string"),
                            "explanation": datasets.Value("string"),
                        }
                    ],
                    "Negative Examples": [
                        {
                            "input": datasets.Value("string"),
                            "output": datasets.Value("string"),
                            "explanation": datasets.Value("string"),
                        }
                    ],
                    "Input_language": [datasets.Value("string")],
                    "Output_language": [datasets.Value("string")],
                    "Instruction_language": [datasets.Value("string")],
                    "Domains": [datasets.Value("string")],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "input": datasets.Value("string"),
                        "output": [datasets.Value("string")],
                    },
                }
            ),
            license="",
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            dl_path = dl_manager.download_and_extract(_RELEASE_URL)
            self.config.data_dir = os.path.join(
                dl_path, os.listdir(dl_path)[0]
            )  # get the extracted directory
        split_dir = os.path.join(self.config.data_dir, self.config.split_subdir)
        task_dir = os.path.join(self.config.data_dir, self.config.task_subdir)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(split_dir, "train_tasks.txt"),
                    "task_dir": task_dir,
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(split_dir, "test_tasks.txt"),
                    "task_dir": task_dir,
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "split": datasets.Split.TEST,
                },
            ),
        ]

    def _generate_examples(
        self, path=None, task_dir=None, max_num_instances_per_task=None, split=None
    ):
        """Yields examples."""
        logger.info(f"Reading {split} tasks from {path}")
        with open(path, encoding="utf-8") as split_f:
            for line in split_f:
                task_name = line.strip()
                task_path = os.path.join(task_dir, task_name + ".json")
                with open(task_path, encoding="utf-8") as task_f:
                    s = task_f.read()
                    task_data = json.loads(s)
                    # rename task name to task_num + source + category
                    task_name = (
                        task_name.split("_")[0]
                        + "_"
                        + "_".join(task_data["Source"]).lower()
                        + "_"
                        + "_".join(task_data["Categories"][0].lower().split())
                    )
                    task_data["Task"] = task_name
                    if "Instruction Source" in task_data:
                        task_data.pop("Instruction Source")
                    all_instances = task_data.pop("Instances")
                    if split == datasets.Split.TEST:
                        # for testing tasks, 100 instances are selected for efficient
                        # evaluation and they are label-balanced.
                        # we put them in the first for reproducibility.
                        # so, we use them here
                        instances = all_instances[:100]
                    else:
                        instances = all_instances
                    if max_num_instances_per_task is not None and max_num_instances_per_task >= 0:
                        random.Random(self.config.seed).shuffle(instances)
                        instances = instances[:max_num_instances_per_task]
                    for idx, instance in enumerate(instances):
                        example = task_data.copy()
                        example["id"] = instance["id"]
                        example["Instance"] = instance
                        yield f"{task_name}_{idx}", example