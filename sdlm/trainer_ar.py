from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Seq2SeqTrainer, Trainer
from transformers.integrations import TensorBoardCallback
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_sagemaker_mp_enabled, logging

if is_datasets_available():
    import datasets

skip_first_batches = None
IS_SAGEMAKER_MP_POST_1_10 = False
GENERATION_RESULTS = "generated"


logger = logging.get_logger(__name__)


class EvalLoopOutput(NamedTuple):
    logits: Union[np.ndarray, Tuple[np.ndarray]]
    simplex: Union[np.ndarray, Tuple[np.ndarray]]
    input_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    results: Optional[Dict[str, List[str]]]
    num_samples: Optional[int]


class ARTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, preprocess_logits_for_metrics=None)
        self.vocab_size = self.model.config.vocab_size
        self.tb_writer = self.get_tb_writer()
        self.eos_token_id = self.tokenizer.eos_token_id

    def get_tb_writer(self):
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, TensorBoardCallback):
                return cb
        return None

    def log_results_to_tensorboard(self, state, length, results):
        # TODO: we need to fix this which happens during the only eval option.
        if self.tb_writer.tb_writer is None:
            return
        for i in range(length):
            total_text = ""
            for k, v in results.items():
                total_text += f"*** {k} ***: {v[i]}" + "  \n"
            self.tb_writer.tb_writer.add_text(
                f"sample_{i}", total_text, state.global_step
            )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator("train")
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator("eval")

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="evaluation"
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def create_optimizer(self):
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        from transformers.trainer_pt_utils import get_parameter_names

        # overriden
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            # only training warping parameters...
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n in decay_parameters
                            and p.requires_grad
                            and not ("cdf" in n or "linear_l" in n or "position_l" in n)
                        )
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": optimizer_kwargs["lr"],
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n not in decay_parameters
                            and p.requires_grad
                            and not ("cdf" in n or "linear_l" in n or "position_l" in n)
                        )
                    ],
                    "weight_decay": 0.0,
                    "lr": optimizer_kwargs["lr"],
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            ("cdf" in n or "linear_l" in n or "position_l" in n)
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": 0.0,
                    "lr": 1e-3,
                },
            ]

            optimizer_kwargs.pop("lr")

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer
