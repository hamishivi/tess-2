import math
import time
from typing import Dict, List, Optional

from torch.utils.data import DataLoader, Dataset
from transformers import Seq2SeqTrainer
from transformers.integrations import TensorBoardCallback
from transformers.trainer_utils import speed_metrics
from transformers.utils import logging

skip_first_batches = None
IS_SAGEMAKER_MP_POST_1_10 = False
GENERATION_RESULTS = "generated"


logger = logging.get_logger(__name__)


class ARTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, preprocess_logits_for_metrics=None)
        self.tb_writer = self.get_tb_writer()
        self.original_data_collator = self.data_collator

    def get_tb_writer(self):
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, TensorBoardCallback):
                return cb
        return None

    def log_results_to_tensorboard(self, output):
        # TODO: we need to fix this which happens during the only eval option.
        if self.tb_writer.tb_writer is None:
            return
        for i, prediction in enumerate(output.predictions):
            print("AAA", self.tokenizer.decode(prediction))
            self.tb_writer.tb_writer.add_text(
                f"sample_{i}", self.tokenizer.decode(prediction), self.state.global_step
            )

    def get_train_dataloader(self) -> DataLoader:
        self.data_collator = self.original_data_collator("train")
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        self.data_collator = self.original_data_collator("eval")
        return super().get_eval_dataloader(eval_dataset)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Copied from
        - https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
        - https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_seq2seq.py
        with added tensorboard text logging.
        """
        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if (
            gen_kwargs.get("num_beams") is None
            and self.args.generation_num_beams is not None
        ):
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs
        # return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        # NOTE: no tpu
        # if self.is_fsdp_xla_v2_enabled:
        #     eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        # NOTE: no tpu
        # if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
        #     # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        #     xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        if self.args.log_generated_texts:
            self.log_results_to_tensorboard(output)

        return output.metrics
