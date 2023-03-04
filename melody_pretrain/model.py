import os
from typing import List, Optional, Sequence, Tuple, Union

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import BasePredictionWriter

from .dataset import DataBatch, GeneralDataBatch, ngram_ids_ignore_index, span_indices_padding_index
from .ngram import get_lexicon_size
from .tokenizer import MIDITokenizer


class CompoundTokenFuser(nn.Module):
    """Fuses multiple token embeddings into a single embedding."""

    def __init__(self, tokenizer: MIDITokenizer, embedding_dim: Union[int, Tuple[int, ...]], model_dim: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.num_features = len(tokenizer.field_names)
        self.field_sizes = tokenizer.field_sizes
        self.total_field_size = sum(self.field_sizes)

        self.model_dim = model_dim
        if isinstance(embedding_dim, int):
            self.embedding_dims = [embedding_dim for _ in range(self.num_features)]
        else:
            assert len(embedding_dim) == self.num_features, "embedding_dim must be int or list of length num_features"
            self.embedding_dims = embedding_dim
        self.total_embedding_dim = sum(self.embedding_dims)

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=field_size, embedding_dim=embedding_dim_, padding_idx=pad_token_id)
                for field_size, embedding_dim_, pad_token_id in zip(
                    self.field_sizes, self.embedding_dims, tokenizer.pad_token_ids
                )
            ]
        )
        self.encoder = nn.Linear(self.total_embedding_dim, model_dim)
        self.decoder = nn.Linear(model_dim, self.total_field_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            input_ids: (batch_size, seq_len, num_features)
        Returns:
            fused: (batch_size, seq_len, model_dim)
        """
        _, _, num_features = x.shape
        assert num_features == self.num_features, f"num_features must be {self.num_features}"

        # embeddings: (batch_size, seq_len, total_embedding_dim)
        x = torch.concat([embedding(x[:, :, i]) for i, embedding in enumerate(self.embeddings)], dim=2)
        # fused: (batch_size, seq_len, model_dim)
        x = self.encoder(x)
        return x

    def decode(self, fused: torch.Tensor) -> List[torch.Tensor]:
        """Args:
            fused: (batch_size, seq_len, model_dim)
        Returns:
            logits: List[torch.Tensor] of length num_features,
            each of shape (batch_size, seq_len, vocab_size of the feature)
        """
        # embeddings: (batch_size, seq_len, total_field_size)
        embeddings = self.decoder(fused)
        return torch.split(embeddings, self.field_sizes, dim=2)


class NgramClassificationHead(nn.Module):
    """Extra classification head placed after transformer encoder, used for n-gram prediction."""

    def __init__(self, model_dim: int, ngram_size: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(model_dim, ngram_size)

    def forward(self, outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            outputs: (batch_size, seq_len, model_dim), transformer encoder outputs
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        return self.classifier(outputs)


class SpanPositionalEncoding(nn.Module):
    """Learnable positional encoding added on [MASK] and [SEP] tokens for advanced span prediction."""

    def __init__(self, model_dim: int, max_length: int = 128) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.max_length = max_length

        self.positional_encoding = nn.Embedding(max_length, model_dim, padding_idx=span_indices_padding_index)

    def forward(self, span_indices: torch.Tensor) -> torch.Tensor:
        """Args:
            span_indices: (batch_size, seq_len)
        Returns:
            span_encoding: (batch_size, seq_len, model_dim)
        """
        return self.positional_encoding(span_indices)


class MelodyModel(pl.LightningModule):
    """Base model for core step logic."""

    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Union[int, Tuple[int, ...]],  # embedding_dim: (bar, position, duration, pitch)
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        use_span_positional_encoding: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        tokenizer_config_path = os.path.join(dataset_dir, "tokenizer_config.json")
        if not os.path.exists(tokenizer_config_path):
            raise ValueError(f"Tokenizer config file not found: {tokenizer_config_path}")
        self.tokenizer = MIDITokenizer.from_config(tokenizer_config_path)

        self.num_features = len(self.tokenizer.field_names)

        self.model_dim = model_dim
        self.num_heads = num_heads

        self.fuser = CompoundTokenFuser(self.tokenizer, embedding_dim, model_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.use_span_positional_encoding = use_span_positional_encoding
        if use_span_positional_encoding:
            self.span_positional_encoding = SpanPositionalEncoding(model_dim)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        span_indices: Optional[torch.Tensor] = None,
        return_outputs: bool = False,
    ) -> List[torch.Tensor]:
        """Args:
            x: (batch_size, seq_len, num_features)
            padding_mask: (batch_size, seq_len), optional
            attn_mask: (batch_size, seq_len, seq_len), optional
            span_indices: (batch_size, seq_len), optional, providing positional information for each span
            return_outputs: bool, whether to return transformer encoder outputs
        Returns:
            decoded: list of num_features * (batch_size, seq_len, vocab_size of the feature)
            outputs: (batch_size, seq_len, model_dim), transformer encoder outputs
        """
        x = self.fuser(x)
        if self.use_span_positional_encoding and span_indices is not None:
            x = x + self.span_positional_encoding(span_indices)
        outputs = self.transformer_encoder(x, src_key_padding_mask=padding_mask, mask=attn_mask)
        decoded = self.fuser.decode(outputs)
        if return_outputs:
            return decoded, outputs
        return decoded

    def _get_loss(
        self, logits: List[torch.Tensor], label_ids: torch.Tensor, return_parts: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        losses = []
        for i, logit in enumerate(logits):
            # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size, seq_len)
            loss = F.cross_entropy(
                logit.transpose(1, 2), label_ids[:, :, i], ignore_index=self.tokenizer.pad_token_ids[i]
            )
            losses.append(loss)
        loss = torch.stack(losses).mean()
        if return_parts:
            return loss, losses
        return loss

    def _get_logits(self, batch: DataBatch, return_outputs: bool = False) -> List[torch.Tensor]:
        """Args:
            batch: DatasetBatch
            return_outputs: bool, whether to return transformer encoder outputs
        Returns:
            decoded: list of num_features * (batch_size, seq_len, vocab_size of the feature)
            outputs: (batch_size, seq_len, model_dim), transformer encoder outputs
        """
        batch_size, seq_len, _ = batch.input_ids.shape
        attention_mask = batch.attention_mask
        if attention_mask is not None and len(attention_mask.shape) == 3:
            # attention_mask: (batch_size * num_heads, seq_len, seq_len)
            attention_mask = attention_mask.expand(self.num_heads, -1, -1, -1).reshape(
                self.num_heads * batch_size, seq_len, seq_len
            )
        return self(
            batch.input_ids,
            batch.padding_mask,
            attention_mask,
            span_indices=batch.span_indices,
            return_outputs=return_outputs,
        )


class TrainingTask:
    is_main_task: bool = False

    def __init__(self, name: str):
        self.name = name

    def register_extra_modules(self, model: MelodyModel) -> None:
        pass

    def __call__(self, model: MelodyModel, batch: DataBatch, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class GeneralTask(TrainingTask):
    def __init__(self, name: str = "general"):
        super().__init__(name)
        self.is_main_task = True

    def __call__(self, model: MelodyModel, batch: DataBatch, **kwargs) -> torch.Tensor:
        logits, outputs = model._get_logits(batch, return_outputs=True)
        return model._get_loss(logits, batch.label_ids), outputs


class NgramClassificationTask(TrainingTask):
    def __init__(self, ngram_type: str, name: str = "ngram_classification"):
        super().__init__(name)
        assert ngram_type in ("pitch", "rhythm")

    def register_extra_modules(self, model: MelodyModel) -> None:
        lexicon_path = os.path.join(model.dataset_dir, "ngram_data", "lexicon.pkl")
        pitch_size, rhythm_size = get_lexicon_size(lexicon_path)
        ngram_size = pitch_size if self.ngram_type == "pitch" else rhythm_size
        model.ngram_head = NgramClassificationHead(model.model_dim, ngram_size)

    def __call__(
        self, model: MelodyModel, batch: DataBatch, model_outputs: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        batch_size, seq_len, _ = model_outputs.shape
        assert batch.ngram_type == self.ngram_type, "ngram types of batch and task must be the same"
        assert batch.ngram_ids.shape == (batch_size, seq_len), "label_ids must be of shape (batch_size, seq_len)"

        logits = model.ngram_head(model_outputs)
        loss = F.cross_entropy(
            logits.transpose(1, 2), batch.ngram_ids, ignore_index=ngram_ids_ignore_index
        )
        return loss


class SpanRewritingTask(TrainingTask):
    def __init__(self, name: str = "span_rewriting"):
        super().__init__(name)

    def __call__(self, model: MelodyModel, batch: DataBatch, **kwargs) -> torch.Tensor:
        pass


class MelodyPretrainModel(MelodyModel):
    """Use this subclass for pretraining or finetuning the model."""

    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Union[int, Tuple[int, ...]],  # embedding_dim: (bar, position, duration, pitch)
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        # Optimizer hyperparameters
        lr: float,
        betas: Tuple[float, float],
        weight_decay: float,
        warmup_percent: float,
        # Training configuration
        tasks: List[TrainingTask],
        task_weights: Optional[List[float]] = None,
        use_span_positional_encoding: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_dir=dataset_dir,
            embedding_dim=embedding_dim,
            model_dim=model_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_span_positional_encoding=use_span_positional_encoding,
        )

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_percent = warmup_percent

        self.tasks = tasks
        self.task_weights = task_weights
        if self.task_weights is None:
            self.task_weights = [1.0 / len(self.tasks)] * len(self.tasks)
        for task in self.tasks:
            task.register_extra_modules(self)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.max_steps,
            anneal_strategy="cos",
            pct_start=self.warmup_percent,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    def _get_batch_size(self, batch: GeneralDataBatch) -> int:
        if isinstance(batch, DataBatch):
            return batch.input_ids.shape[0]
        # get the first batch in the dict
        batch = next(iter(batch.values()))
        return batch.input_ids.shape[0]

    def _shared_step(self, batch: GeneralDataBatch) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if isinstance(batch, DataBatch):
            assert len(self.tasks) == 1, "Only one task is allowed for DataBatch"
            return self.tasks[0](self, batch)

        assert self.tasks[0].is_main_task, "The main task must be the first one in the list of tasks"
        assert len(batch) == len(self.tasks), "The number of tasks must be the same as the number of batches"

        losses = []
        for i, task in enumerate(self.tasks):
            batch_ = batch[task.name]
            if task.is_main_task:
                # Get the model outputs from the main task for the other tasks
                loss, model_outputs = task(self, batch_)
            else:
                loss = task(self, batch_, model_outputs=model_outputs)
            loss *= self.task_weights[i]
            losses.append(loss)
        loss = torch.stack(losses).sum()
        return loss, losses

    def training_step(self, batch: GeneralDataBatch, batch_idx: int) -> torch.Tensor:
        batch_size = self._get_batch_size(batch)
        loss, losses = self._shared_step(batch)
        self.log("train_loss", loss, sync_dist=True, batch_size=batch_size)
        if len(self.tasks) > 1:
            for task, task_loss in zip(self.tasks, losses):
                self.log(f"train_loss:{task.name}", task_loss, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: GeneralDataBatch, batch_idx: int) -> torch.Tensor:
        batch_size = self._get_batch_size(batch)
        loss, _ = self._shared_step(batch)
        self.log("val_loss", loss, sync_dist=True, batch_size=batch_size)
        return loss


class MelodyTestingModel(MelodyModel):
    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Union[int, Tuple[int, ...]],  # embedding_dim: (bar, position, duration, pitch)
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        # Testing hyperparameters
        max_seq_len: int,
        perplexity_stride: int,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_dir=dataset_dir,
            embedding_dim=embedding_dim,
            model_dim=model_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.max_seq_len = max_seq_len
        self.perplexity_stride = perplexity_stride

        self.pad_token_tensor = torch.from_numpy(self.tokenizer.pad_token_ids).long()

    def test_step(self, batch: DataBatch, batch_idx: int) -> torch.Tensor:
        """Calculate strided fixed-length perplexity for CLM model.
        Reference: https://huggingface.co/docs/transformers/perplexity
        """
        _, seq_len, _ = batch.input_ids.shape
        assert (
            len(batch.attention_mask.shape) == 2
        ), "Only support calculating perplexity for CLM model, where attention mask should be 2D."

        nlls = []
        previous_end_index = 0
        for start_index in range(0, seq_len, self.perplexity_stride):
            end_index = min(start_index + self.max_seq_len, seq_len)
            target_length = end_index - previous_end_index

            # Mask out context tokens for labels
            new_label_ids = batch.label_ids[:, start_index:end_index, :].clone()
            new_label_ids[:, :-target_length] = self.pad_token_tensor.to(new_label_ids.device)
            new_batch = DataBatch(
                input_ids=batch.input_ids[:, start_index:end_index, :],
                label_ids=new_label_ids,
                padding_mask=batch.padding_mask[:, start_index:end_index],
                attention_mask=batch.attention_mask[start_index:end_index, start_index:end_index],
            )
            logits = self._get_logits(new_batch)
            loss = self._get_loss(logits, new_batch.label_ids)
            neg_log_likelihood = loss * target_length
            nlls.append(neg_log_likelihood)
            previous_end_index = end_index

        ppl = torch.exp(torch.stack(nlls).sum() / end_index)
        self.log("perplexity", ppl, sync_dist=True)
        return ppl


class MelodyCompletionModel(MelodyModel):
    """Use this subclass for the melody completion downstream task."""

    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Union[int, Tuple[int, ...]],  # embedding_dim: (bar, position, duration, pitch)
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        # Inference hyperparameters
        conditional_bar_length: int,
        prediction_bar_length: int,
        temperature: float,
        top_k: int,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_dir=dataset_dir,
            embedding_dim=embedding_dim,
            model_dim=model_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.conditional_bar_length = conditional_bar_length
        self.prediction_bar_length = prediction_bar_length
        self.temperature = temperature
        self.top_k = top_k

    def predict_step(self, batch: DataBatch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        input_ids = batch.input_ids
        batch_size, _, _ = input_ids.shape
        assert batch_size == 1, "Only support batch size of 1 for prediction for now"

        # Crop the input to the conditional bar length
        bar_field_index = self.tokenizer.field_indices["bar"]
        conditional_bar_token_id = self.tokenizer.encoder["bar"][self.conditional_bar_length]
        prediction_bar_token_id = self.tokenizer.encoder["bar"][self.prediction_bar_length]

        conditional_bar_test = (input_ids[0, :, bar_field_index] == conditional_bar_token_id).nonzero()
        assert conditional_bar_test.shape[1] >= 1, "No conditional bar token found in the input"
        conditional_bar_index = conditional_bar_test[0, 0].item()
        input_ids = input_ids[:, :conditional_bar_index, :]

        # Inference on a single sequence
        while True:
            seq_len = input_ids.shape[1]
            attention_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1).to(
                input_ids.device
            )
            logits = self(input_ids, attn_mask=attention_mask)
            sampled_tokens = []
            for logit in logits:
                # Decode according to the sampling strategy
                sampled_token = top_k_sample(logit[0, -1, :], k=self.top_k, t=self.temperature)
                sampled_tokens.append(sampled_token)
            sampled_tokens = torch.cat(sampled_tokens, dim=-1)

            token = self.tokenizer.convert_id_to_token(sampled_tokens.cpu().numpy())
            # print(token)

            # until the desired bar length is reached
            if sampled_tokens[bar_field_index] == prediction_bar_token_id:
                break

            # Append the sampled token to the input
            input_ids = torch.cat([input_ids, sampled_tokens.unsqueeze(0).unsqueeze(0)], dim=1)

        return input_ids


class MelodyInfillingModel(MelodyModel):
    """Use this subclass for the melody infilling downstream task."""

    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Union[int, Tuple[int, ...]],  # embedding_dim: (bar, position, duration, pitch)
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        # Inference hyperparameters
        num_middle_bars: int,
        temperature: float,
        top_k: int,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_dir=dataset_dir,
            embedding_dim=embedding_dim,
            model_dim=model_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.num_middle_bars = num_middle_bars
        self.temperature = temperature
        self.top_k = top_k

        self.sep_token = torch.from_numpy(self.tokenizer.sep_token_ids)

    def predict_step(self, batch: DataBatch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        input_ids = batch.input_ids
        batch_size, original_len, _ = input_ids.shape
        assert batch_size == 1, "Only support batch size of 1 for prediction for now"

        bar_field_index = self.tokenizer.field_indices["bar"]
        bar_mask_token_id = self.tokenizer.mask_token_ids[bar_field_index]
        bar_sep_token_id = self.tokenizer.sep_token_ids[bar_field_index]
        mask_token_position = (input_ids[0, :, bar_field_index] == bar_mask_token_id).nonzero()[0].item()
        future_bar_token_id = input_ids[0, mask_token_position + 1, bar_field_index]

        # Add one seperator token to the end
        self.sep_token = self.sep_token.to(input_ids.device)
        input_ids = torch.cat([input_ids, self.sep_token.unsqueeze(0).unsqueeze(0)], dim=1)
        # Attention mask for the prefix part
        attention_mask = torch.zeros((original_len, original_len), dtype=torch.bool).to(input_ids.device)

        while True:
            seq_len = input_ids.shape[1]

            # Construct causal part of the attention mask for the next token
            previous_len = attention_mask.shape[0]
            mask_column = torch.ones((previous_len, 1), dtype=torch.bool).to(attention_mask.device)
            attn_row = torch.zeros((1, seq_len), dtype=torch.bool).to(attention_mask.device)
            attention_mask = torch.cat([attention_mask, mask_column], dim=1)
            attention_mask = torch.cat([attention_mask, attn_row], dim=0)
            # (previous_len, previous_len) -> (seq_len, seq_len)

            # print("input_ids:", input_ids)
            # print("attention_mask:", attention_mask)

            logits = self(input_ids, attn_mask=attention_mask)
            sampled_tokens = []
            for logit in logits:
                # Decode according to the sampling strategy
                sampled_token = top_k_sample(logit[0, -1, :], k=self.top_k, t=self.temperature)
                sampled_tokens.append(sampled_token)
            sampled_tokens = torch.cat(sampled_tokens, dim=-1)

            token = self.tokenizer.convert_id_to_token(sampled_tokens.cpu().numpy())
            # print(token)

            # until the model predicts a seperator token, or the desired bar length is reached
            if (
                sampled_tokens[bar_field_index] == bar_sep_token_id
                or sampled_tokens[bar_field_index] == future_bar_token_id
            ):
                break

            # Append the sampled token to the input
            input_ids = torch.cat([input_ids, sampled_tokens.unsqueeze(0).unsqueeze(0)], dim=1)

        # Rearrange the sequence to the original order
        input_ids = torch.cat(
            [
                input_ids[:, :mask_token_position, :],
                input_ids[:, original_len:, :],
                input_ids[:, mask_token_position + 1 : original_len, :],
            ],
            dim=1,
        )
        return input_ids


def top_k_sample(logits: torch.Tensor, k: int, t: float = 1.0) -> torch.Tensor:
    """Sample from the top k logits with temperature t"""
    assert k > 0, "k must be greater than 0"
    assert t > 0, "t must be greater than 0"
    logits = logits / t
    top_k_logits, top_k_indices = torch.topk(logits, k)
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    sampled_index = torch.multinomial(top_k_probs, 1)
    sampled_token = top_k_indices.gather(0, sampled_index)
    return sampled_token


class CustomWriter(BasePredictionWriter):
    """Write the prediction to a MIDI file."""

    # TODO: Write correct file name.

    def __init__(self, output_dir: str):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: MelodyPretrainModel,
        prediction: torch.Tensor,
        batch_indices: Optional[Sequence[int]],
        batch: DataBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        dest_path = os.path.join(self.output_dir, f"{batch_idx}.mid")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        prediction = prediction[0].cpu().numpy()
        midi = pl_module.tokenizer.decode(prediction)
        midi.dump(dest_path)
