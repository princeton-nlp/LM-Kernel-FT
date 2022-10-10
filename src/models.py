"""Custom models for few-shot learning specific operations."""

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import  RobertaModel, RobertaLMHead
from transformers.modeling_outputs import SequenceClassifierOutput

import logging
logger = logging.getLogger(__name__)

def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError

class ModelForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model_type = config.model_type
        if self.model_type == "roberta":
            self.roberta = RobertaModel(config)
            self.lm_head = RobertaLMHead(config)
        elif self.model_type == "bert":
            self.bert = BertModel(config)
            self.cls = BertOnlyMLMHead(config)
        else:
            raise NotImplementedError

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = 0.0
        self.ub = 1.0

        # For auto label search.
        self.return_full_softmax = None

    def get_model_fn(self):
        return self.roberta if self.model_type == "roberta" else self.bert

    def get_lm_head_fn(self):
        return self.lm_head if self.model_type == "roberta" else self.cls

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
    ):
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        model_fn = self.get_model_fn()
        # Encode everything
        outputs = model_fn(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        if mask_pos is not None:
            sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
        else:
            sequence_mask_output = sequence_output[:,0] # <cls> representation
            # sequence_mask_output = sequence_output.mean(dim=1) # average representation

        if self.label_word_list is not None:
            # Logits over vocabulary tokens
            head_fn = self.get_lm_head_fn()
            prediction_mask_scores = head_fn(sequence_mask_output)

            # Exit early and only return mask logits.
            if self.return_full_softmax:
                if labels is not None:
                    return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
                return prediction_mask_scores

            # Return logits for each label
            logits = []
            # use MLM logit
            if self.model_args.use_task_word:
                vocab_logits = self.lm_head(sequence_mask_output)
                for _id in self.label_word_list:
                    logits.append(vocab_logits[:, _id].unsqueeze(-1))
            # use learned linear head logit on top of task word representation (standard LM-BFF)
            else:
                for label_id in range(len(self.label_word_list)):
                    logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
            logits = torch.cat(logits, -1)

            # Regression task
            if self.config.num_labels == 1:
                logsoftmax = nn.LogSoftmax(-1)
                logits = logsoftmax(logits) # Log prob of right polarity
        else:
            logits = self.classifier(sequence_mask_output)


        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                # Regression task
                if self.label_word_list is not None:
                    labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                    loss = nn.KLDivLoss(log_target=True)(logits.view(-1, 2), labels)
                else:
                    labels = (labels.float().view(-1) - self.lb) / (self.ub - self.lb)
                    loss =  nn.MSELoss()(logits.view(-1), labels)
            else:
                if self.model_args.l2_loss:
                    coords = torch.nn.functional.one_hot(labels.squeeze(), self.config.num_labels).float()
                    loss =  nn.MSELoss()(logits.view(-1, logits.size(-1)), coords)
                else:
                    loss =  nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.model_args.use_task_word and self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output
