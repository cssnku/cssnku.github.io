import transformers
from transformers import BertConfig, BertModel
import torch
import torch.nn as nn

class PassBERT(nn.Module):
    """BERT encoder with a per-position linear classification head.

    The model predicts, for each character position of the source password,
    an in-place edit operation (from the edit-path vocabulary).
    """

    def __init__(self, bert_config=None, output_dim=9122):
        super().__init__()
        if bert_config is None:
            bert_config = BertConfig(
                vocab_size=98,
                num_attention_heads=4,
                hidden_size=256,
                num_hidden_layers=4,
                max_position_embeddings=512,
            )

        self.bert = transformers.BertModel(bert_config)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state
        logits = self.fc(sequence_output)

        return logits

    def freeze_layers(self):
        """Freeze all weights except the last two BERT encoder layers.

        Used for the local-to-global and global-to-local fine-tuning stages:
        the embeddings, the pooler and the final two BERT layers are frozen
        so that only the lower encoder layers (and the head) are updated.
        """
        for param in self.fc.parameters():
            param.requires_grad = False

        # Freeze the pooler layer.
        for param in self.bert.pooler.parameters():
            param.requires_grad = False

        # Freeze the last two BERT encoder layers.
        encoder_layers = self.bert.encoder.layer
        for layer in encoder_layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = False

        # Freeze the embedding layers.
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

    def prob_extract(self, input_ids, labels, attention_mask=None):
        """Compute the cumulative probability of a target edit-path label
        sequence under the model, multiplying probabilities up to the
        second EOS token (label id == 0).

        Args:
            input_ids: token ids of the source password, (batch, seq_len).
            labels: ground-truth label ids, (batch, seq_len).
        Returns:
            total_probs: per-sample cumulative probability, (batch,).
        """
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True
            )
            sequence_output = outputs.last_hidden_state
            logits = self.fc(sequence_output)
        probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len, num_labels)

        # Gather the probability of the ground-truth label at each position.
        label_probs = torch.gather(probs, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)

        # Multiply probabilities only up to the second EOS token (label id == 0).
        batch_size, seq_len = labels.shape
        zero_mask = (labels == 0)
        # Number of EOS tokens seen before each position.
        zero_count = zero_mask.cumsum(dim=1)
        # Position of the second EOS token, if any.
        second_zero_pos = (zero_count == 2).float()
        idx = second_zero_pos.argmax(dim=1)
        has_second_zero = (second_zero_pos.sum(dim=1) > 0)
        idx = torch.where(has_second_zero, idx, torch.full_like(idx, seq_len))
        # Keep only positions before the second EOS token.
        arange = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(batch_size, seq_len)
        mask = arange < idx.unsqueeze(1)
        masked_probs = torch.where(mask, label_probs, torch.ones_like(label_probs))
        # Cumulative product over the sequence.
        total_probs = masked_probs.prod(dim=1)
        return total_probs


if __name__ == "__main__":
    # Instantiate the model.
    model = PassBERT()

    # Print an overview of the model architecture.
    print("Model architecture:")
    print(model)
    print("-" * 30)

    # Build a dummy input.
    batch_size = 2
    seq_length = 10
    # input_ids range is 0-97 (vocab_size=98).
    input_ids = torch.randint(0, 98, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))

    print(f"Input shape: {input_ids.shape}")

    # Forward pass.
    logits = model(input_ids, attention_mask)
    print(logits.shape)