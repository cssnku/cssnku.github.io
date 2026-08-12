from model import PassBERT
import torch.nn.functional as F
import argparse
from utils import read_label, LabelDataset, label_collate
import torch
from tqdm import tqdm

def trainer(model,
            dataloader,
            optimizer,
            save_path=None,
            epochs=5):
    """Run the training loop for a given number of epochs.

    The model predicts the in-place edit-path label at every character
    position; cross-entropy is computed per position (PAD labels are ignored).
    """
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc="Training")
        for X_seq, _, labels in loop:
            optimizer.zero_grad()
            labels = torch.tensor(labels).to('cuda')
            X_seq = torch.tensor(X_seq).to('cuda')
            outputs = model(X_seq)
            B, L, num_classes = outputs.shape
            outputs_flat = outputs.view(-1, num_classes)
            labels_flat = labels.view(-1)
            loss = F.cross_entropy(outputs_flat, labels_flat, ignore_index=2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_description(f"{loss.item()}")
        if save_path is not None:
            torch.save({
                'Model': model.state_dict(),
            }, save_path.replace('.pth', f'_last.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--train_file', type=str, default="./dataset/inplace_edits.txt")
    parser.add_argument('--load_ckpt', type=str, default=None)
    parser.add_argument('--save_path', type=str, default="./ckpts/bert_global.pth")
    parser.add_argument('--epochs', type=int, default=5)

    args = parser.parse_args()

    model = PassBERT()
    model.to('cuda')
    if args.load_ckpt is not None:
        # Fine-tune starting from an existing checkpoint, freezing part of
        # the network (global-to-local / no-freeze adaptation).
        model.freeze_layers()
        model.load_state_dict(torch.load(args.load_ckpt)['Model'])
    src_tokens, trg_tokens, padded_labels = read_label(args.train_file)
    dataset = LabelDataset(src_tokens, trg_tokens, padded_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=label_collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer(model, dataloader, optimizer, args.save_path, args.epochs)
