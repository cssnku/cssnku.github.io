




if __name__ == "__main__":
    #import os
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
    import warnings
    warnings.filterwarnings('ignore')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default="../dataset/train/pseudo_train_data-sister.txt")
    parser.add_argument('--save_path', type=str, default="./experiment/model.pth")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--load_ckpt', type=str, default=None)
    parser.add_argument('--src_pos', type=int, default=0)
    parser.add_argument('--trg_pos', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import utils
    import config
    import logging
    import numpy as np
    from data_loader import MTDataset
    from torch.utils.data import DataLoader
    import torch
    from PW2SEQ import PW2SEQ
    from train import train
    from model import make_model, LabelSmoothing
    pw2seq=PW2SEQ()
    pw2seq_dict=pw2seq.pw2seq_dict

    class NoamOpt:
        """Optim wrapper that implements rate."""

        def __init__(self, model_size, factor, warmup, optimizer):
            self.optimizer = optimizer
            self._step = 0
            self.warmup = warmup
            self.factor = factor
            self.model_size = model_size
            self._rate = 0

        def step(self):
            """Update parameters and rate"""
            self._step += 1
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p['lr'] = rate
            self._rate = rate
            self.optimizer.step()

        def rate(self, step=None):
            """Implement `lrate` above"""
            if step is None:
                step = self._step
            return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


    def get_std_opt(model):
        """for batch_size 32, 5530 steps for one epoch, 2 epoch for warm-up"""
        return NoamOpt(model.src_embed[0].d_model, 1, 10000,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


    def run(train_path=None, model_path=None, load_path=None, epochs=1, src_pos=0, trg_pos=1):
        if train_path == None:
            train_path=config.train_data_path
        if model_path == None:
            model_path=config.model_path

        utils.set_logger(config.log_path)

        train_dataset = MTDataset(train_path,mode='train', src_pos=src_pos, trg_pos=trg_pos)
        #logging.info("--------use keypress, use similar--------")
        logging.info("-------- Dataset Build! --------")
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.train_batch_size,
                                    collate_fn=train_dataset.collate_fn)

        logging.info("-------- Get Dataloader! --------")
        # 初始化模型
        model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                        config.d_model, config.d_ff, config.n_heads, config.dropout)
        #model.load_state_dict(torch.load(config.model_path))
        if load_path is not None:
            model.load_state_dict(torch.load(load_path))
            model.freeze_layers()
            freeze_layers=True
        else:
            freeze_layers=False
        model_par = torch.nn.DataParallel(model)
        # 训练
        if config.use_smoothing:
            criterion = LabelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
            criterion.cuda()
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=config.padding_idx, reduction='sum')
        if config.use_noamopt:
            optimizer = get_std_opt(model)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        train(train_dataloader, model, model_par, criterion, optimizer, model_path, freeze_layers=freeze_layers, epochs=epochs)
        #test(test_dataloader, model, criterion)


    def check_opt():
        """check learning rate changes"""
        import numpy as np
        import matplotlib.pyplot as plt
        model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                        config.d_model, config.d_ff, config.n_heads, config.dropout)
        opt = get_std_opt(model)
        # Three settings of the lrate hyperparameters.
        opts = [opt,
                NoamOpt(512, 1, 20000, None),
                NoamOpt(256, 1, 10000, None)]
        plt.plot(np.arange(1, 50000), [[opt.rate(i) for opt in opts] for i in range(1, 50000)])
        plt.legend(["512:10000", "512:20000", "256:10000"])
        plt.show()
    run(train_path=args.train_file, model_path=args.save_path, load_path=args.load_ckpt, epochs=args.epochs, src_pos=args.src_pos, trg_pos=args.trg_pos)
    #translate_example()
