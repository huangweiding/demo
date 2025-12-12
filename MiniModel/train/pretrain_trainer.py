import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.miniModel import miniModelForCausalLM, MiniConfig
from train.train_utils import init_model, PretrainDataset, init_distributed_mode, setup_seed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import PretrainedConfig
import torch.distributed as dist
import argparse


def train_epoch(dataloader, iter, start_step=0, wandb=None):
    loss_fcn = torch.nn.CrossEntropyLoss()
    for step, (X, Y, mask) in enumerate(dataloader, start=start_step+1):
        optimizer.zero_grad()
        X = X.to(args.device)
        Y = Y.to(args.device)
        mask = mask.to(args.device)
        output = model(X)
        # Y.size() = (batch_size, max_length)
        loss = loss_fcn(output.logits.view(-1, args.vocab_size), Y.view(-1)).view(Y.size())
        loss = (loss*mask).sum() / mask.sum()
        loss.backward()
        optimizer.step()
        return loss.item(), step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniModel Configuration")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--embedding_size", type=int, default=2048)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--layer_num", type=int, default=28)
    parser.add_argument("--intermediate_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: Using device {args.device}")
        args.device = f"cuda:{local_rank}"
    setup_seed(42+ local_rank if dist.is_initialized() else 0)



    config = MiniConfig(
        vocab_size=args.vocab_size,
        embedding_size=args.embedding_size,
        dropout_rate=args.dropout_rate,
        num_heads=args.num_heads,
        max_length=args.max_length,
        eps=args.eps,
        hidden_size=args.hidden_size,
        layer_num=args.layer_num,
        intermediate_size=args.intermediate_size
    )

    model, tokenizer = init_model(config, tokenizer_path="/repos/tmp/minimind_periphrals/model_weight", weight_path=None, device="cpu")

    dataset = PretrainDataset(data_path="/repos/tmp/demo/MiniModel/data/test_data.jsonl", tokenizer=tokenizer)
    train_sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()) if dist.is_initialized() else None

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[dist.get_rank()])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0

    
    for epoch in range(start_epoch, args.num_epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:
            dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=0)
            train_epoch(dataloader, len(dataloader), start_step=0, wandb=None)
        else:
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=0)
            train_epoch(dataloader, len(dataloader), start_step=0, wandb=None)








