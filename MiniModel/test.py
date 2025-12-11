import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os


class SimpleDataset(Dataset):
    """Simple dataset for demonstration"""
    
    def __init__(self, size=100):
        self.data = list(range(size))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def setup_distributed():
    """Initialize distributed training"""
    # Initialize process group
    # For single machine, use 'nccl' backend (CUDA) or 'gloo' (CPU)
    # For multiple machines, use 'nccl' with init_method='env://'
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # Single process mode (for testing)
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
        )
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    # breakpoint()
    
    print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    
    # Create dataset
    dataset = SimpleDataset(size=100)
    
    # Create DistributedSampler
    # This ensures each process gets a different subset of data
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,  # Shuffle data
        seed=42,  # Same seed for reproducibility
    )
    
    # Create DataLoader with DistributedSampler
    # Important: Do NOT set shuffle=True in DataLoader when using DistributedSampler
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        sampler=sampler,  # Use sampler instead of shuffle
        num_workers=0,  # Set to 0 for simplicity, increase for better performance
    )
    
    # Training loop
    print(f"\nRank {rank}: Starting training loop...")
    for epoch in range(2):
        # Set epoch for DistributedSampler (important for shuffling across epochs)
        sampler.set_epoch(epoch)
        
        print(f"\nRank {rank}: Epoch {epoch}")
        for batch_idx, batch in enumerate(dataloader):
            # Each process will see different batches
            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}: {batch.tolist()}")
            
            # Simulate training step
            # loss = model(batch)
            # loss.backward()
            # optimizer.step()
            
            # Only show first few batches for demonstration
            if batch_idx >= 2:
                break
    
    # Cleanup
    cleanup_distributed()
    print(f"\nRank {rank}: Training completed")


if __name__ == "__main__":
    # To run with multiple processes, use:
    # torchrun --nproc_per_node=2 test.py
    # or
    # python -m torch.distributed.launch --nproc_per_node=2 test.py
    
    # For single process testing:
    main()

