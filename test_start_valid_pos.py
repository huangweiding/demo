import torch

def test_start_valid_pos():
    window_size = 3
    seq_length = 6
    
    for i in range(seq_length):
        start_valid_pos = max(0, i - window_size)
        
        if start_valid_pos > 0:
            print(f"Position {i}: start_valid_pos={start_valid_pos} -> Block positions [0, {start_valid_pos})")
        else:
            print(f"Position {i}: start_valid_pos={start_valid_pos} -> Block nothing (no positions too far)")
    
    print("\nVisual representation:")
    print("Allowed attention ranges:")
    for i in range(seq_length):
        start_valid_pos = max(0, i - window_size)
        allowed_range = list(range(start_valid_pos, i + 1))
        print(f"  Position {i}: attend to {allowed_range}")

if __name__ == "__main__":
    test_start_valid_pos()
