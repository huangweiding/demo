"""
LeetCode 3770: Largest Prime from Consecutive Prime Sum
Weekly Contest 479

Given an integer n, find the largest prime number <= n that can be 
expressed as the sum of consecutive prime numbers.
"""


def sieve_of_eratosthenes(limit):
    """
    Generate all prime numbers up to limit using Sieve of Eratosthenes.
    
    Time Complexity: O(n log log n) where n = limit
    - Outer loop runs for primes p where p² ≤ n, so p ≤ √n
    - For each prime p, inner loop marks multiples: n/p iterations
    - Total work: n/2 + n/3 + n/5 + n/7 + ... (sum over primes ≤ √n)
    - This sum is approximately n * (sum of 1/p for primes ≤ √n)
    - Sum of reciprocals of primes up to k is approximately log log k
    - Therefore: O(n log log n)
    
    Space Complexity: O(n)
    - is_prime array of size (limit + 1)
    - primes list of size approximately n/ln(n)
    """
    if limit < 2:
        return []
    
    # 步骤1: 初始化 - 假设所有数字都是质数
    # is_prime[i] = True 表示 i 是质数
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False  # 0 和 1 不是质数
    
    # 步骤2: 从 2 开始，遍历到 √limit
    # 原理：如果 n 是合数，它必然有一个因子 ≤ √n
    # 所以只需要检查到 √limit 即可标记所有合数
    p = 2
    while p * p <= limit:
        # 如果 p 仍然是质数（没有被之前的数标记为合数）
        if is_prime[p]:
            # 步骤3: 标记 p 的所有倍数为合数
            # 从 p² 开始，因为 p * k (k < p) 已经被更小的质数标记过了
            # 例如：当 p=5 时，5*2=10 已经被 p=2 标记，5*3=15 已经被 p=3 标记
            # 所以只需要从 5*5=25 开始标记
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False  # 标记为合数
        p += 1
    
    # 步骤4: 收集所有仍然是 True 的数字，它们就是质数
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    return primes


def largest_prime_from_consecutive_prime_sum(n):
    """
    Find the largest prime number <= n that can be expressed as 
    the sum of consecutive prime numbers.
    
    Time Complexity: O(n² / log²(n))
    - Sieve of Eratosthenes: O(n log log n)
    - Prefix sum calculation: O(P) where P ≈ n/ln(n) is the number of primes
    - Nested loops: O(P²) = O(n² / log²(n)) in worst case
      (Early termination helps in practice, but worst case is still O(P²))
    
    Space Complexity: O(n)
    - Sieve array: O(n)
    - Primes list and set: O(n/ln(n))
    - Prefix sums: O(n/ln(n))
    """
    if n < 2:
        return 0
    
    # Generate all primes up to n
    primes = sieve_of_eratosthenes(n)
    if not primes:
        return 0
    
    # Create a set for O(1) prime lookup
    prime_set = set(primes)
    
    # Calculate prefix sums for efficient sum calculation
    prefix_sums = [0] * (len(primes) + 1)
    for i in range(len(primes)):
        prefix_sums[i + 1] = prefix_sums[i] + primes[i]
    
    max_prime = 0
    
    # Check all possible consecutive prime subsequences (at least 2 primes)
    for i in range(len(primes)):
        for j in range(i + 2, len(primes) + 1):  # j starts at i+2 to require at least 2 primes
            # Sum of primes from index i to j-1
            sum_consecutive = prefix_sums[j] - prefix_sums[i]
            
            # Early termination if sum exceeds n
            if sum_consecutive > n:
                break
            
            # Check if the sum is prime
            if sum_consecutive in prime_set:
                max_prime = max(max_prime, sum_consecutive)
    
    return max_prime


# Example usage and test cases
if __name__ == "__main__":
    # Test case 1: n = 1000
    n1 = 1000
    result1 = largest_prime_from_consecutive_prime_sum(n1)
    print(f"n = {n1}, result = {result1}")  # Expected: 953
    
    # Test case 2: n = 10
    n2 = 10
    result2 = largest_prime_from_consecutive_prime_sum(n2)
    print(f"n = {n2}, result = {result2}")  # Expected: 5 (2+3=5)
    
    # Test case 3: n = 50
    n3 = 50
    result3 = largest_prime_from_consecutive_prime_sum(n3)
    print(f"n = {n3}, result = {result3}")

