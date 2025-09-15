from typing import List
class Solution:
    def subsequenceSumAfterCapping(self, nums: List[int], k: int) -> List[bool]:
        def check_subsequence(current_nums, target):
            dp = [False] * (target+1)
            dp[0] = True
            for n in current_nums:
                for i in range(target, n-1, -1):
                    if dp[i-n]:
                        dp[i] = True
            return dp[target]


        final = [False] * len(nums)
        for i in range(len(nums)):
            current_nums = []
            min_number = i+1
            for idx, n in enumerate(nums):
                current_nums.append(min(min_number, n))
            final[i] = check_subsequence(current_nums, k)
        return final

     
if __name__ == "__main__":
    nums = [11,12,2,8,4,19,10,10,14,20,17,10,2,13,20,15,20,9,13,16]
    # nums = [4, 3, 2, 4]
    k = 5
    print(Solution().subsequenceSumAfterCapping(nums, k))
