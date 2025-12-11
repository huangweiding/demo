"""
3583. Count Special Triplets
Medium
Topics
premium lock icon
Companies
Hint
You are given an integer array nums.

A special triplet is defined as a triplet of indices (i, j, k) such that:

0 <= i < j < k < n, where n = nums.length
nums[i] == nums[j] * 2
nums[k] == nums[j] * 2
Return the total number of special triplets in the array.

Since the answer may be large, return it modulo 109 + 7.

 

Example 1:

Input: nums = [6,3,6]

Output: 1

Explanation:

The only special triplet is (i, j, k) = (0, 1, 2), where:

nums[0] = 6, nums[1] = 3, nums[2] = 6
nums[0] = nums[1] * 2 = 3 * 2 = 6
nums[2] = nums[1] * 2 = 3 * 2 = 6
Example 2:

Input: nums = [0,1,0,0]

Output: 1

Explanation:

The only special triplet is (i, j, k) = (0, 2, 3), where:

nums[0] = 0, nums[2] = 0, nums[3] = 0
nums[0] = nums[2] * 2 = 0 * 2 = 0
nums[3] = nums[2] * 2 = 0 * 2 = 0
Example 3:

Input: nums = [8,4,2,8,4]

Output: 2

Explanation:

There are exactly two special triplets:

(i, j, k) = (0, 1, 3)
nums[0] = 8, nums[1] = 4, nums[3] = 8
nums[0] = nums[1] * 2 = 4 * 2 = 8
nums[3] = nums[1] * 2 = 4 * 2 = 8
(i, j, k) = (1, 2, 4)
nums[1] = 4, nums[2] = 2, nums[4] = 4
nums[1] = nums[2] * 2 = 2 * 2 = 4
nums[4] = nums[2] * 2 = 2 * 2 = 4
 

Constraints:

3 <= n == nums.length <= 105
0 <= nums[i] <= 105
"""
from typing import List

class Solution:
    def specialTriplets(self, nums: List[int]) -> int:
        def naive():
            total = 0
            N = len(nums)
            for i in range(N):
                for j in range(i+1, N):
                    for k in range(j+1, N):
                        if nums[j] * nums[j] == nums[i] and nums[j] * nums[j] == nums[k]:
                            total += 1
            return total
        # forward and backward 
        N = len(nums)
        freqBefore = [0] * (N)
        freqAfter = [0] * (N)
        d = {}
        for i in range(N):
            if nums[i] * 2 in d:
                freqBefore[i] = d.get(nums[i]*2)
            d[nums[i]] = d.get(nums[i], 0) + 1

        d = {}
        for j in range(N-1, -1, -1):
            if nums[j] * 2 in d:
                freqAfter[j] = d.get(nums[j] * 2)
            d[nums[j]] = d.get(nums[j], 0)+1

        print(freqBefore)
        print(freqAfter)
        total = 0
        for k in range(N):
            total += freqBefore[k] * freqAfter[k]
        return total



if __name__ == "__main__":
    nums = [6,3,6]
    print(Solution().specialTriplets(nums))
