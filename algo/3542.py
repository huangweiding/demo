from typing import List
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        s = []
        res = 0
        for a in nums:
            while s and s[-1] > a:
                s.pop()
            if a == 0:
                continue
            if not s or s[-1] < a:
                res += 1
                s.append(a)
            print(s)
        return res

if __name__ == "__main__":
    nums = [1, 3, 2]
    print(Solution().minOperations(nums))