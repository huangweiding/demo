from typing import List
import math

class Solution:
    def countElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        N = len(nums)
        # print(nums)

        greater_num = -1
        current_acc = 0
        current_max = math.inf
        cnt = 0
        for i in range(N-1, -1, -1):
            if nums[i] < current_max:
                greater_num += 1
                greater_num += current_acc
                current_acc = 0
                current_max = nums[i]
            else:
                current_acc += 1

            if greater_num >= k:
                cnt += 1
        return cnt

if __name__ == "__main__":
    nums = [4, 3, 5, 5, 6, 7, 7, 1, 2]
    k = 4
    print(Solution().countElements(nums, k))
