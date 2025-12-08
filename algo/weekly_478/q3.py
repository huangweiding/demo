"""
You are given an integer array nums.

A mirror pair is a pair of indices (i, j) such that:

0 <= i < j < nums.length, and
reverse(nums[i]) == nums[j], where reverse(x) denotes the integer formed by reversing the digits of x. Leading zeros are omitted after reversing, for example reverse(120) = 21.
Return the minimum absolute distance between the indices of any mirror pair. The absolute distance between indices i and j is abs(i - j).

If no mirror pair exists, return -1.

 

Example 1:

Input: nums = [12,21,45,33,54]

Output: 1

Explanation:

The mirror pairs are:

(0, 1) since reverse(nums[0]) = reverse(12) = 21 = nums[1], giving an absolute distance abs(0 - 1) = 1.
(2, 4) since reverse(nums[2]) = reverse(45) = 54 = nums[4], giving an absolute distance abs(2 - 4) = 2.
The minimum absolute distance among all pairs is 1.

Example 2:

Input: nums = [120,21]

Output: 1

Explanation:

There is only one mirror pair (0, 1) since reverse(nums[0]) = reverse(120) = 21 = nums[1].

The minimum absolute distance is 1.

Example 3:

Input: nums = [21,120]

Output: -1

Explanation:

There are no mirror pairs in the array.

 

Constraints:

1 <= nums.length <= 10**5
1 <= nums[i] <= 10**9
"""

from typing import List

class Solution:
    def reverse(self, x: int) -> int:
        """反转数字并去掉前导0"""
        # 将数字转为字符串，反转，再转回整数（自动去掉前导0）
        return int(str(x)[::-1])
    
    def minMirrorPairDistance(self, nums: List[int]) -> int:
        """
        优化解法：O(n) 时间复杂度
        使用哈希表记录每个数字最近出现的索引（从右到左遍历）
        对于每个位置 i，哈希表中记录的是所有 j > i 的位置中，每个数字最近出现的索引
        """
        min_distance = float('inf')
        # 哈希表：key 是数字，value 是该数字在当前位置右侧最近出现的索引
        num_to_index = {}
        
        # 从右到左遍历，这样对于每个 i，哈希表中记录的都是 j > i 的索引
        for i in range(len(nums) - 1, -1, -1):
            reversed_i = self.reverse(nums[i])
            
            # 如果哈希表中存在 reverse(nums[i])，说明找到了 mirror pair
            # 由于我们从右到左遍历，哈希表中记录的是该数字在 i 右侧最近出现的索引
            if reversed_i in num_to_index:
                j = num_to_index[reversed_i]
                distance = j - i
                min_distance = min(min_distance, distance)
                # 如果找到距离为1的匹配，这是可能的最小值，可以提前返回
                if min_distance == 1:
                    return 1
            
            # 更新当前数字的索引
            # 由于我们从右到左遍历，当我们第一次遇到某个数字时，它就是最右边的
            # 但我们需要的是最近的，所以每次都要更新（因为 i 在减小，所以当前 i 就是最近的）
            num_to_index[nums[i]] = i
        
        return min_distance if min_distance != float('inf') else -1

if __name__ == "__main__":
    pass
