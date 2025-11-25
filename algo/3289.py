class Solution:
    def getSneakyNumbers(self, nums: List[int]) -> List[int]:
        n = len(nums) - 2
        y = 0
        for x in nums:
            y ^= x
        for i in range(n):
            y ^= i
        lowBit = y & -y
        x1 = x2 = 0
        for x in nums:
            if x & lowBit:
                x1 ^= x
            else:
                x2 ^= x
        for i in range(n):
            if i & lowBit:
                x1 ^= i
            else:
                x2 ^= i
        return [x1, x2]