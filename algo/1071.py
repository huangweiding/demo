"""
1071. Greatest Common Divisor of Strings
Solved
Easy
Topics
premium lock icon
Companies
Hint
For two strings s and t, we say "t divides s" if and only if s = t + t + t + ... + t + t (i.e., t is concatenated with itself one or more times).

Given two strings str1 and str2, return the largest string x such that x divides both str1 and str2.

 

Example 1:

Input: str1 = "ABCABC", str2 = "ABC"
Output: "ABC"
Example 2:

Input: str1 = "ABABAB", str2 = "ABAB"
Output: "AB"
Example 3:

Input: str1 = "LEET", str2 = "CODE"
Output: ""
"""

class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        res = ""
        str1_length = len(str1)
        str2_length = len(str2)
        final = ""

        if str1_length >= str2_length:
            for i in str2:
                res += i
                reach1 = False
                reach2 = False
                for j in range(1, 1001):
                    if res * j == str2:
                        reach2 = True
                    if res * j == str1:
                        reach1 = True
                if reach1 and reach2:
                    final = res
        else:
            for i in str1:
                res += i
                reach1 = False
                reach2 = False
                for j in range(1, 1001):
                    if res * j == str2:
                        reach2 = True
                    if res * j == str1:
                        reach1 = True
                if reach1 and reach2:
                    final = res
        return final



if __name__ == "__main__":
    pass
