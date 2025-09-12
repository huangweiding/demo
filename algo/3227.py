class Solution:
    def doesAliceWin(self, s: str):
        vowels = ['a', 'e', 'i', 'o', 'u']
        cnt = 0
        for i in s:
            if i in vowels:
                cnt += 1
        if cnt == 0:
            return False
        if cnt % 2 != 0:
            return True
        return True


if __name__ == "__main__":
    s = "leetcoder"
    print(Solution().doesAliceWin(s))
