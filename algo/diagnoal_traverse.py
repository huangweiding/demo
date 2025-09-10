from typing import List
class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        ans = []
        R = len(mat)
        C = len(mat[0])

        D = R+C-1
        for d in range(D):
            if d%2 == 0:
                # for j in range(max(0,d-C+1), min(C, d+1)):
                #     x = d-j 
                #     print(x, j)
                #     ans.append(mat[x][j])
                for i in range(max(0, d-R+1), min(R, d+1)):
                    x = d - i
                    print(i, x)
                    ans.append(mat[x][i])
            else:
                for j in range(max(0,d-C+1), min(C, d+1)):
                    y = d -j
                    ans.append(mat[j][y])
                # for i in range(max(0, d-R+1), min(R, d+1)):
                #     y = d - i
                #     print(i, y)
                #     ans.append(mat[i][y])
        return ans
            

if __name__ == "__main__":
    mat = [
    [1,2,3],
    [4,5,6],
    [7,8,9]]
    # mat = [[1,2],[3,4]]
    print(Solution().findDiagonalOrder(mat))
