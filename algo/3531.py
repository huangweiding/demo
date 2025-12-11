"""
3531. Count Covered Buildings
Medium
Topics
premium lock icon
Companies
Hint
You are given a positive integer n, representing an n x n city. You are also given a 2D grid buildings, where buildings[i] = [x, y] denotes a unique building located at coordinates [x, y].

A building is covered if there is at least one building in all four directions: left, right, above, and below.

Return the number of covered buildings
Example 1:



Input: n = 3, buildings = [[1,2],[2,2],[3,2],[2,1],[2,3]]

Output: 1

Explanation:

Only building [2,2] is covered as it has at least one building:
above ([1,2])
below ([3,2])
left ([2,1])
right ([2,3])
Thus, the count of covered buildings is 1.
Example 2:



Input: n = 3, buildings = [[1,1],[1,2],[2,1],[2,2]]

Output: 0

Explanation:

No building has at least one building in all four directions.
Example 3:



Input: n = 5, buildings = [[1,3],[3,2],[3,3],[3,5],[5,3]]

Output: 1

Explanation:

Only building [3,3] is covered as it has at least one building:
above ([1,3])
below ([5,3])
left ([3,2])
right ([3,5])
Thus, the count of covered buildings is 1.
 

Constraints:

2 <= n <= 105
1 <= buildings.length <= 105 
buildings[i] = [x, y]
1 <= x, y <= n
All coordinates of buildings are unique.
"""

from typing import List
class Solution:
    def countCoveredBuildings(self, n: int, buildings: List[List[int]]) -> int:
        # O(nlogn)
        x_mapping = {}
        y_mapping = {}

        for building in buildings:
            x, y = building
            if x not in x_mapping:
                x_mapping[x] = []
            if y not in y_mapping:
                y_mapping[y] = []

            x_mapping[x].append(y)
            y_mapping[y].append(x)

        total = 0
        for building in buildings:
            x, y = building

            y_lst = x_mapping[x]
            x_lst = y_mapping[y]

            if min(y_lst) < y < max(y_lst) and min(x_lst) < x < max(x_lst):
                total += 1
        return total

if __name__ == "__main__":
    n = 5
    buildings = [[1,3],[3,2],[3,3],[3,5],[5,3]]
    print(Solution().countCoveredBuildings(n, buildings))
