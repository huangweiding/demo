"""
You have n computers. You are given the integer n and a 0-indexed integer array batteries where the ith battery can run a computer for batteries[i] minutes. You are interested in running all n computers simultaneously using the given batteries.

Initially, you can insert at most one battery into each computer. After that and at any integer time moment, you can remove a battery from a computer and insert another battery any number of times. The inserted battery can be a totally new battery or a battery from another computer. You may assume that the removing and inserting processes take no time.

Note that the batteries cannot be recharged.

Return the maximum number of minutes you can run all the n computers simultaneously.

"""

from typing import List

class Solution:
    def maxRunTime(self, n: int, batteries: List[int]) -> int:
        # Get the sum of all extra batteries.
        batteries.sort()   
        extra = sum(batteries[:-n])
        
        # live stands for the n largest batteries we chose for n computers.

        live = batteries[-n:]
        
        # We increase the total running time using 'extra' by increasing 
        # the running time of the computer with the smallest battery.
        for i in range(n - 1):
            # If the target running time is between live[i] and live[i + 1].
            if extra // (i + 1) < live[i + 1] - live[i]:
                return live[i] + extra // (i + 1)
            
            # Reduce 'extra' by the total power used.
            extra -= (i + 1) * (live[i + 1] - live[i])
        
        # If there is power left, we can increase the running time 
        # of all computers.
        return live[-1] + extra // n
