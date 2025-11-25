# if k = 5
# if k = 7, 3 x 7 = 21
# if k = 13
# if k = 3 , n = 111
# if 1, 3, 7
#  what do we do if k is 13,  13 * 97 
class solution:
    def smallestRepunitDivByK(self, k: int):
        cnt = 1
        n = 1
        seen = set()

        while True:
            if n % k == 0:
                return cnt
            else:
                remainder = n % k 
                if remainder not in seen:
                    seen.add(remainder)
                else:
                    break

            n = 10 * n +1
            cnt += 1
        
        return -1


        

