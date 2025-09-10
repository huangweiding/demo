# n = 100, k = 2, res = 10, 10
# n = 44, k = 3, res = 2, 2, 11
import math


def test(n, k):
    def dfs(n, k):
        if k == 1:
            return [n]
        min_diff = float('inf')
        best_factors = None

        for i in range(1, int(n**0.5)+1):
            if n % i ==0:
                qutation = n // i
                remaining_factors = dfs(qutation, k-1)
                print(i)
                print(remaining_factors)
                current_factors = [i] + remaining_factors
                print("current_factors")
                print(current_factors)

                sorted_factors = sorted(current_factors)
                diff = sorted_factors[-1] - sorted_factors[0]
                print("current_diff")
                print(diff)
                if diff < min_diff:
                    min_diff = diff
                    best_factors = sorted_factors
                    print("best_factors")
                    print(best_factors)
                    if min_diff == 0:
                        break
        return best_factors

    return list(dfs(n, k))



if __name__ == "__main__":
    n = 44 
    k = 3
    print(test(n, k))

