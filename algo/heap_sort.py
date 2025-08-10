def heapify(arr, n, i):
    """
    将以索引i为根的子树调整为最大堆
    arr: 待排序的数组
    n: 堆的大小
    i: 当前需要调整的根节点索引
    """
    largest = i  # 初始化最大值为根节点
    left = 2 * i + 1  # 左子节点索引
    right = 2 * i + 2  # 右子节点索引
    
    # 如果左子节点存在且大于根节点
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    # 如果右子节点存在且大于最大值
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    # 如果最大值不是根节点，则交换并继续调整
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        # 递归调整受影响的子树
        heapify(arr, n, largest)


def heap_sort(arr):
    """
    堆排序主函数
    时间复杂度: O(n log n)
    空间复杂度: O(1) - 原地排序
    """
    n = len(arr)
    
    # 第一步：构建最大堆
    # 从最后一个非叶子节点开始，自底向上构建堆
    # 最后一个非叶子节点的索引是 (n//2 - 1)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # 第二步：逐个提取堆顶元素（最大值）
    # 将堆顶元素与末尾元素交换，然后重新调整堆
    for i in range(n - 1, 0, -1):
        # 将当前最大值（堆顶）移到数组末尾
        arr[0], arr[i] = arr[i], arr[0]
        # 对剩余元素重新构建最大堆
        heapify(arr, i, 0)
    
    return arr


def heap_sort_descending(arr):
    """
    降序堆排序（构建最小堆）
    """
    n = len(arr)
    
    # 构建最小堆
    for i in range(n // 2 - 1, -1, -1):
        heapify_min(arr, n, i)
    
    # 逐个提取最小值
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify_min(arr, i, 0)
    
    return arr


def heapify_min(arr, n, i):
    """
    构建最小堆的辅助函数
    """
    smallest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] < arr[smallest]:
        smallest = left
    
    if right < n and arr[right] < arr[smallest]:
        smallest = right
    
    if smallest != i:
        arr[i], arr[smallest] = arr[smallest], arr[i]
        heapify_min(arr, n, smallest)


# 测试代码
if __name__ == "__main__":
    # 测试升序排序
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print("原始数组:", test_array)
    
    sorted_array = heap_sort(test_array.copy())
    print("升序排序后:", sorted_array)
    
    # 测试降序排序
    test_array2 = [64, 34, 25, 12, 22, 11, 90]
    descending_array = heap_sort_descending(test_array2.copy())
    print("降序排序后:", descending_array)
    
    # 测试边界情况
    empty_array = []
    single_array = [42]
    duplicate_array = [3, 1, 4, 1, 5, 9, 2, 6]
    
    print("\n边界情况测试:")
    print("空数组:", heap_sort(empty_array.copy()))
    print("单元素数组:", heap_sort(single_array.copy()))
    print("有重复元素的数组:", heap_sort(duplicate_array.copy()))
