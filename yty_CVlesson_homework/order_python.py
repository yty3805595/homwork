#1.bubbleSort
def bubbleSort(arr):
    for i in range(1, len(arr)):
        for j in range(0, len(arr)-i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
	
#2.
def selectionSort(arr):
    for i in range(len(arr) - 1):
        # 记录最小数的索引
        minIndex = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[minIndex]:
                minIndex = j
        # i 不是最小数时，将 i 和最小数进行交换
        if i != minIndex:
            arr[i], arr[minIndex] = arr[minIndex], arr[i]
    return arr
	
#3.
def insertionSort(arr):
    for i in range(len(arr)):
        preIndex = i-1
        current = arr[i]
        while preIndex >= 0 and arr[preIndex] > current:
            arr[preIndex+1] = arr[preIndex]
            preIndex-=1
        arr[preIndex+1] = current
    return arr
	
#4.希尔排序
def shellSort(arr):
    import math
    gap=1
    while(gap < len(arr)/3):
        gap = gap*3+1
    while gap > 0:
        for i in range(gap,len(arr)):
            temp = arr[i]
            j = i-gap
            while j >=0 and arr[j] > temp:
                arr[j+gap]=arr[j]
                j-=gap
            arr[j+gap] = temp
        gap = math.floor(gap/3)
    return arr
	
#5.
def mergeSort(arr):
    import math
    if(len(arr)<2):
        return arr
    middle = math.floor(len(arr)/2)
    left, right = arr[0:middle], arr[middle:]
    return merge(mergeSort(left), mergeSort(right))

def merge(left,right):
    result = []
    while left and right:
        if left[0] <= right[0]:
            result.append(left.pop(0));
        else:
            result.append(right.pop(0));
    while left:
        result.append(left.pop(0));
    while right:
        result.append(right.pop(0));
    return result


# quickSort
def kClosestNumbers(self, A , target, k, left=None, right=None):
	left = 0 if not isinstance(left,(int, float)) else left
	right = len(arr)-1 if not isinstance(right,(int, float)) else right
	if left < right:
		partitionIndex = partition(arr, left, right)
		quickSort(arr, left, partitionIndex-1)
		quickSort(arr, partitionIndex+1, right)
	return arr

def partition(arr, left, right):
	pivot = left
	index = pivot+1
	i = index
	while  i <= right:
		if arr[i] < arr[pivot]:
			swap(arr, i, index)
			index+=1
		i+=1
	swap(arr,pivot,index-1)
	return index-1

def swap(arr, i, j):
	arr[i], arr[j] = arr[j], arr[i]



#KKK
    def kClosestNumbers(self, A, target, k):
        length = len(A)
        if not A or k <=0 or k > length:
            return None
        start = 0
        end = length - 1
        index = self.partition(A, start, end , target)
        while index != k:
            if index > k:
                index = self.partition(A, start, index - 1,target)
            elif index < k:
                index = self.partition(A, index + 1, end, target)
        return A[:k]
    
    def partition(self, alist, start, end ,target):
        if end <= start:
            return 0
        base = abs(alist[start] - target)
        index1, index2 = start, end
        while start < end:
            while start < end and abs(alist[end] - target) >= base:
                end -= 1
            alist[start] = alist[end]
            while start < end and abs(alist[end] - target) <= base:
                start += 1
            alist[end] = alist[start]
       
        return start
 	