from multinews import toh5df

if __name__ == '__main__':
    toh5df('../train.txt.src', '../train.txt.tgt', '../multinews.train2.h5df', filter_small=False)
    toh5df('../test.txt.src', '../test.txt.tgt', '../multinews.test2.h5df', filter_small=False)
    toh5df('../val.txt.src', '../val.txt.tgt', '../multinews.val2.h5df', filter_small=False)