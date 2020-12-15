from DataSet import MyDataSet
if __name__ == '__main__':
    dataset = MyDataSet.MyDataset()
    print(len(dataset))
    img, label, bbox = dataset[0]
    img, label, bbox




