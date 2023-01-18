from torch.utils.data import DataLoader
import torchvision


def get_dataset_resolution_range(dataloader):
    min_width = 100000
    max_width = 0
    min_height = 100000
    max_height = 0

    for batch_i, batch in enumerate(dataloader):
        # print("batch #", batch_i)
        # forward pass
        x = batch[0]
        _, _, height, width = x.size()
        if width < min_width:
            min_width = width
        if width > max_width:
            max_width = width
        if height < min_height:
            min_height = height
        if height > max_height:
            max_height = height

        return min_width, max_width, min_height, max_height


def main():
    root = '/home/user_115/Project/Code/Milestone1/datasets/'
    transform = torchvision.transforms.ToTensor()
    train_data = torchvision.datasets.StanfordCars(root, split="train", transform=transform,
                                                   target_transform=None, download=True)
    test_data = torchvision.datasets.StanfordCars(root, split="test", transform=transform,
                                                  target_transform=None, download=True)

    dataloader_train = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=True)
    dataloader_test = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=True)
    min_width_train, max_width_train, min_height_train, max_height_train = get_dataset_resolution_range(
        dataloader_train)
    min_width_test, max_width_test, min_height_test, max_height_test = get_dataset_resolution_range(dataloader_test)

    min_width = min(min_width_train, min_width_test)
    max_width = max(max_width_train, max_width_test)
    min_height = min(min_height_train, min_height_test)
    max_height = max(max_height_train, max_height_test)

    print(f"min_width:{min_width},min_height:{min_height},max_width:{max_width},max_height:{max_height}")


if __name__ == '__main__':
    main()
