
import matplotlib.pyplot as plt

# helper function for data visualization
def visualize(saveFlag=False, num=0, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    
    if saveFlag:
        plt.tight_layout()
        figSavePath = 'Patch_examples'
        if not os.path.exists(str(figSavePath)): os.mkdir(figSavePath)
        plt.savefig(str(figSavePath / f"{dataName}_{num}.png"))   
    plt.show()

if __name__ == "__main__":

    from data_samping import DataSampler
    from easydict import EasyDict as edict
    from pathlib import Path
    from prettyprinter import pprint

    cfg = edict(
            rootPath = Path("E:\PyProjects\PyTorch_TV_Transfer_Learning\data"),

            patchsize = 256,
            num_patch_per_image = 10,
            train_val_split_rate = 0.7,
            random_state = 42,
        )

    dataSampler = DataSampler(cfg)
    # dataSampler()
    pprint(dataSampler.cfg)


    from data_provider import Dataset
    from data_visualize import visualize

    x_train_dir = dataSampler.train.patchDir
    y_train_dir = dataSampler.train.maskDir_SAR

    dataset = Dataset(x_train_dir, y_train_dir, classes=['fire'])

    for i in range(0, 5):
        image, mask = dataset[i] # get some sample
        visualize(
            saveFlag=False,
            image=image, 
            cars_mask=mask.squeeze(),
        )


    augmented_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        # augmentation=get_training_augmentation(), 
        classes=['fire'],
    )

    # same image with different random transforms
    for i in range(0, 5):
        image, mask = augmented_dataset[i]
        # print(image.shape, mask.shape)
        visualize(image=image, mask=mask.squeeze(-1))
