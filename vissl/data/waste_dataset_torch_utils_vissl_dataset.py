from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image, UnidentifiedImageError

class VisslWasteDataset(Dataset):

    def __init__(self, cfg, data_source, path, split, dataset_name):
        super().__init__()

        assert data_source in [
            "waste_dataset",
        ], "data_source must be waste_dataset"

        self.cfg = cfg
        self.split = split  # TRAIN or VAL
        self.dataset_name = dataset_name  # waste-dataset
        self.data_source = data_source  # waste_dataset
        self._path = Path(path)  # /workspace/data/single_label_images/train

        self.df = pd.read_csv(self._path / 'df.csv')

        self._num_samples = len(self.df)

    def num_samples(self):
        """
        Size of the dataset
        """
        return self._num_samples

    def __len__(self):
        """
        Size of the dataset
        """
        return self.num_samples()

    def __getitem__(self, idx: int):
        """
        implement how to load the data corresponding to idx element in the dataset
        from your data source
        """

        try:
            loaded_data = Image.open(self._path / 'img' / self.df['id'][idx]).convert("RGB")
            is_success = True
        except UnidentifiedImageError:
            loaded_data = None
            is_success = False

        # is_success should be True or False indicating whether loading data was successful or failed
        # loaded data should be Image.Image if image data
        return loaded_data, is_success


if __name__ == '__main__':
    # just for testing purposes
    dataset = VisslWasteDataset(
        cfg={},
        split='TRAIN',
        dataset_name='waste-dataset',
        data_source='waste_dataset',
        path='/workspace/data/single_label_images/train'
    )

    img, success = dataset[0]
