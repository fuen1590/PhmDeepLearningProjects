from torch.utils.data import Dataset


class Sampler:
    """
    This class is used for customize yourself sampling method used for dataset.

    This class is first used in cmapss.CMAPSS dataset, and will be supported more custom dataset sampling.

    When customizing your own sampler, you should:

    1. override the sample(index) method. The sample(index) method should return the sample and label similar to
    torch.utils.data.Dataset class.

    2. Making sure your __init__(dataset) method containing the sampling target argument "dataset".
    The argument "dataset" should be a torch.utils.data.Dataset instance.
    And call the super.__init__(dataset) at the first line in your __init__ method.
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def sample(self, index: int):
        raise NotImplementedError("You must define the Sampler.sample(index) method.")
