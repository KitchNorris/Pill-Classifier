from torch.utils.data import Dataset
from torchvision.transforms.v2 import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.datasets import ImageFolder


# Класс-обёртка для трансформаций
class TransformDataset(Dataset):
  def __init__(self, dataset, transforms):
    super(TransformDataset, self).__init__()
    self.dataset = dataset
    self.transforms = transforms

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    x, y = self.dataset[idx]
    return self.transforms(x), y
  
# Трансформации датасетов
train_transforms = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(p=0.2),
    RandomVerticalFlip(p=0.2),
    RandomRotation([-5, 5], fill=255.),
    ToTensor(),
    Normalize((0.5), (0.5))
])

test_transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize((0.5), (0.5))
])


# Загрузка датасета
train_path = 'data/dataset/ogyeiv2/train'
test_path = 'data/dataset/ogyeiv2/test'

train_data = ImageFolder(train_path)
test_data = ImageFolder(test_path)

# Добавление трансформаций к датасетам
train_dataset = TransformDataset(train_data, train_transforms)
val_dataset = TransformDataset(test_data, test_transforms)

# Список и кол-во классов в датасете
classes = train_data.classes
len_classes = len(train_data.classes)
