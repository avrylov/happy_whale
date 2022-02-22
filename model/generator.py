from torch.utils.data import Dataset
import cv2


class MyDataset(Dataset):
    def __init__(self,
                 df,
                 crop=False,
                 transform=None
                 ):
        self.df = df.copy()
        self.crop = crop
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def crop_image(self, row, image, image_name):
        if image_name == 'first':
            bbox = row['bbox_voc_a']
            xmin, ymin, xmax, ymax = bbox
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            image = image[ymin:ymax, xmin:xmax]
        else:
            bbox = row['bbox_voc_b']
            xmin, ymin, xmax, ymax = bbox
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            image = image[ymin:ymax, xmin:xmax]
        return image

    def __get_input(self, row):

        image_a = cv2.imread(row['image_a'])
        image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)

        image_b = cv2.imread(row['image_b'])
        image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)

        inputs = {'first': image_a, 'second': image_b}

        return inputs

    def __get_output(self, row):
        label = row['label']
        return label

    def __getitem__(self, index):
        row = self.df.iloc[index, :]
        inputs = self.__get_input(row)
        labels = self.__get_output(row)

        if self.crop:
            image_a = inputs['first']
            image_b = inputs['second']

            image_a = self.crop_image(row, image_a, 'first')
            image_b = self.crop_image(row, image_b, 'second')

            inputs = {'first': image_a, 'second': image_b}

        if self.transform:
            image_a = inputs['first']
            image_b = inputs['second']

            transformer = self.transform(image=image_a, image_b=image_b)

            image_a = transformer['image']
            image_b = transformer['image_b']

            inputs = {'first': image_a, 'second': image_b}

        return index, inputs, labels