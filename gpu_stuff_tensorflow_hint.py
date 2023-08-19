import fiftyone as fo
import json
import fiftyone.zoo as foz
from PIL import Image
import numpy as np
import pybboxes as pbx


dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    dataset_name="evaluate-detections-tutorial",
)
count = 0
dataset.persistent = True
x_in = np.array([])
y_in = np.array([])
the_size = 500000
the_coordinates = []
the_super_list = {}
min_predictions = 50
max_predictions = 60
the_answers = []
y_answers = np.array([])
x_predictions = np.array([])
county = 0
print(len(dataset))
the_misses = 0

x_predictions = np.array([])
the_answers = []

the_county  = 0

the_final_counter = 0
for sample in dataset:
    if(the_county > 1040):
        break
    if(the_county < 1020):
        the_county += 1
        continue

    if sample.ground_truth is None:
        the_misses+=1
        continue

    for i in sample.ground_truth.detections:
        img=np.array(Image.open(sample.filepath))
        the_lister = list(i['bounding_box'])
        the_label = i['label']

        the_list = []
        the_list = (img.shape[1],img.shape[0])
        a = pbx.convert_bbox(the_lister, from_type="yolo", to_type="voc", image_size=(the_list))
        the_coordinates.append(a)
        counter = a[1]
        while(counter < a[3]):
            the_list = img[a[0]:a[2]]
            counter += 1

        the_list  = np.resize(the_list, (420, 420, 3))
        the_final_counter += 1
        print(the_list.shape)



        #if(len(new_list) > the_size):
            #new_list = new_list[0:the_size]

        x_predictions = np.append(x_predictions, the_list)
    
        the_answers.append(i['label'])
        
    the_county += 1




x_in = x_predictions
x_in = x_in.reshape(the_final_counter,420,420,3)
print(x_in.shape)
y_in = np.array(the_answers)
np.savez_compressed('the_data_test.npz', A=x_in, B=y_in)
