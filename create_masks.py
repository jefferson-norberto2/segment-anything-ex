from typing import Any
from segment_anything import SamPredictor, sam_model_registry
from numpy import array, save, uint8
from cv2 import imread, imwrite, cvtColor, bitwise_and, threshold
from cv2 import COLOR_BGR2RGB, THRESH_BINARY, COLOR_BGR2GRAY
from os import listdir
from utils.tools import get_dict_annotations, get_annotation

class CreateMasks:
    def __init__(self, root_path: str, path_model='sam_vit_l.pth', model_type='vit_l') -> Any:
        self.path_images = root_path + 'images/'
        self.path_masks = root_path + 'masks/'
        self.path_annotations = root_path + 'annotations.json'
        self.sam_checkpoint = path_model
        self.model_type = model_type
        self.device = "cuda"
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
    
    def generate_masks(self):
        annotations = get_dict_annotations(self.path_annotations)
        print('Start generating: ')
        for filename in listdir(self.path_images):
            annotation = get_annotation(annotations, filename)
            image = imread(self.path_images + filename)
            print(filename, end=', ')
            image = cvtColor(image, COLOR_BGR2RGB)
            self.predictor.set_image(image)

            image_id = filename.split('.')[0]
            input_box = array(annotation[filename])
            input_point = array([[input_box[0] + 120, input_box[1] + 35]])
            input_label = array([1])
            
            masks, _, _ = self.predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                box=input_box[None, :],
                                multimask_output=False,
                            )

            save(self.path_masks + str(image_id) + '.npy', masks)
            
            mask_image = (masks * 255).astype(uint8)
            output_image = bitwise_and(image, image, mask=mask_image[0])
            gray_output_image = cvtColor(output_image, COLOR_BGR2GRAY)
            _, output_image = threshold(gray_output_image, 0, 255, THRESH_BINARY)
            imwrite(self.path_masks + str(image_id) + '.png', output_image)

        
if __name__ == '__main__':
    print('Generating masks...')
    create_masks = CreateMasks('urna/', 'sam_vit_l.pth', 'vit_l')
    create_masks.generate_masks()
    print('Done!')