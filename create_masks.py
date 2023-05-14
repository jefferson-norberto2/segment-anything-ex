from typing import Any
from segment_anything import SamPredictor, sam_model_registry
from random import randint
import numpy as np
import cv2
import os
from interpolation import get_annotations

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

    def generate_masks(self, point=[350, 250], range_pixels=60, confidence=0.98, num_process=5):
        for index, filename in enumerate(os.listdir(self.path_images)):
            image = cv2.imread(self.path_images + filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image)
            
            input_point = np.array([point])
            input_label = np.array([1])
            
            masks, scores, logits = self.process_mask(input_point=input_point, input_label=input_label, mask_input=None, multimask_output=True)

            aux1 = randint(10, range_pixels)
            aux2 = randint(10, range_pixels)
            aux3 = randint(10, range_pixels)
            aux4 = randint(10, range_pixels)

            point_2 = [point[0] - aux1*10, point[1] - aux2*10]
            point_3 = [point[0] + aux3, point[1] + aux4]
            input_point = np.array([point_2, point_3])
            input_label = np.array([1, 1])

            for _ in range(num_process):
                if np.max(scores) > confidence:
                    mask_input = logits[np.argmax(scores), :, :]
                    masks, scores, _ = self.process_mask(
                        input_point=input_point,
                        input_label=input_label,
                        mask_input=mask_input[None, :, :],
                        multimask_output=False )
                    break

                aux1 = randint(10, range_pixels)
                aux2 = randint(10, range_pixels)
                aux3 = randint(10, range_pixels)
                aux4 = randint(10, range_pixels)

                point_2 = [point[0] - aux1*10, point[1] - aux2*10]
                point_3 = [point[0] + aux3, point[1] + aux4]

                input_point = np.array([point_2, point_3])
                input_label = np.array([1, 1])

                mask_input = logits[np.argmax(scores), :, :]

                masks, scores, logits = self.process_mask(
                    input_point=input_point,
                    input_label=input_label,
                    mask_input=mask_input[None, :, :],
                    multimask_output=True,
                )

            # plt.figure(figsize=(10,10))
            # plt.imshow(image)
            # show_mask(masks, plt.gca())
            # show_points(input_point, input_label, plt.gca())
            # plt.axis('off')
            # plt.show() 
        
            np.save(self.path_masks + filename + '.npy', masks)
            mask_image = (masks * 255).astype(np.uint8)
            output_image = cv2.bitwise_and(image, image, mask=mask_image[0])
            cv2.imwrite(self.path_masks + filename, output_image)
            
            print('Images done:' + str(index) + ', ', end='')

    def generate_mask2(self):
        anottations = get_annotations(self.path_annotations)
        print('Start generating: ')
        for index, filename in enumerate(os.listdir(self.path_images)):
            image = cv2.imread(self.path_images + filename)
            print(filename, end=', ')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image)

            image_id = filename.split('.')[0]
            input_box = np.array(anottations[image_id])
            input_point = np.array([[input_box[0] + 120, input_box[1] + 35]])
            input_label = np.array([1])
            
            masks, _, _ = self.predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                box=input_box[None, :],
                                multimask_output=False,
                            )

            np.save(self.path_masks + str(image_id) + '.npy', masks)
            
            mask_image = (masks * 255).astype(np.uint8)
            output_image = cv2.bitwise_and(image, image, mask=mask_image[0])
            gray_output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
            _, output_image = cv2.threshold(gray_output_image, 0, 255, cv2.THRESH_BINARY)
            cv2.imwrite(self.path_masks + str(image_id) + '.png', output_image)
        
    def process_mask(self, input_point, input_label, mask_input, multimask_output=False):
        if mask_input is None:
            return self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                mask_input=mask_input,
                multimask_output=multimask_output,
            )
        else:
            return self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                mask_input=mask_input,
                multimask_output=multimask_output,
            )


if __name__ == '__main__':
    print('Generating masks...')
    create_masks = CreateMasks('urna/', 'sam_vit_l.pth', 'vit_l')
    create_masks.generate_mask2()
    print('Done!')