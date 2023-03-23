import pickle
import cv2
pickle_file_path = "/home/ciccio/Desktop/multi_task_lfd/multitask_dataset/multitask_dataset_baseline_1/pick_place/panda_pick_place/task_00/traj000.pkl"
with open(pickle_file_path, "rb") as f:
    sample = pickle.load(f)
    print(sample)
    
    obs = sample['traj'].get(0)['obs']['image'][:,:,::-1]
    # Display the image using OpenCV's imshow function
    cv2.imshow('Image', obs)
    cv2.waitKey(0) # wait for any key press
    # Save the image using OpenCV's imwrite function
    cv2.imwrite('/home/ciccio/Downloads/saved_image_obs.jpg', obs)

# Define the path to your image file
image_path = "/home/ciccio/Desktop/img/opencv-show-image-imshow.png"
# Load the image using OpenCV
img = cv2.imread(image_path)
# Display the image using OpenCV's imshow function
cv2.imshow('Image', img)
cv2.waitKey(0) # wait for any key press

# Save the image using OpenCV's imwrite function
cv2.imwrite('/home/ciccio/Downloads/saved_image.jpg', img)

# Close all windows
cv2.destroyAllWindows()


import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
# Load the image using the PIL library
img = Image.open(image_path)

# Define the transformation to apply
transform = transforms.RandomResizedCrop(size=(100,180), scale=(0.6,0.6), ratio=(1.8, 1.8))

# Apply the transformation to the image
transformed_img = transform(img)

# Display the original and transformed images using matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('RandomResizedCrop Transformation')

ax1.imshow(img)
ax1.set_title('Original Image')

ax2.imshow(transformed_img)
ax2.set_title('Transformed Image')

plt.show()
