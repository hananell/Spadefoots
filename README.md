This project is a part of biological clock research in spadefoots.
there are many spadefoots, each in a cell with camera that shoots a photo when detects a movement.
Problem is, there are many movements that do not belong to a spadefoots, such as movements of a sunspot or a worm.
This code was built to filter out those images.

We built a YOLOv5x model for detecting spadefoots, trained on about 4000 manually labeled images. The model take an image and return if there is a spadefoot, and its coordinates.
We built a function that given two images, returns where exactly the images are different = where there is a movement.
The program works like this: for each image, detect spadefoots and movements (comparing to previous image). If there is an overlap it's a moving spadefoot, otherwise put that image aside.
For each camera there will be corresponding result directory, with to subdirectories: with_movements for images with a moving spadefoots, and without_movements for others.

To run the program:
Make data directory that contains directories of computers that contain directories of cameras that contain images.
In main.py set 'data_dir' to your data directory, and 'results_dir' to your desired destination of results.
Get spadefoots_yolov5x.pt (not on github because it's big) and put it in the same directory as the scripts.
Run main.py

To compare your results to manually labeled results, run comparing.py with 'results' set to your results' directory, and 'truth' set to manually labeled directory.