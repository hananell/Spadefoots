This project was requested by a Biology MSC student who does a biological clock experiment on spadefoots.
She has about 20 spadefoots, each in different cage with a camera on top, that takes pictures whenever detects any movement.
Problem is, there are movements not coming from the spadefoots, such as moving sunspots or Larvae (spadefoots' food). My task was to filter them out.

I have trained a yolov5x model on about 5000 images of spadefoots (5000 was necessary because it's a very hard task), and then for each pair of consecutive images:
1. Detect spadefoots using the model.
2. Detect movements, defined as any change between the images.
3. If there is an overlap - a spadefoot has moved.