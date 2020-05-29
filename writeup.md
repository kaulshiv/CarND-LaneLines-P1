# **Finding Lane Lines on the Road** 
---

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[ROI]: ./test_images_output/roi_solidWhiteCurve.jpg "ROI"
[screenshot]: ./screenshot.png "screenshot"
[white]: ./test_images_output/solidWhiteRight.jpg "white"
[yellow]: ./test_images_output/solidYellowCurve.jpg "yellow"

---


### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

I first converted the image to grayscale. Then I applied a gaussian blur to the image with a filter size of 3. I then used the canny edge detection algorithm with upper and lower thresholds of 100 and 250, respectively. I used a quarilateral region of interest mask which can be seen in the blue outline fo the image below.


![ROI]

Finally I applied the hough transform with the following parameters

```python
rho = 1
theta = 2*π/180
hough_threshold = 50
min_line_len = 20
max_line_gap = 40
```

All parameters of the algorithms used in the pipeline were chosen with the help of a utility I wrote. You can look at the code in `parameter_tuning.py`. Here is a screen shot from using the tool.

![screenshot]

In order to draw full lines from the bottom to the top of the ROI, I modified the draw_lines function to take the average of all lines with positive slopes (corresponding to the right lane) and those with negative slopes (corresponding to the left lane). I also averaged all the midpoints of lines with positive and negative slope as well. Using the calculated slope and midpoint for each lane, I was able to draw the final "aggregate" line.

Here are some more results 

![white]
![yellow]

### 2. Identify potential shortcomings with your current pipeline


One shortcoming of my current pipeline is that for a small number of frames in the video my pipeline does not draw a lane line. This stems from the fact that the image processing pipeline fails to detect and lines with either positive or negative slope. Subsequently when it takes the average, it returns NaN. In these instances, I do not draw a line.

Also, in some frames of the output video the lane lines cross each other. This can be seen for example in `test_videos_output/solidYellowLeft.mp4` around 12 seconds and towards the end.


### 3. Suggest possible improvements to your pipeline

To address the two shortcomings listed above it would be helpful to average and/or interpolate the lane lines across frames (in time), in addition to averaging them within a single frame. This would prevent wild deviations between frames and would allow you to have exactly two lane lines for each and every frame.