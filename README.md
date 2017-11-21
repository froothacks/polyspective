# Polyspective

## Inspiration
Video editing takes ages to do successfully and tons of manpower. Casual concert-goers or small video creators don't have the time or resources to spend on editing a bunch perspectives together.

## What it does
Perspective is a "hands free" software recording app that will automatically switch cameras based on the activity they are recording. It can be used for live streams, or pre recorded videos. Our software does an entire **paid job**, for **free**.


## How we built it
We use a set of different features to look for in a camera's frame, and turn these into scores. We then combine these scores (some weighted different than others) to a final, "activity score". The camera with the highest activity score is the camera selected for input.

## Challenges we ran into
Here we go...

- Writing an algorithm to find out if a person is talking or not, from video
- Sometimes our facial recognition software cuts out, so we must interpolate previous frame's data to accommodate for sudden "empty" frames (that aren't really empty).
- Since we didn't have loads of time to optimize our code, we had to make recording accommodations for the processing time our algorithms take
 - Faces are not always straight. Frames are. We had to write an algorithm to find the tilt of a face, and then neutralize that in our other facial algorithms.

## Accomplishments that we're proud of

- Detects human speech and switches to the desired camera
- Detects movements, such as actions/gestures, and switches to the desired camera
- Training a recurrent neural network to continuously classify video and audio
- Simulating a phone as a webcam

## What we learned
- Sleep is for the weak.
- Detecting lips is much harder than it seems
- Image processing is very expensive in terms of time complexity

## What's next for Polyspective
- Faster processing time and a more intuitive user interface.
- Server side processing of images for faster image analysis
- Finishing the neural network training of the algorithm on professionally edited YouTube videos in order to use machine learning for the algorithm to improve accuracy
