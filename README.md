# Auto_Virtual_Tour_Gen

## Background
The workflow suggested in this project is an automatic virtual tour generation using a drone:
The drone is automatically guided to several Points of Interest (POIs) to capture images that are stitched together into 
a panorama image for each POI. These panoramas are then used for a virtual tour creation.

The first step in the suggested workflow in acquiring an initial exploration video of the space to be visualized.
This video is manually captured and it can be captured using any camera. It is the only step in the workflow that 
requires a human intervention. The purpose of this step is to use an existing Simultaneous localization and mapping (SLAM)
solution to map the scene and retrieve a sparse point cloud.
Next, the sparse point cloud of the scene, generated from the SLAM solution, is used to find the borders of the scene, 
usually made of walls and doors, and calculate several POIs to which the drone is sent in order to capture images.
The drone gets POIs coordinates which were calculated based on a sparse point cloud, and flies to the POIs.
In each POI, the drone hovers and starts to rotate incrementally by a chosen angle size until it reaches a 360° rotation.
In each angle, the drone acquires a single image. The acquired data is then sent to a connected computer and saved on disk.
Then, an automatic software performs images stitching to create a panoramic image for each POI.
Then, it uses the result panoramas to create a virtual tour, which is the final output of the workflow.


## Virtual Tour Example
A demonstration of the tour can be viewed in this video:
https://www.youtube.com/watch?v=FbnNVpJl6PU

The virtual tour can easily be viewed in your computer: (this flow is also shown in the video)
1. Download the "example_tour" folder
2. Run:
    - Windows: tour\tour_testingserver.exe
    - Linux: tour/tour_testingserver_macos
3. The virtual tour will be opened in the browser.
4. Changes to the tour can be made through the tour.xml file. In the video the transparency value of the hotspot clickable area (alpha parameter) is changed to 0 to get invisible hotspots.

## Code
The code contains 2 different parts, which are used in different parts of the above flow:

1. A code for POI selection derived from a sparse point-cloud created on the first exploration video step.
    In these POIs images will be acquired and used for the virtual tour creation.
    Requirements:
    python=3.6
    libraries: cv2, Open3d, numpy, pandas, matplotlib.pyplot


2. A code for an automatic panorama-based virtual tour creation from single images.
    Requirements:
    python=3.6
    installed libraries: xml, shutil, subprocess, glob, fileinput

    The code is a python wrap which uses these 2 external programs:
    2a. Hugin framework - Used to: create panoramas from single images.
    
    Download from: 
    http://hugin.sourceforge.net/download/

    2b. krpano framework - Used to: create a virtual tour from several panoramas.
        
    Download from: 
	https://krpano.com/download


## More Details
A detailed description of the software installations and the code can be found in the project report Appendix section.


