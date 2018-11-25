# Behavioral transfer for self driving with CARLA

- client_example.py is used for generating training data
- Replace the client_example.py in PythonClient folder in CARLA

### Execution steps
1. run server: ./CarlaUE4.sh -carla-server -benchmark -fps=15 -windowed -ResX=800 -ResY=600
2. run client: python client_example.py -i -q Low -a -l -sem

### Output Information
Output is generated in a folder called _out, with a new folder per episode. Each episode contains 
- measurement
- camerargb
- depthmap
- lidar
- segmentation: the images will appear black, a converter has to be used from Utils/ImageConverter to get the segmented image. 

  Usage: ./bin/image_converter -c semseg -i ../../PythonClient/_out/episode_0000/SemanticSegmentation/ -o output/folder


