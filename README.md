# ros-faster-rcnn
ROS bridge to py-faster-rcnn

### Caffe Changes:

This requires a few changes to your local caffe in order to run, as I haven't figured out how to get ROS to point to a specific directory for Caffe.

1) Move your entire caffe install to a backup folder somewhere
2) Move the caffe fork from here: https://github.com/rbgirshick/caffe-fast-rcnn/tree/0dcd397b29507b8314e252e850518c5695efbb83 into it's old place (probably ~caffe/)
3) Add caffe to your $PYTHONPATH 
```
export PYTHONPATH=~/caffe/python:$PYTHONPATH
```
4) Add a layers directory to your caffe install, so that [caffe_path]/python/caffe/layers exists
5) Add the faster-rcnn layers to your caffe install. Copy all of the directories within ros_faster_rcnn/scripts/lib/ into [caffe_path]/python/caffe/layers
6) Add that to your $PYTHONPATH
```
export PYTHONPATH=~/caffe/python/caffe/layers:$PYTHONPATH
```
7) Get the .caffemodel and .prototxt files from me
8) Edit lines 21 - 22 to point to the .caffemodel and .prototxt files
9) Run!
