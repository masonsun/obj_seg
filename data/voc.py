import subprocess
import os

_CMDS = [
	# download pascal voc 2012 dataset
	"wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
	"tar -xvf VOCtrainval_11-May-2012.tar",
	# create symlinks
	"ln -s ./VOCdevkit/VOC2012/SegmentationClass labels",
	"ln -s ./VOCdevkit/VOC2012/JPEGImages images",
]

if __name__ == "__main__":
	for c in _CMDS:
		subprocess.call(c.split())
	print('Downloaded VOC 2012 successfully.')