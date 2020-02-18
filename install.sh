#!/bin/bash

echo "setup mujoco"
if [ ! -d ~/.mujoco ]; then
	mkdir ~/.mujoco
	cp ~/borg/mujoco/mjkey.txt ~/.mujoco
fi

if [ ! -d ~/.mujoco/mjpro150 ]; then
	cd ~/.mujoco
	wget https://www.roboti.us/download/mjpro150_linux.zip
	unzip mjpro150_linux.zip
	rm -rf mjpro150_linux.zip
	#mv mujoco200_linux mujoco200
	cd -
	#echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/liyuc/.mujoco/mujoco200/bin" >> ~/.bashrc
	#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liyuc/.mujoco/mujoco200/bin
fi

source ~/anaconda3/etc/profile.d/conda.sh
if [ ! -d ~/anaconda3/envs/qmc ]; then
	echo "install qmc conda env"
	conda env create -f environment.yml
else
	conda activate qmc
	conda env update --file environment.yml
fi 
conda activate qmc
pip install numba
pip install particles
cd ../cherry/
python setup.py develop
pip install randopt
pip install mujoco_py==1.50.1.68
