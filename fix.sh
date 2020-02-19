cd ~/.mujoco
wget https://www.roboti.us/download/mjpro150_linux.zip
unzip mjpro150_linux.zip
rm -rf mjpro150_linux.zip
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/liyuc/.mujoco/mjpro150/bin" >> ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liyuc/.mujoco/mjpro150/bin
echo "run: sudo apt-get install patchelf"
su feisha
pip install mujoco_py==1.50.1.68

