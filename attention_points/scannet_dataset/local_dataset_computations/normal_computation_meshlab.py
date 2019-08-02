"""
This script allows to use a Meshlab Server to compute and extract normal vectors from ply files
This script is designed for Windows commands, to adapt the commands, see https://github.com/TheNerdJedi/MeshlabAuto
"""
import os
import subprocess
import time

if __name__ == '__main__':
    source_dir = "C:/scannet"
    target_dir = "C:/scannet_normal"

    for subdir, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith("2.ply"):
                # paths must be adjusted for the command
                command = f"\"C:\\Program Files\\VCG\\MeshLab\\meshlabserver\" -i {subdir + '/' + file} " \
                          f"-o {target_dir + '/' + file} -m vn -s C:\\scannet-pre\\test.mlx "
                print(file)
                # os.system(f"start /wait cmd /c {command}")
                subprocess.call(f'{command}', shell=True)
                time.sleep(0.1)
