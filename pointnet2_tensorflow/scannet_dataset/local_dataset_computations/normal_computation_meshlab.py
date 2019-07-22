import os
import time
import subprocess

target_dir = "C:/scannet_normal"

for subdir, dirs, files in os.walk("C:/scannet"):
    for file in files:
        if file.endswith("2.ply"):
            command = f"\"C:\\Program Files\\VCG\\MeshLab\\meshlabserver\" -i {subdir + '/' + file} " \
                f"-o {target_dir+ '/' + file} -m vn -s C:\\scannet-pre\\test.mlx "
            print(file)
            # os.system(f"start /wait cmd /c {command}")
            subprocess.call(f'{command}', shell=True)
            time.sleep(0.1)
