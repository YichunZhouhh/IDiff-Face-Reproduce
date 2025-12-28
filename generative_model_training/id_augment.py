import os
import shutil
import time
import subprocess

def moveFiles(path, disdir, suffix = ".png"):  # path为原始路径，disdir是移动的目标目录
    for j in range(0,5):
        imagename = os.path.join(disdir, str(j + 50) + suffix)  
        # try-catch
        max_tries=3
        for retry in range(max_tries):
            try:
                if not os.path.exists(imagename):
                    shutil.copy(path, imagename) # 为每个原始图片创建五个副本
                break  # 如果成功，跳出重试循环
            except (BlockingIOError, OSError) as e:
                if retry < max_tries - 1:
                    time.sleep(0.1)  # 等待0.1秒后重试
                else:
                    # 最后一次重试失败，使用subprocess调用cp命令
                    try:
                        subprocess.run(['cp', path, imagename], check=True)
                    except subprocess.CalledProcessError as e:
                        # 如果cp命令也失败，尝试其它方法
                        try:
                            # 读取文件内容并写入新文件
                            with open(path, 'rb') as src:
                                with open(imagename, 'wb') as dst:
                                    dst.write(src.read())
                        except Exception as e:
                            print(f"Failed to copy {path} to {imagename}: {e}")
                            continue

id_images_root = "../dataset/context_database/images"
disdir = './samples'    
suffix = ".png"
# load id images
if os.path.isdir(id_images_root):
    id_images = os.listdir(id_images_root)
    id_images = sorted(id_images)
    ## add origin id x 5 ###
    for i in range(0, len(id_images)):
        origin_id = id_images[i]
        disdir_id = os.path.join(disdir, str(i))
        if os.path.isdir(disdir_id):
            moveFiles(os.path.join(id_images_root, origin_id), disdir_id, suffix = suffix)
        if i % 100 == 0:
            print("done %d"%i)