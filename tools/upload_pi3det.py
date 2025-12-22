from huggingface_hub import HfApi

api = HfApi()

local_file = "/home/alan/AlanLiang/Dataset/M3ED/processed/Spot"   # 替换为你的文件路径

repo_id = "Pi3DET/data"
path_in_repo = "processed/Quadruped"

api.upload_folder(
    folder_path=local_file,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="dataset"
)

print("上传成功！")


local_file = "/home/alan/AlanLiang/Dataset/M3ED/processed/Falcon"   # 替换为你的文件路径

path_in_repo = "processed/Drone"

api.upload_folder(
    folder_path=local_file,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="dataset"
)

print("上传成功！")
