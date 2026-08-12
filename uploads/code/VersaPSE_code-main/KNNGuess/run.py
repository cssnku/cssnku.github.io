import os
import config
try:
    f=open(config.train_data_path,"r",encoding='utf-8')
except:
    print("error train data path")
    exit()
# try:
#     f=open(config.test_data_path,"r",encoding='utf-8')
# except:
#     print("error test data path")
#     exit()
f.close()

# for exp in range(1,11):
#     # config.train_data_path = f'../formatted_data/filtered_target-exp{exp}.txt'
#     # config.model_path = f'./experiment/exp{exp}/model.pth'
#     if not os.path.exists(f'./experiment/exp{exp}'):
#         os.makedirs(f'./experiment/exp{exp}', exist_ok=True)
#     # print(config.train_data_path)
#     # print(config.model_path)

#     cmd=f'python main.py --train_path ../formatted_data/filtered_target-exp{exp}.txt --model_path ./experiment/exp{exp}/model.pth'
#     os.system(cmd)
# exp=1
# if not os.path.exists(f'./datastore/exp{exp}'):
#     os.makedirs(f'./datastore/exp{exp}', exist_ok=True)
# cmd=f'python gen_datastore.py --train_data_path ../formatted_data/filtered_target-exp{exp}.txt --knn_train_data_path ../formatted_data/filtered_target-exp{exp}_knn.txt --model_path ./experiment/exp{exp}/model.pth --knn_datastore_path ./datastore/exp{exp} --gpu_id 1'
# os.system(cmd)
# cmd='python test.py --test_data_path ../formatted_data/exp1/test.txt --guess_ans_path ./guess_ans.txt --guess_ans_path_nomix ./guess_ans_nomix.txt --model_path ./experiment/exp1/model.pt --knn_datastore_path ./knn_datastore'
# os.system(cmd)


# import os
# import subprocess
# import time

# num_gpus = 4
# exp_range = range(1, 8)
# gpu_pool = [i for i in range(num_gpus)]
# processes = []

# for idx, exp in enumerate(exp_range):
#     gpu_id = gpu_pool[idx % num_gpus]
#     if not os.path.exists(f'./datastore/exp{exp}'):
#         os.makedirs(f'./datastore/exp{exp}', exist_ok=True)
#     cmd = [
#         "python", "gen_datastore.py",
#         "--train_data_path", f"../formatted_data/exp{exp}/train_attack.txt",
#         "--knn_train_data_path", f"../formatted_data/train_attack-exp{exp}_knn.txt",
#         "--model_path", f"./experiment/exp{exp}/model.pth",
#         "--knn_datastore_path", f"./datastore/exp{exp}",
#         "--gpu_id", str(gpu_id)
#     ]
#     # 启动进程
#     p = subprocess.Popen(cmd)
#     processes.append((p, gpu_id))
#     # 控制最多同时运行4个进程
#     if len(processes) == num_gpus:
#         # 等待任意一个进程结束
#         while True:
#             for i, (proc, gid) in enumerate(processes):
#                 if proc.poll() is not None:
#                     processes.pop(i)
#                     break
#             else:
#                 time.sleep(2)
#                 continue
#             break

# # 等待所有进程结束
# for p, _ in processes:
#     p.wait()


# import os
# import subprocess
# import time

# num_gpus = 4
# exp_range = range(1, 9)
# gpu_pool = [i for i in range(num_gpus)]
# processes = []

# test_model={1 : 2,
#             2 : 1,
#             3 : 4,
#             4 : 3,
#             5 : 7,
#             6 : 5,
#             7 : 10,
#             8 : 6}

# for idx, exp in enumerate(exp_range):
#     gpu_id = gpu_pool[idx % num_gpus]
#     # if not os.path.exists(f'./datastore/exp{exp}'):
#     #     os.makedirs(f'./datastore/exp{exp}', exist_ok=True)
#     cmd = [
#         "python", "test.py",
#         "--test_data_path", f"../formatted_data/exp{exp}/test.txt",
#         "--guess_ans_path", f"./psm_guesses/successful_exp{exp}.txt",
#         "--guess_ans_path_nomix", f"./psm_guesses/successful_exp{exp}_nomix.txt",
#         "--model_path", f"./experiment/exp{test_model[exp]}/model.pth",
#         "--knn_datastore_path", f"./datastore/exp{test_model[exp]}",
#         "--gpu_id", str(gpu_id)
#     ]
#     # 启动进程
#     p = subprocess.Popen(cmd)
#     processes.append((p, gpu_id))
#     # 控制最多同时运行4个进程
#     if len(processes) == num_gpus:
#         # 等待任意一个进程结束
#         while True:
#             for i, (proc, gid) in enumerate(processes):
#                 if proc.poll() is not None:
#                     processes.pop(i)
#                     break
#             else:
#                 time.sleep(2)
#                 continue
#             break

# # 等待所有进程结束
# for p, _ in processes:
#     p.wait()

# import os
# import subprocess
# import time

# num_gpus = 4
# exp_range = range(1, 11)
# gpu_pool = [i for i in range(num_gpus)]
# processes = []

# for idx, exp in enumerate(exp_range):
#     gpu_id = gpu_pool[idx % num_gpus]
#     # if not os.path.exists(f'./datastore/exp{exp}'):
#     #     os.makedirs(f'./datastore/exp{exp}', exist_ok=True)
#     if exp in [1, 2, 3, 4, 9]:
#         cmd = [
#             "python", "main.py",
#             "--train_file", f"../formatted_data/pwdpair-tianya.txt",
#             "--save_path", f"./experiment/exp{exp}/local_to_global.pth",
#             "--epochs", "1",
#             "--load_ckpt", f"./experiment/exp{exp}/model.pth",
#             "--gpu_id", str(gpu_id)
#         ]
#     else:
#         cmd = [
#             "python", "main.py",
#             "--train_file", f"../formatted_data/pwdpair-rockyou.txt",
#             "--save_path", f"./experiment/exp{exp}/local_to_global.pth",
#             "--epochs", "1",
#             "--load_ckpt", f"./experiment/exp{exp}/model.pth",
#             "--gpu_id", str(gpu_id)
#         ]
#     # 启动进程
#     p = subprocess.Popen(cmd)
#     processes.append((p, gpu_id))
#     # 控制最多同时运行4个进程
#     if len(processes) == num_gpus:
#         # 等待任意一个进程结束
#         while True:
#             for i, (proc, gid) in enumerate(processes):
#                 if proc.poll() is not None:
#                     processes.pop(i)
#                     break
#             else:
#                 time.sleep(2)
#                 continue
#             break

# # 等待所有进程结束
# for p, _ in processes:
#     p.wait()

# for exp in range(1, 11):
#     if exp in [1, 2, 3, 4, 9]:
#         cmd=f'python gen_datastore.py --train_data_path ../formatted_data/filtered_target-exp{exp}.txt --train_data_path_2 ../formatted_data/pwdpair-tianya.txt --knn_train_data_path ../formatted_data/filtered_target_local_to_global-exp{exp}_knn.txt --model_path ./experiment/exp{exp}/local_to_global.pth --knn_datastore_path ./datastore/exp{exp}_local_to_global --gpu_id 0'
#         os.system(cmd)
#     else:
#         cmd=f'python gen_datastore.py --train_data_path ../formatted_data/filtered_target-exp{exp}.txt --train_data_path_2 ../formatted_data/pwdpair-rockyou.txt --knn_train_data_path ../formatted_data/filtered_target_local_to_global-exp{exp}_knn.txt --model_path ./experiment/exp{exp}/local_to_global.pth --knn_datastore_path ./datastore/exp{exp}_local_to_global --gpu_id 0'
#         os.system(cmd)

# for exp in range(1, 11):
#     if exp in [1, 2, 3, 4, 9]:
#         cmd=f'python gen_datastore.py --train_data_path ../formatted_data/filtered_target-exp{exp}.txt --train_data_path_2 ../formatted_data/pwdpair-tianya.txt --knn_train_data_path ../formatted_data/filtered_target_local_to_global-exp{exp}_knn.txt --model_path ./experiment/exp{exp}/local_to_global.pth --knn_datastore_path ./datastore/exp{exp}_local_to_global --gpu_id 0'
#         os.system(cmd)
#     else:
#         cmd=f'python gen_datastore.py --train_data_path ../formatted_data/filtered_target-exp{exp}.txt --train_data_path_2 ../formatted_data/pwdpair-rockyou.txt --knn_train_data_path ../formatted_data/filtered_target_local_to_global-exp{exp}_knn.txt --model_path ./experiment/exp{exp}/local_to_global.pth --knn_datastore_path ./datastore/exp{exp}_local_to_global --gpu_id 0'
#         os.system(cmd)

untar_test_model={
    "CSDN":1,
    "126":1,
    "Dodonew":9,
    "Linkedin":5,
    "Twitter":6,
    "000webhost":10
}

test_model={1 : 2,
            2 : 1,
            3 : 4,
            4 : 3,
            5 : 7,
            6 : 5,
            7 : 10,
            8 : 6}

# for exp in range(1,9):
#     model=test_model[exp]
#     if not os.path.exists(f"./targeted_result/exp{exp}"):
#         os.makedirs(f"./targeted_result/exp{exp}")
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/successful.txt --load_ckpt ./experiment/exp{model}/local_to_global.pth --output_path ./targeted_result/exp{exp}/successful_tar.txt --datastore_path ./datastore/exp{model}_local_to_global'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/successful_pop.txt --load_ckpt ./experiment/exp{model}/local_to_global.pth --output_path ./targeted_result/exp{exp}/successful_pop.txt --datastore_path ./datastore/exp{model}_local_to_global'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/unsuccessful.txt --load_ckpt ./experiment/exp{model}/local_to_global.pth --output_path ./targeted_result/exp{exp}/unsuccessful_tar.txt --datastore_path ./datastore/exp{model}_local_to_global'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/unsuccessful_pop.txt --load_ckpt ./experiment/exp{model}/local_to_global.pth --output_path ./targeted_result/exp{exp}/unsuccessful_pop.txt --datastore_path ./datastore/exp{model}_local_to_global'
#     os.system(cmd)

# for name in untar_test_model.keys():
#     exp=untar_test_model[name]
#     if not os.path.exists(f"./untargeted_result/{name}"):
#         os.makedirs(f"./untargeted_result/{name}")
#     cmd=f'python eval.py --test_file ../formatted_data/untargeted_test/test_pwdpair-{name}.txt --load_ckpt ./experiment/exp{exp}/local_to_global.pth --output_path ./untargeted_result/{name}/knnguess_res.txt --datastore_path ./datastore/exp{exp}_local_to_global'
#     os.system(cmd)


# for exp in range(1,11):
#     cmd=f"python main.py --train_file ../formatted_data/filtered_target-exp{exp}.txt --save_path ./experiment/exp{exp}/global.pth --epochs 1 --gpu_id 0"
#     os.system(cmd)

# for exp in range(1,11):
#     if exp in [1, 2, 3, 4, 9]:
#         cmd=f"python main.py --train_file ../formatted_data/pwdpair-tianya.txt --save_path ./experiment/exp{exp}/global.pth --epochs 1 --gpu_id 0"
#         os.system(cmd)
#     else:
#         cmd=f"python main.py --train_file ../formatted_data/pwdpair-rockyou.txt --save_path ./experiment/exp{exp}/global.pth --epochs 1 --gpu_id 0"
#         os.system(cmd)

# for exp in range(1,11):
#     cmd=f"python main.py --train_file ../formatted_data/filtered_target-exp{exp}.txt --save_path ./experiment/exp{exp}/global_to_local.pth --epochs 30 --gpu_id 0 --load_ckpt ./experiment/exp{exp}/global.pth"
#     os.system(cmd)

# for exp in range(1,11):
#     if exp in [1, 2, 3, 4, 9]:
#         cmd=f"python main.py --train_file ../formatted_data/pwdpair-tianya.txt --save_path ./experiment/exp{exp}/nofreeze.pth --epochs 1 --gpu_id 0 --load_ckpt ./experiment/exp{exp}/model.pth"
#         os.system(cmd)
#     else:
#         cmd=f"python main.py --train_file ../formatted_data/pwdpair-rockyou.txt --save_path ./experiment/exp{exp}/nofreeze.pth --epochs 1 --gpu_id 0 --load_ckpt ./experiment/exp{exp}/model.pth"
#         os.system(cmd)


# """
# Local
# """

# for exp in range(1, 11):
#     if exp in [1, 2, 3, 4, 9]:
#         cmd=f'python gen_datastore.py --train_data_path ../formatted_data/filtered_target-exp{exp}.txt --train_data_path_2 ../formatted_data/pwdpair-tianya.txt --knn_train_data_path ../formatted_data/filtered_target_local_to_global-exp{exp}_knn.txt --model_path ./experiment/exp{exp}/model.pth --knn_datastore_path ./datastore/exp{exp} --gpu_id 0'
#         os.system(cmd)
#     else:
#         cmd=f'python gen_datastore.py --train_data_path ../formatted_data/filtered_target-exp{exp}.txt --train_data_path_2 ../formatted_data/pwdpair-rockyou.txt --knn_train_data_path ../formatted_data/filtered_target_local_to_global-exp{exp}_knn.txt --model_path ./experiment/exp{exp}/model.pth --knn_datastore_path ./datastore/exp{exp} --gpu_id 0'
#         os.system(cmd)

# for exp in range(1,9):
#     model=test_model[exp]
#     if not os.path.exists(f"./targeted_result/exp{exp}"):
#         os.makedirs(f"./targeted_result/exp{exp}")
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/successful.txt --load_ckpt ./experiment/exp{model}/model.pth --output_path ./targeted_result/exp{exp}/successful_tar_local.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/successful_pop.txt --load_ckpt ./experiment/exp{model}/model.pth --output_path ./targeted_result/exp{exp}/successful_pop_local.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/unsuccessful.txt --load_ckpt ./experiment/exp{model}/model.pth --output_path ./targeted_result/exp{exp}/unsuccessful_tar_local.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/unsuccessful_pop.txt --load_ckpt ./experiment/exp{model}/model.pth --output_path ./targeted_result/exp{exp}/unsuccessful_pop_local.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)

# for name in untar_test_model.keys():
#     exp=untar_test_model[name]
#     if not os.path.exists(f"./untargeted_result/{name}"):
#         os.makedirs(f"./untargeted_result/{name}")
#     cmd=f'python eval.py --test_file ../formatted_data/untargeted_test/test_pwdpair-{name}.txt --load_ckpt ./experiment/exp{exp}/model.pth --output_path ./untargeted_result/{name}/knnguess_local_res.txt --datastore_path ./datastore/exp{exp}'
#     os.system(cmd)


# """
# Global
# """

# for exp in range(1, 11):
#     if exp in [1, 2, 3, 4, 9]:
#         cmd=f'python gen_datastore.py --train_data_path ../formatted_data/filtered_target-exp{exp}.txt --train_data_path_2 ../formatted_data/pwdpair-tianya.txt --knn_train_data_path ../formatted_data/filtered_target_local_to_global-exp{exp}_knn.txt --model_path ./experiment/exp{exp}/global.pth --knn_datastore_path ./datastore/exp{exp} --gpu_id 0'
#         os.system(cmd)
#     else:
#         cmd=f'python gen_datastore.py --train_data_path ../formatted_data/filtered_target-exp{exp}.txt --train_data_path_2 ../formatted_data/pwdpair-rockyou.txt --knn_train_data_path ../formatted_data/filtered_target_local_to_global-exp{exp}_knn.txt --model_path ./experiment/exp{exp}/global.pth --knn_datastore_path ./datastore/exp{exp} --gpu_id 0'
#         os.system(cmd)

# for exp in range(1,9):
#     model=test_model[exp]
#     if not os.path.exists(f"./targeted_result/exp{exp}"):
#         os.makedirs(f"./targeted_result/exp{exp}")
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/successful.txt --load_ckpt ./experiment/exp{model}/global.pth --output_path ./targeted_result/exp{exp}/successful_tar_global.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/successful_pop.txt --load_ckpt ./experiment/exp{model}/global.pth --output_path ./targeted_result/exp{exp}/successful_pop_global.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/unsuccessful.txt --load_ckpt ./experiment/exp{model}/global.pth --output_path ./targeted_result/exp{exp}/unsuccessful_tar_global.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/unsuccessful_pop.txt --load_ckpt ./experiment/exp{model}/global.pth --output_path ./targeted_result/exp{exp}/unsuccessful_pop_global.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)

# for name in untar_test_model.keys():
#     exp=untar_test_model[name]
#     if not os.path.exists(f"./untargeted_result/{name}"):
#         os.makedirs(f"./untargeted_result/{name}")
#     cmd=f'python eval.py --test_file ../formatted_data/untargeted_test/test_pwdpair-{name}.txt --load_ckpt ./experiment/exp{exp}/global.pth --output_path ./untargeted_result/{name}/knnguess_global_res.txt --datastore_path ./datastore/exp{exp}'
#     os.system(cmd)


# """
# Global to local
# """

# for exp in range(1, 11):
#     if exp in [1, 2, 3, 4, 9]:
#         cmd=f'python gen_datastore.py --train_data_path ../formatted_data/filtered_target-exp{exp}.txt --train_data_path_2 ../formatted_data/pwdpair-tianya.txt --knn_train_data_path ../formatted_data/filtered_target_local_to_global-exp{exp}_knn.txt --model_path ./experiment/exp{exp}/global_to_local.pth --knn_datastore_path ./datastore/exp{exp} --gpu_id 0'
#         os.system(cmd)
#     else:
#         cmd=f'python gen_datastore.py --train_data_path ../formatted_data/filtered_target-exp{exp}.txt --train_data_path_2 ../formatted_data/pwdpair-rockyou.txt --knn_train_data_path ../formatted_data/filtered_target_local_to_global-exp{exp}_knn.txt --model_path ./experiment/exp{exp}/global_to_local.pth --knn_datastore_path ./datastore/exp{exp} --gpu_id 0'
#         os.system(cmd)

# for exp in range(1,9):
#     model=test_model[exp]
#     if not os.path.exists(f"./targeted_result/exp{exp}"):
#         os.makedirs(f"./targeted_result/exp{exp}")
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/successful.txt --load_ckpt ./experiment/exp{model}/global_to_local.pth --output_path ./targeted_result/exp{exp}/successful_tar_global_to_local.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/successful_pop.txt --load_ckpt ./experiment/exp{model}/global_to_local.pth --output_path ./targeted_result/exp{exp}/successful_pop_global_to_local.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/unsuccessful.txt --load_ckpt ./experiment/exp{model}/global_to_local.pth --output_path ./targeted_result/exp{exp}/unsuccessful_tar_global_to_local.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/unsuccessful_pop.txt --load_ckpt ./experiment/exp{model}/global_to_local.pth --output_path ./targeted_result/exp{exp}/unsuccessful_pop_global_to_local.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)

# for name in untar_test_model.keys():
#     exp=untar_test_model[name]
#     if not os.path.exists(f"./untargeted_result/{name}"):
#         os.makedirs(f"./untargeted_result/{name}")
#     cmd=f'python eval.py --test_file ../formatted_data/untargeted_test/test_pwdpair-{name}.txt --load_ckpt ./experiment/exp{exp}/global_to_local.pth --output_path ./untargeted_result/{name}/knnguess_global_to_local_res.txt --datastore_path ./datastore/exp{exp}'
#     os.system(cmd)


# """
# Nofreeze
# """

# for exp in range(1, 11):
#     if exp in [1, 2, 3, 4, 9]:
#         cmd=f'python gen_datastore.py --train_data_path ../formatted_data/filtered_target-exp{exp}.txt --train_data_path_2 ../formatted_data/pwdpair-tianya.txt --knn_train_data_path ../formatted_data/filtered_target_local_to_global-exp{exp}_knn.txt --model_path ./experiment/exp{exp}/nofreeze.pth --knn_datastore_path ./datastore/exp{exp} --gpu_id 0'
#         os.system(cmd)
#     else:
#         cmd=f'python gen_datastore.py --train_data_path ../formatted_data/filtered_target-exp{exp}.txt --train_data_path_2 ../formatted_data/pwdpair-rockyou.txt --knn_train_data_path ../formatted_data/filtered_target_local_to_global-exp{exp}_knn.txt --model_path ./experiment/exp{exp}/nofreeze.pth --knn_datastore_path ./datastore/exp{exp} --gpu_id 0'
#         os.system(cmd)

# for exp in range(1,9):
#     model=test_model[exp]
#     if not os.path.exists(f"./targeted_result/exp{exp}"):
#         os.makedirs(f"./targeted_result/exp{exp}")
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/successful.txt --load_ckpt ./experiment/exp{model}/nofreeze.pth --output_path ./targeted_result/exp{exp}/successful_tar_nofreeze.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/successful_pop.txt --load_ckpt ./experiment/exp{model}/nofreeze.pth --output_path ./targeted_result/exp{exp}/successful_pop_nofreeze.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/unsuccessful.txt --load_ckpt ./experiment/exp{model}/nofreeze.pth --output_path ./targeted_result/exp{exp}/unsuccessful_tar_nofreeze.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)
#     cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp{exp}/unsuccessful_pop.txt --load_ckpt ./experiment/exp{model}/nofreeze.pth --output_path ./targeted_result/exp{exp}/unsuccessful_pop_nofreeze.txt --datastore_path ./datastore/exp{model}'
#     os.system(cmd)

# for name in untar_test_model.keys():
#     exp=untar_test_model[name]
#     if not os.path.exists(f"./untargeted_result/{name}"):
#         os.makedirs(f"./untargeted_result/{name}")
#     cmd=f'python eval.py --test_file ../formatted_data/untargeted_test/test_pwdpair-{name}.txt --load_ckpt ./experiment/exp{exp}/nofreeze.pth --output_path ./untargeted_result/{name}/knnguess_nofreeze_res.txt --datastore_path ./datastore/exp{exp}'
#     os.system(cmd)
# cmd=f'python gen_datastore.py --train_data_path ../formatted_data/filtered_target-exp1.txt --knn_train_data_path ../formatted_data/time_test.txt --model_path ./experiment/exp1/model.pth --knn_datastore_path ./datastore/exp1 --gpu_id 0'
# os.system(cmd)

# cmd='python test.py --test_data_path ../formatted_data/exp1/test.txt --guess_ans_path ./guess_ans.txt --guess_ans_path_nomix ./guess_ans_nomix.txt --model_path ./experiment/exp1/model.pth --knn_datastore_path ./datastore/exp1'
# os.system(cmd)

cmd=f'python eval.py --test_file ../formatted_data/targeted_dataset_raid/exp1/unsuccessful.txt --load_ckpt ./experiment/exp1/global_to_local.pth --output_path ./time_test.txt --datastore_path ./datastore/exp1'
os.system(cmd)