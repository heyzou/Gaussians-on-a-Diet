
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bicycle_ours python train.py -s ./Dataset/bicycle -i images_4 --eval -m ./output/bicycle/wo_pruning --target_num "800000" --our_door&
wait;
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=flowers_ours python train.py -s ./Dataset/flowers -i images_4 --eval -m ./output/flowers/wo_pruning --target_num "570000" --our_door&
wait;
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=garden_ours python train.py -s ./Dataset/garden -i images_4 --eval -m ./output/garden/wo_pruning --target_num "1900000"  --our_door&
wait;
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=stump_ours python train.py -s ./Dataset/stump -i images_4 --eval -m ./output/stump/wo_pruning --target_num "480000" --our_door&
wait;
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=treehill_ours python train.py -s ./Dataset/treehill -i images_4 --eval -m ./output/treehill/wo_pruning --target_num "780_000"  --our_door&
wait;
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=room_ours python train.py -s ./Dataset/room -i images_2 --eval -m ./output/room/alter_compensate --target_num "225118"&
wait;
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=counter_ours python train.py -s ./Dataset/counter -i images_2 --eval -m ./output/counter/wo_pruning --target_num "310000"&
wait;
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=kitchen_ours python train.py -s ./Dataset/kitchen -i images_2 --eval -m ./output/kitchen/wo_pruning --target_num "480000"&
wait;
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=playroom_ours python train.py -s ./Dataset/playroom --eval -m ./output/playroom/wo_pruning --target_num "184847"&

CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bonsai_ours python train.py -s ./Dataset/bonsai -i images_2 --eval -m ./output/bonsai/wo_pruning --target_num "410000"&

CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=truck_ours python train.py -s ./Dataset/truck --eval -m ./output/truck/wo_pruning --target_num "270000"&
wait;
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=train_ours python train.py -s ./Dataset/train --eval -m ./output/train/wo_pruning --target_num "365000"&

CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=drjohnson_ours python train.py -s ./Dataset/drjohnson --eval -m ./output/drjohnson/wo_pruning --target_num "403919"&
wait;
