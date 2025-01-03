path="--dataset_name=cifar10 --train_log=train_log --train_img=train_img --zca_path=data/zca --data_path=~/tensorflow_datasets --save_image=True"
exp="--learn_label=True --random_seed=0"
arch="--arch=conv --width=128 --depth=3 --normalization=batch"
hyper="--max_online_updates=100 --num_nn_state=10 --num_train_steps=500000"
ckpt="--ckpt_dir=train_log/cifar10/step500K_num100/conv_w128_d3_batch_llTrue/state10_reset100 --ckpt_name=best_ckpt --res_dir=dd/cifar100 --num_eval=5"
CUDA_VISIBLE_DEVICES=0 python -m script.distill $path $exp $arch $hyper --num_prototypes_per_class=10 --epsilon=1.0 --topk=0.8
CUDA_VISIBLE_DEVICES=0 python -m script.eval $ckpt $path $arch
