# # SMD
# python main.py --anormly_ratio 0.5 --num_epochs 100 --batch_size 256 --mode train --dataset SMD --data_path ./data/SMD/SMD/ --input_c 38 --output_c 38 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anormly_ratio 0.5 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset SMD  --data_path ./data/SMD/SMD/  --input_c 38 --output_c 38 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 256  --mode test --dataset SMD  --data_path ./data/SMD/SMD/  --input_c 38 --output_c 38 --n_memory 10 --memory_initial False --phase_type test


# # SMAP
# python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 256 --mode train --dataset SMAP --data_path ./data/SMAP/SMAP/ --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anormly_ratio 1.0 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset SMAP  --data_path ./data/SMAP/SMAP/  --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 256  --mode test --dataset SMAP  --data_path ./data/SMAP/SMAP/  --input_c 25 --output_c 25 --n_memory 10 --memory_initial False --phase_type test

# # PSM
# python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 256 --mode train --dataset PSM --data_path ./data/PSM/PSM/ --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anormly_ratio 1.0 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset PSM  --data_path ./data/PSM/PSM/  --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 256  --mode test --dataset PSM  --data_path ./data/PSM/PSM/  --input_c 25 --output_c 25 --n_memory 10 --memory_initial False --phase_type test

# # MSL

# python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 256 --mode train --dataset MSL --data_path ./data/MSL/MSL/ --input_c 55 --output_c 55 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anormly_ratio 1.0 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset MSL  --data_path ./data/MSL/MSL/  --input_c 55 --output_c 55 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 256  --mode test --dataset MSL  --data_path ./data/MSL/MSL/  --input_c 55 --output_c 55 --n_memory 10 --memory_initial False --phase_type test


# # SWaT

# python main.py --anormly_ratio 0.1 --num_epochs 100 --batch_size 256 --mode train --dataset SWaT --data_path ./data/SWaT/SWaT/ --input_c 51 --output_c 51 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anormly_ratio 0.1 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset SWaT  --data_path ./data/SWaT/SWaT/  --input_c 51 --output_c 51 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anormly_ratio 0.1 --num_epochs 100   --batch_size 256  --mode test --dataset SWaT  --data_path ./data/SWaT/SWaT/  --input_c 51 --output_c 51 --n_memory 10 --memory_initial False --phase_type test
file_path="./data_factory/SIM/l9b1/l9b1_test_label"
folder_path = "./data_factory/SIM/"
curr_path = $(pwd)
if [[ -e $file_path ]]; then
    echo "File exists"
else
    cd "$folder_path" || exit
    python make_np_train.py
    cd "$curr_path" || exit
fi
python main.py --anormly_ratio 10 --num_epochs 100 --batch_size 256 --mode train --dataset b9k1 --data_path ./data_factory/SIM/b9k1/ --input_c 4 --output_c 4 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

python main.py --anormly_ratio 10 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset b9k1  --data_path ./data_factory/SIM/b9k1/  --input_c 4 --output_c 4 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

python main.py --anormly_ratio 10 --num_epochs 10   --batch_size 256  --mode test --dataset b9k1  --data_path ./data_factory/SIM/b9k1/  --input_c 4 --output_c 4 --n_memory 10 --memory_initial False --phase_type test

python main.py --anormly_ratio 10 --num_epochs 100 --batch_size 256 --mode train --dataset k9l1 --data_path ./data_factory/SIM/k9l1/ --input_c 4 --output_c 4 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

python main.py --anormly_ratio 10 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset k9l1  --data_path ./data_factory/SIM/k9l1/  --input_c 4 --output_c 4 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

python main.py --anormly_ratio 10 --num_epochs 10   --batch_size 256  --mode test --dataset k9l1  --data_path ./data_factory/SIM/k9l1/  --input_c 4 --output_c 4 --n_memory 10 --memory_initial False --phase_type test

python main.py --anormly_ratio 10 --num_epochs 100 --batch_size 256 --mode train --dataset l9b1 --data_path ./data_factory/SIM/l9b1/ --input_c 4 --output_c 4 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

python main.py --anormly_ratio 10 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset l9b1  --data_path ./data_factory/SIM/l9b1/  --input_c 4 --output_c 4 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

python main.py --anormly_ratio 10 --num_epochs 10   --batch_size 256  --mode test --dataset l9b1  --data_path ./data_factory/SIM/l9b1/  --input_c 4 --output_c 4 --n_memory 10 --memory_initial False --phase_type test