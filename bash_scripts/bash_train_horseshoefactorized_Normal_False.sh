#!/bin/bash
until CUDA_VISIBLE_DEVICES=2 python3 train.py -mts "{'stochastic':True,'stochastic_f_pass':10,'distr_type':'Normal','discrete_continuous':False,'precip_threshold':0.5,'var_model_type':'dropout' }" -mn "DeepSD" -gidx "[0]" -dd "/media/Data3/akanni/Vandal/Data"; do
    echo " task.py crashed with exit code $?.  Restarting.." >&2
    sleep 10
done &