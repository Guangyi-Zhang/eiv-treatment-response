#!/bin/bash
#SBATCH --time=0-1:00:00
#SBATCH --mem-per-cpu=6000    # 500MB of memory
#SBATCH -c 2 --constraint="hsw|ivb|bdw"

#xcertain='--x_uncertainty'
#other='--skipsample'
#other='--lengthscale=100'
testrun=1
other='--hier_sd_h=1 --hier_sd_ls=3'
#sd=0.05

# BEST
#task='--task=v12302145-nuts-uspgp-pol0-ind-p13-testday-days1-nostd-featlog-notime'
#task='--task=v12302145-nuts-uspgp-pol0-hier-p13-testday-days1-nostd-featlog-notime'
task='--task=v12302145-nuts-uspgp-pol0-hier-p13-testday-days1-nostd-featlog'
#task="--task=v12302145-nuts-uspgp-pol0-hier-p13-testday-days2-nostd-featlog-cov${sd}"


if [[ $task == *"p13"* ]]; then
    ids='--ids=0,1,2,3,4,5,6,7,8,9,10,11,12,13'
elif [[ $task == *"p3"* ]]; then
    ids='--ids=0,1,2'
else
    ids='--ids=0'
fi

if [[ $task == *"ind"* ]]; then
    m='--model=GPTrendIndividualModel'
else
    m='--model=GPTrendHierModel'
fi

if [[ $testrun == 1 ]]; then
    sample='--step=HMC --n_sample=1 --n_tune=1 --target_accept=0.8 --nppc=1 --fast'
else
    if [[ $task == *"metro"* ]]; then
        sample='--step=Metropolis --n_sample=1000 --n_tune=1000 --target_accept=0.9 --nppc=100'
    elif [[ $task == *"nuts1k"* ]]; then
        sample='--step=NUTS --n_sample=1000 --n_tune=1000 --nppc=100'
    elif [[ $task == *"nuts"* ]]; then
        sample='--step=NUTS --n_sample=500 --n_tune=500 --nppc=100'
    else
        sample='--step=HMC --n_sample=500 --n_tune=500 --target_accept=0.8 --nppc=100'
    fi
fi

if [[ $task == *"days1"* ]]; then
    days=1
elif [[ $task == *"days2"* ]]; then
    days=2
else
    days=3
fi

if [[ $task == *"testday"* ]]; then
    testset="--days=$days --testset=day"
else
    #testset="--days=$days --testset=no"
    echo 'err: not impl'
    exit 1
fi

if [[ $task == *"notime"* ]]; then
    :
else
    certain='--time_uncertainty'
fi

if [[ $task == *"nostd"* ]]; then
    std='--nostd'
fi

if [[ $task == *"nointercept"* ]]; then
    intercept='--nointercept'
fi

if [[ $task == *"spgp"* ]]; then
    sparse="--n_inducing_points=${days}0 --sparse"
    if [[ $task == *"pol3"* ]]; then
        sparse="$sparse --inducing_policy=policy3"
    else
        sparse="$sparse --inducing_policy=policy0"
    fi
else
    :
fi

feature='--feature=x'
if [[ $task == *"featlog"* ]]; then
    feature="${feature}log"
fi
if [[ $task == *"featsqrt"* ]]; then
    feature="${feature}sqrt"
fi
if [[ $task == *"featpoly2"* ]]; then
    feature="${feature}poly2"
fi

if [[ $task == *"-cov"* ]]; then
    covariate="--covariate --covariate_sd=$sd"
fi


#module load anaconda3 # needed in Aalto systems
export MKL_NUM_THREADS=1
python run.py $task $m $testset $sparse $sample $certain $xcertain $feature $std $covariate $ids $other $data $intercept
