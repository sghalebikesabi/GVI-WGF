# langevin-deep-ensembles

Run `src/main.py` to start experiment.
Configs are saved in the folder `configs`.

Model updates are defined in `src/{FRAMEWORK}_updates.py`.

Example run command on XXXX cluster
'''
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python src/main.py training.update_rule.args.method=repulsive training.update_rule.distance_on=params
'''

## Reproduce final results

```bash
# optimal measure plot
python src/main.py -m +experiment=optimal_q ++training.update_rule.args.method=standard,langevin,repulsivse
python src/main.py +experiment=optimal_q ++training.update_rule.args.method=langevin
python src/main.py +experiment=optimal_q ++training.update_rule.args.method=repulsivse

# gmm plot
python src/main.py -m +experiment=toy_gmm ++training.update_rule.args.method=langevin ++training.update_rule.args.langevin_reg_param=0.2,0.3,0.4

# small uci standard
python src/main.py -m data.train_args.name=boston,concrete,energy,KIN8NM,naval,power,wine,yacht seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19‚ wandb_args.project='langevin-final-uci' data.train_args.split=train_valid data.eval_args.split=test

# protein standard
python src/main.py -m data.train_args.name=protein seed=0,1,2,3,4 model.model_args.hidden_nodes=100 wandb_args.project=langevin-final-uci data.train_args.split=train_valid data.eval_args.split=test

# small uci langevin
python src/main.py -m data.train_args.name=boston,concrete,energy,KIN8NM,naval,power,wine,yacht seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19‚ wandb_args.project='langevin-final-uci' data.train_args.split=train_valid data.eval_args.split=test training.update_rule.args.lr=5e-1 training.update_rule.args.langevin_reg_param=1e-4

# protein langevin
python src/main.py -m data.train_args.name=protein seed=0,1,2,3,4 model.model_args.hidden_nodes=100 wandb_args.project=langevin-final-uci data.train_args.split=train_valid data.eval_args.split=test training.update_rule.args.lr=5e-1 training.update_rule.args.langevin_reg_param=1e-4


# debug 
python src/main.py -m data.train_args.name=protein seed=${SLURM_ARRAY_TASK_ID} model.model_args.hidden_nodes=100 data.train_args.split=train_valid data.eval_args.split=test
```

## Sweeps

Run
'''
wandb sweep sweeps/lr.yaml --project=langevin
wandb agent XXXX/langevin/SWEEP_ID
'''
