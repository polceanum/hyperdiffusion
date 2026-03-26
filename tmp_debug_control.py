from pathlib import Path
import torch
from hyperdiffusion.experiment import Experiment, ExperimentConfig
from hyperdiffusion.models import functional_target_network
from hyperdiffusion.tasks import CONTROL_FAMILIES

config = ExperimentConfig(
    task_type='control',
    families=['nonlinear_control'],
    eval_families=['nonlinear_control'],
    train_steps_stage1=40,
    train_steps_stage2=40,
    eval_batches=5,
    batch_size=16,
    support_size=16,
    query_size=32,
    visualization_count=2,
    visualization_grid_size=40,
    device='cpu',
    seed=42,
)
exp = Experiment(config=config, output_dir=Path('runs/tmp'))
for _ in range(10):
    exp.stage1_step(exp.sample_batch())
for _ in range(10):
    exp.stage2_step(exp.sample_batch())

batch = exp.to_device(exp.sample_batch(batch_size=1, family_names=['nonlinear_control']))
context, latent = exp.system.encode(batch.support_x, batch.support_y)
params = exp.system.decode(latent[0:1], context[0:1])

state = batch.support_x[0,0]
act = functional_target_network(state.view(1,1,-1), params).squeeze().item()

family = CONTROL_FAMILIES['nonlinear_control']

def model_policy(s):
    with torch.no_grad():
        return functional_target_network(s.view(1,1,-1), params).squeeze()

def zero_policy(s):
    return torch.zeros(1, dtype=torch.float32)

lr = family.rollout(model_policy, state.clone(), num_steps=10, dt=0.05)
zr = family.rollout(zero_policy, state.clone(), num_steps=10, dt=0.05)

print('state', state)
print('model action', act)
print('learned first rewards', lr['rewards'][:5])
print('zero first rewards', zr['rewards'][:5])
print('diff cum', lr['cum_rewards'][-1] - zr['cum_rewards'][-1])
