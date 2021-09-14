import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model_AC import set_decode_type
from utils.log_utils import log_values
from utils import move_to
import numpy as np


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, problem, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, problem, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, problem, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            # make state
            x = move_to(bat, opts.device)
            state = problem.make_state(x)
            mask = state.get_mask()
            mask_first = mask

            # step count
            step_i = 0
            sequences = []
            while not (state.all_finished() and step_i < opts.num_env_steps):
                log_p, value = model(x, mask, mask_first, state, step_i)

                # Select the indices of the next nodes in the sequences, result (batch_size) long
                selected = model._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension

                next_state, r = state.update(selected)
                next_mask = next_state.get_mask()
                if not problem.NAME == 'cvrp' and step_i == 0:
                    mask_first = mask

                state = next_state
                mask = next_mask

                step_i += 1
                sequences.append(selected)
            cost, mask = problem.get_costs(x, torch.stack(sequences, 1))
        return cost.data.cpu()
    # for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar):
    #     print(bat)

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)



def train_AC(problem, model, opts, epoch, optimizer, lr_scheduler, val_dataset, tb_logger):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    training_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, filename=None,
                                            distribution=opts.data_distribution)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    # num_updates = int(opts.num_env_steps) // opts.num_steps // opts.num_processes

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(
            model,
            optimizer,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts,
            problem
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, problem, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts,
        problem
):
    x = move_to(batch, opts.device)

    episode_buffer = []

    # make state
    state = problem.make_state(x)
    mask = state.get_mask()
    mask_first = mask

    # step count
    step_i = 0
    while not (state.all_finished() and step_i < opts.num_env_steps):
        log_p, value = model(x, mask, mask_first, state, step_i)

        # Select the indices of the next nodes in the sequences, result (batch_size) long
        selected = model._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
        next_state, r = state.update(selected)
        # print('selected: ', selected)
        #
        next_mask = next_state.get_mask()
        if not problem.NAME == 'cvrp' and step_i == 0:
            mask_first = mask

        # insert data to buffer [s, a, r, done, s']
        episode_buffer.append([x, state, mask, step_i, selected, log_p.exp()[:, 0, :], r, next_state, next_mask, mask_first])

        state = next_state
        mask = next_mask

        step_i += 1

    # update net according to experience
    data_length = len(episode_buffer)
    # data = replay_buffer.get_trajectory()
    # # print('data: ', data['actions'])
    # agent.learn(data, data_length, replay_buffer)

    inputs = []
    states = []
    masks = []
    mask_firsts = []
    step_is = []
    acts = []
    rewards = []
    old_act_ps = []
    for i in range(data_length):
        inputs.append(episode_buffer[i][0])  # x, state, mask, step_i
        states.append(episode_buffer[i][1])
        masks.append(episode_buffer[i][2])
        mask_firsts.append(episode_buffer[i][9])
        step_is.append(episode_buffer[i][3])
        acts.append(np.array(episode_buffer[i][4]))  # graph_size(step) x batch_size
        rewards.append(np.array(episode_buffer[i][6].squeeze(-1)))  # 列是batch_size，行是graph_size
        old_act_ps.append(np.array(episode_buffer[i][5].detach().cpu()))  # graph_size(step) x batch_size x graph_size
    # print('act: ', acts)

    # 计算reward-to-go
    R = torch.zeros(1, opts.batch_size).to(opts.device)
    Gt = torch.Tensor([]).to(opts.device)
    rewards = torch.tensor(rewards, dtype=torch.float).to(opts.device)
    acts = torch.tensor(acts, dtype=torch.int64).to(opts.device)
    old_act_ps = torch.tensor(old_act_ps, dtype=torch.float).to(opts.device)
    for jj in range(rewards.shape[0]-1, -1, -1):  # rewards.shape[0]是graph_size
        R = rewards[jj, :][None, :] + opts.gamma * R
        Gt = torch.cat((R, Gt), 0)
    Gt = Gt.t()  # row is batch_size, col is graph_size
    # print("Gt: ", Gt, Gt.shape)

    for _ in range(opts.update_freq):
        V = torch.Tensor([]).to(opts.device)
        prob = torch.Tensor([]).to(opts.device)
        for jj in range(data_length):
            log_p, value = model(inputs[jj], masks[jj], mask_firsts[jj], states[jj], step_is[jj])

            value = value.squeeze(-1)
            V = torch.cat((V, value), -1)  # row is batch_size, col is graph_size

            # print('log_p: ', log_p.shape)
            prob = torch.cat((prob, log_p.exp()), 1)  # batch_size x graph_size(step) x graph_size(num)

        delta = Gt - V  # batch_size x graph_size
        # print(delta)
        advantage = delta.detach()
        action_prob = prob.permute(1, 0, 2).gather(-1, acts.unsqueeze(-1))  # ???取值是否对  graph_size x batch_size x 1
        action_prob = torch.transpose(action_prob, 0, 1)
        old_act_prob = old_act_ps.gather(-1, acts.unsqueeze(-1)).permute(1, 0, 2)
        ratio = (action_prob / old_act_prob).squeeze(-1)  # batch_size x graph_size(step)
        # print('ratio: ', ratio, ratio.shape)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - opts.clip_param, 1 + opts.clip_param) * advantage
        # print(surr1.shape, surr2.shape)

        action_loss = -torch.min(surr1, surr2).mean()
        # print(action_loss)

        value_loss = F.mse_loss(Gt, V)
        # print(value_loss)

        entropy = (prob * torch.log(torch.clamp(prob, 1e-10, 1.0))).mean()
        loss = action_loss + value_loss * 0.5 - entropy * 0.01

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), opts.max_grad_norm)
        optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(loss, action_loss, value_loss, entropy, epoch, batch_id, step,
                   tb_logger, opts)













