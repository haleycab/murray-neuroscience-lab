import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import pandas as pd

data = pd.read_csv('normI_forben.csv')


def get_data(cell, speed, freq, time_warped=True, time_shift=0):
    '''
    Get Moneeza's data for a particular cell, speed, and swim frequency.

    Parameters
    --
    cell : Can be 'V2aD' or 'V2aB'.

    speed : Can be 'fast' or 'slow'.

    freq : Can be one of ['15', '25', '35', '45+'].

    time_warped : If True, the traces all have the same number of points, as in
        Moneeza's original data. If False, the traces are subsampled in proportion
        to the swim frequency.

    time_shift : Number of phase steps (where 100 phase steps equals one period)
        by which to shift the pseudo_output, where negative values advance the
        trace and positive values delay it.

    Returns
    --
    phases : 1D array of phases in the swim cycle (-0.5 < phase < 0.5).

    e_current, i_current : 1D arrays of excitatory and inhibitory input currents.

    pseudo_output : The "pseudo output" is the rectified difference between E and I
        input currents.
    '''
    phases = data['phase_values_for_norm_current'][(data['cell_type'] == cell)
                                                  & (data['speed_class'] == speed)
                                                  & (data['vr_freq_cat'] == freq)
                                                  & (data['current_type'] == 'excitatory')]
    e_current = data['average_current'][(data['cell_type'] == cell)
                                      & (data['speed_class'] == speed)
                                      & (data['vr_freq_cat'] == freq)
                                      & (data['current_type'] == 'excitatory')]
    i_current = data['average_current'][(data['cell_type'] == cell)
                                      & (data['speed_class'] == speed)
                                      & (data['vr_freq_cat'] == freq)
                                      & (data['current_type'] == 'inhibitory')]
    phases, e_current, i_current = np.array(phases), np.array(e_current), np.array(i_current)
    pseudo_output = 0.5 * (1 + np.sign(e_current - i_current)) * (e_current - i_current)
    pseudo_output = np.roll(pseudo_output, time_shift)

    n_timesteps = len(phases)
    min_freq = 15
    freq_int = int(freq[:2])

    if time_warped == False:
        phases_interp = np.linspace(-0.5, 0.5, int(n_timesteps * 15 / freq_int))
        e_current_interp = np.interp(phases_interp, phases, e_current)
        i_current_interp = np.interp(phases_interp, phases, i_current)
        pseudo_output_interp = np.interp(phases_interp, phases, pseudo_output)
        return np.array(phases_interp), np.array(e_current_interp), np.array(i_current_interp), np.array(pseudo_output_interp)
    else:
        return np.array(phases), np.array(e_current), np.array(i_current), np.array(pseudo_output)


def get_e_currents(freq):
    '''
    Get the E input currents for a given frequency. Returns a 2D array of size n_timesteps by 4.
    '''
    f = int(freq[:2])
    fast_recruit_factor = 1 #(f - 15) / (45 - 15)
    slow_recruit_factor = 1 #(45 - f) / (45 - 15)
    _, e_currents, _, _ =  get_data('V2aD', 'fast', freq, time_warped=False)
    e_currents *= fast_recruit_factor
    _, foo, _, _ = get_data('V2aD', 'slow', freq, time_warped=False)
    e_currents = np.vstack((e_currents, slow_recruit_factor * foo))
    _, foo, _, _ = get_data('V2aB', 'fast', freq, time_warped=False)
    e_currents = np.vstack((e_currents, fast_recruit_factor * foo))
    _, foo, _, _ = get_data('V2aB', 'slow', freq, time_warped=False)
    e_currents = np.vstack((e_currents, slow_recruit_factor * foo))

    return e_currents.T


def get_i_currents(freq):
    '''
    Get the I input currents for a given frequency. Returns a 2D array of size n_timesteps by 4.
    '''
    f = int(freq[:2])
    fast_recruit_factor = 1 #(f - 15) / (45 - 15)
    slow_recruit_factor = 1 #(45 - f) / (45 - 15)
    _, _, i_currents, _ = get_data('V2aD', 'fast', freq, time_warped=False)
    i_currents *= fast_recruit_factor
    _, _, foo, _ = get_data('V2aD', 'slow', freq, time_warped=False)
    i_currents = np.vstack((i_currents, slow_recruit_factor * foo))
    _, _, foo, _ = get_data('V2aB', 'fast', freq, time_warped=False)
    i_currents = np.vstack((i_currents, fast_recruit_factor * foo))
    _, _, foo, _ = get_data('V2aB', 'slow', freq, time_warped=False)
    i_currents = np.vstack((i_currents, slow_recruit_factor * foo))

    return i_currents.T


def get_pseudo_outputs(freq, n_lags=1, tonic_only=False):
    '''
    Get the pseudo_outputs for a given frequency. Returns a 2D array of size 8*n_lags+2 by n_timesteps.
    The inputs are ordered as follows:

    V2aD-f, lag 1, phase=0
    V2aD-s, lag 1, phase=0
    V2aB-f, lag 1, phase=0
    V2aB-s, lag 1, phase=0
    V2aD-f, lag 2, phase=0
    ...
    V2aB-s, lag n_lags, phase=0
    V2aD-f, lag 1, phase=pi
    ...
    V2aB-s, lag n_lags, phase=pi
    tonic f
    tonic s

    '''
    f = int(freq[:2])
    fast_recruit_factor_D = (f - 15) / (45 - 15)
    slow_recruit_factor_D = (45 - f) / (45 - 15)
    fast_recruit_factor_B = 0.2 + 0.8 * (f - 15) / (45 - 15)
    slow_recruit_factor_B = 0.2 + 0.8 * (45 - f) / (45 - 15)

    for n in range(n_lags):
        if n==0:
            _, _, _, pseudo_outputs = get_data('V2aD', 'fast', freq, time_warped=False, time_shift=-3*n_lags)
            pseudo_outputs *= fast_recruit_factor_D
        else:
            _, _, _, foo = get_data('V2aD', 'fast', freq, time_warped=False, time_shift=-3*n_lags)
            pseudo_outputs = np.vstack((pseudo_outputs, fast_recruit_factor_D * foo))

        _, _, _, foo = get_data('V2aD', 'slow', freq, time_warped=False, time_shift=-3*n_lags)
        pseudo_outputs = np.vstack((pseudo_outputs, slow_recruit_factor_D * foo))
        _, _, _, foo = get_data('V2aB', 'fast', freq, time_warped=False, time_shift=-3*n_lags)
        pseudo_outputs = np.vstack((pseudo_outputs, fast_recruit_factor_B * foo))
        _, _, _, foo = get_data('V2aB', 'slow', freq, time_warped=False, time_shift=-3*n_lags)
        pseudo_outputs = np.vstack((pseudo_outputs, slow_recruit_factor_B * foo))

    # Combine with a copy of the data that has opposite phase for commissural inhibition:
    pseudo_outputs = np.vstack((pseudo_outputs, np.roll(pseudo_outputs, len(pseudo_outputs.T)//2, axis=1)))

    if tonic_only:
        pseudo_outputs = np.vstack((0*pseudo_outputs,
                                    fast_recruit_factor_D * np.ones(len(pseudo_outputs.T)),
                                    slow_recruit_factor_D * np.ones(len(pseudo_outputs.T))))
    else:
        pseudo_outputs = np.vstack((pseudo_outputs,
                                    fast_recruit_factor_D * np.ones(len(pseudo_outputs.T)),
                                    slow_recruit_factor_D * np.ones(len(pseudo_outputs.T))))

    return pseudo_outputs.T

def model(x, w_given=None):
    '''
    Forward pass of the RNN.

    Inputs
    --
    x : 2D array of inputs (n_timesteps by n_inputs)

    Returns
    --
    h : 2D array giving RNN activity (n_timesteps by n_hidden)

    y : 2D array giving RNN output (n_timesteps by n_outputs)

    input currents: ...

    '''

    h = torch.zeros(n_hidden)
    if n_outputs > 0:
        y_list = [torch.zeros(n_outputs)]
    else:
        y_list = None
    h_list = [h]
    inputs_e_list, inputs_i_list = [torch.zeros(n_hidden)], [torch.zeros(n_hidden)]
    for t in range(len(x) - 1):
        if w_given is not None:
            w = w_given[0]
            w_in = w_given[1]
        else:
            w = w_pos * torch.relu(w_rec0) - w_neg * torch.relu(-w_rec0)
            #w_in = w_in_mask * torch.relu(w_in0)
            w_in = w_in_mask * w_in0
        if not sign_constrain_weights:
            w = w_rec
        #inputs_e = h[:4*pop_size] @ w[:4*pop_size, :] + x[t,:] @ torch.relu(w_in)
        inputs_e = torch.matmul(h[:4*pop_size], w[:4*pop_size, :])
        inputs_e = torch.cat((inputs_e, torch.zeros(2*pop_size)))
        inputs_e += torch.matmul(x[t,:], torch.relu(w_in))
        #inputs_i = -h[4*pop_size:8*pop_size] @ w[4*pop_size:8*pop_size, :] - x[t,:] @ torch.relu(-w_in)
        inputs_i = -torch.matmul(h[4*pop_size:8*pop_size], w[4*pop_size:8*pop_size, :])
        inputs_i = torch.cat((inputs_i, torch.zeros(2*pop_size)))
        inputs_i += - torch.matmul(x[t,:], torch.relu(-w_in))
        inputs_e_list.append(inputs_e)
        inputs_i_list.append(inputs_i)
        if sign_constrain_weights:
            #h = (1 - 1/tau) * h + 1/tau * torch.relu(x[t,:] @ w_in + inputs_e - inputs_i + bias)
            #h = (1 - 1/tau) * h + 1/tau * torch.relu(inputs_e - inputs_i + bias)
            #h = (1 - tau_inv) * h + tau_inv * torch.relu(inputs_e - inputs_i + bias)
            h = (1 - tau_inv) * h + tau_inv * torch.relu(inputs_e - inputs_i)
        else:
            #h = (1 - 1/tau) * h + 1/tau * torch.relu(torch.matmul(x[t,:], w_in) + torch.matmul(h, w) + bias)
            h = (1 - 1/tau) * h + 1/tau * torch.relu(torch.matmul(x[t,:], w_in) + torch.matmul(h, w))
        h_list.append(h)
        if n_outputs > 0:
            y = torch.matmul(h, w_out)
            y_list.append(y)

    h = torch.stack(h_list).detach().numpy()[-x.shape[0]:,:]

    if n_outputs > 0:
        return h, torch.stack(y_list), torch.stack(inputs_e_list), torch.stack(inputs_i_list)
    else:
        return h, None, torch.stack(inputs_e_list), torch.stack(inputs_i_list)

def loss_func(inputs_e, inputs_i, targets_e, targets_i, input_vars_e, input_vars_i):
    loss_e = torch.mean((targets_e - inputs_e)**2)
    loss_i = torch.mean((targets_i - inputs_i)**2)
    loss_var = 0.1 * torch.mean(input_vars_e + input_vars_i)

    return loss_e + loss_i + loss_var# + loss_w

def plot_currents(w_given=None, alpha=1, tonic_only=False):
    r_squared = 0
    for i, cell in enumerate(cell_types):
        for j, speed in enumerate(speeds):
            for k, freq in enumerate(freqs):
                f = int(freq[:2])
                fast_recruit_factor_D = (f - 15) / (45 - 15)
                slow_recruit_factor_D = (45 - f) / (45 - 15)
                fast_recruit_factor_B = 0.2 + 0.8 * (f - 15) / (45 - 15)
                slow_recruit_factor_B = 0.2 + 0.8 * (45 - f) / (45 - 15)


                plt.subplot(4, 4, 1 + 8*i + 4*j + k)
                plt.ylim(0, 0.8)

                # Get experimental data:
                phases, e_current, i_current, pseudo_output = get_data(cell, speed, freq,
                                                                       time_warped=False,
                                                                       time_shift=-3)
                pseudo_output *= fast_recruit_factor_D * ((speed=='fast') & (cell[-1]=='D')) \
                                + slow_recruit_factor_D * ((speed=='slow') & (cell[-1]=='D')) \
                                + fast_recruit_factor_B * ((speed=='fast') & (cell[-1]=='B')) \
                                + slow_recruit_factor_B * ((speed=='slow') & (cell[-1]=='B'))

                # Get RNN outputs:
                pseudo_outputs = torch.tensor(get_pseudo_outputs(freq, n_lags=n_lags, tonic_only=tonic_only),
                                              dtype=torch.float)
                if w_given is not None:
                    _, _, input_currents_e, input_currents_i = model(torch.cat((pseudo_outputs,
                                                                            pseudo_outputs,
                                                                            pseudo_outputs,
                                                                            pseudo_outputs)), w_given)
                else:
                    _, _, input_currents_e, input_currents_i = model(torch.cat((pseudo_outputs,
                                                                            pseudo_outputs,
                                                                            pseudo_outputs,
                                                                            pseudo_outputs)))
                inputs_e = input_currents_e.detach().numpy()[-len(phases):]
                inputs_i = input_currents_i.detach().numpy()[-len(phases):]

                # Average over populations:
                inputs_e_avg = np.zeros((len(inputs_e), 4))
                inputs_i_avg = np.zeros((len(inputs_i), 4))
                inputs_e_sem = np.zeros((len(inputs_e), 4))
                inputs_i_sem = np.zeros((len(inputs_i), 4))
                for n in range(4):
                    inputs_e_avg[:, n] = np.mean(inputs_e[:, n*pop_size:(n+1)*pop_size], axis=1)
                    inputs_i_avg[:, n-4] = np.mean(inputs_i[:, n*pop_size:(n+1)*pop_size], axis=1)
                    inputs_e_sem[:, n] = np.std(inputs_e[:, n*pop_size:(n+1)*pop_size], axis=1)/pop_size**0.5
                    inputs_i_sem[:, n-4] = np.std(inputs_i[:, n*pop_size:(n+1)*pop_size], axis=1)/pop_size**0.5

                r_squared += 1/32 * np.corrcoef(inputs_e_avg[:,2*i+j], e_current)[0,1]**2
                r_squared += 1/32 * np.corrcoef(inputs_i_avg[:,2*i+j], i_current)[0,1]**2

                plt.plot(phases, inputs_e_avg[:,2*i+j], c='tab:blue', alpha=alpha)
                plt.plot(phases, inputs_i_avg[:,2*i+j], c='tab:orange', alpha=alpha)
                plt.fill_between(phases, inputs_e_avg[:,2*i+j] - inputs_e_sem[:,2*i+j],
                                 inputs_e_avg[:,2*i+j] + inputs_e_sem[:,2*i+j], color='tab:blue', alpha=0.1)
                plt.fill_between(phases, inputs_i_avg[:,2*i+j] - inputs_i_sem[:,2*i+j],
                                 inputs_i_avg[:,2*i+j] + inputs_i_sem[:,2*i+j], color='tab:orange', alpha=0.1)
                if w_given is None:
                    for n in range(n_lags):
                        plt.plot(phases, np.roll(pseudo_output, -3*n, axis=0), c='k', alpha=0.5)
                    plt.plot(phases, e_current, ls=':', c='tab:blue')
                    plt.plot(phases, i_current, ls=':', c='tab:orange')
                plt.xlim(-0.5, 0.5)
                plt.ylim(-0.02, 0.77)


                if 1 + 8*i + 4*j + k == 1:
                    plt.legend(currents)
                if 1 + 8*i + 4*j + k in [1, 2, 3, 4]:
                    plt.title(freq + ' Hz')
                if 1 + 8*i + 4*j + k in [1, 5, 9, 13]:
                    plt.ylabel(cell + ' ' + speed)
                else:
                    plt.yticks([0, 0.2, 0.4, 0.6], ['', '', '', ''])
                if 1 + 8*i + 4*j + k in [13, 14, 15, 16]:
                    plt.xlabel('Phase')
                    plt.xticks([-0.5, 0, 0.5])
                else:
                    plt.xticks([-0.5, 0, 0.5], ['', '', ''])

    plt.tight_layout()
    print('r-squared: ', r_squared)

