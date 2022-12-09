import os
import numpy as np
from yaml import load
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt
from IPython.display import display

def plot_img(model, frame_source, frames_driving, save=False):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.set_title('frame_source')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.set_title('frame_diving')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3.set_title('frame_generated')
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax1.imshow(frame_source.permute(1, 2, 0))
    out = model(frame_source.unsqueeze(0).repeat(frames_driving.shape[0], 1, 1, 1), frames_driving)['frame_generated']
    for i in range(out.shape[0]):
        ax2.imshow(frames_driving[i].permute(1, 2, 0))
        ax3.imshow(out[i].permute(1, 2, 0).detach())
        display(fig)
        if save:
            if not os.path.exists('frames'):
                os.mkdir('frames')
            fig.savefig('frames/frame_{}.png'.format(i), transparent=True)
    fig.clf()

def plot_loss(path_log, step_loss=100):
    with open(path_log, 'r') as f:
        log = load(f, Loader=SafeLoader)
    loss_total = []
    loss_ec = []
    for epoch in log.values():
        for i in range(0, len(epoch['total']), step_loss):
            loss_total.append(epoch['total'][i])
            if 'ec' in epoch.keys():
                loss_ec.append(epoch['ec'][i])
    time = np.linspace(1, len(log.keys()) + 1, len(loss_total))
    if 'ec' in log['epoch_1'].keys():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.set_xlabel('Total loss')
        ax1.set_ylabel('Loss value')
        ax2.set_xlabel('Equivariance Constraint loss')
        ax2.set_ylabel('Loss value')
        ax1.plot(time, loss_total)
        ax2.plot(time, loss_ec)
        display(fig)
        fig.clf()
    else:
        fig, ax = plt.figure(figsize=(5, 5))
        ax.set_xlabel('Total loss')
        ax.set_ylabel('Loss value')
        ax.plot(time, loss_total)
        display(fig)
        fig.clf()