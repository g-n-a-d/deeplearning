import torch
from yaml import dump

def train(model, data_loader, loss_function, checkpoint_epoch=0, num_epochs=10, lr=1e-4, milestones_lr=[7, 9], gamma=0.1, device=torch.device('cpu'), display=False, log=False):
    criterion = loss_function
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_lr, gamma=gamma)
    model.train()
    for epoch in range(checkpoint_epoch, num_epochs):
        log_loss = {}
        log_loss_ = {
            'total': [],
            'ec': []
        }
        for i, (s, d) in enumerate(data_loader):
            s = s.to(device)
            d = d.to(device)
            pred = model(s, d)
            loss = criterion(pred, d)
            optimizer.zero_grad()
            loss['total'].backward()
            optimizer.step()
            if display:
                print('epoch {}/{}: [{}/{}] -------> loss_total: {} | loss_ec: {}'.format(
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    len(data_loader),
                    loss['total'].item(),
                    loss['ec'].item() if 'ec' in loss.keys() else 'na'))
            log_loss_['total'].append(loss['total'].item())
            if 'ec' in loss.keys():
                log_loss_['ec'].append(loss['ec'].item())
        scheduler.step()
        log_loss['epoch_{}'.format(epoch + 1)] = log_loss_
        if log:
            with open('log_loss_{}.yaml'.format(type(model).__name__), 'a') as f:
                dump(log_loss, f, default_flow_style=False)