import torch

def train(model, data_loader, loss_function, num_epochs=10, lr=1e-4, milestones_lr=[7, 9], gamma=0.1, display=False, log=False, device=torch.device('cpu')):
    criterion = loss_function
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_lr, gamma=gamma)
    model.train()
    if log:
        f = open('log/log_loss_{}.txt'.format(type(model).__name__), 'a')
    for epoch in range(num_epochs):
        log_loss = []
        for i, (s, d) in enumerate(data_loader):
            s = s.to(device)
            d = d.to(device)
            pred = model(s, d)
            loss = criterion(pred['frame_generated'], d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if display:
                print('epoch {}/{}: [{}/{}] -------> loss: {}'.format(epoch + 1, num_epochs, i + 1, len(data_loader), loss.item()))
            log_loss.append(loss.item())
        scheduler.step()
        if log:
            for i in log_loss:
                f.write(str(i) + ' ')
        f.write('\n')
    f.close()