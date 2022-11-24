import torch

def train(model, data_loader, loss_function, num_epochs=10, lr=1e-4, milestones_lr=[7, 9], display=False, log=False):
    criterion = loss_function
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_lr, gamma=0.1)
    if log:
        log_loss = []
    for epoch in range(num_epochs):
        for i, (s, d) in enumerate(data_loader):
            pred = model(s, d)
            loss = criterion(pred, d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if display:
                print('epoch {}/{}: [{}/{}] -------> loss: {}'.format(epoch + 1, num_epochs, i + 1, len(data_loader), loss.item()))
        scheduler.step()
        if log:
            log_loss.append(loss)
    if log:
        f = open('log/log_loss_{}.txt'.format(type(model).__name__), 'w')
        for i in log_loss:
            f.write(i + ' ')
        f.write('\n')
        f.close()
        