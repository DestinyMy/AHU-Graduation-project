import torch
import torchvision
import torch.backends.cudnn as cudnn

import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(individual, num_epochs, batch_size, learning_rate):
    train_transform, test_transform = utils._data_transforms_cifar10()

    train_dataset = torchvision.datasets.CIFAR10(root='../../data',
                                               train=True,
                                               transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='../../data',
                                              train=False,
                                              transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    if individual.model_dict:
        individual.model.load_state_dict(individual.model_dict)

    structure = individual.model.to(device)

    individual.size = utils.count_parameters_in_MB(structure)

    parameters = filter(lambda p: p.requires_grad, structure.parameters())

    cudnn.enabled = True
    cudnn.benchmark = True

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=3e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0.0)

    best_acc = 0

    for epoch in range(num_epochs):
        print('epoch[{}/{}]:'.format(epoch + 1, num_epochs))
        train(train_loader, structure, criterion, optimizer)
        scheduler.step()
        valid_acc = test(test_loader, structure, criterion)
        print()
        if valid_acc > best_acc:
            best_acc = valid_acc

    individual.accuracy = best_acc


# train
def train(train_loader, structure, criterion, optimizer):
    structure.train()
    total_step = len(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = structure(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(structure.parameters(), 5)
        optimizer.step()
        if (i + 1) % 100 == 0:
            print('train: step[{}/{}], loss:{:.4f}'.format(i + 1, total_step, loss.item()))


# test
def test(test_loader, structure, criterion):
    structure.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        total_loss = 0
        total_step = len(test_loader)
        for step, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = structure(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        accuracy = 100 * correct / total
        print('test: step[{}/{}], loss:{:.4f}, accuracy:{:.4f}'.format(step + 1, total_step, total_loss / total_step,
                                                                       accuracy))
        return accuracy
