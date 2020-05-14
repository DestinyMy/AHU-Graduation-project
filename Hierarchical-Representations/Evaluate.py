import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from Assemble import assemble
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(individual, num_classes, num_epochs, batch_size, learning_rate):
    # train_transform, test_transform = utils._data_transforms_cifar10()

    train_dataset = torchvision.datasets.MNIST(root='../../data',
                                               train=True,
                                               transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    structure = assemble(individual.genotype, [(1, 32), (32, 64), (64, 128)], num_classes).to(device)
    print('model constructed')
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
        print('test: loss:{:.4f}, accuracy:{:.4f}'.format(total_loss / total_step, accuracy))
        return accuracy
