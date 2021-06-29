import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import argparse
import timeit
import os

def train(train_loader, net, criterion, optimizer):
    net.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return loss

def test(test_loader, net, criterion, optimizer):
    net.eval()
    test_loss = 0
    nsamples = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = net(data)
            test_loss += criterion(output, target).item() * len(data) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            nsamples += len(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    return test_loss, nsamples, correct

def main(args):
    world_size = int(os.environ[args.env_size]) if args.env_size in os.environ else 1
    local_rank = int(os.environ[args.env_rank]) if args.env_rank in os.environ else 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if local_rank == 0:
        print(vars(args))

    if world_size > 1:
        print('rank: {}/{}'.format(local_rank+1, world_size))
        torch.distributed.init_process_group(
                backend='gloo',
                init_method='file://%s' % args.tmpname,
                rank=local_rank,
                world_size=world_size)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            transform=transform,
            download=True)

    sampler_train = None
    if world_size > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            sampler=sampler_train)

    test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            transform=transform)

    sampler_test = None
    if world_size > 1:
        sampler_test = torch.utils.data.distributed.DistributedSampler(test_dataset)

    test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            sampler=sampler_test)

    net = torchvision.models.resnet50()
    if world_size > 1:
        net = torch.nn.parallel.DistributedDataParallel(net)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = args.lr, momentum=0.9)

    for epoch in range(args.epochs):
        epoch_start = timeit.default_timer()
        loss = train(train_loader, net, criterion, optimizer)
        print('Train Epoch: {} Loss: {:.6f}'.format(epoch, loss.item()), end='')

        test_loss, nsamples, correct = test(test_loader, net, criterion, optimizer)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, nsamples, 100. * correct / nsamples), end='')

        print(' {:.2f}sec'.format(timeit.default_timer() - epoch_start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--data_dir', default='datasets/cifar10', type=str)
    parser.add_argument('--tmpname', default='tmpfile', type=str)
    parser.add_argument('--env_size', default='WORLD_SIZE', type=str)
    parser.add_argument('--env_rank', default='RANK', type=str)
    args = parser.parse_args()

    torch.manual_seed(10)

    main(args)
