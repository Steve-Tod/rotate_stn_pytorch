import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_output)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class ClsNet(nn.Module):

    def __init__(self):
        super(ClsNet, self).__init__()
        self.cnn = CNN(10)

    def forward(self, x):
        return F.log_softmax(self.cnn(x), dim=1)


# get rotate theta
class LocNet(nn.Module):

    def __init__(self):
        super(LocNet, self).__init__()
        self.cnn = CNN(1)

        # zero init
        bias = torch.zeros(1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size()[0]
        theta = self.cnn(x)
        return theta.view(batch_size)


# rotate
class RotateGridGen(nn.Module):
    
    def __init__(self):
        super(RotateGridGen, self).__init__()

    def forward(self, theta, out_size):
        assert len(theta.size()) == 1
        assert type(out_size) == torch.Size
        batch_size = theta.size()[0]
        affine_mat = theta.new(batch_size, 2, 3)
        affine_mat[:, :, 2] = 0
        affine_mat[:, 0, 0] = torch.cos(theta)
        affine_mat[:, 1, 1] = torch.cos(theta)
        affine_mat[:, 0, 1] = -torch.sin(theta)
        affine_mat[:, 1, 0] = torch.sin(theta)
        grid = F.affine_grid(affine_mat, out_size)
        return grid


# Full net
class STNClsNet(nn.Module):

    def __init__(self):
        super(STNClsNet, self).__init__()

        self.loc_net = LocNet()
        self.rotate = RotateGridGen()
        self.cls_net = ClsNet()

    def forward(self, x):
        batch_size = x.size()[0]
        theta = self.loc_net(x)
        grid = self.rotate(theta, x.size())
        transformed_x = F.grid_sample(x, grid)
        logit = self.cls_net(transformed_x)
        return logit
