import torch
from torch import nn

w = torch.tensor(2.4, requires_grad=True)
"""
opt = torch.optim.SGD([w],  lr = 0.01)

for i in range (100): 
    y = (w * 3 + 2) ** 2
    opt.zero_grad()
    y.backward()
    opt.step()
    #print(w.grad)


print(w)



net = nn.Sequential(nn.Linear(100,1), nn.ReLU(), nn.Linear(4, 2))
x = torch.tensor([1.3, 2.2, 2.5])
print(net(x))           # слой вызывают как функцию

count = 0

for p in net.parameters():
    print(p.shape)
    count += p.numel()
print(count)  

"""
"""
net = nn.Sequential(nn.Linear(1, 1))
net.load_state_dict(torch.load("model.pth", weights_only=True))

#x = torch.rand(100,1)

test_x = torch.tensor([[0], [0.5], [1]])
#correct_ans = 2 * x + 1
with torch.no_grad():
    print(net(test_x))

opt = torch.optim.SGD(net.parameters(), lr = 0.01)
loss_fn = nn.MSELoss()

for i in range (2000):
    opt.zero_grad()
    loss = loss_fn(net(x), correct_ans)
    loss.backward()
    opt.step()
    if i % 10 == 0:
       print(loss.item()) 

torch.save(net.state_dict(), "model.pth")
"""