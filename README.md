# Demo of resnet50_ddp with PyTorch in Intel(R) DevCloud

Distributed training (Data Parallel) demo with Resnet50 in Intel(R) DevCloud.

#### How to run
```bash
qsub run.sh
```

```
nodes=2
rank: 4/4
Train Epoch: 0 Loss: 2.026701Test set: Average loss: 0.5106, Accuracy: 598/2500 (24%) 6072.58sec
Train Epoch: 1 Loss: 2.065476Test set: Average loss: 0.4767, Accuracy: 699/2500 (28%) 6207.86sec
rank: 3/4
Train Epoch: 0 Loss: 1.950460Test set: Average loss: 0.5083, Accuracy: 585/2500 (23%) 6080.41sec
Train Epoch: 1 Loss: 1.842680Test set: Average loss: 0.4800, Accuracy: 730/2500 (29%) 6204.83sec
rank: 2/4
Train Epoch: 0 Loss: 2.146861Test set: Average loss: 0.5126, Accuracy: 585/2500 (23%) 6074.88sec
Train Epoch: 1 Loss: 2.026654Test set: Average loss: 0.4792, Accuracy: 690/2500 (28%) 6218.96sec
rank: 1/4
Train Epoch: 0 Loss: 1.826375Test set: Average loss: 0.5014, Accuracy: 613/2500 (25%) 6076.09sec
Train Epoch: 1 Loss: 1.672386Test set: Average loss: 0.4751, Accuracy: 704/2500 (28%) 6249.39sec
```

```
nodes=4
rank: 5/8
Train Epoch: 0 Loss: 2.138356Test set: Average loss: 0.2695, Accuracy: 251/1250 (20%) 3062.71sec
Train Epoch: 1 Loss: 1.932474Test set: Average loss: 0.2494, Accuracy: 333/1250 (27%) 3195.63sec
rank: 7/8
Train Epoch: 0 Loss: 2.095016Test set: Average loss: 0.2753, Accuracy: 257/1250 (21%) 3073.66sec
Train Epoch: 1 Loss: 1.883697Test set: Average loss: 0.2508, Accuracy: 340/1250 (27%) 3203.40sec
rank: 3/8
Train Epoch: 0 Loss: 2.177709Test set: Average loss: 0.2704, Accuracy: 286/1250 (23%) 3082.07sec
Train Epoch: 1 Loss: 2.047086Test set: Average loss: 0.2489, Accuracy: 338/1250 (27%) 3200.46sec
rank: 8/8
Train Epoch: 0 Loss: 2.292927Test set: Average loss: 0.2764, Accuracy: 283/1250 (23%) 3078.25sec
Train Epoch: 1 Loss: 2.215290Test set: Average loss: 0.2520, Accuracy: 336/1250 (27%) 3207.51sec
rank: 1/8
Train Epoch: 0 Loss: 2.091822Test set: Average loss: 0.2719, Accuracy: 264/1250 (21%) 3083.06sec
Train Epoch: 1 Loss: 1.967184Test set: Average loss: 0.2500, Accuracy: 347/1250 (28%) 3204.86sec
rank: 6/8
Train Epoch: 0 Loss: 2.048297Test set: Average loss: 0.2770, Accuracy: 253/1250 (20%) 3073.23sec
Train Epoch: 1 Loss: 1.930759Test set: Average loss: 0.2504, Accuracy: 363/1250 (29%) 3214.40sec
rank: 4/8
Train Epoch: 0 Loss: 2.162984Test set: Average loss: 0.2718, Accuracy: 261/1250 (21%) 3081.36sec
Train Epoch: 1 Loss: 1.957624Test set: Average loss: 0.2480, Accuracy: 349/1250 (28%) 3211.35sec
rank: 2/8
Train Epoch: 0 Loss: 2.085200Test set: Average loss: 0.2743, Accuracy: 262/1250 (21%) 3080.26sec
Train Epoch: 1 Loss: 1.976073Test set: Average loss: 0.2493, Accuracy: 331/1250 (26%) 3214.03sec
```
