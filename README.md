# autotvm-test
Benchmark AutoTVM against cuDNN and MIOpen


## GTX 1070 ti

| | VGG16 | Resnet50 | Densenet121
|------|--------|-------|-------|
| AutoTVM |  6.68  | 4.03  | 4.64 |
| TVM + cuDNN | 7.10   | 5.14  | 7.14 |
| MXNet + cuDNN | 8.07 | 5.89| 8.39 |
| PyTorch + cuDNN| 7.79 | 6.55| 12.3|


<br/>

## R9 Nano

| | VGG16 | Resnet50 |
|------|--------|-------|
| AutoTVM |  7.44 | 6.45  |
| TVM + MIOpen | 7.18 | 6.13  |
