### Training order (Worst Case) : Decathlon --> Promise12 --> ISBI --> Prostate158

|   Method  | Accuracy (ACC) &#8593;&#8593; | Backward Transfer (BWT) &#8593;&#8593; | Average Forgetting (AFGT) &#8595;&#8595;|
|:---------:|:--------------:|:-----------------------:|:-------------------------:|
|Sequential(SGD) |      66.7      |          20.4           |           -1.29            |
|Sequential(Adam) |     73.3       |          -10.6          |           11.1            |
|Sequential (SGD 100)   |       72.5      |         -10.9        |          11.8        |
|Joint      |      -      |          -           |           -            |
|Reservoir Replay |     80.8       |        -1.6            |           2.6            |
|Class 1 Selective Replay |      81.2      |          -1.3           |           2.3           |
|Class Representative Replay |      80.2      |          -1.6           |           2.4            |
|Sample Importance Replay |      81.6      |          0.4           |           0.8            |
|Sample Importance Representative Replay |      -      |          -           |           -            |


### Training order (Practical Case) : Prostate158 --> ISBI --> Promise12 --> Decathlon

|   Method  | Accuracy (ACC) &#8593;&#8593; | Backward Transfer (BWT) &#8593;&#8593; | Average Forgetting (AFGT) &#8595;&#8595;|
|:---------:|:--------------:|:-----------------------:|:-------------------------:|
|Sequential(SGD) |      73.5      |          -8.6           |           9.2            |
|Sequential(Adam) |      75.7      |          -11.0           |           11.8            |
|Sequential (SGD 100)   |       68.9      |         12.2        |          7.8        |
|Joint      |      -      |          -           |           -            |
|Reservoir Replay |      79.6      |          -5.8           |           6.7           |
|Class 1 Selective Replay |      81.2      |          -3.6           |           4.6            |
|Class Representative Replay |      79.1      |          -5.2           |           6.1            |
|Sample Importance Replay |      81.55      |          -2.1           |           2.7            |
|Sample Importance Representative Replay |      -      |          -           |           -            |
