### Training order (Worst Case) : Decathlon --> Promise12 --> ISBI --> Prostate158

|   Method  |Prostate158|NCI-ISBI|Promise12|Decathlon| Accuracy (ACC) &#8593;&#8593; | Backward Transfer (BWT) &#8593;&#8593; | Average Forgetting (AFGT) &#8595;&#8595;|
|:---------:|:---------:|:---------:|:---------:|:---------:|:--------------:|:-----------------------:|:-------------------------:|
|Sequential(SGD) |  |  |  |  |    66.7      |          20.4           |           -1.29            |
|Sequential(Adam) |  |  |  |  |     73.3       |          -10.6          |           11.1            |
|Sequential (SGD 100)   |  |  |  |  |       72.5      |         -10.9        |          11.8        |
|Joint      |  |  |  |  |      -      |          -           |           -            |
|Reservoir Replay |82.3 ± 1.7| 82.9 ± 4.3| 74.8 ± 5.4| 82.1 ± 2.3| 80.5 ± 1.5| -1.3 ± 2.2| 1.9 ± 2.1|
|Class 1 Selective Replay |  |  |  |  |      81.2      |          -1.3           |           2.3           |
|Class Representative Replay |  |  |  |  |      80.2      |          -1.6           |           2.4            |
|Sample Importance Replay |  |  |  |  |      81.6      |          0.4           |           0.8            |
|Sample Importance Representative Replay |  |  |  |  |      -      |          -           |           -            |
|L2 Regularization | 37.9 ± 1.6                | 61.6 ± 10.8        | 34.9 ± 16.5             | 81.4 ± 2.0              | 53.9 ± 3.9       | 0.2 ± 0.2                 | -0.2 ± 0.2                  |
|EWC | 83.5 ± 1.4                | 76.9 ± 4.2         | 60.2 ± 11.2             | 74.9 ± 5.8              | 73.8 ± 2.8       | -10.4 ± 4.5               | 11.4 ± 5.5                  |

### Training order (Practical Case) : Prostate158 --> ISBI --> Promise12 --> Decathlon

|   Method  |Prostate158|  NCI - ISBI  |Promise12|Decathlon| Accuracy (ACC) &#8593;&#8593; | Backward Transfer (BWT) &#8593;&#8593; | Average Forgetting (AFGT) &#8595;&#8595;|
|:---------:|:---------:|:---------:|:---------:|:---------:|:--------------:|:-----------------------:|:-------------------------:|
|Sequential(SGD) |  |  |  |  |      73.5      |          -8.6           |           9.2            |
|Sequential(Adam) |  |  |  |  |      75.7      |          -11.0           |           11.8            |
|Sequential (SGD 100)   |  |  |  |  |       68.9      |         12.2        |          7.8        |
|Joint      |  |  |  |  |      -      |          -           |           -            |
|Reservoir Replay |67.5 ± 3.3| 85.7 ± 3.0| 74.8 ± 10.3| 85.7 ± 1.3| 78.4 ± 2.8| -6.8 ± 3.2| 7.7 ± 2.9|
|Class 1 Selective Replay |  |  |  |  |      81.2      |          -3.6           |           4.6            |
|Class Representative Replay |  |  |  |  |      79.1      |          -5.2           |           6.1            |
|Sample Importance Replay |  |  |  |  |      81.55      |          -2.1           |           2.7            |
|Sample Importance Representative Replay |  |  |  |  |      -      |          -           |           -            |
|L2 Regularization | 83.5 ± 1.6                | 67.6 ± 5.2         | 43.8 ± 10.6             | 69.6 ± 4.8              | 66.1 ± 3.8       | -0.0 ± 0.0                | 0.0 ± 0.0                   |
|EWC | 61.6 ± 3.2                | 82.6 ± 2.6         | 64.2 ± 10.9             | 85.3 ± 1.3              | 73.4 ± 3.1       | -13.6 ± 4.3               | 14.8 ± 3.5                  |
