# Setup

```
conda create -n openevo python=3.8
conda activate openevo
```

and install evogym.
https://evolutiongym.github.io/tutorials/getting-started.html


# Experiments

## Novelty Search with NEAT
### execution
```
$python run_ns_neat.py
```
#### options:
| option        | abbrev  | default         | detail  |
| :---          | :---:   | :---:           | :---    |
| --name        | -n      | "{task}_{robot}"| experiment name |
| --task        | -t      | Walker-v0       | evogym environment id |
| --robot       | -r      | cat             | robot structure name <br> built on "robot_files/" <br> if "default", load default robot for the task |
| --pop-size    | -p      | 200             | population size of NEAT |
| --generation  | -g      | 500             | iterations of NEAT |
| --ns-threshold|         | 0.1             | initial threshold to add to novelty archive |
| --num-knn     |         | 5               | number of nearest neighbors to calculate novelty |
| --mcns        |         | 0.0             | minimal reward criterion. if not satisfy, die. |
| --eval-num    |         | 1               | evaluation times. if probabilistic task, need more. |
| --num-cores   | -c      | 4               | number of parallel evaluation processes |
| --no-view     |         | *false*         | not open simulation window of best robot |

### make figure
after run_evogym, make {gif, jpg} file for each of all genomes written in history files.
output to "out/ns_neat/{expt name}/figure/{gif, jpg}/"
```
$python make_figures_ns_neat.py {experiment name}
```
#### options:
| option              | abbrev  | default | detail  |
| :---                | :---:   | :---:   | :---    |
|                     |         |         | name of experiment for making figures |
| --specified         | -s      |         | input id, make figure for the only specified genome |
| --save-type         | -st     | gif     | file type (choose from [gif, jpg])
| --resolution-ratio  | -r      | 0.2     | gif resolution ratio (0.2 -> (256,144)) |
| --interval          | -i      | timestep| in case of save type is jpg, type of interval for robot drawing <br>(choose from [timestep, distance, hybrid]) |
| --resolution-scale  | -rs     | 32.0    | jpg resolution scale <br> when output monochrome image, try this argument change. |
| --timestep-interval | -ti     | 80      | timestep interval for robot drawing <br>(if interval is hybrid, it should be about 40) |
| --distance-interval | -di     | 0.8     | distance interval for robot drawing |
| --display-timestep  |         | *false* | display timestep above robot |
| --num-cores         | -c      | 1       | number of parallel making processes |
| --not-overwrite     |         | *false* | skip process if already figure exists |
| --no-multi          |         | *false* | do without using multiprocessing. if error occur, try this option. |


## Novelty Search with Hyper-NEAT
### execution
```
$python run_ns_hyper.py
```
#### options:
| option        | abbrev  | default         | detail  |
| :---          | :---:   | :---:           | :---    |
| --name        | -n      | "{task}_{robot}"| experiment name |
| --task        | -t      | Walker-v0       | evogym environment id |
| --robot       | -r      | cat             | robot structure name <br> built on "robot_files/" <br> if "default", load default robot for the task |
| --pop-size    | -p      | 200             | population size of NEAT |
| --generation  | -g      | 500             | iterations of NEAT |
| --ns-threshold|         | 0.1             | initial threshold to add to novelty archive |
| --num-knn     |         | 5               | number of nearest neighbors to calculate novelty |
| --mcns        |         | 0.0             | minimal reward criterion. if not satisfy, die. |
| --use-hideen  |         | *false*         | make hidden nodes on NN substrate |
| --eval-num    |         | 1               | evaluation times. if probabilistic task, need more. |
| --num-cores   | -c      | 4               | number of parallel evaluation processes |
| --no-view     |         | *false*         | not open simulation window of best robot |

### make figure
after run_evogym_hyper, make {gif, jpg} file for each of all genomes written in history files.
output to "out/ns_hyper/{expt name}/figure/{gif, jpg}/"
```
$python make_figures_ns_hyper.py {experiment name}
```
#### options:
| option              | abbrev  | default | detail  |
| :---                | :---:   | :---:   | :---    |
|                     |         |         | name of experiment for making figures |
| --specified         | -s      |         | input id, make figure for the only specified genome |
| --save-type         | -st     | gif     | file type (choose from [gif, jpg])
| --resolution-ratio  | -r      | 0.2     | gif resolution ratio (0.2 -> (256,144)) |
| --interval          | -i      | timestep| in case of save type is jpg, type of interval for robot drawing <br>(choose from [timestep, distance, hybrid]) |
| --resolution-scale  | -rs     | 32.0    | jpg resolution scale <br> when output monochrome image, try this argument change. |
| --timestep-interval | -ti     | 80      | timestep interval for robot drawing <br>(if interval is hybrid, it should be about 40) |
| --distance-interval | -di     | 0.8     | distance interval for robot drawing |
| --display-timestep  |         | *false* | display timestep above robot |
| --num-cores         | -c      | 1       | number of parallel making processes |
| --not-overwrite     |         | *false* | skip process if already figure exists |
| --no-multi          |         | *false* | do without using multiprocessing. if error occur, try this option. |


## PPO
### execution
```
$python run_ppo.py
```
#### options:
| option                | abbrev  | default         | detail  |
| :---                  | :---:   | :---:           | :---    |
| --name                | -n      | "{task}_{robot}"| experiment name |
| --task                | -t      | Walker-v0       | evogym environment id |
| --robot               | -r      | default         | robot structure name <br> built on "robot_files/" <br> if "default", load default robot for the task |
| --num-processes       | -p      | 4               | how many training CPU processes to use |
| --steps               | -s      | 256             | num steps to use in PPO |
| --num-mini-batch      | -b      | 8               | number of batches for ppo |
| --epochs              | -e      | 8               | number of ppo epochs |
| --train-iters         | -i      | 2500            | learning iterations of PPO |
| --evaluation-interval | -ei     | 25              | frequency to evaluate policy |
| --lerning-rate        | -lr     | 3e-4            | learning rate |
| --gamma               |         | 0.99            | discount factor for rewards |
| --clip-range          | -c      | 0.3             | ppo clip parameter |
| --init-log-std        | -std    | 0.1             | initial log std of action distribution |
| --deterministic       | -d      | *false*         | robot act deterministic |
| --no-view             |         | *false*         | not open simulation window of best robot |


### make figure
after run_evogym, make {gif, jpg} file for each of all controllers.
output to "out/ppo/{expt name}/figure/{gif, jpg}/"
```
$python make_figures_ppo.py {experiment name}
```
#### options:
| option              | abbrev  | default | detail  |
| :---                | :---:   | :---:   | :---    |
|                     |         |         | name of experiment for making figures |
| --specified         | -s      |         | input iter, make figure for the only specified controller |
| --save-type         | -st     | gif     | file type (choose from [gif, jpg])
| --resolution-ratio  | -r      | 0.2     | gif resolution ratio (0.2 -> (256,144)) |
| --interval          | -i      | timestep| in case of save type is jpg, type of interval for robot drawing <br>(choose from [timestep, distance, hybrid]) |
| --resolution-scale  | -rs     | 32.0    | jpg resolution scale <br> when output monochrome image, try this argument change. |
| --timestep-interval | -ti     | 80      | timestep interval for robot drawing <br>(if interval is hybrid, it should be about 40) |
| --distance-interval | -di     | 0.8     | distance interval for robot drawing |
| --display-timestep  |         | *false* | display timestep above robot |
| --num-cores         | -c      | 1       | number of parallel making processes |
| --not-overwrite     |         | *false* | skip process if already figure exists |
| --no-multi          |         | *false* | do without using multiprocessing. if error occur, try this option. |
