<div align="center">

[![Example](https://raw.githubusercontent.com/Tviskaron/pogema-svg/main/learn-to-follow-ep00001-lab-maze_010-seed0.svg)](https://github.com/AIRI-Institute/learn-to-follow) 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CnC47qbc4Z3sHfiR6sIX0ngXi6UfTx8o?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/AIRI-Institute/learn-to-follow/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2310.01207-b31b1b.svg)](https://arxiv.org/abs/2310.01207)
[![Paper](https://img.shields.io/badge/AAAI-2024-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/29704)

**Learn to Follow: Lifelong Multi-agent Pathfinding with Decentralized Replanning**

</div> 

This study addresses the challenging problem of decentralized lifelong multi-agent pathfinding. The proposed **Follower** 
approach utilizes a combination of a planning algorithm for constructing a long-term plan and reinforcement learning
for resolving local conflicts.

**Paper:** [Learn to Follow: Decentralized Lifelong Multi-agent Pathfinding via Planning and Learning
](https://arxiv.org/abs/2310.01207)



## Installation:

```bash
pip3 install -r docker/requirements.txt
```


Installation of ONNX runtime:
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz \
    && tar -xf onnxruntime-linux-x64-1.14.1.tgz \
    && cp onnxruntime-linux-x64-1.14.1/lib/* /usr/lib/ && cp onnxruntime-linux-x64-1.14.1/include/* /usr/include/
```

Optionally, you could use the Dockerfile to build the image:
```bash
cd "/path/to/learn-to-follow"
docker build -t learn-to-follow -f docker/dockerfile .
```

To start the container, use the following command:
```bash
docker run --rm -it learn-to-follow:latest bash
```

## Inference Example:

To execute the **Follower** algorithm and produce an animation using pre-trained weights, use the following command:

```bash
python3 example.py
```

The animation will be stored in the `renders` folder.

It's recommended to set environment variable to restrict Numpy CPU threads to 1,  avoiding performance issues:

```bash
export OMP_NUM_THREADS="1" 
export MKL_NUM_THREADS="1" 
export OPENBLAS_NUM_THREADS="1"
```

You can adjust the environment and algorithm parameter using arguments. For example:
```
python3 example.py --map_name wfi_warehouse --num_agents 128
python3 example.py --map_name pico_s00_od20_na32 --num_agents 32 --algorithm FollowerLite
```


We offer a Google Colab example that simplifies the process:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CnC47qbc4Z3sHfiR6sIX0ngXi6UfTx8o?usp=sharing)


## Training:

To train **Follower** from scratch, use the following command:

```bash
python3 main.py  --actor_critic_share_weights=True --batch_size=16384 --env=PogemaMazes-v0 --exploration_loss_coeff=0.023 --extra_fc_layers=1 --gamma=0.9756 --hidden_size=512 --intrinsic_target_reward=0.01 --learning_rate=0.00022 --lr_schedule=constant --network_input_radius=5 --num_filters=64 --num_res_blocks=8 --num_workers=8 --optimizer=adam --ppo_clip_ratio=0.2   --train_for_env_steps=1000000000 --use_rnn=True
```

To train **FollowerLite** from scratch, use the following command:
```bash
python3 main.py  --actor_critic_share_weights=True --batch_size=16384 --env=PogemaMazes-v0 --exploration_loss_coeff=0.0156 --extra_fc_layers=0 --gamma=0.9716 --hidden_size=16 --intrinsic_target_reward=0.01 --learning_rate=0.00013 --lr_schedule=kl_adaptive_minibatch --network_input_radius=3 --num_filters=8 --num_res_blocks=1 --num_workers=4 --optimizer=adam --ppo_clip_ratio=0.2     --train_for_env_steps=20000000 --use_rnn=False
```
The parameters are set to the values used in the paper.

### Testing and Results Visualization 
To reproduce the main results of **Follower** and **FollowerLite** using [pogema-toolbox](https://github.com/AIRI-Institute/pogema-toolbox), use the following command:
```bash
python3 eval.py
```
This script will run all the experiments, the configurations for which are placed in the experiments folder. The raw data will be saved in the corresponding folders (including plots) and optionally saved to wandb.

#### Example Configuration:

```yaml
environment:
  name: Pogema-v0
  on_target: restart
  max_episode_steps: 512
  observation_type: POMAPF
  collision_system: soft  
  map_name: wfi_warehouse
  num_agents:
    grid_search: [ 32, 64, 96, 128, 160, 192 ]
  seed:
    grid_search: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

algorithms:
  Follower:
    name: Follower
    num_process: 4
    parallel_backend: 'balanced_dask'


  No dynamic cost:
    name: Follower
    num_process: 4
    parallel_backend: 'balanced_dask'
    
    override_config:
      preprocessing:
        use_dynamic_cost: False

  No static cost:
    name: Follower
    num_process: 4
    num_threads: 4
    parallel_backend: 'balanced_dask'
    
    override_config:
      preprocessing:
        use_static_cost: False

results_views:
  TabularResults:
    type: tabular
    drop_keys: [ seed ]
    print_results: True

  05-warehouse:
    type: plot
    x: num_agents
    y: avg_throughput
    name: Warehouse $46 \times 33$
```

#### Description of Configuration:

The configuration defines the environment settings and the algorithms used for the experiments. It specifies the following:
- **Environment**: Includes parameters of the POGEMA environment, behavior on target (restart, corresponding to LifeLong), maximum episode steps (512), observation type, collision system, etc. It also sets up grid searches for the number of agents and seed values. The `grid_search` can be used for any environment parameter.
- **Algorithms**: Details the algorithms to be tested. The primary algorithm is **Follower**. Variants include "No dynamic cost" and "No static cost," which override specific preprocessing configurations. All algorithms are configurable to use `4` processes and the `balanced_dask` backend for parallelization, enhancing computational efficiency.
- **Results Views**: Defines how the results will be presented, including tabular and plot views.

This example configuration demonstrates how to set up experiments for the Pogema-v0 environment, varying the number of agents and seeds, and comparing different versions of the Follower algorithm.
#### Raw Data

The raw data, comprising the results of our experiments for Follower and FollowerLite, can be downloaded from the following link:
[Download Raw Data](https://github.com/AIRI-Institute/learn-to-follow/releases/download/v0/learn-to-follow-raw-data.zip)


## Citation:

```bibtex
@inproceedings{skrynnik2024learn,
  title={Learn to Follow: Decentralized Lifelong Multi-Agent Pathfinding via Planning and Learning},
  author={Skrynnik, Alexey and Andreychuk, Anton and Nesterova, Maria and Yakovlev, Konstantin and Panov, Aleksandr},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={17541--17549},
  year={2024}
}
```


# Rezultati eksperimenta na isti mapi

## Follower.json
| num_agents | algorithm | throughput | success_rate | num_collisions | makespan | planner_time | ISR | CSR | ep_length | runtime            |
|------------|-----------|------------|--------------|----------------|----------|--------------|-----|-----|-----------|--------------------|
| 8          | Follower  | 0.4961     | 1            | 30             | 129      | 0.002045     | 1   | 1   | 63.546875 | 45.46448551600315  |
| 8          | Follower  | 0.4961     | 1            | 17             | 129      | 0.002112     | 1   | 1   | 56.421875 | 45.818045076998715 |
| 8          | Follower  | 0.64       | 1            | 26             | 100      | 0.002127     | 1   | 1   | 49.375    | 35.757246230999954 |
| 8          | Follower  | 0.4923     | 1            | 13             | 130      | 0.002183     | 1   | 1   | 60.1875   | 47.63647588100321  |
| 8          | Follower  | 0.6214     | 1            | 25             | 103      | 0.002319     | 1   | 1   | 49.8125   | 39.72165113500341  |
| 8          | Follower  | 0.3902     | 1            | 16             | 164      | 0.002107     | 1   | 1   | 64.328125 | 60.360865859001024 |
| 8          | Follower  | 0.5079     | 1            | 19             | 126      | 0.002097     | 1   | 1   | 51.53125  | 47.87792698699195  |
| 8          | Follower  | 0.5565     | 1            | 22             | 115      | 0.0021       | 1   | 1   | 49.21875  | 42.23164787900805  |
| 8          | Follower  | 0.5926     | 1            | 8              | 108      | 0.002203     | 1   | 1   | 58.421875 | 38.92080548799731  |
| 8          | Follower  | 0.5424     | 1            | 22             | 118      | 0.002098     | 1   | 1   | 67.34375  | 41.98614891900161  |
| 8          | Follower  | 0.5517     | 1            | 17             | 116      | 0.001973     | 1   | 1   | 56.25     | 41.06083853300788  |
| 8          | Follower  | 0.5714     | 1            | 22             | 112      | 0.00212      | 1   | 1   | 65.203125 | 39.645642148999286 |
| 8          | Follower  | 0.4183     | 1            | 36             | 153      | 0.002066     | 1   | 1   | 67.734375 | 54.16616111300846  |
| 8          | Follower  | 0.5565     | 1            | 20             | 115      | 0.002035     | 1   | 1   | 55.875    | 40.43345647699334  |
| 8          | Follower  | 0.4923     | 1            | 36             | 130      | 0.002126     | 1   | 1   | 54.5      | 46.07366224900761  |
| 8          | Follower  | 0.6275     | 1            | 17             | 102      | 0.002061     | 1   | 1   | 50.6875   | 35.80462476399771  |
| 8          | Follower  | 0.6531     | 1            | 26             | 98       | 0.002142     | 1   | 1   | 53.484375 | 35.14552382300553  |
| 8          | Follower  | 0.4476     | 1            | 17             | 143      | 0.001918     | 1   | 1   | 58.53125  | 45.944083928005966 |
| 8          | Follower  | 0.5039     | 1            | 28             | 127      | 0.002031     | 1   | 1   | 56.953125 | 44.26301804900504  |
| 8          | Follower  | 0.5079     | 1            | 24             | 126      | 0.002125     | 1   | 1   | 60.203125 | 45.455123187000936 |
| 16         | Follower  | 0.4961     | 1            | 20             | 129      | 0.002372     | 1   | 1   | 52.75     | 52.88827782400131  |
| 16         | Follower  | 0.6465     | 1            | 16             | 99       | 0.00241      | 1   | 1   | 55.015625 | 40.23752821600283  |
| 16         | Follower  | 0.3721     | 1            | 48             | 172      | 0.002354     | 1   | 1   | 69.953125 | 68.41760637199695  |
| 16         | Follower  | 0.6214     | 1            | 30             | 103      | 0.002366     | 1   | 1   | 50.078125 | 41.962066654998125 |
| 16         | Follower  | 0.6809     | 1            | 21             | 94       | 0.002398     | 1   | 1   | 41.71875  | 36.14037145099701  |
| 16         | Follower  | 0.4354     | 1            | 24             | 147      | 0.00236      | 1   | 1   | 56.890625 | 61.37676904200089  |
| 16         | Follower  | 0.4812     | 1            | 24             | 133      | 0.002317     | 1   | 1   | 57.0625   | 55.41685210000742  |
| 16         | Follower  | 0.5517     | 1            | 36             | 116      | 0.002609     | 1   | 1   | 59.59375  | 48.46699161599918  |
| 16         | Follower  | 0.5981     | 1            | 26             | 107      | 0.002182     | 1   | 1   | 60.5625   | 37.874676171999454 |
| 16         | Follower  | 0.547      | 1            | 13             | 117      | 0.002373     | 1   | 1   | 48.84375  | 46.97718487200473  |
| 16         | Follower  | 0.6214     | 1            | 24             | 103      | 0.002381     | 1   | 1   | 55.28125  | 40.69480337800724  |
| 16         | Follower  | 0.5818     | 1            | 22             | 110      | 0.002054     | 1   | 1   | 49.9375   | 38.72879303599984  |
| 16         | Follower  | 0.4672     | 1            | 21             | 137      | 0.002072     | 1   | 1   | 51.46875  | 47.78346834400054  |
| 16         | Follower  | 0.3351     | 1            | 21             | 191      | 0.001963     | 1   | 1   | 60.9375   | 66.642924734997    |
| 16         | Follower  | 0.5161     | 1            | 26             | 124      | 0.002019     | 1   | 1   | 63.234375 | 43.571782440999414 |
| 16         | Follower  | 0.547      | 1            | 23             | 117      | 0.002094     | 1   | 1   | 58.8125   | 41.23594908799532  |
| 16         | Follower  | 0.4604     | 1            | 32             | 139      | 0.002126     | 1   | 1   | 66.1875   | 50.594069693000165 |
| 16         | Follower  | 0.5039     | 1            | 26             | 127      | 0.0021       | 1   | 1   | 63.53125  | 44.87863402300354  |
| 16         | Follower  | 0.6275     | 1            | 30             | 102      | 0.001955     | 1   | 1   | 49.09375  | 35.90138131299773  |
| 16         | Follower  | 0.5872     | 1            | 24             | 109      | 0.001999     | 1   | 1   | 47.453125 | 38.27091187400856  |
| 32         | Follower  | 0.5        | 1            | 14             | 128      | 0.002963     | 1   | 1   | 56.21875  | 58.96613205299764  |
| 32         | Follower  | 0.512      | 1            | 26             | 125      | 0.003004     | 1   | 1   | 51.28125  | 60.185614134994466 |
| 32         | Follower  | 0.5565     | 1            | 16             | 115      | 0.002933     | 1   | 1   | 54.390625 | 55.04578729899458  |
| 32         | Follower  | 0.5766     | 1            | 28             | 111      | 0.002956     | 1   | 1   | 46.890625 | 53.75979511900459  |
| 32         | Follower  | 0.4776     | 1            | 37             | 134      | 0.002932     | 1   | 1   | 54.34375  | 64.63124594799865  |
| 32         | Follower  | 0.4741     | 1            | 25             | 135      | 0.004016     | 1   | 1   | 55.796875 | 73.30605065200234  |
| 32         | Follower  | 0.64       | 1            | 19             | 100      | 0.003051     | 1   | 1   | 48.328125 | 50.53537286599476  |
| 32         | Follower  | 0.5714     | 1            | 19             | 112      | 0.003323     | 1   | 1   | 60.46875  | 61.255993336002575 |
| 32         | Follower  | 0.5079     | 1            | 19             | 126      | 0.002441     | 1   | 1   | 53.609375 | 51.48485161600456  |
| 32         | Follower  | 0.5664     | 1            | 6              | 113      | 0.002609     | 1   | 1   | 48.46875  | 49.254734606004604 |
| 32         | Follower  | 0.4051     | 1            | 29             | 158      | 0.002519     | 1   | 1   | 75.625    | 66.32460221000565  |
| 32         | Follower  | 0.4706     | 1            | 18             | 136      | 0.002307     | 1   | 1   | 59.828125 | 55.92930215999695  |
| 32         | Follower  | 0.6038     | 1            | 21             | 106      | 0.002637     | 1   | 1   | 57.09375  | 43.968878113001665 |
| 32         | Follower  | 0.5565     | 1            | 17             | 115      | 0.002348     | 1   | 1   | 58.8125   | 46.2018411660074   |
| 32         | Follower  | 0.4885     | 1            | 26             | 131      | 0.002423     | 1   | 1   | 60.390625 | 54.16979605800043  |
| 32         | Follower  | 0.5039     | 1            | 21             | 127      | 0.002421     | 1   | 1   | 64.171875 | 50.962335215001985 |
| 32         | Follower  | 0.4156     | 1            | 26             | 154      | 0.00249      | 1   | 1   | 66.421875 | 65.23658326399891  |
| 32         | Follower  | 0.6275     | 1            | 25             | 102      | 0.00238      | 1   | 1   | 49.3125   | 41.94764465199387  |
| 32         | Follower  | 0.5424     | 1            | 18             | 118      | 0.002467     | 1   | 1   | 58.0625   | 47.79646782399868  |
| 32         | Follower  | 0.5714     | 1            | 22             | 112      | 0.002323     | 1   | 1   | 57.421875 | 46.31408224300321  |
| 64         | Follower  | 0.512      | 1            | 20             | 125      | 0.001763     | 1   | 1   | 57.71875  | 37.51964562500143  |
| 64         | Follower  | 0.5203     | 1            | 18             | 123      | 0.001589     | 1   | 1   | 56.671875 | 34.76939119699637  |
| 64         | Follower  | 0.6882     | 1            | 9              | 93       | 0.001836     | 1   | 1   | 50.546875 | 27.45545798100011  |
| 64         | Follower  | 0.4885     | 1            | 30             | 131      | 0.001647     | 1   | 1   | 60.203125 | 37.9755268960007   |
| 64         | Follower  | 0.5079     | 1            | 23             | 126      | 0.001658     | 1   | 1   | 58.8125   | 36.35029306700562  |
| 64         | Follower  | 0.4885     | 1            | 22             | 131      | 0.001589     | 1   | 1   | 53.34375  | 36.560090446994764 |
| 64         | Follower  | 0.5565     | 1            | 22             | 115      | 0.001598     | 1   | 1   | 52.890625 | 32.37789456700739  |
| 64         | Follower  | 0.5424     | 1            | 18             | 118      | 0.001566     | 1   | 1   | 57.765625 | 33.39837425300357  |
| 64         | Follower  | 0.5246     | 1            | 10             | 122      | 0.001791     | 1   | 1   | 59.453125 | 36.74605739600338  |
| 64         | Follower  | 0.5203     | 1            | 33             | 123      | 0.00193      | 1   | 1   | 57.390625 | 39.97614411600625  |
| 64         | Follower  | 0.5517     | 1            | 16             | 116      | 0.001848     | 1   | 1   | 54.671875 | 36.71566547600287  |
| 64         | Follower  | 0.4672     | 1            | 18             | 137      | 0.002941     | 1   | 1   | 63.3125   | 65.22698111799673  |
| 64         | Follower  | 0.4384     | 1            | 27             | 146      | 0.002912     | 1   | 1   | 59.015625 | 68.05215977000353  |
| 64         | Follower  | 0.5289     | 1            | 32             | 121      | 0.002674     | 1   | 1   | 53.890625 | 54.51403965100417  |
| 64         | Follower  | 0.4571     | 1            | 23             | 140      | 0.002888     | 1   | 1   | 60.234375 | 65.13319188299829  |
| 64         | Follower  | 0.5079     | 1            | 16             | 126      | 0.001812     | 1   | 1   | 61.828125 | 38.88122075100273  |
| 64         | Follower  | 0.64       | 1            | 26             | 100      | 0.002388     | 1   | 1   | 50.28125  | 40.63506390600378  |
| 64         | Follower  | 0.4923     | 1            | 14             | 130      | 0.003176     | 1   | 1   | 55.03125  | 67.74751924899556  |
| 64         | Follower  | 0.4961     | 1            | 10             | 129      | 0.003119     | 1   | 1   | 57.15625  | 64.09709055899748  |
| 64         | Follower  | 0.4571     | 1            | 39             | 140      | 0.002898     | 1   | 1   | 58.125    | 69.3959333469993   |
| 128        | Follower  | 0.4672     | 1            | 32             | 137      | 0.001697     | 1   | 1   | 62.84375  | 40.806729392002126 |
| 128        | Follower  | 0.4211     | 1            | 26             | 152      | 0.002101     | 1   | 1   | 62.125    | 53.62025652399825  |
| 128        | Follower  | 0.5203     | 1            | 19             | 123      | 0.001868     | 1   | 1   | 57.421875 | 41.08617812499233  |
| 128        | Follower  | 0.5        | 1            | 37             | 128      | 0.001993     | 1   | 1   | 59.703125 | 44.04300081800102  |
| 128        | Follower  | 0.4885     | 1            | 25             | 131      | 0.001871     | 1   | 1   | 62.828125 | 41.97008338100295  |
| 128        | Follower  | 0.5714     | 1            | 24             | 112      | 0.002136     | 1   | 1   | 58.21875  | 39.53564672700304  |
| 128        | Follower  | 0.5517     | 1            | 19             | 116      | 0.001687     | 1   | 1   | 52.859375 | 34.54248380299941  |
| 128        | Follower  | 0.5        | 1            | 18             | 128      | 0.001976     | 1   | 1   | 49.578125 | 43.65288662900093  |
| 128        | Follower  | 0.5333     | 1            | 14             | 120      | 0.001669     | 1   | 1   | 59.109375 | 35.76073869400443  |
| 128        | Follower  | 0.5378     | 1            | 30             | 119      | 0.002008     | 1   | 1   | 60.875    | 39.582192599003974 |
| 128        | Follower  | 0.5818     | 1            | 26             | 110      | 0.001812     | 1   | 1   | 59.78125  | 33.980891864987825 |
| 128        | Follower  | 0.4539     | 1            | 22             | 141      | 0.001681     | 1   | 1   | 52.390625 | 41.36186107999856  |
| 128        | Follower  | 0.5818     | 1            | 13             | 110      | 0.001793     | 1   | 1   | 55.0625   | 32.889212375998795 |
| 128        | Follower  | 0.5079     | 1            | 18             | 126      | 0.001766     | 1   | 1   | 58.890625 | 37.382419064001624 |
| 128        | Follower  | 0.6095     | 1            | 27             | 105      | 0.001771     | 1   | 1   | 53.0625   | 31.49016339499849  |
| 128        | Follower  | 0.5203     | 1            | 31             | 123      | 0.001725     | 1   | 1   | 56.578125 | 37.14289571899553  |
| 128        | Follower  | 0.5039     | 1            | 25             | 127      | 0.001717     | 1   | 1   | 65.375    | 38.506909936996635 |
| 128        | Follower  | 0.5766     | 1            | 34             | 111      | 0.001728     | 1   | 1   | 51.546875 | 33.15818953699727  |
| 128        | Follower  | 0.4211     | 1            | 32             | 152      | 0.001694     | 1   | 1   | 67.703125 | 45.590521271995385 |
| 128        | Follower  | 0.3459     | 1            | 27             | 185      | 0.001665     | 1   | 1   | 58.9375   | 54.545801201998984 |

## FolloweLite.json
| num_agents | algorithm    | throughput | success_rate | num_collisions | makespan | planner_time | ISR      | CSR | ep_length | runtime            |
|------------|--------------|------------|--------------|----------------|----------|--------------|----------|-----|-----------|--------------------|
| 8          | FollowerLite | 0.4324     | 1            | 14             | 148      | 0.001889     | 1        | 1   | 56.390625 | 1.0864712869997675 |
| 8          | FollowerLite | 0.1803     | 1            | 25             | 355      | 0.003281     | 1        | 1   | 80.140625 | 3.058877547998236  |
| 8          | FollowerLite | 0.237      | 1            | 22             | 270      | 0.003094     | 1        | 1   | 79.09375  | 2.4019279839803858 |
| 8          | FollowerLite | 0.123      | 0.9844       | 27             | 512      | 0.003297     | 0.984375 | 0   | 94.078125 | 4.224071929996171  |
| 8          | FollowerLite | 0.4848     | 1            | 9              | 132      | 0.003333     | 1        | 1   | 65.4375   | 1.6025131210071777 |
| 8          | FollowerLite | 0.4        | 1            | 9              | 160      | 0.003102     | 1        | 1   | 57.078125 | 1.7756040779922841 |
| 8          | FollowerLite | 0.5289     | 1            | 6              | 121      | 0.003505     | 1        | 1   | 49.34375  | 1.830850268001086  |
| 8          | FollowerLite | 0.4539     | 1            | 12             | 141      | 0.003129     | 1        | 1   | 56.25     | 1.5642372500042256 |
| 8          | FollowerLite | 0.6095     | 1            | 4              | 105      | 0.003271     | 1        | 1   | 45.703125 | 1.3174914580049517 |
| 8          | FollowerLite | 0.4923     | 1            | 11             | 130      | 0.00364      | 1        | 1   | 59.859375 | 1.7108190569988437 |
| 8          | FollowerLite | 0.4156     | 1            | 5              | 154      | 0.00305      | 1        | 1   | 54.796875 | 1.6243637480083635 |
| 8          | FollowerLite | 0.3855     | 1            | 6              | 166      | 0.002786     | 1        | 1   | 60.515625 | 1.8547006590060846 |
| 8          | FollowerLite | 0.4706     | 1            | 7              | 136      | 0.002878     | 1        | 1   | 62.578125 | 1.6341293890054658 |
| 8          | FollowerLite | 0.3855     | 1            | 7              | 166      | 0.002944     | 1        | 1   | 54.234375 | 1.6484339929984344 |
| 8          | FollowerLite | 0.3497     | 1            | 10             | 183      | 0.003292     | 1        | 1   | 65.671875 | 2.176285248999193  |
| 8          | FollowerLite | 0.3855     | 1            | 19             | 166      | 0.003396     | 1        | 1   | 78.59375  | 2.099230728999828  |
| 8          | FollowerLite | 0.4604     | 1            | 4              | 139      | 0.002659     | 1        | 1   | 62.625    | 1.3358296740061633 |
| 8          | FollowerLite | 0.1509     | 1            | 16             | 424      | 0.002394     | 1        | 1   | 92.5      | 2.6079205630203433 |
| 8          | FollowerLite | 0.5517     | 1            | 8              | 116      | 0.002676     | 1        | 1   | 51.328125 | 1.1865800320065318 |
| 8          | FollowerLite | 0.4961     | 1            | 3              | 129      | 0.003123     | 1        | 1   | 56.203125 | 1.4759262969964766 |
| 16         | FollowerLite | 0.4444     | 1            | 4              | 144      | 0.003032     | 1        | 1   | 49.140625 | 1.6278657250013566 |
| 16         | FollowerLite | 0.6465     | 1            | 5              | 99       | 0.002894     | 1        | 1   | 50.671875 | 1.1754591010003423 |
| 16         | FollowerLite | 0.4776     | 1            | 5              | 134      | 0.003085     | 1        | 1   | 50.90625  | 1.4995861679981317 |
| 16         | FollowerLite | 0.5424     | 1            | 3              | 118      | 0.003208     | 1        | 1   | 50.71875  | 1.625744076999581  |
| 16         | FollowerLite | 0.6275     | 1            | 9              | 102      | 0.003448     | 1        | 1   | 52.578125 | 1.4912773390024086 |
| 16         | FollowerLite | 0.2261     | 1            | 10             | 283      | 0.003147     | 1        | 1   | 94.015625 | 2.4038107499936814 |
| 16         | FollowerLite | 0.4812     | 1            | 6              | 133      | 0.003304     | 1        | 1   | 63.78125  | 1.6465084539931922 |
| 16         | FollowerLite | 0.32       | 1            | 9              | 200      | 0.003423     | 1        | 1   | 73.828125 | 2.090250467995247  |
| 16         | FollowerLite | 0.1793     | 1            | 7              | 357      | 0.00281      | 1        | 1   | 65.375    | 2.8013803970006848 |
| 16         | FollowerLite | 0.5161     | 1            | 16             | 124      | 0.002897     | 1        | 1   | 53.71875  | 1.35826294499293   |
| 16         | FollowerLite | 0.5289     | 1            | 11             | 121      | 0.00304      | 1        | 1   | 54.1875   | 1.5391275739993944 |
| 16         | FollowerLite | 0.512      | 1            | 5              | 125      | 0.003246     | 1        | 1   | 54.703125 | 1.5677406330005397 |
| 16         | FollowerLite | 0.4        | 1            | 14             | 160      | 0.00294      | 1        | 1   | 72.640625 | 1.6170813379931133 |
| 16         | FollowerLite | 0.255      | 1            | 11             | 251      | 0.002774     | 1        | 1   | 66.921875 | 2.1614309860051435 |
| 16         | FollowerLite | 0.5        | 1            | 4              | 128      | 0.002967     | 1        | 1   | 59.71875  | 1.4466942489998473 |
| 16         | FollowerLite | 0.2344     | 1            | 11             | 273      | 0.003035     | 1        | 1   | 83.21875  | 2.4321700070076986 |
| 16         | FollowerLite | 0.287      | 1            | 27             | 223      | 0.003111     | 1        | 1   | 79        | 2.02621122500841   |
| 16         | FollowerLite | 0.4848     | 1            | 7              | 132      | 0.003207     | 1        | 1   | 55.875    | 1.5174760490008339 |
| 16         | FollowerLite | 0.3459     | 1            | 14             | 185      | 0.003022     | 1        | 1   | 65.984375 | 1.7407687880022422 |
| 16         | FollowerLite | 0.3616     | 1            | 19             | 177      | 0.003124     | 1        | 1   | 76.875    | 1.7285779059966444 |
| 32         | FollowerLite | 0.5872     | 1            | 11             | 109      | 0.003093     | 1        | 1   | 49.765625 | 1.4121793540007275 |
| 32         | FollowerLite | 0.4211     | 1            | 5              | 152      | 0.002936     | 1        | 1   | 51.96875  | 1.7613308739955755 |
| 32         | FollowerLite | 0.4414     | 1            | 5              | 145      | 0.003123     | 1        | 1   | 62.15625  | 1.6541548800068995 |
| 32         | FollowerLite | 0.2025     | 1            | 19             | 316      | 0.002907     | 1        | 1   | 76.046875 | 2.524101779013108  |
| 32         | FollowerLite | 0.4354     | 1            | 6              | 147      | 0.003103     | 1        | 1   | 55.90625  | 1.6721316049925008 |
| 32         | FollowerLite | 0.4812     | 1            | 7              | 133      | 0.003        | 1        | 1   | 51.015625 | 1.60466684599578   |
| 32         | FollowerLite | 0.2025     | 1            | 13             | 316      | 0.002875     | 1        | 1   | 71.75     | 2.5433407410037034 |
| 32         | FollowerLite | 0.4961     | 1            | 16             | 129      | 0.003262     | 1        | 1   | 57.15625  | 1.4844246459942951 |
| 32         | FollowerLite | 0.4129     | 1            | 6              | 155      | 0.003036     | 1        | 1   | 61.8125   | 1.6689444599996932 |
| 32         | FollowerLite | 0.4812     | 1            | 7              | 133      | 0.003065     | 1        | 1   | 58.6875   | 1.5825714229995356 |
| 32         | FollowerLite | 0.5926     | 1            | 6              | 108      | 0.003318     | 1        | 1   | 54.21875  | 1.4568097270075668 |
| 32         | FollowerLite | 0.3005     | 1            | 8              | 213      | 0.002854     | 1        | 1   | 67.984375 | 1.831010605001211  |
| 32         | FollowerLite | 0.3765     | 1            | 13             | 170      | 0.002968     | 1        | 1   | 71.359375 | 1.6767316610021226 |
| 32         | FollowerLite | 0.4885     | 1            | 4              | 131      | 0.003169     | 1        | 1   | 46.84375  | 1.4890249530044457 |
| 32         | FollowerLite | 0.4776     | 1            | 5              | 134      | 0.00315      | 1        | 1   | 46.3125   | 1.5491115849927155 |
| 32         | FollowerLite | 0.3951     | 1            | 15             | 162      | 0.002993     | 1        | 1   | 56.984375 | 1.6152879169958396 |
| 32         | FollowerLite | 0.3975     | 1            | 5              | 161      | 0.002962     | 1        | 1   | 56.90625  | 1.590278501004832  |
| 32         | FollowerLite | 0.4848     | 1            | 9              | 132      | 0.00321      | 1        | 1   | 60.8125   | 1.5034427189948474 |
| 32         | FollowerLite | 0.4295     | 1            | 5              | 149      | 0.003186     | 1        | 1   | 58.140625 | 1.680585025002074  |
| 32         | FollowerLite | 0.3902     | 1            | 12             | 164      | 0.003173     | 1        | 1   | 75.34375  | 1.6377919240012488 |
| 64         | FollowerLite | 0.2922     | 1            | 9              | 219      | 0.002986     | 1        | 1   | 53.9375   | 1.8310129180044896 |
| 64         | FollowerLite | 0.3497     | 1            | 11             | 183      | 0.002901     | 1        | 1   | 68.828125 | 1.706027463997998  |
| 64         | FollowerLite | 0.5872     | 1            | 11             | 109      | 0.00316      | 1        | 1   | 53.078125 | 1.2034155439987444 |
| 64         | FollowerLite | 0.5        | 1            | 12             | 128      | 0.003203     | 1        | 1   | 61.53125  | 1.401484280007935  |
| 64         | FollowerLite | 0.4211     | 1            | 15             | 152      | 0.002984     | 1        | 1   | 72.03125  | 1.457023787003891  |
| 64         | FollowerLite | 0.4384     | 1            | 8              | 146      | 0.003091     | 1        | 1   | 58.8125   | 1.5097780269989016 |
| 64         | FollowerLite | 0.3975     | 1            | 12             | 161      | 0.002979     | 1        | 1   | 58.109375 | 1.7289255240002603 |
| 64         | FollowerLite | 0.4672     | 1            | 11             | 137      | 0.003204     | 1        | 1   | 62.578125 | 1.4512465649995647 |
| 64         | FollowerLite | 0.6275     | 1            | 7              | 102      | 0.002774     | 1        | 1   | 48.171875 | 1.3070319120015483 |
| 64         | FollowerLite | 0.2689     | 1            | 14             | 238      | 0.00286      | 1        | 1   | 72.953125 | 2.0816746599994076 |
| 64         | FollowerLite | 0.5714     | 1            | 5              | 112      | 0.003055     | 1        | 1   | 64.359375 | 1.4177844500018182 |
| 64         | FollowerLite | 0.4672     | 1            | 6              | 137      | 0.002832     | 1        | 1   | 56.890625 | 1.5151246479981637 |
| 64         | FollowerLite | 0.3556     | 1            | 6              | 180      | 0.002963     | 1        | 1   | 59.859375 | 1.858276219995787  |
| 64         | FollowerLite | 0.3404     | 1            | 17             | 188      | 0.002993     | 1        | 1   | 70.8125   | 1.75373421299264   |
| 64         | FollowerLite | 0.3787     | 1            | 10             | 169      | 0.00297      | 1        | 1   | 63.421875 | 1.690638371010209  |
| 64         | FollowerLite | 0.5039     | 1            | 8              | 127      | 0.002846     | 1        | 1   | 51.375    | 1.5230435540015606 |
| 64         | FollowerLite | 0.4267     | 1            | 10             | 150      | 0.002928     | 1        | 1   | 60.328125 | 1.5717148299954715 |
| 64         | FollowerLite | 0.4051     | 1            | 6              | 158      | 0.002898     | 1        | 1   | 63.625    | 1.6629134509976211 |
| 64         | FollowerLite | 0.3497     | 1            | 18             | 183      | 0.003164     | 1        | 1   | 65.3125   | 1.7141343469993444 |
| 64         | FollowerLite | 0.27       | 1            | 10             | 237      | 0.003094     | 1        | 1   | 74.234375 | 2.060323348997372  |
| 128        | FollowerLite | 0.3478     | 1            | 15             | 184      | 0.003123     | 1        | 1   | 61.359375 | 1.803160749993367  |
| 128        | FollowerLite | 0.4051     | 1            | 7              | 158      | 0.003291     | 1        | 1   | 60.6875   | 1.7341091920025065 |
| 128        | FollowerLite | 0.5378     | 1            | 8              | 119      | 0.00327      | 1        | 1   | 53.734375 | 1.3737785519988392 |
| 128        | FollowerLite | 0.381      | 1            | 5              | 168      | 0.003235     | 1        | 1   | 66.953125 | 1.7358586270011074 |
| 128        | FollowerLite | 0.5        | 1            | 4              | 128      | 0.003164     | 1        | 1   | 53.953125 | 1.476670942997771  |
| 128        | FollowerLite | 0.257      | 1            | 18             | 249      | 0.003145     | 1        | 1   | 67.40625  | 1.965434836998611  |
| 128        | FollowerLite | 0.4672     | 1            | 18             | 137      | 0.003175     | 1        | 1   | 63.578125 | 1.4150600000075428 |
| 128        | FollowerLite | 0.4604     | 1            | 12             | 139      | 0.002766     | 1        | 1   | 55.0625   | 1.4804338990006727 |
| 128        | FollowerLite | 0.5424     | 1            | 3              | 118      | 0.003381     | 1        | 1   | 52.125    | 1.3984264120035732 |
| 128        | FollowerLite | 0.4103     | 1            | 9              | 156      | 0.00301      | 1        | 1   | 66.9375   | 1.6873004280032546 |
| 128        | FollowerLite | 0.3265     | 1            | 23             | 196      | 0.002903     | 1        | 1   | 64.453125 | 1.8043894809979975 |
| 128        | FollowerLite | 0.2353     | 1            | 12             | 272      | 0.002915     | 1        | 1   | 77.734375 | 2.2374504900062675 |
| 128        | FollowerLite | 0.5664     | 1            | 7              | 113      | 0.003268     | 1        | 1   | 54.796875 | 1.5458056420056892 |
| 128        | FollowerLite | 0.5203     | 1            | 7              | 123      | 0.003175     | 1        | 1   | 50.84375  | 1.5340545140024915 |
| 128        | FollowerLite | 0.3556     | 1            | 16             | 180      | 0.003178     | 1        | 1   | 78.171875 | 1.8024380249971728 |
| 128        | FollowerLite | 0.5        | 1            | 11             | 128      | 0.00328      | 1        | 1   | 59.46875  | 1.5187711579992538 |
| 128        | FollowerLite | 0.3721     | 1            | 11             | 172      | 0.003005     | 1        | 1   | 66.578125 | 1.6378618650078351 |
| 128        | FollowerLite | 0.5        | 1            | 7              | 128      | 0.003344     | 1        | 1   | 52.515625 | 1.6300187239967272 |
| 128        | FollowerLite | 0.3536     | 1            | 10             | 181      | 0.003166     | 1        | 1   | 73.3125   | 1.6862965530062866 |
| 128        | FollowerLite | 0.3575     | 1            | 14             | 179      | 0.003286     | 1        | 1   | 69.890625 | 1.7222201160020632 |


