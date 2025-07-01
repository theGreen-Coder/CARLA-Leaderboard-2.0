#  Bench2Drive in Carla Garage

## Goal

<span style="color:red">TODO</span> Explain what is Carla Garage, and why is it better than what we have (CIL++ framework), point to the original README

<span style="color:red">TODO</span> The goal of this repo/readme is to to make a really starter kit: explain how to use and adapt Garage 2 to train and asses your driving model, what are the best practices and why (references to papers).

<span style="color:red">TODO</span> for the moment we will just address how to evaluate with the Bench2drive benchmark, with TF++ and (hopefully) CIL++


## Why evaluate with Bench2Drive ?

Bench2Drive -> Bench2Drive is an extension of the official CARLA Leaderboard 2.0 and evaluates 220 *short*, safety‑critical routes instead of the long, variable routes used in earlier challenges. 

<span style="color:red">TODO</span> reasons from the abstract and body of the B2D paper and the paper of biases in datasets


## Background

- Carla leaderboard 2.0
    - https://leaderboard.carla.org/get_started_v2_0/ **must read**
    - [Tackling CARLA Leaderboard 2.0 with End-to-End Imitation Learning](https://kashyap7x.github.io/assets/pdf/students/Zimmerlin2024.pdf). MSc thesis by Julian Zimmerlin at U. Tubingen, 2024
    - [Hidden Biases of End-to-End Driving Datasets](https://arxiv.org/abs/2412.09602). Julian Zimmerlin, Jens Beißwenger, Bernhard Jaeger, Andreas Geiger, Kashyap Chitta. CVPR 2024 Workshop on Foundation Models for Autonomous Systems. A summary of previous reference.

- [Hidden Biases of End-to-End Driving Models](https://arxiv.org/abs/2306.07957)
Bernhard Jaeger, Kashyap Chitta, Andreas Geiger. ICCV'23. A companion paper to the previous one, with some discussion on TransFuser++.

- [Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving](https://github.com/Thinklab-SJTU/Bench2Drive). Xiaosong Jia, Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, Junchi Yan. NeurIPS 2024 Datasets and Benchmarks Track. 

## Evaluating TF++ on Bench2Drive


- The scripts involved are, in order of invocation 
    1. [`Bench2Drive/leaderboard/scripts/run_evaluation_tf++.sh`](./Bench2Drive/leaderboard/scripts/run_evaluation_tf++.sh)<br>
    Sets a number of important environment variables like 
        - BASE_ROUTES the .xml file where the 220 routes of Bench2Drive are specified, each consisting of route id, list of waypoints, scenario (one per route, belonging to the set of 38 scenarios defined in Leaderboard 2, like *HighwayExit*, *AccidentTwoWays*, *YieldToEmergencyVehicle* etc.) and list of weathers, see file [`Bench2Drive/leaderboard/data/bench2drive220.xml`](./Bench2Drive/leaderboard/data/bench2drive220.xml)
        - TEAM_AGENT this is the driving agent, like [`team_code/sensor_agent.py`](./team_code/sensor_agent.py), is a "Carla agent", see [here](https://leaderboard.carla.org/get_started_v2_0/#3-creating-your-own-autonomous-agent). The actual agent is *not* directly the driving agent, for instance a Python class "TransFuserPlusPlus", it's more complicated, see [below](#evaluating_tf++)
        - TASK_NUM, number of parts in which we want to partition all the routes, typically one per GPU, related to TASK_LIST, see below
        - With `GPU_RANK_LIST=(0 1 2 3 4 5 6 7)` and `TASK_LIST=(0 1 2 3 4 5 6 7)` the script launches a Carla server in a different port (so there's no clashes) and evaluates one of the 8 parts (assuming we have previously set TASK_NUM=8) in each GPU. This is done by the next script.        
        - CHECKPOINT_ENDPOINT, the directory where results = metrics are saved as .json files. This path is made of other variables.
        - TEAM_CONFIG is the directory where the checkpoints of the model of the agent are, plus an args.txt and a config.json. See for instance [`pretrained_models/all_towns`](./pretrained_models/all_towns).
        - These two last variables and some other to be passed to run_evaluation.sh are explained in the Leaderboard [documentation](https://leaderboard.carla.org/get_started_v2_1/#22-understanding-the-leaderboard-components)


    1. [`Bench2Drive/leaderboard/scripts/run_evaluation.sh`](./Bench2Drive/leaderboard/scripts/run_evaluation.sh)<br>
    Simply sets 4 new variables and forwards the variables defined in  run_evaluation_tf++.sh to the next Python script, so this could be already done inside the loop of run_evaluation_tf++.sh. However, they must have kept this script because it's the same as [`run_leaderboard.sh`](leaderboard/run_leaderboard.sh) provided in Leaderboard 2, that also calls leaderboard_evaluator.py.

    1. [`Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py`](./Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py)<br>
    Contains the LeaderboardEvaluator class that according to the comment, is "the main class of the leaderboard. Everything is handled from here, from parsing the given files, to preparing the simulation, to running the route".

    1. [`team_code/sensor_agent.py`](./team_code/sensor_agent.py)<br>
    According to the Leaderboard 2 docs, in section [3. Creating your own Autonomous Agent](https://leaderboard.carla.org/get_started_v2_1/#3-creating-your-own-autonomous-agent), one has to create an agent class that inherits from the ``leaderboard.autoagents.autonomous_agent.AutonomousAgent`` class and override some methods like ``setup()`` and ``sensors()``. It is in this later than cameras, lidar etc are defined. In the case of Transfuser, the values of the many parameters of the sensors and model are in a file ``config.json`` to be found in CHECKPOINT_ENDPOINT directory. The full path to this file reaches the agent class through parameter ``path_to_conf_file`` of the ``setup()`` method. There's an [`agent_wrapper.py`](Bench2Drive/leaderboard/leaderboard/autoagents/agent_wrapper.py) that sets some limits to the sensors used, for a fair competition of methods. <br><br>


- [`split_xml.py`](./Bench2Drive/tools/split_xml.py) lets you shard the workload so that eight GPUs can each work on a quarter of the routes in parallel (or any ratio you set). The bash loop in your earlier script pairs each <splitID> with a distinct port range (30000 + 150×i) and GPU ID, which prevents port clashes between CARLA servers.
- Most times `GPU_RANK_LIST` and `TASK_LIST` should be the same since optimally you might one task for each GPU 

    - Input Routes: `Bench2Drive/leaderboard/data/bench2drive220.xml` has all the data. When running ```split_xml.py```, it generates the following files:

        ```
        bench2drive220_<splitID>_<agentKey>_traj.xml
        └───────┬──────┘ └─┬─┘  └────┬────┘
        master route set   │         │
           220 routes      │         │
                           │         └── short tag of the agent/planner you evaluate
                           └────────── zero‑based index after XML splitting

        ```
    
        - By default the repo already has a split done for tfpp: `bench2drive220_0_tfpp_traj.xml`, `bench2drive220_1_tfpp_traj.xml`, `bench2drive220_2_tfpp_traj.xml`, `bench2drive220_3_tfpp_traj.xml`. Which have 55 routes each (hence 55*4=220 routes). A split of task by (0 1 2 3 4 5 6 7) would split the routes with (28 routes * 4 files + 27 routes * 4 files = 220 routes).

    - Output in [`Bench2Drive/tfpp_b2d_traj`](./Bench2Drive/tfpp_b2d_traj/) and [`merge_route_json.py`](./Bench2Drive/tools/merge_route_json.py) merges all json files generated by the splits into [`merged_ability`](./Bench2Drive/tfpp_b2d_traj/merged_ability.json) and [`merged.json`](./Bench2Drive/tfpp_b2d_traj/merged.json).
    - [`scenario_runner`](./Bench2Drive/scenario_runner/) is CARLA’s companion module that defines traffic scenarios (via a Python API or the OpenSCENARIO standard) and drives the simulator through them so you can test, train, or benchmark an autonomous‑driving agent. It ships with ready‑made scenarios, an execution engine, metrics collection and helper scripts (e.g., no‑rendering mode, manual control).
    - To see the terminal output of the benchmark run this command `cat ${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.log`, it's this file over [here](./Bench2Drive/leaderboard/data/bench2drive220_0_tfpp_traj.log).


<a id="evaluating_tf++"></a>
## Evaluating TransFuser++ on Bench2Drive

<span style="color:red">TODO</span> primer explicar els arxius i mecanisme per TF++ i a la següent seccion que hem canviat per fer-ho amb el cil++, si ho aconseguim

## Evaluating CIL++ on Bench2Drive
- [`sensor_agent.py`](./team_code/sensor_agent.py) seems to be loading the model by calling [`model.py`](./team_code/model.py). Therefore, I will be changing this to correctly load CIL++ (will have to create a new file with the class CIL++).