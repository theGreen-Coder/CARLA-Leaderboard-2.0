### 17th of July
- Discovered GPU 1 on the 141 doesn't work with CARLA
- Tried adding retry logic to leaderboard_evaluator.py (will see how it goes, I started evaluation)
- Tried commiting everything so everyone can reproduce this
- Redirected all input to log files (to actually see what happens)
- Overnight experiment evaluating Bench2Drive
    - eval_bench2drive220_0.json - Completed
    - eval_bench2drive220_1.json - 18/37
    - eval_bench2drive220_2.json - 20/37
    - eval_bench2drive220_3.json - Completed
    - eval_bench2drive220_4.json - Completed
    - eval_bench2drive220_5.json - Completed

### 18th of July
- Overnight experiment evaluating Bench2Drive failed
    - terminate called after throwing an instance of 'carla::client::TimeoutException' what():  time-out of 600000ms while waiting for the simulator, make sure the simulator is ready and connected to localhost:30300
    - I've read online and there doesn't seem to be a way to catch this error and restart CARLA (will look more into it)


To DO:
- Git clone and see if it works
- Evaluate CIL attention
- Add Other evaluators??