# Balanced MotionSense View Resampled to 20Hz - Multiplied acc by 9.81m/s²

This is a view from [MotionSense] that was spllited into 3s windows and was resampled to 20Hz using the [FFT method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html#scipy.signal.resample). 

The data was first splitted in three sets: train, validation and test. Each one with the following proportions:
- Train: 70% of samples
- Validation: 10% of samples
- Test: 20% of samples

After splits, the datasets were balanced in relation to the activity code column, that is, each subset have the same number of activitiy samples.

**NOTE**: Each subset contain samples from distinct users, that is, samples of one user belongs exclusivelly to one of three subsets.

## Activity codes
- 0: downstairs (579 train, 103 validation, 156 test) 
- 1: upstairs (579 train, 103 validation, 156 test) 
- 2: sitting (579 train, 103 validation, 156 test) 
- 3: standing (579 train, 103 validation, 156 test) 
- 4: walking (579 train, 103 validation, 156 test) 
- 5: jogging (579 train, 103 validation, 156 test) 
 

## Standartized activity codes
- 0: sit (579 train, 103 validation, 156 test) 
- 1: stand (579 train, 103 validation, 156 test) 
- 2: walk (579 train, 103 validation, 156 test) 
- 3: stair up (579 train, 103 validation, 156 test) 
- 4: stair down (579 train, 103 validation, 156 test) 
- 5: run (579 train, 103 validation, 156 test) 
      


