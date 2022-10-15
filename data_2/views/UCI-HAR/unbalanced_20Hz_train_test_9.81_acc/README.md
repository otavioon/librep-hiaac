# Unbalanced UCI-HAR View Resampled to 20Hz withou gravity

This view contain only the train and test files for [UCI-HAR dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones#) (70% samples train and 30% test). The data was spllited into 3s windows and was resampled to 20Hz using the [FFT method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html#scipy.signal.resample). The accelerometer meansure is in m/s² and without gravity.

## Activity codes
- 1: walking (413 train, 0 validation, 168 test) 
- 2: walking upstairs (334 train, 0 validation, 157 test) 
- 3: walking downstairs (315 train, 0 validation, 141 test) 
- 4: sitting (475 train, 0 validation, 165 test) 
- 5: standing (455 train, 0 validation, 187 test) 
 

## Standartized activity codes
- 0: sit (475 train, 0 validation, 165 test) 
- 1: stand (455 train, 0 validation, 187 test) 
- 2: walk (413 train, 0 validation, 168 test) 
- 3: stair up (334 train, 0 validation, 157 test) 
- 4: stair down (315 train, 0 validation, 141 test) 
      


