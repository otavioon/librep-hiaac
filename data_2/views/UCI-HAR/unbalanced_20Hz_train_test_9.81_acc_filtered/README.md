# Unbalanced UCI-HAR View Resampled to 20Hz without gravity. 
            
The data used was the samples with gravity by autors.

The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used and the signal filtered was subtracted from the original signal.

This view contain only the train and test files for [UCI-HAR dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones#) (70% samples train and 30% test). The data was spllited into 3s windows and was resampled to 20Hz using the [FFT method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html#scipy.signal.resample). The accelerometer meansure is in m/s² and without gravity.

## Activity codes
- 1: walking (506 train, 0 validation, 204 test) 
- 2: walking upstairs (439 train, 0 validation, 189 test) 
- 3: walking downstairs (395 train, 0 validation, 173 test) 
- 4: sitting (544 train, 0 validation, 204 test) 
- 5: standing (575 train, 0 validation, 227 test) 
 

## Standartized activity codes
- 0: sit (544 train, 0 validation, 204 test) 
- 1: stand (575 train, 0 validation, 227 test) 
- 2: walk (506 train, 0 validation, 204 test) 
- 3: stair up (439 train, 0 validation, 189 test) 
- 4: stair down (395 train, 0 validation, 173 test) 
      


